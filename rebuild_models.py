"""
Rebuilds and saves all the football models from matches.csv.
Run it from the folder that contains matches.csv:   python rebuild_models.py

It trains each model twice, once with the original stats and once with the extra stats
(head-to-head + goals scored/conceded + home/away form), prints both results, and keeps
whichever set did better on matches the model has not seen.

Needs:  pip install xgboost lightgbm catboost scikit-learn pandas scipy joblib
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

import features as F

WEIGHTS = {"xgb": 0.79, "lgb": 0.11, "cat": 0.10}  # blend: XGBoost, LightGBM, CatBoost

# 1. Load and tidy
df = pd.read_csv("matches.csv", low_memory=False)
df["MatchDate"] = pd.to_datetime(df["MatchDate"], format="mixed", errors="coerce")
RAW = [c for c in ["HomeElo", "AwayElo", "Form3Home", "Form5Home", "Form3Away", "Form5Away",
                   "OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway",
                   "Over25", "Under25", "HandiSize", "C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]
       if c in df.columns]
for c in RAW + [c for c in ("FTHome", "FTAway", "HTHome", "HTAway") if c in df.columns]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"])
df = df.sort_values("MatchDate", kind="stable").reset_index(drop=True)

df["EloDifference"] = df["HomeElo"] - df["AwayElo"]
df["Form5Difference"] = df["Form5Home"] - df["Form5Away"]
df["BookmakerMargin"] = (1 / df["OddHome"] + 1 / df["OddDraw"] + 1 / df["OddAway"]) - 1
BASE = RAW + ["EloDifference", "Form5Difference", "BookmakerMargin"]

# 2. Head-to-head and goal stats, using only matches played before each one
df = F.add_extra_features(df)
EXTENDED = BASE + F.EXTRA

df = df.dropna(subset=BASE + ["FTResult"]).copy()
df["Target"] = df["FTResult"].map({"D": 0, "H": 1, "A": 2})
df = df.dropna(subset=["Target"])
print(f"Usable matches: {len(df):,}")

# 3. Train on matches before 2024, test on 2024 onward
cut = pd.Timestamp("2024-01-01")
train, test = df[df["MatchDate"] < cut], df[df["MatchDate"] >= cut]
print(f"Training: {len(train):,}  |  Testing: {len(test):,}")
if len(train) == 0 or len(test) == 0:
    raise SystemExit("Split came out empty - check the MatchDate column in matches.csv")
ytr, yte = train["Target"].astype(int), test["Target"].astype(int)


def fit_outcome_models(feats):
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8,
                                colsample_bytree=0.8, random_state=42, eval_metric="mlogloss")
    xgb_clf.fit(train[feats], ytr, sample_weight=compute_sample_weight("balanced", ytr))
    lgb_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
                                 random_state=42, verbose=-1).fit(train[feats], ytr)
    cat_clf = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.03,
                                 random_state=42, verbose=0).fit(train[feats], ytr)
    probs = (WEIGHTS["xgb"] * xgb_clf.predict_proba(test[feats])
             + WEIGHTS["lgb"] * lgb_clf.predict_proba(test[feats])
             + WEIGHTS["cat"] * cat_clf.predict_proba(test[feats]))
    acc_x = accuracy_score(yte, xgb_clf.predict(test[feats]))
    acc_b = accuracy_score(yte, np.argmax(probs, axis=1))
    return (xgb_clf, lgb_clf, cat_clf), acc_x, acc_b


def fit_goal_models(feats, home_col, away_col, data_train, data_test):
    def make():
        return xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.04, random_state=42)
    rh = make().fit(data_train[feats], data_train[home_col])
    ra = make().fit(data_train[feats], data_train[away_col])
    mae = (np.abs(rh.predict(data_test[feats]) - data_test[home_col]).mean()
           + np.abs(ra.predict(data_test[feats]) - data_test[away_col]).mean()) / 2
    return rh, ra, mae


# 4. Win / draw / loss: original stats vs original + extra stats
base_clf, bx, bb = fit_outcome_models(BASE)
ext_clf, ex, eb = fit_outcome_models(EXTENDED)
print(f"Win/draw/loss, original stats : XGBoost {bx*100:.2f}%  | blend {bb*100:.2f}%")
print(f"Win/draw/loss, with extra stats: XGBoost {ex*100:.2f}%  | blend {eb*100:.2f}%")
use_ext_clf = eb >= bb
clf_feats, (xgb_clf, lgb_clf, cat_clf) = (EXTENDED, ext_clf) if use_ext_clf else (BASE, base_clf)
print("-> keeping:", "extra stats" if use_ext_clf else "original stats", "for win/draw/loss")

# 5. Goals (average error in goals per team: lower is better)
b_h, b_a, b_mae = fit_goal_models(BASE, "FTHome", "FTAway", train, test)
e_h, e_a, e_mae = fit_goal_models(EXTENDED, "FTHome", "FTAway", train, test)
print(f"Goals, original stats : average error {b_mae:.3f}")
print(f"Goals, with extra stats: average error {e_mae:.3f}")
use_ext_goal = e_mae <= b_mae
goal_feats, (reg_home, reg_away) = (EXTENDED, (e_h, e_a)) if use_ext_goal else (BASE, (b_h, b_a))
print("-> keeping:", "extra stats" if use_ext_goal else "original stats", "for goals")

# 6. Half-time goal models (same stats as full-time goals)
ht_train = train.dropna(subset=["HTHome", "HTAway"])
ht_test = test.dropna(subset=["HTHome", "HTAway"])
reg_home_ht, reg_away_ht, _ = fit_goal_models(goal_feats, "HTHome", "HTAway", ht_train, ht_test)

# 7. Save everything
xgb_clf.save_model("clf_3way.json")
joblib.dump(lgb_clf, "lgb_3way.joblib")
cat_clf.save_model("cat_3way.cbm")
reg_home.save_model("reg_home_ft.json")
reg_away.save_model("reg_away_ft.json")
reg_home_ht.save_model("reg_home_ht.json")
reg_away_ht.save_model("reg_away_ht.json")
with open("blend.json", "w") as f:
    json.dump(WEIGHTS, f)
with open("features.json", "w") as f:
    json.dump({"all": EXTENDED, "clf": clf_feats, "goal": goal_feats}, f)
with open("feature_defaults.json", "w") as f:  # typical values for anything the app can't supply
    json.dump({k: float(v) for k, v in train[EXTENDED].median().items()}, f)
print("Saved all models + features.json. Your app can load these now.")
