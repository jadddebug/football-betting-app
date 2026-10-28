"""
Rebuilds and saves the 5 football models from matches.csv.
Run it from the folder that contains matches.csv:   python rebuild_models.py

Needs:  pip install xgboost lightgbm catboost scikit-learn pandas scipy joblib

Saves: clf_3way.json (XGBoost), lgb_3way.joblib (LightGBM), cat_3way.cbm (CatBoost),
       blend.json (blend weights), feature_defaults.json, reg_home_ft.json, reg_away_ft.json,
       reg_home_ht.json, reg_away_ht.json, features.json
"""
import json
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# 1. Load the data (low_memory=False avoids the mixed-type warning)
df = pd.read_csv("matches.csv", low_memory=False)

# 2. Dates: turn text into real dates, drop rows where that fails
df["MatchDate"] = pd.to_datetime(df["MatchDate"], format="mixed", errors="coerce")
df = df.dropna(subset=["MatchDate"]).sort_values("MatchDate").reset_index(drop=True)

# 3. Extra columns built from existing ones
df["EloDifference"] = df["HomeElo"] - df["AwayElo"]
df["Form5Difference"] = df["Form5Home"] - df["Form5Away"]
df["BookmakerMargin"] = (1 / df["OddHome"] + 1 / df["OddDraw"] + 1 / df["OddAway"]) - 1

# 4. A SHORT FIXED list of columns (this is what fixes the "0 matches" problem)
FEATURES = [
    "HomeElo", "AwayElo", "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "HandiSize",
    "C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB",
    "EloDifference", "Form5Difference", "BookmakerMargin",
]
FEATURES = [c for c in FEATURES if c in df.columns]
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")

# 5. Only keep rows that have what we need, and play-by-play results
needed = FEATURES + ["FTResult", "FTHome", "FTAway"]
df = df.dropna(subset=needed).copy()
df["Target"] = df["FTResult"].map({"D": 0, "H": 1, "A": 2})
df = df.dropna(subset=["Target"])
print(f"Usable matches: {len(df):,}")

# 6. Train on matches before 2024, test on 2024 onward
cut = pd.Timestamp("2024-01-01")
train, test = df[df["MatchDate"] < cut], df[df["MatchDate"] >= cut]
print(f"Training: {len(train):,}  |  Testing: {len(test):,}")
if len(train) == 0 or len(test) == 0:
    raise SystemExit("Split came out empty - check the MatchDate column in matches.csv")

# 7. Win/draw/loss model
clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric="mlogloss")
clf.fit(train[FEATURES], train["Target"].astype(int),
        sample_weight=compute_sample_weight("balanced", train["Target"].astype(int)))
acc = accuracy_score(test["Target"].astype(int), clf.predict(test[FEATURES]))
print(f"XGBoost alone, accuracy on unseen matches: {acc*100:.2f}%")

# 7b. LightGBM and CatBoost, then blend all three
y_train = train["Target"].astype(int)
y_test = test["Target"].astype(int)

lgb_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
                             random_state=42, verbose=-1)
lgb_clf.fit(train[FEATURES], y_train)

cat_clf = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.03,
                             random_state=42, verbose=0)
cat_clf.fit(train[FEATURES], y_train)

# Blend weights: XGBoost, LightGBM, CatBoost
WEIGHTS = {"xgb": 0.79, "lgb": 0.11, "cat": 0.10}
blend = (WEIGHTS["xgb"] * clf.predict_proba(test[FEATURES])
         + WEIGHTS["lgb"] * lgb_clf.predict_proba(test[FEATURES])
         + WEIGHTS["cat"] * cat_clf.predict_proba(test[FEATURES]))
blend_acc = accuracy_score(y_test, np.argmax(blend, axis=1))
print(f"Three-model blend, accuracy on unseen matches: {blend_acc*100:.2f}%")

# 8. Goal models (full-time and half-time)
def goal_model():
    return xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.04, random_state=42)

reg_home_ft = goal_model().fit(train[FEATURES], train["FTHome"])
reg_away_ft = goal_model().fit(train[FEATURES], train["FTAway"])

ht = train.dropna(subset=["HTHome", "HTAway"])
reg_home_ht = goal_model().fit(ht[FEATURES], ht["HTHome"])
reg_away_ht = goal_model().fit(ht[FEATURES], ht["HTAway"])

# 9. Save everything
clf.save_model("clf_3way.json")
joblib.dump(lgb_clf, "lgb_3way.joblib")
cat_clf.save_model("cat_3way.cbm")
with open("blend.json", "w") as f:
    json.dump(WEIGHTS, f)
reg_home_ft.save_model("reg_home_ft.json")
reg_away_ft.save_model("reg_away_ft.json")
reg_home_ht.save_model("reg_home_ht.json")
reg_away_ht.save_model("reg_away_ht.json")
with open("features.json", "w") as f:
    json.dump(FEATURES, f)
# Typical values for columns the app can't ask a person for (odds extras, etc.)
with open("feature_defaults.json", "w") as f:
    json.dump({k: float(v) for k, v in train[FEATURES].median().items()}, f)
print("Saved all models + blend.json + features.json. Your app can load these now.")
