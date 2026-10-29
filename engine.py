"""Shared prediction engine used by both the Streamlit app and the daily pipeline."""
import json, os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
EV_THRESHOLD = 0.05  # flag a bet when expected value is above 5%


def _p(name):
    return os.path.join(HERE, name)


class Engine:
    def __init__(self):
        self.features = json.load(open(_p("features.json")))
        self.defaults = json.load(open(_p("feature_defaults.json")))
        self.w = json.load(open(_p("blend.json")))
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(_p("clf_3way.json"))
        self.lgb = joblib.load(_p("lgb_3way.joblib"))
        self.cat = CatBoostClassifier()
        self.cat.load_model(_p("cat_3way.cbm"))
        self.reg_home = xgb.XGBRegressor()
        self.reg_home.load_model(_p("reg_home_ft.json"))
        self.reg_away = xgb.XGBRegressor()
        self.reg_away.load_model(_p("reg_away_ft.json"))
        self.reg_home_ht = xgb.XGBRegressor()
        self.reg_home_ht.load_model(_p("reg_home_ht.json"))
        self.reg_away_ht = xgb.XGBRegressor()
        self.reg_away_ht.load_model(_p("reg_away_ht.json"))

    def build_row(self, home_elo, away_elo, form5_home, form5_away,
                  odd_home, odd_draw, odd_away, over25, under25=None):
        r = dict(self.defaults)  # start from typical values, then overwrite what we know
        if under25 is None:
            rest = 1.05 - 1 / over25
            under25 = 1 / rest if rest > 0.05 else r.get("Under25", 1.9)
        r.update(
            HomeElo=home_elo, AwayElo=away_elo,
            Form5Home=form5_home, Form5Away=form5_away,
            Form3Home=round(form5_home * 0.6, 1), Form3Away=round(form5_away * 0.6, 1),
            OddHome=odd_home, OddDraw=odd_draw, OddAway=odd_away,
            MaxHome=odd_home, MaxDraw=odd_draw, MaxAway=odd_away,
            Over25=over25, Under25=under25,
            EloDifference=home_elo - away_elo,
            Form5Difference=form5_home - form5_away,
            BookmakerMargin=1 / odd_home + 1 / odd_draw + 1 / odd_away - 1,
        )
        return pd.DataFrame([[r[f] for f in self.features]], columns=self.features)

    def predict(self, X):
        probs = (self.w["xgb"] * self.xgb.predict_proba(X)
                 + self.w["lgb"] * self.lgb.predict_proba(X)
                 + self.w["cat"] * self.cat.predict_proba(X))[0]
        p_draw, p_home, p_away = (float(v) for v in probs)  # class order: 0=Draw, 1=Home, 2=Away
        lam_h = max(0.05, float(self.reg_home.predict(X)[0]))
        lam_a = max(0.05, float(self.reg_away.predict(X)[0]))
        g = np.arange(7)
        M = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
        M /= M.sum()
        I, J = np.meshgrid(g, g, indexing="ij")
        # Best score that agrees with the predicted result (home win / draw / away win)
        mask = max((p_home, I > J), (p_draw, I == J), (p_away, I < J), key=lambda t: t[0])[1]
        i, j = np.unravel_index(np.where(mask, M, -1).argmax(), M.shape)
        # The 5 most likely scores overall
        top = [(f"{a}-{b}", float(M[a, b]))
               for a, b in (np.unravel_index(k, M.shape) for k in np.argsort(M, axis=None)[::-1][:5])]
        # Half-time: same idea with the half-time goal models
        lh_ht = max(0.05, float(self.reg_home_ht.predict(X)[0]))
        la_ht = max(0.05, float(self.reg_away_ht.predict(X)[0]))
        H = np.outer(poisson.pmf(g, lh_ht), poisson.pmf(g, la_ht))
        H /= H.sum()
        return dict(p_home=p_home, p_draw=p_draw, p_away=p_away,
                    xg_home=lam_h, xg_away=lam_a, score=f"{i}-{j}",
                    score_prob=float(M[i, j]), top_scores=top, p_over25=float(M[(I + J) > 2].sum()),
                    p_btts=float(M[1:, 1:].sum()),
                    p_ht_home=float(H[I > J].sum()), p_ht_draw=float(H[I == J].sum()),
                    p_ht_away=float(H[I < J].sum()))

    def run(self, **kw):
        return self.predict(self.build_row(**kw))


def value_table(res, odd_home, odd_draw, odd_away, over25):
    """Expected value per market: probability x odds - 1."""
    rows = []
    for name, p, odd in [("Home win", res["p_home"], odd_home),
                         ("Draw", res["p_draw"], odd_draw),
                         ("Away win", res["p_away"], odd_away),
                         ("Over 2.5 goals", res["p_over25"], over25)]:
        ev = p * odd - 1
        rows.append({"Market": name, "Model prob": p, "Bookmaker odds": odd,
                     "Fair odds": 1 / p if p > 0 else np.inf,
                     "EV": ev, "Value bet": ev > EV_THRESHOLD})
    return pd.DataFrame(rows)
