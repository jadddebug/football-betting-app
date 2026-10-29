"""Daily job: update team ratings from new results, score upcoming fixtures, write todays_value_bets.csv.
Data comes free from football-data.co.uk (no API key needed)."""
import datetime as dt

import pandas as pd

import football_data as fd
import ratings
from engine import Engine, value_table

OUT = "todays_value_bets.csv"
COLUMNS = ["Date", "League", "Home", "Away", "P_Home", "P_Draw", "P_Away", "xG_Home", "xG_Away",
           "Score", "P_Over25", "P_BTTS", "P_HT_Home", "P_HT_Draw", "P_HT_Away", "Odd_Home", "Odd_Draw",
           "Odd_Away", "Odd_Over25", "BestMarket", "BestEV", "ValueBet", "RatingsFound"]


def update_ratings(R):
    """Move Elo + form using every finished match since the last update."""
    last = dt.date.fromisoformat(R.get("updated_through") or R["base_date"])
    end = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    if last >= end:
        return 0
    games = fd.new_results(last, end)
    for date, home, away, hg, ag, div in games:
        ratings.apply_result(R["teams"], home, away, hg, ag, date, div)
    R["updated_through"] = str(end)
    return len(games)


def rating_of(name, teams):
    t = teams.get(ratings.resolve(name, teams))
    if t:
        return t["elo"], ratings.form_points(t["results"]), True
    return 1500.0, 7.0, False  # unknown team: average defaults


def main():
    eng = Engine()
    R = ratings.load_or_build()
    n = update_ratings(R)
    ratings.save(R)
    print(f"Ratings updated with {n} new results (through {R['updated_through']})")

    today = dt.datetime.now(dt.timezone.utc).date()
    rows = []
    for fx in fd.upcoming(today, days=3):
        h_elo, h_form, h_ok = rating_of(fx["home_team"], R["teams"])
        a_elo, a_form, a_ok = rating_of(fx["away_team"], R["teams"])
        res = eng.run(home_elo=h_elo, away_elo=a_elo, form5_home=h_form, form5_away=a_form,
                      odd_home=fx["home"], odd_draw=fx["draw"], odd_away=fx["away"], over25=fx["over25"])
        vt = value_table(res, fx["home"], fx["draw"], fx["away"], fx["over25"])
        best = vt.loc[vt["EV"].idxmax()]
        rows.append({
            "Date": fx["date"], "League": ratings.LEAGUE_NAMES.get(fx["div"], fx["div"]),
            "Home": fx["home_team"], "Away": fx["away_team"],
            "P_Home": round(res["p_home"], 3), "P_Draw": round(res["p_draw"], 3),
            "P_Away": round(res["p_away"], 3),
            "xG_Home": round(res["xg_home"], 2), "xG_Away": round(res["xg_away"], 2),
            "Score": res["score"], "P_Over25": round(res["p_over25"], 3),
            "P_BTTS": round(res["p_btts"], 3), "P_HT_Home": round(res["p_ht_home"], 3),
            "P_HT_Draw": round(res["p_ht_draw"], 3), "P_HT_Away": round(res["p_ht_away"], 3),
            "Odd_Home": fx["home"], "Odd_Draw": fx["draw"], "Odd_Away": fx["away"],
            "Odd_Over25": fx["over25"],
            "BestMarket": best["Market"], "BestEV": round(best["EV"], 3),
            "ValueBet": bool(best["Value bet"]), "RatingsFound": h_ok and a_ok})
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} fixtures ({int(out['ValueBet'].sum()) if len(out) else 0} value bets) to {OUT}")


if __name__ == "__main__":
    main()
