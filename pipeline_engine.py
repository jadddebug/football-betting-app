"""Daily job: update team ratings from new results, score upcoming fixtures, write todays_value_bets.csv."""
import datetime as dt
import os
import sys

import pandas as pd
import requests

import ratings
from engine import Engine, value_table

KEY = os.environ.get("API_FOOTBALL_KEY")
BASE = "https://v3.football.api-sports.io"
LEAGUES = {39: ("Premier League", "E0"), 140: ("La Liga", "SP1"), 135: ("Serie A", "I1"),
           78: ("Bundesliga", "D1"), 61: ("Ligue 1", "F1")}
BOOKMAKER = 8  # Bet365
OUT = "todays_value_bets.csv"
COLUMNS = ["Date", "League", "Home", "Away", "P_Home", "P_Draw", "P_Away", "xG_Home", "xG_Away",
           "Score", "P_Over25", "P_BTTS", "P_HT_Home", "P_HT_Draw", "P_HT_Away", "Odd_Home", "Odd_Draw",
           "Odd_Away", "Odd_Over25", "BestMarket", "BestEV", "ValueBet", "RatingsFound"]


def api(path, **params):
    r = requests.get(BASE + path, headers={"x-apisports-key": KEY}, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):  # API-Football reports plan/limit problems here, with a normal 200 reply
        sys.exit(f"API-Football refused the request {path} {params}: {data['errors']}")
    return data


def season_of(d):
    return d.year if d.month >= 7 else d.year - 1


def update_ratings(R):
    """Fetch finished matches since the last update and move Elo + form accordingly."""
    last = dt.date.fromisoformat(R.get("updated_through") or R["base_date"])
    start = last + dt.timedelta(days=1)
    end = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    if start > end:
        return 0
    done, games = set(R.get("processed_ids", [])), {}
    for season in sorted({season_of(start), season_of(end)}):
        for lid, (_, code) in LEAGUES.items():
            data = api("/fixtures", league=lid, season=season, status="FT",
                       **{"from": str(start), "to": str(end)})
            print(f"  {lid} season {season}: {len(data['response'])} finished matches returned")
            for f in data["response"]:
                fid = f["fixture"]["id"]
                if fid in done or f["goals"]["home"] is None:
                    continue
                games[fid] = (f["fixture"]["date"][:10], f["teams"]["home"]["name"],
                              f["teams"]["away"]["name"], f["goals"]["home"], f["goals"]["away"], code)
    for fid, (date, home, away, hg, ag, code) in sorted(games.items(), key=lambda kv: kv[1][0]):
        teams = R["teams"]
        ratings.apply_result(teams, ratings.resolve(home, teams), ratings.resolve(away, teams), hg, ag, date, code)
    R["updated_through"] = str(end)
    R["processed_ids"] = (list(done) + list(games))[-3000:]
    return len(games)


def fetch_odds(league, season, date):
    """{fixture_id: {home, draw, away, over25}} from one bookmaker."""
    odds, page = {}, 1
    while True:
        data = api("/odds", league=league, season=season, date=date, bookmaker=BOOKMAKER, page=page)
        for item in data.get("response", []):
            o = {}
            for bk in item.get("bookmakers", [])[:1]:
                for bet in bk.get("bets", []):
                    vals = {v["value"]: float(v["odd"]) for v in bet["values"]}
                    if bet["id"] == 1:
                        o.update(home=vals.get("Home"), draw=vals.get("Draw"), away=vals.get("Away"))
                    elif bet["id"] == 5:
                        o["over25"] = vals.get("Over 2.5")
            odds[item["fixture"]["id"]] = o
        if page >= data.get("paging", {}).get("total", 1):
            return odds
        page += 1


def rating_of(name, teams):
    t = teams.get(ratings.resolve(name, teams))
    if t:
        return t["elo"], ratings.form_points(t["results"]), True
    return 1500.0, 7.0, False  # unknown team: average defaults


def main():
    if not KEY:
        sys.exit("API_FOOTBALL_KEY is not set")
    eng = Engine()
    R = ratings.load_or_build()
    n = update_ratings(R)
    ratings.save(R)
    print(f"Ratings updated with {n} new results (through {R['updated_through']})")

    today = dt.datetime.now(dt.timezone.utc).date()
    rows = []
    for day in (today, today + dt.timedelta(days=1)):
        season = season_of(day)
        for lid, (lname, _) in LEAGUES.items():
            fixtures = [f for f in api("/fixtures", league=lid, season=season, date=str(day))["response"]
                        if f["fixture"]["status"]["short"] == "NS"]
            if not fixtures:
                continue
            odds = fetch_odds(lid, season, str(day))
            for f in fixtures:
                o = odds.get(f["fixture"]["id"], {})
                if not all(o.get(k) for k in ("home", "draw", "away", "over25")):
                    continue  # no complete odds yet, can't judge value
                home, away = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
                h_elo, h_form, h_ok = rating_of(home, R["teams"])
                a_elo, a_form, a_ok = rating_of(away, R["teams"])
                res = eng.run(home_elo=h_elo, away_elo=a_elo, form5_home=h_form, form5_away=a_form,
                              odd_home=o["home"], odd_draw=o["draw"], odd_away=o["away"], over25=o["over25"])
                vt = value_table(res, o["home"], o["draw"], o["away"], o["over25"])
                best = vt.loc[vt["EV"].idxmax()]
                rows.append({
                    "Date": f["fixture"]["date"][:16].replace("T", " "), "League": lname,
                    "Home": home, "Away": away,
                    "P_Home": round(res["p_home"], 3), "P_Draw": round(res["p_draw"], 3),
                    "P_Away": round(res["p_away"], 3),
                    "xG_Home": round(res["xg_home"], 2), "xG_Away": round(res["xg_away"], 2),
                    "Score": res["score"], "P_Over25": round(res["p_over25"], 3),
                    "P_BTTS": round(res["p_btts"], 3), "P_HT_Home": round(res["p_ht_home"], 3),
                    "P_HT_Draw": round(res["p_ht_draw"], 3), "P_HT_Away": round(res["p_ht_away"], 3),
                    "Odd_Home": o["home"], "Odd_Draw": o["draw"], "Odd_Away": o["away"],
                    "Odd_Over25": o["over25"],
                    "BestMarket": best["Market"], "BestEV": round(best["EV"], 3),
                    "ValueBet": bool(best["Value bet"]), "RatingsFound": h_ok and a_ok})
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} fixtures ({int(out['ValueBet'].sum()) if len(out) else 0} value bets) to {OUT}")


if __name__ == "__main__":
    main()
