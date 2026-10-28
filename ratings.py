"""Team ratings (Elo + last 5 results). Built from matches.csv, then kept up to date from live results."""
import datetime as dt
import difflib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "team_ratings.json")
K = 20  # Elo K-factor (same as the notebook)
LEAGUE_NAMES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
POINTS = {"W": 3, "D": 1, "L": 0}

# API-Football team names -> names used in matches.csv (extend as you spot misses)
ALIASES = {"Manchester United": "Man United", "Manchester City": "Man City",
           "Atletico Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
           "Borussia Monchengladbach": "M'gladbach", "Paris Saint Germain": "Paris SG",
           "AC Milan": "Milan", "Real Betis": "Betis", "Real Sociedad": "Sociedad",
           "Rayo Vallecano": "Vallecano", "Borussia Dortmund": "Dortmund",
           "Bayer Leverkusen": "Leverkusen", "Eintracht Frankfurt": "Ein Frankfurt",
           "VfB Stuttgart": "Stuttgart", "Nottingham Forest": "Nott'm Forest"}


def form_points(results):
    return sum(POINTS[r] for r in results[-5:])


def resolve(name, teams):
    """Match an outside team name to ours. Returns our name, or the original if nothing fits."""
    name = ALIASES.get(name, name)
    if name in teams:
        return name
    close = difflib.get_close_matches(name, list(teams), n=1, cutoff=0.6)
    return close[0] if close else name


def apply_result(teams, home, away, hg, ag, date, league):
    for t in (home, away):
        teams.setdefault(t, {"elo": 1500.0, "results": [], "league": league, "last_date": str(date)})
    h, a = teams[home], teams[away]
    s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    exp = 1 / (1 + 10 ** ((a["elo"] - h["elo"]) / 400))
    h["elo"] += K * (s - exp)
    a["elo"] -= K * (s - exp)
    h["results"] = (h["results"] + ["W" if s == 1 else ("D" if s == 0.5 else "L")])[-5:]
    a["results"] = (a["results"] + ["L" if s == 1 else ("D" if s == 0.5 else "W")])[-5:]
    h["last_date"] = a["last_date"] = str(date)
    h["league"], a["league"] = league, league


def build_initial():
    """Latest rating for every team, from matches.csv."""
    df = pd.read_csv(os.path.join(HERE, "matches.csv"), low_memory=False)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], format="mixed", errors="coerce")
    df = df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam", "HomeElo", "AwayElo", "FTHome", "FTAway"])
    df = df.sort_values("MatchDate")
    teams = {}
    for r in df.itertuples():
        for t, elo in ((r.HomeTeam, r.HomeElo), (r.AwayTeam, r.AwayElo)):
            teams.setdefault(t, {"results": []})["elo"] = float(elo)  # start from the dataset's own Elo
        apply_result(teams, r.HomeTeam, r.AwayTeam, r.FTHome, r.FTAway, r.MatchDate.date(), r.Division)
    return {"base_date": str(df["MatchDate"].max().date()), "updated_through": None,
            "processed_ids": [], "teams": teams}


def load_or_build():
    if os.path.exists(FILE):
        return json.load(open(FILE))
    return build_initial()


def save(R):
    json.dump(R, open(FILE, "w"), indent=1)
