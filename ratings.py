"""Team ratings (Elo, last results, goal history, head-to-head). Built from matches.csv, then kept up to date."""
import difflib
import json
import os

import pandas as pd

import features

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "team_ratings.json")
VERSION = 2  # bump when the saved format changes (old files are rebuilt automatically)
K = 20  # Elo K-factor (same as the notebook)
LEAGUE_NAMES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
POINTS = {"W": 3, "D": 1, "L": 0}

# Outside team names -> names used in matches.csv (extend as you spot misses)
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


def apply_result(R, home, away, hg, ag, date, league, update_elo=True):
    teams = R["teams"]
    for t in (home, away):
        teams.setdefault(t, {"elo": 1500.0, "results": [], "league": league, "last_date": str(date)})
    h, a = teams[home], teams[away]
    s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    if update_elo:
        exp = 1 / (1 + 10 ** ((a["elo"] - h["elo"]) / 400))
        h["elo"] += K * (s - exp)
        a["elo"] -= K * (s - exp)
    h["results"] = (h["results"] + ["W" if s == 1 else ("D" if s == 0.5 else "L")])[-5:]
    a["results"] = (a["results"] + ["L" if s == 1 else ("D" if s == 0.5 else "W")])[-5:]
    h["last_date"] = a["last_date"] = str(date)
    h["league"], a["league"] = league, league
    features.record(R["hist"], R["h2h"], home, away, hg, ag)


def build_initial():
    """Ratings and history for every team, replayed from matches.csv."""
    df = pd.read_csv(os.path.join(HERE, "matches.csv"), low_memory=False)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], format="mixed", errors="coerce")
    for c in ("HomeElo", "AwayElo", "FTHome", "FTAway"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"])
    df = df.sort_values("MatchDate", kind="stable")  # same order the training script uses
    R = {"version": VERSION, "base_date": str(df["MatchDate"].max().date()), "updated_through": None,
         "processed_ids": [], "teams": {}, "hist": {}, "h2h": {}}
    for r in df.itertuples():
        have_elo = pd.notna(r.HomeElo) and pd.notna(r.AwayElo)
        if have_elo:  # start from the dataset's own Elo
            R["teams"].setdefault(r.HomeTeam, {"results": []})["elo"] = float(r.HomeElo)
            R["teams"].setdefault(r.AwayTeam, {"results": []})["elo"] = float(r.AwayElo)
        apply_result(R, r.HomeTeam, r.AwayTeam, r.FTHome, r.FTAway, r.MatchDate.date(), r.Division,
                     update_elo=have_elo)
    return R


def load_or_build():
    if os.path.exists(FILE):
        R = json.load(open(FILE))
        if R.get("version") == VERSION:
            return R
    return build_initial()  # no file, or an older format: rebuild


def save(R):
    json.dump(R, open(FILE, "w"), separators=(",", ":"))


def extras(R, home, away):
    """Goal-based stats and head-to-head for a match; {} if either team is unknown."""
    if home not in R["teams"] or away not in R["teams"]:
        return {}
    x = features.extras(R["hist"].get(home, []), R["hist"].get(away, []),
                        R["h2h"].get(features.pair_key(home, away), []), home)
    x["Form3Home"] = form_points(R["teams"][home]["results"][-3:])
    x["Form3Away"] = form_points(R["teams"][away]["results"][-3:])
    return x


def h2h_summary(R, home, away):
    """(wins, draws, losses, games) for `home` in the last meetings with `away`."""
    w = d = l = 0
    for h, a, hg, ag in R["h2h"].get(features.pair_key(home, away), []):
        gf, ga = (hg, ag) if h == home else (ag, hg)
        w, d, l = w + (gf > ga), d + (gf == ga), l + (gf < ga)
    return w, d, l, w + d + l
