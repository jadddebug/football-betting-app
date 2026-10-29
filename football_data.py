"""Free data from football-data.co.uk: finished results + upcoming fixtures with odds. No API key needed."""
import datetime as dt
import io

import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
DIVS = ["E0", "SP1", "I1", "D1", "F1"]  # Premier League, La Liga, Serie A, Bundesliga, Ligue 1
# Odds columns, in order of preference (Bet365 first, then market average / max / Pinnacle)
HOME = ["B365H", "AvgH", "MaxH", "PSH"]
DRAW = ["B365D", "AvgD", "MaxD", "PSD"]
AWAY = ["B365A", "AvgA", "MaxA", "PSA"]
OVER = ["B365>2.5", "Avg>2.5", "Max>2.5", "P>2.5"]


def fetch_csv(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    try:
        text = r.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = r.content.decode("latin-1")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def season_code(d):
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"  # 2026/27 -> "2627"


def pick(row, names):
    for n in names:
        v = row.get(n)
        try:
            if pd.notna(v) and float(v) > 1:
                return float(v)
        except (TypeError, ValueError):
            pass
    return None


def new_results(last, end):
    """Finished matches with last < date <= end, oldest first: (date, home, away, hg, ag, div)."""
    games = []
    for code in sorted({season_code(last + dt.timedelta(days=1)), season_code(end)}):
        for div in DIVS:
            try:
                df = fetch_csv(f"{BASE}/mmz4281/{code}/{div}.csv")
            except Exception as e:
                print(f"  {div} {code}: could not download ({e})")
                continue
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
            df["d"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
            df = df[(df["d"] > last) & (df["d"] <= end)]
            print(f"  {div} {code}: {len(df)} new results")
            for r in df.itertuples():
                games.append((r.d, r.HomeTeam, r.AwayTeam, int(r.FTHG), int(r.FTAG), div))
    return sorted(games, key=lambda g: g[0])


def upcoming(today, days=3):
    """Fixtures from today to today+days that already have odds."""
    df = fetch_csv(f"{BASE}/fixtures.csv")
    df = df[df["Div"].isin(DIVS)].copy()
    df["d"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    window = df[(df["d"] >= today) & (df["d"] <= today + dt.timedelta(days=days))]
    print(f"  fixtures.csv: {len(df)} fixtures in the 5 leagues, {len(window)} in the next {days} days")
    out = []
    for _, r in window.iterrows():
        o = {"home": pick(r, HOME), "draw": pick(r, DRAW), "away": pick(r, AWAY), "over25": pick(r, OVER)}
        if all(o.values()):
            out.append({"date": f"{r['d']} {r.get('Time', '') if pd.notna(r.get('Time')) else ''}".strip(),
                        "div": r["Div"], "home_team": r["HomeTeam"], "away_team": r["AwayTeam"], **o})
    print(f"  {len(out)} of them have complete odds")
    return out
