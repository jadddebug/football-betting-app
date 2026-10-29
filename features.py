"""Head-to-head and goal-based team stats. Used both to train the models and to predict live,
so the two always calculate them the same way."""
import pandas as pd

DEFAULT_GOALS = 1.4   # typical goals per team per match, used when there is no history yet
DEFAULT_PTS = 1.4     # typical points per match
KEEP = 12             # matches remembered per team
EXTRA = ["GF5Home", "GA5Home", "GF5Away", "GA5Away", "VenueFormHome", "VenueFormAway",
         "H2HPts", "H2HGoalDiff", "H2HGames", "AttackDiff", "DefenceDiff", "FormGoalDiff"]


def pts(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


def _mean(vals, default):
    return sum(vals) / len(vals) if vals else default


def team_block(hist, venue):
    """hist: oldest-first list of [goals_for, goals_against, 'H' or 'A']."""
    last5 = hist[-5:]
    gf = _mean([m[0] for m in last5], DEFAULT_GOALS)
    ga = _mean([m[1] for m in last5], DEFAULT_GOALS)
    same = [m for m in hist if m[2] == venue][-5:]
    return gf, ga, _mean([pts(m[0], m[1]) for m in same], DEFAULT_PTS)


def h2h_block(meetings, home):
    """meetings: oldest-first list of [home_team, away_team, home_goals, away_goals]."""
    p, gd = [], []
    for h, a, hg, ag in meetings[-5:]:
        if h == home:
            p.append(pts(hg, ag)); gd.append(hg - ag)
        else:
            p.append(pts(ag, hg)); gd.append(ag - hg)
    return _mean(p, DEFAULT_PTS), _mean(gd, 0.0), len(p)


def extras(home_hist, away_hist, meetings, home):
    gfh, gah, vph = team_block(home_hist, "H")
    gfa, gaa, vpa = team_block(away_hist, "A")
    hp, hgd, hn = h2h_block(meetings, home)
    return {"GF5Home": gfh, "GA5Home": gah, "GF5Away": gfa, "GA5Away": gaa,
            "VenueFormHome": vph, "VenueFormAway": vpa,
            "H2HPts": hp, "H2HGoalDiff": hgd, "H2HGames": hn,
            "AttackDiff": gfh - gfa, "DefenceDiff": gah - gaa,
            "FormGoalDiff": (gfh - gah) - (gfa - gaa)}


def pair_key(a, b):
    return "|".join(sorted([a, b]))


def record(hist_map, pair_map, home, away, hg, ag):
    """Remember a finished match (works on plain dicts, so it can be saved as JSON)."""
    for team, gf, ga, venue in ((home, hg, ag, "H"), (away, ag, hg, "A")):
        lst = hist_map.setdefault(team, [])
        lst.append([int(gf), int(ga), venue])
        del lst[:-KEEP]
    m = pair_map.setdefault(pair_key(home, away), [])
    m.append([home, away, int(hg), int(ag)])
    del m[:-5]


def add_extra_features(df):
    """Add the EXTRA columns to a date-sorted table, using only matches played BEFORE each row."""
    hist, pair, rows = {}, {}, []
    for h, a, hg, ag in zip(df["HomeTeam"], df["AwayTeam"], df["FTHome"], df["FTAway"]):
        rows.append(extras(hist.get(h, []), hist.get(a, []), pair.get(pair_key(h, a), []), h))
        record(hist, pair, h, a, hg, ag)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
