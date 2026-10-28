"""Football value-betting dashboard.  Run with:  streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

import ratings
from engine import Engine, value_table, EV_THRESHOLD

st.set_page_config(page_title="Football Value Engine", page_icon="⚽", layout="wide")


@st.cache_resource
def get_engine():
    return Engine()


@st.cache_data(ttl=3600)
def get_ratings():
    return ratings.load_or_build()


eng = get_engine()
R = get_ratings()
TEAMS = R["teams"]
NAMES = sorted(TEAMS, key=lambda t: (ratings.LEAGUE_NAMES.get(TEAMS[t]["league"], "z"), t))


def label(t):
    return f"{t}  ·  {ratings.LEAGUE_NAMES.get(TEAMS[t]['league'], '')}"


def team_info(t):
    d = TEAMS[t]
    return f"Last 5: {' '.join(d['results']) or 'n/a'} (rating as of {d['last_date']})"
st.title("⚽ Football Value-Betting Engine")
st.caption("Three-model outcome blend + Poisson goal model. Flags a bet when expected value is above "
           f"{EV_THRESHOLD:.0%}. For information only; no prediction is a guarantee.")

tab1, tab2 = st.tabs(["Single match", "Daily scanner"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.caption("Ratings through " + str(R.get("updated_through") or R["base_date"])
               + ". Click a team box and start typing to search.")
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", NAMES, index=None, format_func=label, placeholder="Type a team name...")
    away = c2.selectbox("Away team", NAMES, index=None, format_func=label, placeholder="Type a team name...")
    if not (home and away):
        st.info("Pick both teams to continue.")
        st.stop()
    c1.caption(team_info(home))
    c2.caption(team_info(away))
    c1, c2, c3, c4 = st.columns(4)
    h_elo = c1.number_input("Home Elo", 1000.0, 2400.0, round(TEAMS[home]["elo"], 1), 5.0, key=f"he_{home}")
    a_elo = c2.number_input("Away Elo", 1000.0, 2400.0, round(TEAMS[away]["elo"], 1), 5.0, key=f"ae_{away}")
    h_form = c3.number_input("Home form (points, last 5)", 0, 15, ratings.form_points(TEAMS[home]["results"]),
                             key=f"hf_{home}")
    a_form = c4.number_input("Away form (points, last 5)", 0, 15, ratings.form_points(TEAMS[away]["results"]),
                             key=f"af_{away}")
    st.caption("Elo and form are filled in automatically and can be changed. Odds you type in yourself.")
    c1, c2, c3, c4 = st.columns(4)
    o_h = c1.number_input("Odds: home win", 1.01, 50.0, 2.30, 0.05)
    o_d = c2.number_input("Odds: draw", 1.01, 50.0, 3.40, 0.05)
    o_a = c3.number_input("Odds: away win", 1.01, 50.0, 3.10, 0.05)
    o_o = c4.number_input("Odds: over 2.5 goals", 1.01, 20.0, 1.95, 0.05)

    if st.button("Predict", type="primary"):
        res = eng.run(home_elo=h_elo, away_elo=a_elo, form5_home=h_form, form5_away=a_form,
                      odd_home=o_h, odd_draw=o_d, odd_away=o_a, over25=o_o)
        st.subheader(f"{home} vs {away}")
        m = st.columns(5)
        m[0].metric("Home win", f"{res['p_home']:.1%}")
        m[1].metric("Draw", f"{res['p_draw']:.1%}")
        m[2].metric("Away win", f"{res['p_away']:.1%}")
        m[3].metric("Likely score", res["score"], f"{res['score_prob']:.1%} chance")
        m[4].metric("Over 2.5 goals", f"{res['p_over25']:.1%}")
        m2 = st.columns(4)
        m2[0].metric("Half-time: home ahead", f"{res['p_ht_home']:.1%}")
        m2[1].metric("Half-time: draw", f"{res['p_ht_draw']:.1%}")
        m2[2].metric("Half-time: away ahead", f"{res['p_ht_away']:.1%}")
        m2[3].metric("Both teams to score", f"{res['p_btts']:.1%}")
        st.write(f"Expected goals: **{home} {res['xg_home']:.2f}**, **{away} {res['xg_away']:.2f}**")

        vt = value_table(res, o_h, o_d, o_a, o_o)
        st.dataframe(vt.style.format({"Model prob": "{:.1%}", "Fair odds": "{:.2f}", "EV": "{:+.1%}"})
                     .apply(lambda r: ["background-color:#14532d" if r["Value bet"] else "" for _ in r], axis=1),
                     use_container_width=True, hide_index=True)
        if vt["Value bet"].any():
            best = vt.loc[vt["EV"].idxmax()]
            st.success(f"Value found: **{best['Market']}** at {best['Bookmaker odds']} "
                       f"(expected value {best['EV']:+.1%})")
        else:
            st.info("No market clears the value threshold for this match.")

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Today's slate (from the daily pipeline)")
    if os.path.exists("todays_value_bets.csv"):
        slate = pd.read_csv("todays_value_bets.csv")
        if slate.empty:
            st.info("The pipeline ran but found no fixtures with complete odds today.")
        else:
            only_value = st.checkbox("Show value bets only", value=True)
            view = slate[slate["ValueBet"]] if only_value else slate
            st.caption(f"{len(slate)} fixtures scored, {int(slate['ValueBet'].sum())} flagged as value.")
            st.dataframe(view, use_container_width=True, hide_index=True)
            if not slate["RatingsFound"].all():
                st.warning("Some teams weren't found in matches.csv and used average ratings. "
                           "Treat those rows with caution.")
    else:
        st.info("No todays_value_bets.csv yet. The daily pipeline creates it.")

    st.divider()
    st.subheader("Batch upload")
    template = pd.DataFrame([{"HomeTeam": "Home FC", "AwayTeam": "Away FC", "HomeElo": 1650, "AwayElo": 1600,
                              "Form5Home": 7, "Form5Away": 7, "OddHome": 2.3, "OddDraw": 3.4,
                              "OddAway": 3.1, "Over25": 1.95}])
    st.download_button("Download CSV template", template.to_csv(index=False), "template.csv")
    up = st.file_uploader("Upload your matches (CSV)", type="csv")
    if up:
        df = pd.read_csv(up)
        need = list(template.columns)
        missing = [c for c in need if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            out = []
            for _, r in df.iterrows():
                res = eng.run(home_elo=r.HomeElo, away_elo=r.AwayElo, form5_home=r.Form5Home,
                              form5_away=r.Form5Away, odd_home=r.OddHome, odd_draw=r.OddDraw,
                              odd_away=r.OddAway, over25=r.Over25)
                vt = value_table(res, r.OddHome, r.OddDraw, r.OddAway, r.Over25)
                best = vt.loc[vt["EV"].idxmax()]
                out.append({"Home": r.HomeTeam, "Away": r.AwayTeam,
                            "P_Home": round(res["p_home"], 3), "P_Draw": round(res["p_draw"], 3),
                            "P_Away": round(res["p_away"], 3), "Score": res["score"],
                            "P_Over25": round(res["p_over25"], 3),
                            "P_BTTS": round(res["p_btts"], 3), "P_HT_Home": round(res["p_ht_home"], 3),
                            "P_HT_Draw": round(res["p_ht_draw"], 3), "P_HT_Away": round(res["p_ht_away"], 3),
                            "BestMarket": best["Market"],
                            "BestEV": round(best["EV"], 3), "ValueBet": bool(best["Value bet"])})
            result = pd.DataFrame(out)
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("Download results", result.to_csv(index=False), "predictions.csv")
