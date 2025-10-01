# app.py
import json, math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kings | Intl Scouting", layout="wide")

@st.cache_data
def load_reports():
    df = pd.read_csv("final_scouting_report.csv")
    try:
        with open("ml_metrics.json","r") as f: metrics = json.load(f)
    except Exception:
        metrics = {}
    try:
        fi = pd.read_csv("feature_importance.csv")
    except Exception:
        fi = pd.DataFrame()
    return df, metrics, fi

df, metrics, fi = load_reports()

# ---- SIDEBAR ----
st.sidebar.header("Filters")
leagues = sorted([l for l in df["league"].dropna().unique()])
sel_leagues = st.sidebar.multiselect("League", leagues, default=leagues)
min_mpg = float(df["mpg"].min()) if "mpg" in df else 0.0
max_mpg = float(df["mpg"].max()) if "mpg" in df else 40.0
mpg_range = st.sidebar.slider("Minutes per game", min_mpg, max_mpg, (20.0, max_mpg))
age_max = st.sidebar.slider("Max age (2021)", 20, int(df["age_2021"].max()), 30)
exp_opt = st.sidebar.selectbox("NBA experience", ["All","With NBA exp","Intl only"])
top_k = st.sidebar.slider("Top-K to highlight", 5, 30, 20)
sort_by = st.sidebar.selectbox("Sort by", ["scout_score","nba_success_prob","ppg","efficiency"])
st.sidebar.caption("Anonymized data per assignment; probabilities are calibrated.")

# ---- METRICS HEADER ----
left, mid, right, extra = st.columns(4)
left.metric("PR-AUC", f"{metrics.get('test_pr_auc', float('nan')):.3f}" if metrics else "—")
mid.metric("ROC-AUC", f"{metrics.get('test_auc', float('nan')):.3f}" if metrics else "—")
right.metric("Precision@10", f"{metrics.get('precision_at_10', float('nan')):.2%}" if metrics else "—")
extra.metric("Brier", f"{metrics.get('brier_score_calibrated', float('nan')):.3f}" if metrics else "—")

# ---- FILTER DATA ----
f = df.copy()
f = f[f["league"].isin(sel_leagues)]
f = f[(f["mpg"] >= mpg_range[0]) & (f["mpg"] <= mpg_range[1])]
f = f[f["age_2021"] <= age_max]
if exp_opt == "With NBA exp":
    f = f[f["has_nba_exp"] == True]
elif exp_opt == "Intl only":
    f = f[f["has_nba_exp"] == False]

# Sort and rank
if sort_by in f.columns:
    f = f.sort_values(sort_by, ascending=False)
f["rank"] = range(1, len(f)+1)

# ---- MAIN LAYOUT ----
st.title("International Scouting Recommendations (Anonymized)")
st.write("Data: NBA + Euro leagues (2010–2021). Scores reflect process quality, not real-world identities.")

tab_rec, tab_eda, tab_model, tab_limits = st.tabs(
    ["Recommendations", "EDA", "Model Diagnostics", "Limitations"]
)

with tab_rec:
    # highlight Top-K
    styled = f.head(100).style.apply(
        lambda s: ["background-color:#fff4d6" if (i < top_k) else "" for i in range(len(s))],
        axis=0
    )
    display_cols = [c for c in [
        "rank","first_name","last_name","age_2021","league","team",
        "games","mpg","ppg","apg","rpg","three_pt_pct",
        "true_shooting_percentage","efficiency","nba_success_prob","scout_score"
    ] if c in f.columns]
    st.dataframe(f[display_cols].head(200), use_container_width=True)
    st.download_button("Download filtered CSV", f.to_csv(index=False).encode("utf-8"),
                       file_name="scouting_recs_filtered.csv", mime="text/csv")

    # details panel
    st.subheader("Player details")
    pid = st.selectbox("Select player", f["first_name"]+" "+f["last_name"])
    row = f[(f["first_name"]+" "+f["last_name"])==pid].iloc[0]
    colA,colB,colC = st.columns(3)
    colA.metric("MPG", f"{row.get('mpg', float('nan')):.1f}")
    colB.metric("PPG", f"{row.get('ppg', float('nan')):.1f}")
    colC.metric("TS%", f"{row.get('true_shooting_percentage', float('nan')):.1%}")
    st.write(f"**Team**: {row.get('team','—')} | **League**: {row.get('league','—')} | **Age**: {int(row.get('age_2021',0))}")
    if not math.isnan(row.get("nba_success_prob", float("nan"))):
        st.info(f"Calibrated NBA success probability: {row['nba_success_prob']:.1%}")

with tab_eda:
    st.subheader("Exploratory Data Analysis")
    try:
        st.image("eda_visualizations.png", caption="Distributions & Age–MPG")
    except Exception:
        st.write("EDA plot not found.")

with tab_model:
    st.subheader("Diagnostic Plots")
    try:
        st.image("ml_diagnostic_plots.png", caption="Calibration, PR, ROC")
    except Exception:
        st.write("Diagnostics image not found.")
    if not fi.empty:
        st.subheader("Feature importance (permutation)")
        st.dataframe(fi.sort_values("importance_perm", ascending=False).head(15), use_container_width=True)

with tab_limits:
    st.markdown("""
- Player identities are **anonymized**; outcomes cannot be validated post-2021.
- Dataset is **imbalanced (~12–13% success)**; PR-AUC emphasized over ROC-AUC.
- Reported metrics include **calibration** (Brier) and **Precision@K** for decision support.
""")
