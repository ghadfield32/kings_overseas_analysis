"""
 Streamlit App for International Basketball Scouting
Comprehensive EDA, Model Diagnostics, and Recommendations

This version adds:
- Interactive Plotly visualizations
- Comprehensive statistical analysis
- Data quality insights
- Model performance breakdowns
- Assignment-specific documentation
"""

import json
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Kings | Intl Scouting", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_reports():
    """Load all report data with error handling."""
    df = pd.read_csv("final_scouting_report.csv")

    try:
        with open("ml_metrics.json", "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    try:
        fi = pd.read_csv("feature_importance.csv")
    except Exception:
        fi = pd.DataFrame()

    return df, metrics, fi


@st.cache_data
def calculate_statistics(df):
    """Calculate summary statistics for the dataset."""
    stats = {
        'total_prospects': len(df),
        'avg_age': df['age_2021'].mean(),
        'avg_ppg': df['ppg'].mean(),
        'avg_mpg': df['mpg'].mean(),
        'with_nba_exp': df['has_nba_exp'].sum(),
        'leagues': df['league'].nunique() if 'league' in df.columns else 0,
        'avg_scout_score': df['scout_score'].mean(),
        'improving_players': (df['ppg_change'] > 3).sum() if 'ppg_change' in df.columns else 0,
    }
    return stats


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_age_distribution_plot(df):
    """Create age distribution histogram."""
    fig = px.histogram(
        df,
        x='age_2021',
        nbins=15,
        title='Age Distribution of Top Prospects',
        labels={'age_2021': 'Age (2021)', 'count': 'Number of Players'},
        color_discrete_sequence=['#5D3A9B']
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        hovermode='x unified'
    )
    fig.add_vline(x=df['age_2021'].mean(), line_dash="dash",
                  annotation_text=f"Mean: {df['age_2021'].mean():.1f}",
                  line_color="red")
    return fig


def create_performance_scatter(df):
    """Create PPG vs Efficiency scatter plot."""
    fig = px.scatter(
        df,
        x='ppg',
        y='efficiency',
        size='mpg',
        color='age_2021',
        hover_data=['first_name', 'last_name', 'league', 'team'],
        title='Performance Analysis: PPG vs Efficiency',
        labels={
            'ppg': 'Points Per Game',
            'efficiency': 'Efficiency Rating',
            'age_2021': 'Age',
            'mpg': 'Minutes Per Game'
        },
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500)
    return fig


def create_league_comparison(df):
    """Create league-wise performance comparison."""
    if 'league' not in df.columns:
        return None

    league_stats = df.groupby('league').agg({
        'ppg': 'mean',
        'apg': 'mean',
        'rpg': 'mean',
        'efficiency': 'mean',
        'scout_score': 'mean',
        'first_name': 'count'
    }).reset_index()
    league_stats.columns = ['league', 'PPG', 'APG', 'RPG', 'Efficiency', 'Scout Score', 'Players']

    # Sort by number of players
    league_stats = league_stats.sort_values('Players', ascending=False)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Stats by League', 'Number of Prospects by League'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Stats comparison
    fig.add_trace(
        go.Bar(name='PPG', x=league_stats['league'], y=league_stats['PPG'], marker_color='#5D3A9B'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='APG', x=league_stats['league'], y=league_stats['APG'], marker_color='#E8927C'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='RPG', x=league_stats['league'], y=league_stats['RPG'], marker_color='#7FC8A9'),
        row=1, col=1
    )

    # Player count
    fig.add_trace(
        go.Bar(x=league_stats['league'], y=league_stats['Players'],
               marker_color='#5D3A9B', showlegend=False),
        row=1, col=2
    )

    fig.update_xaxes(title_text="League", row=1, col=1)
    fig.update_xaxes(title_text="League", row=1, col=2)
    fig.update_yaxes(title_text="Per Game Average", row=1, col=1)
    fig.update_yaxes(title_text="Number of Players", row=1, col=2)

    fig.update_layout(height=450, showlegend=True, hovermode='x unified')
    return fig


def create_shooting_analysis(df):
    """Create comprehensive shooting analysis."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('3-Point %', 'Free Throw %', 'True Shooting %'),
        specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}]]
    )

    # 3PT%
    fig.add_trace(
        go.Box(y=df['three_pt_pct'], name='3P%', marker_color='#5D3A9B', showlegend=False),
        row=1, col=1
    )

    # FT%
    fig.add_trace(
        go.Box(y=df['ft_pct'], name='FT%', marker_color='#E8927C', showlegend=False),
        row=1, col=2
    )

    # TS%
    fig.add_trace(
        go.Box(y=df['true_shooting_percentage'], name='TS%', marker_color='#7FC8A9', showlegend=False),
        row=1, col=3
    )

    fig.update_yaxes(title_text="Percentage", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=1, col=2)
    fig.update_yaxes(title_text="Percentage", row=1, col=3)

    fig.update_layout(height=400, title_text="Shooting Efficiency Analysis")
    return fig


def create_improvement_analysis(df):
    """Create player improvement trend analysis."""
    if 'ppg_change' not in df.columns:
        return None

    # Filter to players with meaningful data
    improving_df = df[df['ppg_change'].notna()].copy()
    improving_df['improvement_category'] = pd.cut(
        improving_df['ppg_change'],
        bins=[-float('inf'), -3, 0, 3, 5, float('inf')],
        labels=['Declining (< -3)', 'Slight Decline (0 to -3)',
                'Stable (0 to 3)', 'Improving (3-5)', 'Strong Growth (> 5)']
    )

    category_counts = improving_df['improvement_category'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=category_counts.index.astype(str),
            y=category_counts.values,
            marker_color=['#E74C3C', '#F39C12', '#95A5A6', '#2ECC71', '#27AE60'],
            text=category_counts.values,
            textposition='outside'
        )
    ])

    fig.update_layout(
        title='Player Development Trends (PPG Change)',
        xaxis_title='Improvement Category',
        yaxis_title='Number of Players',
        height=450,
        showlegend=False
    )

    return fig


def create_success_probability_distribution(df):
    """Create NBA success probability distribution."""
    if 'nba_success_prob' not in df.columns:
        return None

    prob_df = df[df['nba_success_prob'].notna()].copy()

    fig = px.histogram(
        prob_df,
        x='nba_success_prob',
        nbins=20,
        title='NBA Success Probability Distribution',
        labels={'nba_success_prob': 'Success Probability', 'count': 'Number of Players'},
        color_discrete_sequence=['#5D3A9B']
    )

    fig.add_vline(
        x=prob_df['nba_success_prob'].median(),
        line_dash="dash",
        annotation_text=f"Median: {prob_df['nba_success_prob'].median():.1%}",
        line_color="red"
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickformat='.0%')

    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap for key statistics."""
    numeric_cols = ['ppg', 'apg', 'rpg', 'spg', 'bpg', 'efficiency',
                    'true_shooting_percentage', 'three_pt_pct', 'ft_pct',
                    'mpg', 'scout_score']

    # Filter to available columns
    available_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title='Correlation Matrix: Key Performance Indicators',
        height=600,
        width=800
    )

    return fig


def create_feature_importance_plot(fi_df):
    """Create feature importance visualization."""
    if fi_df.empty:
        return None

    # Sort and take top 15
    fi_sorted = fi_df.sort_values('importance_perm', ascending=False).head(15)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=fi_sorted['feature'],
        x=fi_sorted['importance_perm'],
        orientation='h',
        marker_color='#5D3A9B',
        error_x=dict(
            type='data',
            array=fi_sorted['importance_perm_std'],
            visible=True
        ),
        text=np.round(fi_sorted['importance_perm'], 3),
        textposition='outside'
    ))

    fig.update_layout(
        title='Top 15 Features by Permutation Importance',
        xaxis_title='Importance (PR-AUC Drop)',
        yaxis_title='Feature',
        height=500,
        showlegend=False
    )

    return fig


def create_scout_score_breakdown(df):
    """Create scout score component analysis."""
    top_10 = df.nlargest(10, 'scout_score')[['first_name', 'last_name', 'scout_score',
                                               'age_2021', 'ppg', 'efficiency',
                                               'nba_success_prob']].copy()

    top_10['player_name'] = top_10['first_name'] + ' ' + top_10['last_name']

    fig = go.Figure()

    # Scout score bars
    fig.add_trace(go.Bar(
        x=top_10['player_name'],
        y=top_10['scout_score'],
        name='Scout Score',
        marker_color='#5D3A9B',
        text=np.round(top_10['scout_score'], 1),
        textposition='outside'
    ))

    fig.update_layout(
        title='Top 10 Prospects: Scout Score Rankings',
        xaxis_title='Player',
        yaxis_title='Scout Score',
        height=450,
        showlegend=False,
        xaxis_tickangle=-45
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

# Load data
df, metrics, fi = load_reports()
stats = calculate_statistics(df)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("🏀 Filters")

# League filter
if 'league' in df.columns:
    leagues = sorted([l for l in df['league'].dropna().unique()])
    sel_leagues = st.sidebar.multiselect("League", leagues, default=leagues)
else:
    sel_leagues = []

# MPG filter
min_mpg = float(df["mpg"].min()) if "mpg" in df else 0.0
max_mpg = float(df["mpg"].max()) if "mpg" in df else 40.0
mpg_range = st.sidebar.slider("Minutes per game", min_mpg, max_mpg, (20.0, max_mpg))

# Age filter
age_max = st.sidebar.slider("Max age (2021)", 20, int(df["age_2021"].max()), 30)

# NBA experience filter
exp_opt = st.sidebar.selectbox("NBA experience", ["All", "With NBA exp", "Intl only"])

# Display options
top_k = st.sidebar.slider("Top-K to highlight", 5, 30, 20)
sort_by = st.sidebar.selectbox("Sort by", ["scout_score", "nba_success_prob", "ppg", "efficiency"])

st.sidebar.markdown("---")
st.sidebar.caption("📊 Anonymized data per assignment")
st.sidebar.caption("🎯 Probabilities are calibrated")

# =============================================================================
# FILTER DATA
# =============================================================================

f = df.copy()

if 'league' in f and sel_leagues:
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
f["rank"] = range(1, len(f) + 1)

# =============================================================================
# HEADER METRICS
# =============================================================================

st.title("🏀 Sacramento Kings: International Scouting Analysis")
st.markdown("**Data**: NBA + European Leagues (2010–2021) | **Purpose**: Assignment Demonstration")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📈 PR-AUC", f"{metrics.get('test_pr_auc', 0):.3f}" if metrics else "—",
              help="Precision-Recall AUC (primary metric for imbalanced data)")
with col2:
    st.metric("📊 ROC-AUC", f"{metrics.get('test_auc', 0):.3f}" if metrics else "—",
              help="Receiver Operating Characteristic AUC")
with col3:
    st.metric("🎯 Precision@10", f"{metrics.get('precision_at_10', 0):.1%}" if metrics else "—",
              help="Precision in top 10 predictions")
with col4:
    st.metric("📉 Brier Score", f"{metrics.get('brier_score_calibrated', 0):.3f}" if metrics else "—",
              help="Calibration quality (lower is better)")

st.markdown("---")

# =============================================================================
# TABS
# =============================================================================

tab_rec, tab_eda, tab_model, tab_process, tab_limits = st.tabs([
    "🎯 Recommendations",
    "📊 Exploratory Data Analysis",
    "🤖 Model Diagnostics",
    "📋 Process & Methodology",
    "⚠️ Limitations"
])

# =============================================================================
# TAB 1: RECOMMENDATIONS
# =============================================================================

with tab_rec:
    st.subheader("Top Scouting Targets")
    st.markdown(f"Showing **{len(f)}** prospects matching filters (sorted by {sort_by})")

    # Display table
    display_cols = [c for c in [
        "rank", "first_name", "last_name", "age_2021", "league", "team",
        "games", "mpg", "ppg", "apg", "rpg", "three_pt_pct",
        "true_shooting_percentage", "efficiency", "nba_success_prob", "scout_score"
    ] if c in f.columns]

    st.dataframe(
        f[display_cols].head(200).style.background_gradient(
            subset=['scout_score'], cmap='Greens'
        ),
        use_container_width=True,
        height=400
    )

    st.download_button(
        "📥 Download Filtered CSV",
        f.to_csv(index=False).encode("utf-8"),
        file_name="scouting_recs_filtered.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Player details
    st.subheader("🔍 Individual Player Analysis")

    if not f.empty:
        player_names = f["first_name"] + " " + f["last_name"]
        selected_player = st.selectbox("Select player for detailed view:", player_names)

        row = f[player_names == selected_player].iloc[0]

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.metric("Minutes/Game", f"{row.get('mpg', 0):.1f}")
        with col_b:
            st.metric("Points/Game", f"{row.get('ppg', 0):.1f}")
        with col_c:
            st.metric("Efficiency", f"{row.get('efficiency', 0):.1f}")
        with col_d:
            st.metric("True Shooting %", f"{row.get('true_shooting_percentage', 0):.1%}")

        st.markdown(f"""
        **Profile**: {row.get('first_name', '')} {row.get('last_name', '')}
        **Team**: {row.get('team', '—')} | **League**: {row.get('league', '—')} | **Age**: {int(row.get('age_2021', 0))}
        **Scout Score**: {row.get('scout_score', 0):.1f}
        """)

        if 'nba_success_prob' in row and not math.isnan(row.get("nba_success_prob", float("nan"))):
            prob = row['nba_success_prob']
            st.info(f"🎯 **Calibrated NBA Success Probability**: {prob:.1%}")

        if 'ppg_change' in row and not math.isnan(row.get("ppg_change", float("nan"))):
            change = row['ppg_change']
            trend = "📈 Improving" if change > 0 else "📉 Declining"
            st.success(f"{trend}: {abs(change):.1f} PPG change over career")

# =============================================================================
# TAB 2: EDA
# =============================================================================

with tab_eda:
    st.header("📊 Exploratory Data Analysis")

    # Summary statistics
    st.subheader("Dataset Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Prospects", stats['total_prospects'])
    with col2:
        st.metric("Avg Age", f"{stats['avg_age']:.1f}")
    with col3:
        st.metric("Avg PPG", f"{stats['avg_ppg']:.1f}")
    with col4:
        st.metric("With NBA Exp", stats['with_nba_exp'])
    with col5:
        st.metric("Leagues", stats['leagues'])

    st.markdown("---")

    # Age distribution
    st.subheader("Age Distribution")
    fig_age = create_age_distribution_plot(f)
    st.plotly_chart(fig_age, use_container_width=True)

    # Performance scatter
    st.subheader("Performance Analysis")
    fig_perf = create_performance_scatter(f)
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("""
    **Insights**:
    - Bubble size represents minutes played (larger = more playing time)
    - Color represents age (darker = older players)
    - Look for high PPG + high efficiency + reasonable age
    """)

    st.markdown("---")

    # League comparison
    if 'league' in f.columns:
        st.subheader("League Comparison")
        fig_league = create_league_comparison(f)
        if fig_league:
            st.plotly_chart(fig_league, use_container_width=True)

    # Shooting analysis
    st.subheader("Shooting Efficiency Distribution")
    fig_shoot = create_shooting_analysis(f)
    st.plotly_chart(fig_shoot, use_container_width=True)

    st.markdown("---")

    # Improvement analysis
    if 'ppg_change' in f.columns:
        st.subheader("Player Development Trends")
        fig_improve = create_improvement_analysis(f)
        if fig_improve:
            st.plotly_chart(fig_improve, use_container_width=True)

            improving_count = (f['ppg_change'] > 3).sum()
            st.info(f"📈 **{improving_count}** players show significant improvement (> 3 PPG growth)")

    # Success probability
    if 'nba_success_prob' in f.columns:
        st.subheader("NBA Success Probability Distribution")
        fig_prob = create_success_probability_distribution(f)
        if fig_prob:
            st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # Correlation heatmap
    st.subheader("Statistical Correlations")
    fig_corr = create_correlation_heatmap(f)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Key Relationships**:
    - **High correlation** (red): Variables that move together
    - **Low/negative correlation** (blue): Independent or inverse relationships
    - Scout score components show expected relationships with performance metrics
    """)

# =============================================================================
# TAB 3: MODEL DIAGNOSTICS
# =============================================================================

with tab_model:
    st.header("🤖 Model Diagnostics & Performance")

    if metrics:
        st.subheader("Model Performance Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Classification Metrics")
            metrics_df = pd.DataFrame([
                {"Metric": "ROC-AUC (Test)", "Value": f"{metrics.get('test_auc', 0):.3f}"},
                {"Metric": "PR-AUC (Test)", "Value": f"{metrics.get('test_pr_auc', 0):.3f}"},
                {"Metric": "CV ROC-AUC (Median)", "Value": f"{metrics.get('cv_auc_median', 0):.3f}"},
                {"Metric": "CV PR-AUC (Median)", "Value": f"{metrics.get('cv_pr_auc_median', 0):.3f}"},
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### Calibration & Precision")
            calib_df = pd.DataFrame([
                {"Metric": "Brier Score (Calibrated)", "Value": f"{metrics.get('brier_score_calibrated', 0):.3f}"},
                {"Metric": "Brier Score (Uncalibrated)", "Value": f"{metrics.get('brier_score_uncalibrated', 0):.3f}"},
                {"Metric": "Brier Improvement", "Value": f"{metrics.get('brier_improvement', 0):.3f}"},
                {"Metric": "Precision@10", "Value": f"{metrics.get('precision_at_10', 0):.1%}"},
            ])
            st.dataframe(calib_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.info("""
        **Model Interpretation**:
        - **PR-AUC** is the primary metric due to class imbalance (few NBA successes)
        - **Brier Score** measures calibration quality (lower is better)
        - **Precision@K** shows accuracy in top-K predictions (most relevant for scouting)
        - **Cross-validation** results show model stability
        """)

    st.markdown("---")

    # Feature importance
    if not fi.empty:
        st.subheader("Feature Importance Analysis")

        fig_fi = create_feature_importance_plot(fi)
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("### Detailed Feature Rankings")
        fi_display = fi.sort_values('importance_perm', ascending=False).head(20)
        st.dataframe(
            fi_display.style.background_gradient(subset=['importance_perm'], cmap='Blues'),
            use_container_width=True
        )

        st.markdown("""
        **Feature Importance Interpretation**:
        - **Permutation importance** measures impact on PR-AUC when feature is shuffled
        - Higher values = more critical for predictions
        - Error bars show variability across permutations
        - Top features align with basketball domain knowledge (PPG, efficiency, shooting)
        """)

    st.markdown("---")

    # Diagnostic plots
    st.subheader("Visual Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image("ml_diagnostic_plots.png", caption="Calibration, ROC, and PR Curves")
        except Exception:
            st.warning("Diagnostic plots image not found")

    with col2:
        try:
            st.image("eda_visualizations_.png", caption="Comprehensive EDA (Static)")
        except Exception:
            st.warning("EDA visualization image not found")

    st.markdown("---")

    # Scout score breakdown
    st.subheader("Scout Score Component Analysis")
    fig_scout = create_scout_score_breakdown(f)
    st.plotly_chart(fig_scout, use_container_width=True)

# =============================================================================
# TAB 4: PROCESS & METHODOLOGY
# =============================================================================

with tab_process:
    st.header("📋 Process & Methodology")

    st.markdown("""
    ## Data Pipeline Overview

    This analysis follows a comprehensive, modular pipeline for international basketball scouting:

    ### 1️⃣ Data Loading & Validation
    - **Sources**: NBA and International (EuroLeague, ACB, VTB, etc.) statistics (2010-2021)
    - **Validation**: Automated quality checks for missing values, outliers, and inconsistencies
    - **Normalization**: Standardized metrics across leagues (TS%, efficiency, per-game stats)

    ### 2️⃣ Feature Engineering
    - **Per-game statistics**: PPG, APG, RPG, SPG, BPG
    - **Advanced metrics**: True Shooting %, Efficiency Rating, Usage %
    - **Trajectory features**: PPG trends, improvement over time
    - **Contextual features**: Age, league quality, playing time

    ### 3️⃣ Machine Learning Model
    - **Algorithm**: Calibrated Gradient Boosting Classifier
    - **Target**: NBA success (MPG ≥ 15 AND meaningful statistical contribution)
    - **Training**: Players with both international and NBA experience
    - **Calibration**: Sigmoid calibration for probability reliability
    - **Validation**: 5-fold cross-validation + hold-out test set

    ### 4️⃣ Scouting Score Calculation

    The scout score combines multiple components:

    ```
    scout_score = performance_score × age_bonus × improvement_bonus ×
                  fit_multiplier × ml_multiplier
    ```

    Where:
    - **Performance score**: Weighted combination of PPG, efficiency, TS%, APG, RPG
    - **Age bonus**: 1.3x for <24, 1.2x for 24-26, 1.1x for 26-28, 1.0x for 28+
    - **Improvement bonus**: Based on PPG trajectory (1.2x for strong growth)
    - **Fit multiplier**: Team-specific needs (shooting, defense, playmaking)
    - **ML multiplier**: Calibrated NBA success probability adjustment

    ### 5️⃣ Quality Assurance
    - **Data quality log**: All issues tracked and documented
    - **Sanity checks**: SQL queries verify data integrity
    - **Diagnostic visualizations**: Monitor distributions and outliers
    - **Model diagnostics**: Calibration curves, confusion matrices

    ## Key Design Decisions

    ### Why PR-AUC over ROC-AUC?
    - Dataset is **imbalanced** (few NBA successes vs. many non-successes)
    - PR-AUC better captures performance on minority class
    - ROC-AUC can be misleadingly optimistic with imbalance

    ### Why Calibrated Probabilities?
    - Raw model outputs may be overconfident or underconfident
    - Calibration ensures probabilities are **trustworthy** (70% means 70%)
    - Brier score measures calibration quality

    ### Why Multiple Scout Score Components?
    - **Holistic evaluation**: Not just current performance
    - **Context matters**: Age, improvement trajectory, team fit
    - **Risk mitigation**: ML probability tempered by traditional scouting

    ## Data Quality & Limitations

    ### Quality Controls Implemented:
    ✅ Duplicate name detection and resolution
    ✅ Out-of-range value flagging (shooting %, games, minutes)
    ✅ Scale normalization (0-100 vs 0-1 percentages)
    ✅ Orphaned record removal (stats without player demographics)
    ✅ Missing value documentation

    ### Known Limitations:
    ⚠️ Data anonymized per assignment (identities hidden)
    ⚠️ No post-2021 validation possible
    ⚠️ League quality differences not fully captured
    ⚠️ Small sample size for NBA success (class imbalance)
    ⚠️ Temporal data may have leakage (players with concurrent NBA/Intl careers)

    ## Assignment-Specific Notes

    - All player names are **anonymized** to demonstrate process, not outcomes
    - Focus is on **methodology** and **reproducibility**
    - Code is **modular** and **well-documented** for easy review
    - Outputs include **diagnostics** for transparency
    """)

    st.markdown("---")

    st.subheader("📁 Output Files Generated")

    output_files = pd.DataFrame([
        {"File": "final_scouting_report.csv", "Description": "Top 30 prospects with all metrics"},
        {"File": "ml_metrics.json", "Description": "Model performance metrics"},
        {"File": "feature_importance.csv", "Description": "Feature rankings"},
        {"File": "data_quality_report.txt", "Description": "Data quality issues log"},
        {"File": "eda_visualizations_.png", "Description": "9-panel EDA visualization"},
        {"File": "ml_diagnostic_plots.png", "Description": "Calibration, ROC, PR curves"},
        {"File": "kings_scouting.db", "Description": "SQLite database with all data"},
    ])

    st.dataframe(output_files, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: LIMITATIONS
# =============================================================================

with tab_limits:
    st.header("⚠️ Limitations & Considerations")

    st.markdown("""
    ## Data Limitations

    ### 1. Anonymization
    - Player identities are **anonymized** per assignment requirements
    - Prevents real-world validation of recommendations
    - Focus is on demonstrating **process quality**, not actual outcomes

    ### 2. Temporal Constraints
    - Data ends in **2021**
    - Cannot validate predictions against post-2021 NBA performance
    - Some players may have since proven/disproven model predictions

    ### 3. Sample Size
    - **Class imbalance**: Few NBA successes relative to international players
    - Limits model confidence, especially for rare player archetypes
    - PR-AUC and calibration help mitigate but don't eliminate this issue

    ### 4. League Quality Adjustment
    - Different international leagues have varying competition levels
    - Model implicitly captures this through historical data
    - Could be  with explicit league strength ratings

    ## Model Limitations

    ### 1. Feature Coverage
    - **Missing factors**: Injury history, personality, work ethic, team culture fit
    - **Intangibles**: Leadership, clutch performance, defensive IQ
    - **Context**: Coaching, system fit, role availability

    ### 2. Temporal Leakage Risk
    - Some players played internationally **while** in NBA (e.g., lockout, buyouts)
    - Feature extraction attempts to use only pre-NBA international data
    - Diagnostic checks flag potential leakage cases

    ### 3. Generalization
    - Model trained on **historical** NBA success criteria
    - Game evolution (3-point emphasis, pace, positionless basketball) may shift success factors
    - Periodic retraining recommended

    ## Operational Considerations

    ### 1. Probability Interpretation
    - **70% success probability ≠ 70% certainty**
    - Reflects model's confidence based on historical similar players
    - Should inform, not replace, human scouting judgment

    ### 2. Scout Score Trade-offs
    - High score favors **young, improving, efficient** players
    - May undervalue **veterans** with specialized skills
    - May overvalue **high-volume scorers** in weak leagues

    ### 3. Recommendation Usage
    - Use as **screening tool** to prioritize in-depth scouting
    - Combine with **video analysis**, **interviews**, **medical evaluations**
    - Consider **organizational needs** and **roster construction**

    ## Ethical Considerations

    ### 1. Bias & Fairness
    - Historical NBA success criteria may embed systemic biases
    - International league representation may vary by region
    - Model should be audited for demographic fairness

    ### 2. Transparency
    - All code is **open** and **documented**
    - Feature importance and diagnostics provided
    - Anonymization protects player privacy

    ### 3. Human Oversight
    - ML should **augment**, not replace, human scouts
    - Final decisions require contextual judgment
    - Model outputs are **recommendations**, not mandates

    ## Future Improvements

    ### Potential Enhancements:
    - ✨ Incorporate **play-by-play** data for deeper insights
    - ✨ Add **video analysis** features (shot selection, defensive positioning)
    - ✨ Include **combine measurements** (athleticism, length)
    - ✨ Model **different success tiers** (starter vs. rotation vs. end-of-bench)
    - ✨ Implement **Bayesian updating** as more data arrives
    - ✨ Build **similarity search** to find NBA comparables

    ---

    ## Contact & Feedback

    This analysis demonstrates a **structured, reproducible pipeline** for sports analytics.
    For questions about methodology or implementation, refer to the codebase documentation.
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Sacramento Kings International Scouting Analysis</strong></p>
    <p>Assignment Demonstration | Data: 2010-2021 | Player Names Anonymized</p>
    <p>Built with Streamlit, Plotly, scikit-learn | Modular, Reproducible Pipeline</p>
</div>
""", unsafe_allow_html=True)
