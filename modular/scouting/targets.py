"""
Scouting target identification module - FIXED VERSION
Addresses: Always document ml_multiplier value, even when ML disabled
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from modular.config import Config



class ScoutingAnalyzer:
    """Analyzes and ranks international prospects for NBA scouting."""

    def __init__(self, team_weights: Optional[Dict] = None, nba_prob_weight: float = 1.10):
        self.team_weights = team_weights or Config.DEFAULT_TEAM_WEIGHTS
        self.nba_prob_weight = nba_prob_weight

    def calculate_team_weights(self, nba_df: pd.DataFrame) -> 'ScoutingAnalyzer':
        """Calculate team fit weights based on successful NBA players."""
        print("\nCalculating team fit weights...")
        print("Definition: NBA seasons with MPG ≥ 20. Weights reflect prevalence above ~75th percentile.")

        successful_nba = nba_df[nba_df['mpg'] >= 20].copy()

        if len(successful_nba) < 50:
            print("Not enough NBA data; using conservative defaults.")
            self.team_weights = Config.DEFAULT_TEAM_WEIGHTS
        else:
            p75_3pt = successful_nba['three_pt_pct'].quantile(0.75)
            p75_def = (successful_nba['spg'] + successful_nba['bpg']).quantile(0.75)
            p75_ast = successful_nba['apg'].quantile(0.75)
            p75_reb = successful_nba['rpg'].quantile(0.75)

            high_3pt = (successful_nba['three_pt_pct'] >= p75_3pt).sum()
            high_def = ((successful_nba['spg'] + successful_nba['bpg']) >= p75_def).sum()
            high_ast = (successful_nba['apg'] >= p75_ast).sum()
            high_reb = (successful_nba['rpg'] >= p75_reb).sum()

            total = len(successful_nba)
            self.team_weights = {
                'shooting_3pt': 1.0 + (high_3pt / total) * 0.5,
                'defense': 1.0 + (high_def / total) * 0.4,
                'playmaking': 1.0 + (high_ast / total) * 0.4,
                'rebounding': 1.0 + (high_reb / total) * 0.3,
                'youth': 1.05
            }

        print("Team fit weights:")
        for k, v in self.team_weights.items():
            print(f"  {k}: {v:.3f}")
        return self

    def normalize_metrics(self, df: pd.DataFrame, metrics: list) -> pd.DataFrame:
        """Normalize metrics to 0-100 scale."""
        def _norm(s):
            s = s.astype(float)
            mn, mx = s.min(), s.max()
            if pd.isna(mn) or pd.isna(mx) or mx <= mn:
                return pd.Series(np.nan, index=s.index)
            return (s - mn) / (mx - mn) * 100.0

        for metric in metrics:
            if metric in df.columns:
                df[f'{metric}_norm'] = _norm(df[metric])
        return df

    def calculate_performance_score(self, prospects: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted performance score."""
        metrics = ['ppg', 'apg', 'rpg', 'efficiency', 'true_shooting_percentage']
        prospects = self.normalize_metrics(prospects, metrics)

        prospects['performance_score'] = (
            prospects.get('ppg_norm', np.nan).fillna(50) * 0.30 +
            prospects.get('efficiency_norm', np.nan).fillna(50) * 0.25 +
            prospects.get('true_shooting_percentage_norm', np.nan).fillna(50) * 0.20 +
            prospects.get('apg_norm', np.nan).fillna(50) * 0.15 +
            prospects.get('rpg_norm', np.nan).fillna(50) * 0.10
        )

        return prospects

    def calculate_age_bonus(self, prospects: pd.DataFrame) -> pd.DataFrame:
        """Calculate age-based bonus multiplier."""
        prospects['age_bonus'] = prospects['age_2021'].apply(
            lambda x: 1.3 if x < 24 else (1.2 if x < 26 else (1.1 if x < 28 else 1.0))
        )
        return prospects

    def calculate_improvement_bonus(self, prospects: pd.DataFrame) -> pd.DataFrame:
        """Calculate improvement trend bonus."""
        prospects['improvement_bonus'] = prospects['ppg_change'].fillna(0).apply(
            lambda x: 1.2 if x > 5 else (1.15 if x > 3 else (1.1 if x > 1 else 1.0))
        )
        return prospects

    def calculate_team_fit_multiplier(self, prospects: pd.DataFrame) -> pd.DataFrame:
        """Calculate team fit multiplier based on skill components."""
        prospects['def_events_pg'] = (
            prospects.get('spg', 0).fillna(0) + prospects.get('bpg', 0).fillna(0)
        )

        fit_components = {
            'shooting_3pt': self.normalize_metrics(prospects.copy(), ['three_pt_pct'])['three_pt_pct_norm'],
            'defense': self.normalize_metrics(prospects.copy(), ['def_events_pg'])['def_events_pg_norm'],
            'playmaking': self.normalize_metrics(prospects.copy(), ['apg'])['apg_norm'],
            'rebounding': self.normalize_metrics(prospects.copy(), ['rpg'])['rpg_norm']
        }

        prospects['fit_multiplier'] = (
            (1.0 + (fit_components['shooting_3pt'].fillna(50)/100) * (self.team_weights.get('shooting_3pt', 1.0) - 1.0)) *
            (1.0 + (fit_components['defense'].fillna(50)/100) * (self.team_weights.get('defense', 1.0) - 1.0)) *
            (1.0 + (fit_components['playmaking'].fillna(50)/100) * (self.team_weights.get('playmaking', 1.0) - 1.0)) *
            (1.0 + (fit_components['rebounding'].fillna(50)/100) * (self.team_weights.get('rebounding', 1.0) - 1.0)) *
            (prospects['age_bonus'] ** (self.team_weights.get('youth', 1.0) - 1.0))
        )

        return prospects

    def calculate_ml_multiplier(self, prospects: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Calculate ML prediction multiplier with explicit documentation.
        """
        # FIXED: Always initialize to 1.0 and document
        prospects['ml_multiplier'] = 1.0

        if 'nba_success_prob' in prospects.columns and prospects['nba_success_prob'].notna().any():
            prospects['ml_multiplier'] = (
                1.0 + (prospects['nba_success_prob'].fillna(0.5) - 0.5) * 
                (self.nba_prob_weight - 1.0) * 2.0
            )
            print(f"NBA success probability weight: {self.nba_prob_weight:.2f}x")
        else:
            # FIXED: Explicitly document when ML not used
            print(f"ML predictions not available; ml_multiplier = 1.0 for all prospects")

        return prospects

    def calculate_scout_score(self, prospects: pd.DataFrame) -> pd.DataFrame:
        """Calculate final scout score with formula documentation."""
        print("\nScout score formula:")
        print("  performance_score × age_bonus × improvement_bonus × fit_multiplier × ml_multiplier")

        prospects['scout_score'] = (
            prospects['performance_score'] *
            prospects['age_bonus'] *
            prospects['improvement_bonus'] *
            prospects['fit_multiplier'] *
            prospects['ml_multiplier']
        )

        # DEBUG ISSUE #4: Decompose top 5 scout scores
        print(f"\n[DEBUG ISSUE #4] Top 5 scout score decomposition:")
        top5 = prospects.nlargest(5, 'scout_score')

        for rank, (idx, row) in enumerate(top5.iterrows(), 1):
            name = f"{row.get('first_name', 'Unknown')} {row.get('last_name', 'Unknown')}"
            print(f"\nRank #{rank}: {name}")
            print(f"  Final scout_score: {row['scout_score']:.1f}")
            print(f"  Components:")
            print(f"    performance_score: {row.get('performance_score', np.nan):.1f}")
            print(f"    age_bonus: {row.get('age_bonus', 1.0):.3f} (age {int(row.get('age_2021', 0))})")
            print(f"    improvement_bonus: {row.get('improvement_bonus', 1.0):.3f} (Δppg {row.get('ppg_change', 0):.1f})")
            print(f"    fit_multiplier: {row.get('fit_multiplier', 1.0):.3f}")
            print(f"    ml_multiplier: {row.get('ml_multiplier', 1.0):.3f}")

            # Show ML probability if available
            if 'nba_success_prob' in row and pd.notna(row['nba_success_prob']):
                print(f"    nba_success_prob: {row['nba_success_prob']:.1%}")
            else:
                print(f"    nba_success_prob: N/A")

            # Verify calculation
            calculated = (row['performance_score'] * row['age_bonus'] *
                         row['improvement_bonus'] * row['fit_multiplier'] *
                         row['ml_multiplier'])
            diff = abs(calculated - row['scout_score'])
            if diff > 0.1:
                print(f"  ⚠️ CALCULATION MISMATCH: Expected {calculated:.1f}, got {row['scout_score']:.1f}")

        return prospects

    def identify_scouting_targets(self, player_df: pd.DataFrame, 
                                nba_df: pd.DataFrame,
                                intl_df: pd.DataFrame,
                                traj_df: Optional[pd.DataFrame] = None,
                                ml_predictions: Optional[pd.DataFrame] = None,
                                season: int = 2021,
                                min_games: int = 10,
                                min_mpg: int = 20,
                                max_age: int = 30,
                                top_n: int = 30) -> pd.DataFrame:
        """Identify and rank scouting targets."""
        print(f"\nSCOUTING TARGETS")
        self.calculate_team_weights(nba_df)

        prospects = intl_df[intl_df['season'] == season].copy()
        prospects = prospects.merge(
            player_df[['player_id', 'age_2021']], on='player_id', how='left'
        )

        print(f"Criteria: season={season}, games≥{min_games}, mpg≥{min_mpg}, age<{max_age}")
        prospects = prospects[
            (prospects['games'] >= min_games) &
            (prospects['mpg'] >= min_mpg) &
            (prospects['age_2021'] < max_age)
        ].copy()
        print(f"Eligible players: {len(prospects)}")

        prospects['has_nba_exp'] = prospects['player_id'].isin(nba_df['player_id'].unique())

        if traj_df is not None:
            traj_merge = traj_df[['ppg_change', 'efficiency_change', 'seasons_played']].reset_index()
            prospects = prospects.merge(traj_merge, on='player_id', how='left')
        else:
            prospects['ppg_change'] = np.nan
            prospects['efficiency_change'] = np.nan

        if ml_predictions is not None:
            prospects = prospects.merge(ml_predictions, on='player_id', how='left')
        else:
            prospects['nba_success_prob'] = np.nan

        prospects = self.calculate_performance_score(prospects)
        prospects = self.calculate_age_bonus(prospects)
        prospects = self.calculate_improvement_bonus(prospects)
        prospects = self.calculate_team_fit_multiplier(prospects)
        prospects = self.calculate_ml_multiplier(prospects)  # FIXED: Always documents value
        prospects = self.calculate_scout_score(prospects)

        prospects = prospects.sort_values('scout_score', ascending=False).drop_duplicates(
            subset='player_id', keep='first'
        )
        top_prospects = prospects.nlargest(top_n, 'scout_score').copy()

        print(f"\nTop {top_n} prospects identified.")
        print(f"With NBA experience: {int(top_prospects['has_nba_exp'].sum())}")
        print(f"International only: {int((~top_prospects['has_nba_exp']).sum())}")

        return top_prospects


def identify_scouting_targets(player_df: pd.DataFrame, 
                            nba_df: pd.DataFrame,
                            intl_df: pd.DataFrame,
                            traj_df: Optional[pd.DataFrame] = None,
                            ml_predictions: Optional[pd.DataFrame] = None,
                            team_weights: Optional[Dict] = None,
                            nba_prob_weight: float = 1.10,
                            season: int = 2021,
                            top_n: int = 30) -> pd.DataFrame:
    """Main function to identify scouting targets."""
    print("\n" + "=" * 100)
    print("IDENTIFYING SCOUTING TARGETS")
    print("=" * 100)

    analyzer = ScoutingAnalyzer(team_weights, nba_prob_weight)

    prospects = analyzer.identify_scouting_targets(
        player_df, nba_df, intl_df, traj_df, ml_predictions, season, top_n=top_n
    )

    print("\nScouting target identification complete.")
    return prospects


if __name__ == "__main__":
    """Smoke test."""
    print("="*100)
    print("SCOUTING MODULE SMOKE TEST (FIXED VERSION)")
    print("="*100)

    try:
        test_player_df = pd.DataFrame({
            'player_id': ['p1', 'p2', 'p3'],
            'age_2021': [22, 25, 28]
        })

        test_nba_df = pd.DataFrame({
            'player_id': ['p1', 'p2'],
            'season': [2021, 2021],
            'mpg': [25, 22],
            'ppg': [15, 12],
            'apg': [5, 4],
            'rpg': [6, 5],
            'spg': [1.5, 1.2],
            'bpg': [0.8, 0.6],
            'three_pt_pct': [0.35, 0.32]
        })

        test_intl_df = pd.DataFrame({
            'player_id': ['p1', 'p2', 'p3'],
            'season': [2021, 2021, 2021],
            'league': ['EuroLeague', 'ACB', 'EuroLeague'],
            'team': ['TeamA', 'TeamB', 'TeamC'],
            'games': [30, 25, 35],
            'mpg': [25, 22, 28],
            'ppg': [12, 10, 15],
            'apg': [4, 3, 6],
            'rpg': [5, 4, 7],
            'spg': [1.2, 1.0, 1.5],
            'bpg': [0.6, 0.5, 0.9],
            'efficiency': [10, 8, 12],
            'true_shooting_percentage': [0.55, 0.52, 0.58],
            'three_pt_pct': [0.35, 0.32, 0.38]
        })

        # Test without ML predictions
        analyzer = ScoutingAnalyzer()
        prospects = analyzer.identify_scouting_targets(
            test_player_df, test_nba_df, test_intl_df, top_n=3
        )

        assert 'ml_multiplier' in prospects.columns, "Missing ml_multiplier"
        assert (prospects['ml_multiplier'] == 1.0).all(), "ml_multiplier should be 1.0 when no ML"
        print(f"\n✓ ml_multiplier = 1.0 when ML not available (documented)")

        # Test with ML predictions
        test_ml = pd.DataFrame({
            'player_id': ['p1', 'p2', 'p3'],
            'nba_success_prob': [0.7, 0.6, 0.8]
        })

        prospects_ml = analyzer.identify_scouting_targets(
            test_player_df, test_nba_df, test_intl_df, ml_predictions=test_ml, top_n=3
        )

        assert (prospects_ml['ml_multiplier'] != 1.0).any(), "ml_multiplier should vary with predictions"
        print(f"✓ ml_multiplier varies when ML available: {prospects_ml['ml_multiplier'].min():.2f}–{prospects_ml['ml_multiplier'].max():.2f}")

        print("\n✓ SCOUTING MODULE (FIXED): PASSED")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
