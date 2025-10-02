"""
Performance analysis module - FIXED VERSION
Addresses: Consistent definitions for trending/improving players with clear thresholds
"""

import pandas as pd
import numpy as np



class PerformanceAnalyzer:
    """Analyzes player performance patterns and development trajectories."""

    @staticmethod
    def calculate_player_trajectories(intl_df: pd.DataFrame, min_seasons: int = 2) -> pd.DataFrame:
        """Calculate player development trajectories from international data."""
        print(f"\nCalculating player trajectories (min {min_seasons} seasons)...")

        intl_sorted = intl_df.sort_values('season')
        intl_grouped = intl_sorted.groupby('player_id')

        first_season = intl_grouped.first()
        last_season = intl_grouped.last()
        season_count = intl_grouped.size()

        traj_df = pd.DataFrame({
            'seasons_played': season_count,
            'ppg_change': last_season['ppg'] - first_season['ppg'],
            'efficiency_change': last_season['efficiency'] - first_season['efficiency'],
            'first_season': first_season['season'],
            'last_season': last_season['season']
        })

        traj_df = traj_df[traj_df['seasons_played'] >= min_seasons]

        print(f"Players with {min_seasons}+ seasons: {len(traj_df):,}")
        if len(traj_df) > 0:
            print(f"Average PPG change: {traj_df['ppg_change'].mean():+.2f}")
            print(f"Average efficiency change: {traj_df['efficiency_change'].mean():+.2f}")

        return traj_df

    @staticmethod
    def identify_trending_players(traj_df: pd.DataFrame, 
                                  ppg_threshold: float = 3.0,
                                  eff_threshold: float = 2.0) -> pd.DataFrame:
        """
        FIXED: Identify trending players with SINGLE consistent definition.

        Trending = PPG improved by 3.0+ OR efficiency improved by 2.0+
        """
        trending = traj_df[
            (traj_df['ppg_change'] >= ppg_threshold) |
            (traj_df['efficiency_change'] >= eff_threshold)
        ].copy()

        # Add categorization for reporting
        trending['improvement_category'] = 'Moderate'
        trending.loc[
            (trending['ppg_change'] >= 5.0) | (trending['efficiency_change'] >= 3.0),
            'improvement_category'
        ] = 'Strong'

        moderate = (trending['improvement_category'] == 'Moderate').sum()
        strong = (trending['improvement_category'] == 'Strong').sum()

        print(f"\nTrending players (PPG≥{ppg_threshold} or EFF≥{eff_threshold}): {len(trending)}")
        print(f"  - Moderate improvement: {moderate}")
        print(f"  - Strong improvement: {strong}")

        return trending

    @staticmethod
    def analyze_league_performance(intl_df: pd.DataFrame, 
                                   season: int = 2021,
                                   min_games: int = 10) -> pd.DataFrame:
        """
        FIXED: Analyze performance with consistent game filter.
        """
        season_data = intl_df[
            (intl_df['season'] == season) & 
            (intl_df['games'] >= min_games)
        ].copy()

        if len(season_data) == 0:
            print(f"\nNo data for season {season} with min {min_games} games")
            return pd.DataFrame()

        league_stats = season_data.groupby('league').agg({
            'ppg': ['mean', 'median', 'std'],
            'apg': ['mean', 'median'],
            'rpg': ['mean', 'median'],
            'efficiency': ['mean', 'median'],
            'three_pt_pct': ['mean', 'median'],
            'player_id': 'count'
        }).round(2)

        league_stats.columns = ['_'.join(col).strip() for col in league_stats.columns.values]
        league_stats = league_stats.rename(columns={'player_id_count': 'num_players'})

        total_players = league_stats['num_players'].sum()
        print(f"\nLeague performance summary ({season}, min {min_games} games):")
        print(f"  Leagues analyzed: {len(league_stats)}")
        print(f"  Total players: {int(total_players)}")

        return league_stats.reset_index()

    @staticmethod
    def compare_nba_vs_international(nba_df: pd.DataFrame, 
                                    intl_df: pd.DataFrame,
                                    season: int = 2021,
                                    min_games: int = 10) -> dict:
        """
        FIXED: Compare with consistent filters applied to both datasets.
        """
        # Apply same filter to both
        nba_season = nba_df[(nba_df['season'] == season) & (nba_df['games'] >= min_games)]
        intl_season = intl_df[(intl_df['season'] == season) & (intl_df['games'] >= min_games)]

        comparison = {
            'season': season,
            'min_games': min_games,
            'nba_players': len(nba_season['player_id'].unique()),
            'intl_players': len(intl_season['player_id'].unique()),
            'nba_ppg_mean': nba_season['ppg'].mean(),
            'intl_ppg_mean': intl_season['ppg'].mean(),
            'nba_efficiency_mean': nba_season['efficiency'].mean(),
            'intl_efficiency_mean': intl_season['efficiency'].mean(),
            'nba_3pt_pct_mean': nba_season['three_pt_pct'].mean(),
            'intl_3pt_pct_mean': intl_season['three_pt_pct'].mean(),
        }

        print(f"\nNBA vs International comparison ({season}, min {min_games} games):")
        print(f"  NBA: {comparison['nba_players']} players")
        print(f"  International: {comparison['intl_players']} players")
        print(f"  Total: {comparison['nba_players'] + comparison['intl_players']} players")
        print(f"  PPG - NBA: {comparison['nba_ppg_mean']:.1f}, Intl: {comparison['intl_ppg_mean']:.1f}")
        print(f"  EFF - NBA: {comparison['nba_efficiency_mean']:.1f}, Intl: {comparison['intl_efficiency_mean']:.1f}")

        return comparison


def analyze_performance_patterns(intl_df: pd.DataFrame, 
                                nba_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main function to analyze performance patterns.

    FIXED: Consistent filtering and clear definitions throughout.
    """
    print("\n" + "=" * 100)
    print("ANALYZING PERFORMANCE PATTERNS")
    print("=" * 100)

    analyzer = PerformanceAnalyzer()

    traj_df = analyzer.calculate_player_trajectories(intl_df, min_seasons=2)

    if len(traj_df) > 0:
        trending = analyzer.identify_trending_players(traj_df)

    # FIXED: Use consistent min_games threshold
    league_stats = analyzer.analyze_league_performance(intl_df, season=2021, min_games=10)

    if nba_df is not None:
        # FIXED: Apply same filter to both datasets
        comparison = analyzer.compare_nba_vs_international(nba_df, intl_df, season=2021, min_games=10)

    print("\nPerformance analysis complete.")

    return traj_df


if __name__ == "__main__":
    """Smoke test."""
    print("="*100)
    print("ANALYSIS MODULE SMOKE TEST (FIXED VERSION)")
    print("="*100)

    try:
        test_intl = pd.DataFrame({
            'player_id': ['p1', 'p1', 'p1', 'p2', 'p2', 'p3'],
            'season': [2019, 2020, 2021, 2020, 2021, 2021],
            'league': ['EuroLeague', 'EuroLeague', 'EuroLeague', 'ACB', 'ACB', 'EuroLeague'],
            'games': [30, 35, 40, 25, 30, 20],
            'ppg': [10, 12, 15, 8, 9, 12],
            'apg': [3, 4, 5, 2, 3, 4],
            'rpg': [5, 5, 6, 4, 4, 5],
            'efficiency': [8, 10, 12, 6, 7, 9],
            'three_pt_pct': [0.35, 0.37, 0.40, 0.30, 0.32, 0.38]
        })

        test_nba = pd.DataFrame({
            'player_id': ['n1', 'n2'],
            'season': [2021, 2021],
            'games': [50, 60],
            'ppg': [15, 18],
            'efficiency': [12, 14],
            'three_pt_pct': [0.38, 0.40]
        })

        analyzer = PerformanceAnalyzer()

        # Test trajectories
        traj = analyzer.calculate_player_trajectories(test_intl, min_seasons=2)
        assert len(traj) > 0, "No trajectories calculated"
        print(f"\n✓ Trajectories: {len(traj)} players")

        # Test trending with clear definition
        trending = analyzer.identify_trending_players(traj, ppg_threshold=3.0)
        print(f"✓ Trending players: {len(trending)}")

        # Verify categories exist
        if len(trending) > 0:
            assert 'improvement_category' in trending.columns, "Missing improvement category"
            print(f"✓ Improvement categories assigned")

        # Test league analysis with filter
        league_stats = analyzer.analyze_league_performance(test_intl, season=2021, min_games=10)
        print(f"✓ League analysis: {len(league_stats)} leagues")

        # Test comparison with consistent filters
        comparison = analyzer.compare_nba_vs_international(test_nba, test_intl, season=2021, min_games=10)
        assert 'min_games' in comparison, "Missing filter documentation"
        total = comparison['nba_players'] + comparison['intl_players']
        print(f"✓ Comparison: {total} total players (consistent filter)")

        print("\n✓ ANALYSIS MODULE (FIXED): PASSED")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
