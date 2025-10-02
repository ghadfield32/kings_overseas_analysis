"""
Statistics calculation module - FIXED VERSION
Addresses: Proper counting and documentation of all calculated statistics
"""

import pandas as pd
import numpy as np

from modular.utils import safe_div, calculate_efg_percentage


class StatisticsCalculator:
    """Calculates basketball statistics from raw game data."""

    # FIXED: Define all statistics that will be calculated
    CALCULATED_STATS = {
        'per_game': ['ppg', 'apg', 'rpg', 'spg', 'bpg', 'mpg', 'topg', 'fpg'],
        'per_36': ['pts_per_36', 'ast_per_36', 'reb_per_36'],
        'shooting': ['fg_pct', 'three_pt_pct', 'ft_pct', 'efg_pct'],
        'efficiency': ['efficiency'],
        'other': ['usage_percentage', 'plus_minus_per_game']
    }

    @staticmethod
    def get_stat_count():
        """Return total number of calculated statistics."""
        return sum(len(stats) for stats in StatisticsCalculator.CALCULATED_STATS.values())

    @staticmethod
    def get_stat_list():
        """Return flat list of all calculated statistics."""
        all_stats = []
        for stats in StatisticsCalculator.CALCULATED_STATS.values():
            all_stats.extend(stats)
        return all_stats

    @staticmethod
    def calculate_per_game_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-game statistics."""
        games = df['games'].replace(0, np.nan)

        df['ppg'] = safe_div(df['points'], games)
        df['apg'] = safe_div(df['assists'], games)
        df['rpg'] = safe_div(
            df['offensive_rebounds'].fillna(0) + df['defensive_rebounds'].fillna(0), 
            games
        )
        df['spg'] = safe_div(df['steals'], games)
        df['bpg'] = safe_div(df['blocked_shots'], games)
        df['mpg'] = safe_div(df['minutes'], games)
        df['topg'] = safe_div(df['turnovers'], games)
        df['fpg'] = safe_div(df.get('personal_fouls', 0), games)

        return df

    @staticmethod
    def calculate_shooting_percentages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate shooting percentage statistics."""
        df['fg_pct'] = safe_div(df['two_points_made'], df['two_points_attempted'])
        df['three_pt_pct'] = safe_div(df['three_points_made'], df['three_points_attempted'])
        df['ft_pct'] = safe_div(df['free_throws_made'], df['free_throws_attempted'])

        df['efg_pct'] = calculate_efg_percentage(
            df['two_points_made'],
            df['three_points_made'],
            df['two_points_attempted'],
            df['three_points_attempted']
        )

        return df

    @staticmethod
    def calculate_efficiency(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate efficiency rating."""
        games = df['games'].replace(0, np.nan)

        misses = (
            (df['two_points_attempted'].fillna(0) - df['two_points_made'].fillna(0)) +
            (df['three_points_attempted'].fillna(0) - df['three_points_made'].fillna(0)) +
            (df['free_throws_attempted'].fillna(0) - df['free_throws_made'].fillna(0))
        )

        positive = (
            df['points'].fillna(0) +
            df['offensive_rebounds'].fillna(0) + df['defensive_rebounds'].fillna(0) +
            df['assists'].fillna(0) + 
            df['steals'].fillna(0) + 
            df['blocked_shots'].fillna(0)
        )

        negative = misses + df['turnovers'].fillna(0)

        df['efficiency'] = safe_div(positive - negative, games)

        return df

    @staticmethod
    def calculate_per_36_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-36-minute statistics."""
        minutes = df['minutes'].replace(0, np.nan)

        df['pts_per_36'] = safe_div(df['points'], minutes) * 36
        df['ast_per_36'] = safe_div(df['assists'], minutes) * 36
        df['reb_per_36'] = safe_div(
            df['offensive_rebounds'].fillna(0) + df['defensive_rebounds'].fillna(0),
            minutes
        ) * 36

        return df

    @staticmethod
    def calculate_usage_percentage(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate usage percentage if not already present."""
        if 'usage_percentage' not in df.columns or df['usage_percentage'].isna().all():
            total_fga = df['two_points_attempted'].fillna(0) + df['three_points_attempted'].fillna(0)

            team_poss = df.get(
                'team_possessions', 
                total_fga + 0.44 * df['free_throws_attempted'].fillna(0) + df['turnovers'].fillna(0)
            )

            player_poss = (
                total_fga + 
                0.44 * df['free_throws_attempted'].fillna(0) + 
                df['turnovers'].fillna(0)
            )

            df['usage_percentage'] = safe_div(player_poss, team_poss)

        return df

    @staticmethod
    def calculate_plus_minus_per_game(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate plus/minus per game if plus_minus column exists."""
        if 'plus_minus' in df.columns:
            games = df['games'].replace(0, np.nan)
            df['plus_minus_per_game'] = safe_div(df['plus_minus'], games)
        else:
            df['plus_minus_per_game'] = np.nan

        return df

    @classmethod
    def calculate_all_statistics(cls, df: pd.DataFrame, dataset_name: str = '') -> pd.DataFrame:
        """Calculate all statistics for a dataset."""
        if dataset_name:
            print(f"Calculating statistics for {dataset_name}...")

        df = cls.calculate_per_game_stats(df)
        df = cls.calculate_shooting_percentages(df)
        df = cls.calculate_efficiency(df)
        df = cls.calculate_per_36_stats(df)
        df = cls.calculate_usage_percentage(df)
        df = cls.calculate_plus_minus_per_game(df)

        return df


def calculate_statistics(nba_df: pd.DataFrame, intl_df: pd.DataFrame) -> tuple:
    """
    Calculate all statistics for NBA and International datasets.

    FIXED: Properly reports count of calculated statistics.
    """
    print("\n" + "=" * 100)
    print("CALCULATING STATISTICS")
    print("=" * 100)

    calc = StatisticsCalculator()

    nba_df = calc.calculate_all_statistics(nba_df, 'NBA')
    intl_df = calc.calculate_all_statistics(intl_df, 'International')

    # FIXED: Report accurate count
    stat_count = calc.get_stat_count()
    stat_list = calc.get_stat_list()

    print(f"\n✓ Calculated {stat_count} statistics:")
    print(f"  Per-game (8): {', '.join(calc.CALCULATED_STATS['per_game'])}")
    print(f"  Per-36 (3): {', '.join(calc.CALCULATED_STATS['per_36'])}")
    print(f"  Shooting (4): {', '.join(calc.CALCULATED_STATS['shooting'])}")
    print(f"  Efficiency (1): {', '.join(calc.CALCULATED_STATS['efficiency'])}")
    print(f"  Other (2): {', '.join(calc.CALCULATED_STATS['other'])}")

    print("\nStatistics calculation complete.")

    return nba_df, intl_df


if __name__ == "__main__":
    """Smoke test."""
    print("="*100)
    print("STATISTICS MODULE SMOKE TEST (FIXED VERSION)")
    print("="*100)

    try:
        test_data = pd.DataFrame({
            'player_id': ['p1', 'p2'],
            'games': [50, 60],
            'minutes': [1200, 1500],
            'points': [600, 900],
            'assists': [200, 300],
            'offensive_rebounds': [50, 40],
            'defensive_rebounds': [150, 180],
            'steals': [30, 40],
            'blocked_shots': [20, 15],
            'turnovers': [80, 100],
            'personal_fouls': [100, 120],
            'two_points_made': [150, 200],
            'two_points_attempted': [300, 400],
            'three_points_made': [50, 100],
            'three_points_attempted': [150, 300],
            'free_throws_made': [100, 100],
            'free_throws_attempted': [120, 130],
        })

        calc = StatisticsCalculator()
        result = calc.calculate_all_statistics(test_data.copy(), 'Test')

        # Verify stat count
        expected_count = calc.get_stat_count()
        calculated_stats = [col for col in result.columns if col not in test_data.columns]
        actual_count = len(calculated_stats)

        print(f"\n✓ Expected {expected_count} stats, calculated {actual_count}")
        assert actual_count == expected_count, f"Stat count mismatch"

        # Verify all expected stats present
        expected_stats = calc.get_stat_list()
        for stat in expected_stats:
            assert stat in result.columns, f"Missing {stat}"

        print(f"✓ All {expected_count} statistics present in output")

        # Verify calculations
        assert abs(result['ppg'].iloc[0] - 12.0) < 0.01, "PPG wrong"
        assert abs(result['apg'].iloc[0] - 4.0) < 0.01, "APG wrong"
        print("✓ Sample calculations verified")

        print("\n✓ STATISTICS MODULE (FIXED): PASSED")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
