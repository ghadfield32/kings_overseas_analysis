"""
Statistics calculation module for basketball scouting analysis.
Calculates per-game, per-36, and efficiency metrics for player data.
"""

import pandas as pd
import numpy as np

from utils import safe_div, calculate_efg_percentage


class StatisticsCalculator:
    """Calculates basketball statistics from raw game data."""

    @staticmethod
    def calculate_per_game_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-game statistics.

        Args:
            df: DataFrame with raw game stats

        Returns:
            DataFrame with added per-game columns
        """
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
        """
        Calculate shooting percentage statistics.

        Args:
            df: DataFrame with shot attempt/made data

        Returns:
            DataFrame with added shooting percentage columns
        """
        # Basic shooting percentages
        df['fg_pct'] = safe_div(df['two_points_made'], df['two_points_attempted'])
        df['three_pt_pct'] = safe_div(df['three_points_made'], df['three_points_attempted'])
        df['ft_pct'] = safe_div(df['free_throws_made'], df['free_throws_attempted'])

        # Effective field goal percentage
        df['efg_pct'] = calculate_efg_percentage(
            df['two_points_made'],
            df['three_points_made'],
            df['two_points_attempted'],
            df['three_points_attempted']
        )

        return df

    @staticmethod
    def calculate_efficiency(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate efficiency rating.

        Efficiency = (Points + Rebounds + Assists + Steals + Blocks 
                      - Missed FG - Missed FT - Turnovers) / Games

        Args:
            df: DataFrame with game stats

        Returns:
            DataFrame with added efficiency column
        """
        games = df['games'].replace(0, np.nan)

        # Calculate misses
        misses = (
            (df['two_points_attempted'].fillna(0) - df['two_points_made'].fillna(0)) +
            (df['three_points_attempted'].fillna(0) - df['three_points_made'].fillna(0)) +
            (df['free_throws_attempted'].fillna(0) - df['free_throws_made'].fillna(0))
        )

        # Positive contributions
        positive = (
            df['points'].fillna(0) +
            df['offensive_rebounds'].fillna(0) + df['defensive_rebounds'].fillna(0) +
            df['assists'].fillna(0) + 
            df['steals'].fillna(0) + 
            df['blocked_shots'].fillna(0)
        )

        # Negative contributions
        negative = misses + df['turnovers'].fillna(0)

        df['efficiency'] = safe_div(positive - negative, games)

        return df

    @staticmethod
    def calculate_per_36_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-36-minute statistics.

        Args:
            df: DataFrame with game stats

        Returns:
            DataFrame with added per-36 columns
        """
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
        """
        Calculate usage percentage if not already present.

        Usage% = Player Possessions / Team Possessions (while player on court)
        Approximated as: (FGA + 0.44*FTA + TOV) / (Team FGA + 0.44*Team FTA + Team TOV)

        Args:
            df: DataFrame with possession data

        Returns:
            DataFrame with usage_percentage column
        """
        # Only calculate if missing or all NaN
        if 'usage_percentage' not in df.columns or df['usage_percentage'].isna().all():
            total_fga = df['two_points_attempted'].fillna(0) + df['three_points_attempted'].fillna(0)

            # Use team possessions if available, otherwise estimate
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
        """
        Calculate plus/minus per game if plus_minus column exists.

        Args:
            df: DataFrame with plus_minus and games

        Returns:
            DataFrame with plus_minus_per_game column
        """
        if 'plus_minus' in df.columns:
            games = df['games'].replace(0, np.nan)
            df['plus_minus_per_game'] = safe_div(df['plus_minus'], games)
        else:
            df['plus_minus_per_game'] = np.nan

        return df

    @classmethod
    def calculate_all_statistics(cls, df: pd.DataFrame, dataset_name: str = '') -> pd.DataFrame:
        """
        Calculate all statistics for a dataset.

        Args:
            df: DataFrame with raw stats
            dataset_name: Name for logging purposes

        Returns:
            DataFrame with all calculated statistics
        """
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

    Args:
        nba_df: NBA statistics DataFrame
        intl_df: International statistics DataFrame

    Returns:
        Tuple of (nba_df, intl_df) with calculated statistics
    """
    print("\n" + "=" * 100)
    print("CALCULATING STATISTICS")
    print("=" * 100)

    calc = StatisticsCalculator()

    nba_df = calc.calculate_all_statistics(nba_df, 'NBA')
    intl_df = calc.calculate_all_statistics(intl_df, 'International')

    print("Statistics calculation complete.")

    return nba_df, intl_df


if __name__ == "__main__":
    """Smoke test for statistics module."""
    print("=" * 100)
    print("STATISTICS MODULE SMOKE TEST")
    print("=" * 100)

    try:
        # Create test data
        test_data = pd.DataFrame({
            'player_id': ['player1', 'player2', 'player3'],
            'games': [50, 60, 70],
            'minutes': [1200, 1500, 1800],
            'points': [600, 900, 1200],
            'assists': [200, 300, 250],
            'offensive_rebounds': [50, 40, 60],
            'defensive_rebounds': [150, 180, 200],
            'steals': [30, 40, 35],
            'blocked_shots': [20, 15, 25],
            'turnovers': [80, 100, 90],
            'personal_fouls': [100, 120, 110],
            'two_points_made': [150, 200, 250],
            'two_points_attempted': [300, 400, 500],
            'three_points_made': [50, 100, 100],
            'three_points_attempted': [150, 300, 300],
            'free_throws_made': [100, 100, 100],
            'free_throws_attempted': [120, 130, 140],
        })

        print("\nTest data created:")
        print(f"  {len(test_data)} players")
        print(f"  Columns: {', '.join(test_data.columns[:8])}...")

        # Calculate statistics
        calc = StatisticsCalculator()
        result = calc.calculate_all_statistics(test_data.copy(), 'Test')

        print("\n[OK] Statistics calculated")

        # Verify new columns exist
        expected_cols = ['ppg', 'apg', 'rpg', 'spg', 'bpg', 'mpg', 
                        'fg_pct', 'three_pt_pct', 'ft_pct', 'efg_pct',
                        'efficiency', 'pts_per_36', 'usage_percentage']

        missing = [col for col in expected_cols if col not in result.columns]
        if missing:
            print(f"\n[FAIL] Missing columns: {missing}")
        else:
            print(f"[OK] All expected columns present")

        # Display sample results
        print("\nSample statistics (Player 1):")
        player1 = result.iloc[0]
        print(f"  PPG: {player1['ppg']:.1f}")
        print(f"  APG: {player1['apg']:.1f}")
        print(f"  RPG: {player1['rpg']:.1f}")
        print(f"  FG%: {player1['fg_pct']:.3f}")
        print(f"  3P%: {player1['three_pt_pct']:.3f}")
        print(f"  EFF: {player1['efficiency']:.1f}")
        print(f"  Pts/36: {player1['pts_per_36']:.1f}")

        # Verify calculations
        print("\nVerifying calculations...")
        assert abs(player1['ppg'] - (600/50)) < 0.01, "PPG calculation incorrect"
        assert abs(player1['apg'] - (200/50)) < 0.01, "APG calculation incorrect"
        assert abs(player1['rpg'] - (200/50)) < 0.01, "RPG calculation incorrect"
        print("[OK] Calculations verified")

        # Test with missing data
        print("\nTesting with missing data...")
        test_missing = test_data.copy()
        test_missing.loc[0, 'games'] = 0  # Should handle division by zero
        test_missing.loc[1, 'minutes'] = np.nan  # Should handle NaN

        result_missing = calc.calculate_all_statistics(test_missing, 'Missing Data')
        assert pd.isna(result_missing.loc[0, 'ppg']), "Should return NaN for zero games"
        assert pd.isna(result_missing.loc[1, 'pts_per_36']), "Should return NaN for missing minutes"
        print("[OK] Missing data handled correctly")

        print("\n" + "=" * 100)
        print("STATISTICS MODULE: PASSED")
        print("=" * 100)

    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 100)
        print("STATISTICS MODULE: FAILED")
        print("=" * 100)
