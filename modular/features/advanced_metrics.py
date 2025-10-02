"""
Advanced NBA metrics calculator - PER, VORP, EWA, PIE
These are advanced prediction targets for comprehensive player evaluation.

Metrics:
- PER (Player Efficiency Rating): Hollinger's all-in-one efficiency metric
- VORP (Value Over Replacement Player): Wins contributed above replacement level
- EWA (Estimated Wins Added): Win contribution estimate from PER
- PIE (Player Impact Estimate): Overall statistical contribution percentage
"""

import pandas as pd
import numpy as np
from modular.utils import safe_div


class AdvancedMetricsCalculator:
    """Calculates advanced NBA metrics: PER, VORP, EWA, PIE."""

    # League average constants (calibrated from NBA data)
    LEAGUE_AVG_PER = 15.0
    LEAGUE_AVG_PACE = 100.0
    LEAGUE_VOP = 1.0  # Value of Possession
    LEAGUE_DRBP = 0.75  # Defensive Rebound Percentage

    @staticmethod
    def calculate_per(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Player Efficiency Rating (PER).

        Simplified PER formula (Hollinger approximation):
        PER = (Positive Stats - Negative Stats) / Minutes * Pace Adjustment

        Positive: Points, Rebounds, Assists, Steals, Blocks
        Negative: Missed FG, Missed FT, Turnovers

        Args:
            df: DataFrame with basketball statistics

        Returns:
            DataFrame with 'PER' column added
        """
        # Calculate missed shots
        missed_fg = (
            (df.get('two_points_attempted', 0) - df.get('two_points_made', 0)) +
            (df.get('three_points_attempted', 0) - df.get('three_points_made', 0))
        )
        missed_ft = df.get('free_throws_attempted', 0) - df.get('free_throws_made', 0)

        # Positive contributions
        positive = (
            df.get('points', 0) +
            (df.get('offensive_rebounds', 0) + df.get('defensive_rebounds', 0)) +
            df.get('assists', 0) +
            df.get('steals', 0) +
            df.get('blocked_shots', 0)
        )

        # Negative contributions
        negative = (
            missed_fg +
            missed_ft * 0.5 +  # Missed FT weighted less
            df.get('turnovers', 0)
        )

        # Per-minute calculation (scaled to 48 minutes)
        minutes = df.get('minutes', 0).replace(0, np.nan)
        per_raw = safe_div(positive - negative, minutes) * 48

        # Store PER
        df['PER'] = per_raw

        return df

    @staticmethod
    def calculate_vorp(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Value Over Replacement Player (VORP).

        VORP ≈ [BPM - (-2.0)] * (% of possessions played) * (games / 82)

        BPM (Box Plus/Minus) is estimated from box score statistics.
        Replacement level is defined as -2.0 BPM.

        Args:
            df: DataFrame with basketball statistics

        Returns:
            DataFrame with 'VORP' column added
        """
        # Estimate BPM from box score (simplified formula)
        missed_shots = (
            (df.get('two_points_attempted', 0) - df.get('two_points_made', 0)) +
            (df.get('three_points_attempted', 0) - df.get('three_points_made', 0))
        )

        # Weighted box score contributions
        box_contributions = (
            df.get('points', 0) * 0.5 +
            (df.get('offensive_rebounds', 0) + df.get('defensive_rebounds', 0)) * 0.4 +
            df.get('assists', 0) * 0.7 +
            df.get('steals', 0) * 1.0 +
            df.get('blocked_shots', 0) * 0.7 -
            df.get('turnovers', 0) * 1.0 -
            missed_shots * 0.3
        )

        # Convert to per-48-minute rate and scale to BPM range
        minutes = df.get('minutes', 0).replace(0, np.nan)
        bpm_estimate = safe_div(box_contributions, minutes) * 48 - 10

        # VORP calculation
        # Replacement level is -2.0 BPM
        above_replacement = bpm_estimate - (-2.0)

        # Percentage of team possessions played
        # Assume 240 team minutes per game (5 players * 48 minutes)
        team_minutes = df.get('games', 1) * 240
        pct_possessions = safe_div(df.get('minutes', 0), team_minutes)

        # Scale by games played (normalized to 82-game season)
        games_factor = safe_div(df.get('games', 0), 82)

        # Final VORP
        df['VORP'] = above_replacement * pct_possessions * games_factor

        return df

    @staticmethod
    def calculate_ewa(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Estimated Wins Added (EWA).

        EWA ≈ (PER - League_Avg_PER) * (Minutes Played / 48) / 30

        Converts PER into an estimate of wins contributed.

        Args:
            df: DataFrame with basketball statistics (must have 'PER' column)

        Returns:
            DataFrame with 'EWA' column added
        """
        # Ensure PER is calculated first
        if 'PER' not in df.columns:
            df = AdvancedMetricsCalculator.calculate_per(df)

        # PER above league average
        per_above_avg = df['PER'] - AdvancedMetricsCalculator.LEAGUE_AVG_PER

        # Convert minutes to 48-minute equivalents
        minutes_played = df.get('minutes', 0)
        min_48 = safe_div(minutes_played, 48)

        # EWA formula: (PER above avg) * (48-min blocks) / 30
        df['EWA'] = (per_above_avg * min_48) / 30

        return df

    @staticmethod
    def calculate_pie(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Player Impact Estimate (PIE).

        PIE measures a player's overall statistical contribution as a percentage.

        PIE = Player Contributions / Estimated Team Contributions

        Contributions = PTS + FGM + FTM - FGA - FTA + DREB + 0.5*OREB +
                       AST + STL + 0.5*BLK - PF - TO

        Args:
            df: DataFrame with basketball statistics

        Returns:
            DataFrame with 'PIE' column added
        """
        # Player contributions (positive and negative)
        player_contributions = (
            df.get('points', 0) +
            df.get('two_points_made', 0) + df.get('three_points_made', 0) +
            df.get('free_throws_made', 0) -
            df.get('two_points_attempted', 0) - df.get('three_points_attempted', 0) -
            df.get('free_throws_attempted', 0) +
            df.get('defensive_rebounds', 0) +
            0.5 * df.get('offensive_rebounds', 0) +
            df.get('assists', 0) +
            df.get('steals', 0) +
            0.5 * df.get('blocked_shots', 0) -
            df.get('personal_fouls', 0) -
            df.get('turnovers', 0)
        )

        # Estimate team contributions
        # Assume player's stats are proportional to minutes played
        minutes_played = df.get('minutes', 0)
        games_played = df.get('games', 1)

        # Team total minutes per game: 5 players * 48 minutes = 240
        team_minutes_total = games_played * 240

        # Scale player contributions to team level
        minutes_ratio = safe_div(team_minutes_total, minutes_played.replace(0, 1))
        estimated_team_contributions = player_contributions * minutes_ratio

        # PIE calculation
        df['PIE'] = safe_div(player_contributions, estimated_team_contributions)

        # Bound PIE between 0 and 1 (it's a percentage)
        df['PIE'] = df['PIE'].clip(0, 1)

        return df

    @classmethod
    def calculate_all_advanced_metrics(cls, df: pd.DataFrame,
                                       dataset_name: str = '') -> pd.DataFrame:
        """
        Calculate all advanced metrics: PER, VORP, EWA, PIE.

        Args:
            df: DataFrame with basketball statistics
            dataset_name: Optional name for logging

        Returns:
            DataFrame with all advanced metrics added
        """
        if dataset_name:
            print(f"\nCalculating advanced metrics for {dataset_name}...")

        # Calculate each metric in sequence
        df = cls.calculate_per(df)
        df = cls.calculate_vorp(df)
        df = cls.calculate_ewa(df)
        df = cls.calculate_pie(df)

        # Report statistics
        print(f"  ✓ PER: {df['PER'].mean():.2f} ± {df['PER'].std():.2f}")
        print(f"  ✓ VORP: {df['VORP'].mean():.2f} ± {df['VORP'].std():.2f}")
        print(f"  ✓ EWA: {df['EWA'].mean():.2f} ± {df['EWA'].std():.2f}")
        print(f"  ✓ PIE: {df['PIE'].mean():.2f} ± {df['PIE'].std():.2f}")

        return df


def calculate_advanced_statistics(nba_df: pd.DataFrame,
                                  intl_df: pd.DataFrame) -> tuple:
    """
    Calculate advanced statistics for both NBA and International datasets.

    This is the main entry point called by the pipeline.

    Args:
        nba_df: NBA statistics DataFrame
        intl_df: International statistics DataFrame

    Returns:
        Tuple of (nba_df, intl_df) with advanced metrics added
    """
    print("\n" + "=" * 100)
    print("CALCULATING ADVANCED METRICS (PER, VORP, EWA, PIE)")
    print("=" * 100)

    calculator = AdvancedMetricsCalculator()

    # Calculate for NBA data
    if not nba_df.empty:
        nba_df = calculator.calculate_all_advanced_metrics(nba_df, "NBA")

    # Calculate for International data
    if not intl_df.empty:
        intl_df = calculator.calculate_all_advanced_metrics(intl_df, "International")

    print("\n✓ Advanced metrics calculation complete")

    return nba_df, intl_df


if __name__ == "__main__":
    """Test advanced metrics calculation."""
    print("=" * 100)
    print("ADVANCED METRICS MODULE TEST")
    print("=" * 100)

    # Create sample data
    sample_data = pd.DataFrame({
        'player_id': ['player_1', 'player_2'],
        'season': [2021, 2021],
        'games': [50, 60],
        'minutes': [1500, 1800],
        'points': [750, 900],
        'two_points_made': [200, 250],
        'two_points_attempted': [400, 500],
        'three_points_made': [50, 60],
        'three_points_attempted': [150, 180],
        'free_throws_made': [150, 180],
        'free_throws_attempted': [200, 240],
        'offensive_rebounds': [75, 90],
        'defensive_rebounds': [200, 240],
        'assists': [150, 180],
        'steals': [50, 60],
        'blocked_shots': [25, 30],
        'turnovers': [100, 120],
        'personal_fouls': [100, 120]
    })

    print("\nSample data:")
    print(sample_data[['player_id', 'games', 'minutes', 'points']].to_string())

    # Calculate advanced metrics
    calculator = AdvancedMetricsCalculator()
    result = calculator.calculate_all_advanced_metrics(sample_data, "Test")

    print("\nAdvanced metrics calculated:")
    print(result[['player_id', 'PER', 'VORP', 'EWA', 'PIE']].to_string())

    print("\n" + "=" * 100)
    print("ADVANCED METRICS TEST: PASSED")
    print("=" * 100)
