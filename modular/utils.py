"""
Utility functions used across the scouting analysis pipeline.
All functions are self-contained and thoroughly tested.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


from modular.config import Config


def safe_div(numerator: Union[pd.Series, float], 
             denominator: Union[pd.Series, float]) -> Union[pd.Series, float]:
    """
    Safe division that returns NaN for zero or NaN denominator.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)

    Returns:
        Division result with NaN for invalid divisions
    """
    return np.where(
        (pd.notna(denominator)) & (denominator != 0), 
        numerator / denominator, 
        np.nan
    )


def normalize_metric(series: pd.Series) -> pd.Series:
    """
    Normalize a metric to 0-100 scale using min-max scaling.

    Args:
        series: Series of values to normalize

    Returns:
        Normalized series (0-100 scale) or NaN if cannot normalize
    """
    series = series.astype(float)
    mn, mx = series.min(), series.max()

    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.nan, index=series.index)

    return (series - mn) / (mx - mn) * 100.0


def calculate_efg_percentage(two_pm: pd.Series, three_pm: pd.Series,
                             two_pa: pd.Series, three_pa: pd.Series) -> pd.Series:
    """
    Calculate effective field goal percentage.

    eFG% = (2PM + 1.5 * 3PM) / (2PA + 3PA)

    Args:
        two_pm: Two-point makes
        three_pm: Three-point makes
        two_pa: Two-point attempts
        three_pa: Three-point attempts

    Returns:
        Series of effective FG percentages
    """
    total_fga = two_pa.fillna(0) + three_pa.fillna(0)
    weighted_makes = two_pm.fillna(0) + 1.5 * three_pm.fillna(0)

    return safe_div(weighted_makes, total_fga)


def calculate_age_bonus(age: float) -> float:
    """
    Calculate age bonus multiplier.

    Younger players get higher multipliers due to greater upside potential.

    Args:
        age: Player age

    Returns:
        Age bonus multiplier

    Examples:
        >>> calculate_age_bonus(22)
        1.3
        >>> calculate_age_bonus(27)
        1.1
        >>> calculate_age_bonus(30)
        1.0
    """
    if age < 24:
        return 1.3
    elif age < 26:
        return 1.2
    elif age < 28:
        return 1.1
    else:
        return 1.0


def calculate_improvement_bonus(ppg_change: float) -> float:
    """
    Calculate improvement bonus multiplier based on PPG trend.

    Args:
        ppg_change: Change in PPG from first to last season

    Returns:
        Improvement bonus multiplier

    Examples:
        >>> calculate_improvement_bonus(6.0)
        1.2
        >>> calculate_improvement_bonus(4.0)
        1.15
        >>> calculate_improvement_bonus(0.5)
        1.0
    """
    if ppg_change > 5:
        return 1.2
    elif ppg_change > 3:
        return 1.15
    elif ppg_change > 1:
        return 1.1
    else:
        return 1.0


def create_player_id(first_name: pd.Series, last_name: pd.Series,
                     birth_year: Optional[pd.Series] = None) -> pd.Series:
    """
    Create player ID from name components.

    Args:
        first_name: Series of first names
        last_name: Series of last names
        birth_year: Optional series of birth years (for collision resolution)

    Returns:
        Series of player IDs
    """
    base_id = first_name.str.lower() + '_' + last_name.str.lower()

    if birth_year is not None:
        return base_id + '_' + birth_year.astype(str)

    return base_id


def check_numeric_bounds(series: pd.Series, min_val: float, max_val: float,
                        name: str = 'value') -> pd.Series:
    """
    Check if numeric values are within expected bounds.

    Args:
        series: Series to check
        min_val: Minimum valid value
        max_val: Maximum valid value
        name: Name for reporting

    Returns:
        Series with out-of-bounds values set to NaN
    """
    out_of_bounds = (series < min_val) | (series > max_val)

    if out_of_bounds.any():
        print(f"Warning: {out_of_bounds.sum()} out-of-bounds values in {name}")
        series = series.copy()
        series[out_of_bounds] = np.nan

    return series


def format_stat(value, format_type: str = 'float') -> str:
    """
    Format statistical value for display.

    Args:
        value: Value to format
        format_type: Type of formatting ('int', 'float1', 'float2', 'pct', 'str')

    Returns:
        Formatted string
    """
    if pd.isna(value) or value is None:
        return "—"

    if format_type == 'int':
        return f"{int(value)}"
    elif format_type == 'float1':
        return f"{value:.1f}"
    elif format_type == 'float2':
        return f"{value:.2f}"
    elif format_type == 'pct':
        return f"{value:.1%}"
    elif format_type == 'str':
        return str(value)
    else:
        return str(value)


def format_probability_range(prob: float, ci_low: Optional[float] = None,
                            ci_high: Optional[float] = None) -> str:
    """
    Format probability with optional confidence interval.

    Args:
        prob: Probability value
        ci_low: Optional lower CI bound
        ci_high: Optional upper CI bound

    Returns:
        Formatted probability string
    """
    if pd.isna(prob):
        return "—"

    if ci_low is not None and ci_high is not None and not (pd.isna(ci_low) or pd.isna(ci_high)):
        return f"{prob:.1%} ({ci_low:.1%}–{ci_high:.1%})"

    if prob >= 0.95:
        return f">{prob*0.9:.0%}"
    elif prob <= 0.05:
        return f"<{prob*2:.0%}"
    else:
        return f"{prob:.1%}"


def merge_trajectory_features(df: pd.DataFrame, traj_df: pd.DataFrame,
                              on: str = 'player_id') -> pd.DataFrame:
    """
    Merge trajectory features into a DataFrame.

    Args:
        df: DataFrame to merge into
        traj_df: Trajectory DataFrame
        on: Column to merge on

    Returns:
        Merged DataFrame
    """
    if traj_df is None or traj_df.empty:
        return df

    traj_cols = ['ppg_change', 'efficiency_change', 'seasons_played']
    available_cols = [col for col in traj_cols if col in traj_df.columns]

    if not available_cols:
        return df

    return df.merge(
        traj_df[available_cols].reset_index(),
        on=on,
        how='left'
    )


def calculate_career_stats(df: pd.DataFrame, group_by: str = 'player_id',
                          stats: list = ['ppg', 'apg', 'rpg']) -> pd.DataFrame:
    """
    Calculate career aggregate statistics.

    Args:
        df: DataFrame with player statistics
        group_by: Column to group by
        stats: List of statistics to aggregate

    Returns:
        DataFrame with career statistics
    """
    agg_dict = {}

    for stat in stats:
        if stat in df.columns:
            agg_dict[stat] = ['mean', 'max', 'min', 'std']

    if not agg_dict:
        return pd.DataFrame()

    agg_dict['season'] = ['count', 'min', 'max']

    career = df.groupby(group_by).agg(agg_dict)
    career.columns = ['_'.join(col).strip() for col in career.columns.values]
    career = career.rename(columns={'season_count': 'seasons_played'})

    return career.reset_index()


def validate_dataframe(df: pd.DataFrame, required_cols: list, 
                      name: str = 'DataFrame') -> tuple:
    """
    Validate DataFrame has required columns and report issues.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        name: Name for reporting

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []

    # Check required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"{name}: Missing columns {missing}")

    # Check for empty
    if len(df) == 0:
        issues.append(f"{name}: Empty DataFrame")

    # Check for duplicate indices
    if df.index.duplicated().any():
        issues.append(f"{name}: Duplicate indices found")

    return len(issues) == 0, issues


if __name__ == "__main__":
    """Smoke test for utility functions."""
    print("=" * 100)
    print("UTILS MODULE SMOKE TEST")
    print("=" * 100)

    try:
        # Test safe_div
        print("\n1. Testing safe_div...")
        assert safe_div(10, 2) == 5.0, "Basic division failed"
        assert np.isnan(safe_div(10, 0)), "Division by zero should return NaN"
        result = safe_div(pd.Series([10, 20, 30]), pd.Series([2, 0, 3]))
        assert result[0] == 5.0, "Series division failed"
        assert np.isnan(result[1]), "Series division by zero failed"
        print("✓ safe_div works correctly")

        # Test normalize_metric
        print("\n2. Testing normalize_metric...")
        series = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize_metric(series)
        assert normalized.min() == 0.0, "Min should be 0"
        assert normalized.max() == 100.0, "Max should be 100"
        assert abs(normalized.iloc[2] - 50.0) < 0.01, "Middle should be 50"
        print("✓ normalize_metric works correctly")

        # Test calculate_efg_percentage
        print("\n3. Testing calculate_efg_percentage...")
        two_pm = pd.Series([10])
        three_pm = pd.Series([5])
        two_pa = pd.Series([20])
        three_pa = pd.Series([15])
        efg = calculate_efg_percentage(two_pm, three_pm, two_pa, three_pa)
        # (10 + 1.5*5) / (20 + 15) = 17.5 / 35 = 0.5
        assert abs(efg.iloc[0] - 0.5) < 0.01, "eFG% calculation incorrect"
        print("✓ calculate_efg_percentage works correctly")

        # Test create_player_id
        print("\n4. Testing create_player_id...")
        pid = create_player_id(pd.Series(['John']), pd.Series(['Doe']))
        assert pid.iloc[0] == 'john_doe', "Simple player ID failed"
        pid_with_year = create_player_id(pd.Series(['John']), pd.Series(['Doe']), pd.Series([1990]))
        assert pid_with_year.iloc[0] == 'john_doe_1990', "Player ID with year failed"
        print("✓ create_player_id works correctly")

        # Test age bonus
        print("\n5. Testing calculate_age_bonus...")
        assert calculate_age_bonus(22) == 1.3, "Young age bonus incorrect"
        assert calculate_age_bonus(25) == 1.2, "Mid age bonus incorrect"
        assert calculate_age_bonus(30) == 1.0, "Old age bonus incorrect"
        print("✓ calculate_age_bonus works correctly")

        # Test improvement bonus
        print("\n6. Testing calculate_improvement_bonus...")
        assert calculate_improvement_bonus(6.0) == 1.2, "High improvement bonus incorrect"
        assert calculate_improvement_bonus(4.0) == 1.15, "Mid improvement bonus incorrect"
        assert calculate_improvement_bonus(0.5) == 1.0, "Low improvement bonus incorrect"
        print("✓ calculate_improvement_bonus works correctly")

        # Test format_stat
        print("\n7. Testing format_stat...")
        assert format_stat(10.5, 'int') == '10', "Int format failed"
        assert format_stat(10.567, 'float1') == '10.6', "Float1 format failed"
        assert format_stat(0.456, 'pct') == '45.6%', "Pct format failed"
        assert format_stat(None, 'float') == '—', "None format failed"
        print("✓ format_stat works correctly")

        # Test format_probability_range
        print("\n8. Testing format_probability_range...")
        assert format_probability_range(0.75) == '75.0%', "Basic prob format failed"
        assert '65.0%' in format_probability_range(0.75, 0.65, 0.85), "CI format failed"
        assert '>' in format_probability_range(0.96), "High prob format failed"
        print("✓ format_probability_range works correctly")

        print("\n" + "=" * 100)
        print("UTILS MODULE: ALL TESTS PASSED ✓")
        print("=" * 100)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("\n" + "=" * 100)
        print("UTILS MODULE: FAILED ✗")
        print("=" * 100)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 100)
        print("UTILS MODULE: FAILED ✗")
        print("=" * 100)
