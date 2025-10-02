"""
Data loading and validation module - FIXED VERSION
Addresses: Usage % scale issue, consistent issue counting
"""

import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

from modular.config import Config
from modular.utils import safe_div, create_player_id



class DataLoader:
    """Handles loading and validation of player data from JSON files."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or Config.DATA_DIR
        self.player_df = None
        self.nba_df = None
        self.intl_df = None
        # FIXED: Track both issue types AND affected row counts
        self.data_quality_issues = []  # List of issue descriptions
        self.issue_stats = {
            'total_issue_types': 0,
            'total_rows_affected': 0,
            'by_dataset': {}
        }

    def _log_issue(self, dataset: str, issue_type: str, description: str, rows_affected: int = 0):
        """
        Centralized issue logging with consistent counting.

        Args:
            dataset: Dataset name (Player/NBA/International)
            issue_type: Type of issue (duplicate_names, out_of_range, etc.)
            description: Full description
            rows_affected: Number of rows affected
        """
        self.data_quality_issues.append(description)
        self.issue_stats['total_issue_types'] += 1
        self.issue_stats['total_rows_affected'] += rows_affected

        if dataset not in self.issue_stats['by_dataset']:
            self.issue_stats['by_dataset'][dataset] = {
                'issue_types': 0,
                'rows_affected': 0
            }
        self.issue_stats['by_dataset'][dataset]['issue_types'] += 1
        self.issue_stats['by_dataset'][dataset]['rows_affected'] += rows_affected

    def load_raw_data(self) -> 'DataLoader':
        """Load raw JSON data files into DataFrames."""
        print("=" * 100)
        print("LOADING DATA")
        print("=" * 100)

        with open(Path(self.data_dir) / Config.PLAYER_FILE, 'r') as f:
            self.player_df = pd.DataFrame(json.load(f))
        with open(Path(self.data_dir) / Config.NBA_FILE, 'r') as f:
            self.nba_df = pd.DataFrame(json.load(f))
        with open(Path(self.data_dir) / Config.INTL_FILE, 'r') as f:
            self.intl_df = pd.DataFrame(json.load(f))

        print(f"Loaded {len(self.player_df):,} player rows")
        print(f"Loaded {len(self.nba_df):,} NBA player-seasons")
        print(f"Loaded {len(self.intl_df):,} International player-seasons")
        return self

    def process_demographics(self) -> 'DataLoader':
        """Process player demographics including birth dates and ages."""
        print("\nProcessing demographics...")
        self.player_df['birth_date'] = pd.to_datetime(self.player_df['birth_date'])
        self.player_df['birth_year'] = self.player_df['birth_date'].dt.year
        # FIXED: Age calculated as of 2021 (not current date)
        self.player_df['age_2021'] = 2021 - self.player_df['birth_year']
        print(f"Age range (as of 2021): {self.player_df['age_2021'].min():.0f}–{self.player_df['age_2021'].max():.0f}")
        return self

    def create_player_ids(self) -> 'DataLoader':
        """Create initial player IDs for all datasets."""
        print("\nCreating player IDs...")
        for df in [self.player_df, self.nba_df, self.intl_df]:
            df['player_id'] = (df['first_name'].str.lower() + '_' + df['last_name'].str.lower())
        return self

    def check_duplicate_names(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Check for duplicate player names and report them."""
        name_counts = df.groupby(['first_name', 'last_name']).size()
        duplicates = name_counts[name_counts > 1]

        if len(duplicates) > 0:
            rows_affected = duplicates.sum()
            issue = f"[{name}] Found {len(duplicates)} duplicate name(s) affecting {rows_affected} rows:"
            for (fname, lname), count in duplicates.items():
                issue += f"\n  - {fname} {lname}: {count} occurrences"

            self._log_issue(name, 'duplicate_names', issue, rows_affected)
            print(issue)

            if 'birth_year' in df.columns:
                print(f"[{name}] Enhancing player_id with birth_year to resolve collisions")
                df['player_id'] = create_player_id(df['first_name'], df['last_name'], df['birth_year'])

        return df

    def detect_collisions(self) -> 'DataLoader':
        """Detect and resolve player name collisions across datasets."""
        print("\n" + "=" * 100)
        print("CHECKING FOR DUPLICATE NAMES")
        print("=" * 100)

        self.player_df = self.check_duplicate_names(self.player_df, 'Player')
        nba_temp = self.nba_df[['first_name', 'last_name']].drop_duplicates()
        self.check_duplicate_names(nba_temp, 'NBA')
        intl_temp = self.intl_df[['first_name', 'last_name']].drop_duplicates()
        self.check_duplicate_names(intl_temp, 'International')

        if any('birth_year' in issue for issue in self.data_quality_issues):
            print("\nUpdating player_id in NBA and International data...")
            player_id_map = self.player_df.set_index(
                self.player_df['first_name'].str.lower() + '_' + self.player_df['last_name'].str.lower()
            )['player_id'].to_dict()

            simple_id = self.nba_df['first_name'].str.lower() + '_' + self.nba_df['last_name'].str.lower()
            self.nba_df['player_id'] = simple_id.map(player_id_map).fillna(simple_id)

            simple_id = self.intl_df['first_name'].str.lower() + '_' + self.intl_df['last_name'].str.lower()
            self.intl_df['player_id'] = simple_id.map(player_id_map).fillna(simple_id)

        return self

    def validate_data(self, df: pd.DataFrame, name: str, required_cols: List[str]) -> pd.DataFrame:
        """ validation with usage% scale fix."""
        print(f"\n[{name}] Validating data...")

        # Check required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            issue = f"[{name}] Missing required columns (adding as NaN): {missing}"
            self._log_issue(name, 'missing_columns', issue, len(df))
            print(issue)
            for col in missing:
                df[col] = np.nan

        # Validate numeric positive fields
        for col in Config.NUMERIC_POSITIVE_FIELDS:
            if col not in df.columns:
                continue

            invalid_count = 0
            negative_mask = df[col] < 0
            if negative_mask.any():
                invalid_count += negative_mask.sum()
                df.loc[negative_mask, col] = np.nan

            if col == 'games':
                extreme_mask = df[col] > Config.MAX_GAMES_PER_SEASON
                if extreme_mask.any():
                    invalid_count += extreme_mask.sum()

                    # DEBUG ISSUE #5: Show which players have games > 100
                    print(f"\n[DEBUG ISSUE #5] Found {extreme_mask.sum()} records with games > {Config.MAX_GAMES_PER_SEASON}:")
                    extreme_records = df[extreme_mask][['player_id', 'first_name', 'last_name', 'season', 'games', 'minutes', 'points']].head(10)
                    for idx, row in extreme_records.iterrows():
                        print(f"  {row['first_name']} {row['last_name']} ({row['season']}): games={row['games']}, min={row.get('minutes', 'N/A')}, pts={row.get('points', 'N/A')}")
                    if extreme_mask.sum() > 10:
                        print(f"  ... and {extreme_mask.sum() - 10} more")

                    issue = f"[{name}] {extreme_mask.sum()} extreme 'games' values (>{Config.MAX_GAMES_PER_SEASON}) - SET TO NaN"
                    self._log_issue(name, 'extreme_values', issue, extreme_mask.sum())
                    print(f"[DEBUG ISSUE #5] Setting games to NaN for these records (will invalidate per-game stats)")

                    # Also invalidate dependent stats since games is unreliable
                    df.loc[extreme_mask, col] = np.nan
                    # Note: Don't invalidate other columns yet - let user decide

            if col == 'minutes':
                extreme_mask = df[col] > Config.MAX_MINUTES_PER_SEASON
                if extreme_mask.any():
                    invalid_count += extreme_mask.sum()
                    issue = f"[{name}] {extreme_mask.sum()} extreme 'minutes' values (>{Config.MAX_MINUTES_PER_SEASON})"
                    self._log_issue(name, 'extreme_values', issue, extreme_mask.sum())
                    print(issue)
                    df.loc[extreme_mask, col] = np.nan

            if invalid_count > 0 and col in ['games', 'minutes', 'points']:
                issue = f"[{name}] Total invalid '{col}' values: {invalid_count}"
                self._log_issue(name, 'invalid_numeric', issue, invalid_count)
                print(issue)

        # FIXED: Validate percentage fields with scale detection
        for col in Config.PERCENTAGE_FIELDS:
            if col not in df.columns:
                continue

            # Check if values are in 0-100 scale (should be 0-1)
            if col == 'usage_percentage' and (df[col] > 1.5).sum() > len(df) * 0.5:
                # More than 50% of values > 1.5 suggests 0-100 scale
                print(f"[{name}] Detected {col} in 0-100 scale, converting to 0-1 scale")
                df[col] = df[col] / 100.0
                issue = f"[{name}] Converted {col} from 0-100 to 0-1 scale"
                self._log_issue(name, 'scale_conversion', issue, len(df[df[col].notna()]))

            # Now validate 0-1 range
            out_of_range = ((df[col] < 0) | (df[col] > 1)).fillna(False)
            if out_of_range.any():
                issue = f"[{name}] {out_of_range.sum()} out-of-range '{col}' values (should be 0-1)"
                self._log_issue(name, 'out_of_range', issue, out_of_range.sum())
                print(issue)
                df.loc[out_of_range, col] = np.nan

        # Validate shooting logic
        for made_col, att_col in Config.SHOT_PAIRS:
            if made_col not in df.columns or att_col not in df.columns:
                continue

            invalid = ((df[made_col] > df[att_col]) & df[made_col].notna() & df[att_col].notna())
            if invalid.any():
                issue = f"[{name}] {invalid.sum()} rows where {made_col} > {att_col}"
                self._log_issue(name, 'invalid_shooting', issue, invalid.sum())
                print(issue)
                df.loc[invalid, [made_col, att_col]] = np.nan

        # Check for zero-stat records
        if all(col in df.columns for col in ['games', 'minutes', 'points']):
            zero_stats = (
                (df['games'].fillna(0) == 0) & 
                (df['minutes'].fillna(0) == 0) & 
                (df['points'].fillna(0) == 0)
            )
            if zero_stats.any():
                issue = f"[{name}] {zero_stats.sum()} records with zero games/minutes/points"
                self._log_issue(name, 'zero_stats', issue, zero_stats.sum())
                print(issue)

        print(f"[{name}] Validation complete.")
        return df

    def validate_all_data(self) -> 'DataLoader':
        """Validate all datasets."""
        print("\n" + "=" * 100)
        print("VALIDATING DATA")
        print("=" * 100)

        self.player_df = self.validate_data(self.player_df, 'Player', Config.REQUIRED_PLAYER_COLS)
        self.nba_df = self.validate_data(self.nba_df, 'NBA', Config.REQUIRED_NBA_COLS)
        self.intl_df = self.validate_data(self.intl_df, 'International', Config.REQUIRED_INTL_COLS)
        return self

    def print_summary(self) -> 'DataLoader':
        """FIXED: Print consistent data quality summary with detailed breakdown."""
        print("\n" + "=" * 100)
        print("DATA QUALITY SUMMARY")
        print("=" * 100)
        print(f"Total issue types identified: {self.issue_stats['total_issue_types']}")
        print(f"Total rows affected: {self.issue_stats['total_rows_affected']}")

        # DEBUG ISSUE #2: Breakdown by issue type
        print(f"\n[DEBUG ISSUE #2] Breakdown by issue category:")
        issue_type_counts = {}
        for issue in self.data_quality_issues:
            # Categorize issues
            if 'scale' in issue.lower() or 'converted' in issue.lower():
                category = 'scale_conversion'
            elif 'out-of-range' in issue.lower() or 'out_of_range' in issue:
                category = 'out_of_range'
            elif 'extreme' in issue.lower():
                category = 'extreme_values'
            elif 'duplicate' in issue.lower():
                category = 'duplicate_names'
            elif 'orphan' in issue.lower() or 'filtered' in issue.lower():
                category = 'orphaned_records'
            elif 'invalid' in issue.lower():
                category = 'invalid_values'
            elif 'missing' in issue.lower():
                category = 'missing_columns'
            elif 'zero' in issue.lower():
                category = 'zero_stats'
            else:
                category = 'other'

            issue_type_counts[category] = issue_type_counts.get(category, 0) + 1

        for category, count in sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} issue(s)")

        print("\nBy dataset:")
        for dataset, stats in self.issue_stats['by_dataset'].items():
            # Calculate percentage of dataset
            if dataset == 'NBA':
                total_rows = len(self.nba_df)
            elif dataset == 'International':
                total_rows = len(self.intl_df)
            elif dataset == 'Player':
                total_rows = len(self.player_df)
            else:
                total_rows = 0

            pct = (stats['rows_affected'] / total_rows * 100) if total_rows > 0 else 0
            print(f"  {dataset}: {stats['issue_types']} issue type(s), {stats['rows_affected']} row(s) affected ({pct:.1f}% of dataset)")

        # DEBUG ISSUE #2: Show sample issues
        print(f"\n[DEBUG ISSUE #2] Sample issues (first 5):")
        for i, issue in enumerate(self.data_quality_issues[:5], 1):
            # Truncate long issues
            issue_short = issue[:120] + '...' if len(issue) > 120 else issue
            print(f"  {i}. {issue_short}")

        return self

    def load_and_validate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Complete data loading and validation pipeline."""
        self.load_raw_data()
        self.process_demographics()
        self.create_player_ids()
        self.detect_collisions()
        self.validate_all_data()
        self.print_summary()
        return self.player_df, self.nba_df, self.intl_df, self.data_quality_issues


if __name__ == "__main__":
    """Smoke test with synthetic data."""
    print("="*100)
    print("DATA LOADER SMOKE TEST (FIXED VERSION)")
    print("="*100)

    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test data with usage_percentage in wrong scale
        test_players = [{"first_name": "Test", "last_name": "Player1", "birth_date": "1995-01-15"}]
        test_nba = [{
            "first_name": "Test", "last_name": "Player1", "season": 2021,
            "games": 50, "minutes": 1200, "points": 600, "assists": 200,
            "offensive_rebounds": 50, "defensive_rebounds": 150,
            "usage_percentage": 25.5  # Wrong scale (0-100 instead of 0-1)
        }]
        test_intl = []

        for fname, data in [
            ('player.json', test_players),
            ('nba_box_player_season.json', test_nba),
            ('international_box_player_season.json', test_intl)
        ]:
            with open(Path(temp_dir) / fname, 'w') as f:
                json.dump(data, f)

        loader = DataLoader(data_dir=temp_dir)
        player_df, nba_df, intl_df, issues = loader.load_and_validate()

        # Check usage_percentage was converted
        assert nba_df['usage_percentage'].iloc[0] == 0.255, "Usage % not converted correctly"
        print("\n✓ Usage percentage scale conversion works")

        # Check issue counting
        assert loader.issue_stats['total_issue_types'] > 0, "Should have logged issues"
        print(f"✓ Issue tracking works: {loader.issue_stats['total_issue_types']} types, {loader.issue_stats['total_rows_affected']} rows")

        shutil.rmtree(temp_dir)
        print("\n✓ DATA LOADER (FIXED): PASSED")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
