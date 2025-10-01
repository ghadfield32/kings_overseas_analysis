# overseas_analysis.py

"""
Sacramento Kings – International Scouting Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import sqlite3
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


class ImprovedBasketballAnalyzer:
    """
    End-to-end pipeline:
      - load/validate data
      - build SQLite schema and ETL
      - EDA and vectorized stats
      - simple ML model for NBA success
      - ranked scouting list and report
    """

    def __init__(self, data_dir='api_data_files', db_path='kings_scouting.db'):
        self.data_dir = data_dir
        self.db_path = db_path
        self.conn = None

        # DataFrames
        self.player_df = None
        self.nba_df = None
        self.intl_df = None
        self.data_profile = {}

        # Analysis products
        self.career_df = None
        self.traj_df = None
        self.top_prospects = None

        # Team context (derived later)
        self.team_weights = {}
        self.position_priority = {}
        self.nba_prob_weight = 1.10

        # ML artifacts
        self.ml_artifacts = {
            'enabled': False,
            'model': None,
            'scaler': None,
            'features': None,
            'metrics': None,
            'feature_importance': None
        }

    def _safe_div(self, num, den):
        """Safe division returning np.nan for zero/nan denominator."""
        return np.where((pd.notna(den)) & (den != 0), num / den, np.nan)

    def _validate_data(self, df: pd.DataFrame, name: str, required_cols: List[str]) -> pd.DataFrame:
        """Validate DataFrame has required columns and check basic numeric ranges."""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"[{name}] adding missing columns as NaN: {missing}")
            for col in missing:
                df[col] = np.nan

        numeric_positive = ['games', 'minutes', 'points', 'assists', 'rebounds']
        for col in numeric_positive:
            if col in df.columns:
                invalid = df[col] < 0
                if invalid.any():
                    print(f"[{name}] {invalid.sum()} negative values in '{col}' -> set to NaN")
                    df.loc[invalid, col] = np.nan

        return df

    def load_data(self):
        """Load and validate data; create a single player_id consistently."""
        print("=" * 100)
        print("SACRAMENTO KINGS – INTERNATIONAL SCOUTING")
        print("=" * 100)

        with open(f'{self.data_dir}/player.json', 'r') as f:
            self.player_df = pd.DataFrame(json.load(f))
        with open(f'{self.data_dir}/nba_box_player_season.json', 'r') as f:
            self.nba_df = pd.DataFrame(json.load(f))
        with open(f'{self.data_dir}/international_box_player_season.json', 'r') as f:
            self.intl_df = pd.DataFrame(json.load(f))

        print(f"Loaded {len(self.player_df):,} player rows")
        print(f"Loaded {len(self.nba_df):,} NBA player-seasons")
        print(f"Loaded {len(self.intl_df):,} International player-seasons")

        # Unified identifier
        for df in [self.player_df, self.nba_df, self.intl_df]:
            df['player_id'] = (df['first_name'].str.lower() + '_' + df['last_name'].str.lower())

        # Basic dates
        self.player_df['birth_date'] = pd.to_datetime(self.player_df['birth_date'])
        self.player_df['birth_year'] = self.player_df['birth_date'].dt.year
        self.player_df['age_2021'] = 2021 - self.player_df['birth_year']

        # Validation
        self.player_df = self._validate_data(self.player_df, 'Player', ['first_name', 'last_name', 'birth_date'])
        self.nba_df = self._validate_data(self.nba_df, 'NBA', ['season', 'games', 'minutes', 'points', 'assists'])
        self.intl_df = self._validate_data(self.intl_df, 'International', ['season', 'games', 'minutes', 'points', 'assists'])

        print("Data validation complete.")
        return self

    def setup_database(self):
        """Create database and enable useful pragmas."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        print(f"SQLite database ready: {self.db_path}")
        return self

    def etl_to_database(self):
        """Build schema with keys/constraints and load filtered data."""
        assert self.conn is not None, "Database not initialized"

        # Check for orphaned stats (no demographics)
        player_ids_demo = set(self.player_df['player_id'].unique())
        nba_orphans = set(self.nba_df['player_id'].unique()) - player_ids_demo
        intl_orphans = set(self.intl_df['player_id'].unique()) - player_ids_demo

        if len(nba_orphans) > 0:
            before = len(self.nba_df)
            self.nba_df = self.nba_df[self.nba_df['player_id'].isin(player_ids_demo)].copy()
            print(f"Filtered {before - len(self.nba_df)} NBA rows without demographics.")

        if len(intl_orphans) > 0:
            before = len(self.intl_df)
            self.intl_df = self.intl_df[self.intl_df['player_id'].isin(player_ids_demo)].copy()
            print(f"Filtered {before - len(self.intl_df)} International rows without demographics.")

        # Temp tables (helpful when inspecting intermediate data)
        self.player_df.to_sql('players_temp', self.conn, if_exists='replace', index=False)
        self.nba_df.to_sql('nba_stats_temp', self.conn, if_exists='replace', index=False)
        self.intl_df.to_sql('intl_stats_temp', self.conn, if_exists='replace', index=False)

        # Schema
        self.conn.execute("PRAGMA foreign_keys=OFF;")
        self.conn.executescript("""
            DROP TABLE IF EXISTS players;
            CREATE TABLE players (
                player_id TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                birth_date TEXT,
                birth_year INTEGER,
                age_2021 INTEGER CHECK(age_2021 >= 15 AND age_2021 <= 60)
            );

            DROP TABLE IF EXISTS nba_stats;
            CREATE TABLE nba_stats (
                player_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                team TEXT,
                games INTEGER CHECK(games > 0),
                minutes REAL CHECK(minutes >= 0),
                points REAL CHECK(points >= 0),
                assists REAL CHECK(assists >= 0),
                offensive_rebounds REAL,
                defensive_rebounds REAL,
                steals REAL,
                blocked_shots REAL,
                turnovers REAL,
                personal_fouls REAL,
                two_points_made REAL,
                two_points_attempted REAL,
                three_points_made REAL,
                three_points_attempted REAL,
                free_throws_made REAL,
                free_throws_attempted REAL,
                true_shooting_percentage REAL,
                usage_percentage REAL,
                plus_minus REAL,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                PRIMARY KEY (player_id, season, team)
            );

            DROP TABLE IF EXISTS intl_stats;
            CREATE TABLE intl_stats (
                player_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                league TEXT NOT NULL,
                team TEXT,
                games INTEGER CHECK(games > 0),
                starts INTEGER,
                minutes REAL CHECK(minutes >= 0),
                points REAL CHECK(points >= 0),
                assists REAL CHECK(assists >= 0),
                offensive_rebounds REAL,
                defensive_rebounds REAL,
                steals REAL,
                blocked_shots REAL,
                turnovers REAL,
                personal_fouls REAL,
                two_points_made REAL,
                two_points_attempted REAL,
                three_points_made REAL,
                three_points_attempted REAL,
                free_throws_made REAL,
                free_throws_attempted REAL,
                true_shooting_percentage REAL,
                usage_percentage REAL,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                PRIMARY KEY (player_id, season, league, team)
            );
        """)
        self.conn.execute("PRAGMA foreign_keys=ON;")

        # Insert final data
        player_cols = ['player_id', 'first_name', 'last_name', 'birth_date', 'birth_year', 'age_2021']
        self.player_df[player_cols].drop_duplicates(subset='player_id').to_sql('players', self.conn, if_exists='append', index=False)

        nba_cols = ['player_id', 'season', 'team', 'games', 'minutes', 'points', 'assists',
                    'offensive_rebounds', 'defensive_rebounds', 'steals', 'blocked_shots',
                    'turnovers', 'personal_fouls', 'two_points_made', 'two_points_attempted',
                    'three_points_made', 'three_points_attempted', 'free_throws_made',
                    'free_throws_attempted', 'true_shooting_percentage', 'usage_percentage', 'plus_minus']
        (self.nba_df[[c for c in nba_cols if c in self.nba_df.columns]]
         .to_sql('nba_stats', self.conn, if_exists='append', index=False))

        intl_cols = ['player_id', 'season', 'league', 'team', 'games', 'starts', 'minutes', 'points',
                     'assists', 'offensive_rebounds', 'defensive_rebounds', 'steals', 'blocked_shots',
                     'turnovers', 'personal_fouls', 'two_points_made', 'two_points_attempted',
                     'three_points_made', 'three_points_attempted', 'free_throws_made',
                     'free_throws_attempted', 'true_shooting_percentage', 'usage_percentage']
        (self.intl_df[[c for c in intl_cols if c in self.intl_df.columns]]
         .to_sql('intl_stats', self.conn, if_exists='append', index=False))

        # Cleanup temp and add indexes
        self.conn.execute("DROP TABLE IF EXISTS players_temp;")
        self.conn.execute("DROP TABLE IF EXISTS nba_stats_temp;")
        self.conn.execute("DROP TABLE IF EXISTS intl_stats_temp;")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nba_season ON nba_stats(season);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intl_season ON intl_stats(season);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intl_league ON intl_stats(league);")
        self.conn.commit()

        print("ETL complete: tables = players, nba_stats, intl_stats")
        return self

    def run_sql_examples(self):
        """A couple of quick SQL checks."""
        assert self.conn is not None, "Database not initialized."

        print("\nSQL sanity checks:")
        query = """
        SELECT
            'NBA' as dataset,
            COUNT(*) as total_records,
            SUM(CASE WHEN games IS NULL OR games <= 0 THEN 1 ELSE 0 END) as invalid_games
        FROM nba_stats
        UNION ALL
        SELECT
            'International' as dataset,
            COUNT(*) as total_records,
            SUM(CASE WHEN games IS NULL OR games <= 0 THEN 1 ELSE 0 END) as invalid_games
        FROM intl_stats;
        """
        result = pd.read_sql_query(query, self.conn)
        print(result.to_string(index=False))

        print("\nTop scorers by league (2021, min 10 games):")
        query = """
        SELECT league,
               SUBSTR(p.first_name, 1, 1) || '. ' || p.last_name as player,
               ROUND(SUM(points) * 1.0 / SUM(games), 1) as ppg,
               SUM(games) as games
        FROM intl_stats i
        JOIN players p ON i.player_id = p.player_id
        WHERE season = 2021
        GROUP BY i.player_id, league
        HAVING SUM(games) >= 10
        ORDER BY league, ppg DESC
        LIMIT 10;
        """
        result = pd.read_sql_query(query, self.conn)
        print(result.to_string(index=False))
        return self

    def profile_data_structure(self):
        """Print available columns and simple coverage stats; save a basic EDA plot."""
        print("\nDATA STRUCTURE")

        print(f"\nPlayer columns ({len(self.player_df.columns)}):")
        print(f"  {', '.join(sorted(self.player_df.columns))}")

        print(f"\nNBA columns ({len(self.nba_df.columns)}):")
        print(f"  {', '.join(sorted(self.nba_df.columns))}")

        print(f"\nInternational columns ({len(self.intl_df.columns)}):")
        print(f"  {', '.join(sorted(self.intl_df.columns))}")

        nba_seasons = sorted(self.nba_df['season'].unique())
        intl_seasons = sorted(self.intl_df['season'].unique())
        print(f"\nNBA seasons: {nba_seasons[0]}–{nba_seasons[-1]} ({len(nba_seasons)})")
        print(f"International seasons: {intl_seasons[0]}–{intl_seasons[-1]} ({len(intl_seasons)})")

        nba_players = set(self.nba_df['player_id'].unique())
        intl_players = set(self.intl_df['player_id'].unique())
        both_leagues = nba_players.intersection(intl_players)
        print(f"Players with NBA data: {len(nba_players):,}")
        print(f"Players with International data: {len(intl_players):,}")
        print(f"Players with both: {len(both_leagues):,}")

        # EDA plots (saved to file)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            nba_2021 = self.nba_df[self.nba_df['season'] == 2021]
            intl_2021 = self.intl_df[self.intl_df['season'] == 2021]

            if 'true_shooting_percentage' in nba_2021.columns and 'true_shooting_percentage' in intl_2021.columns:
                nba_ts = nba_2021['true_shooting_percentage'].dropna()
                intl_ts = intl_2021['true_shooting_percentage'].dropna()
                axes[0].hist(nba_ts, bins=20, alpha=0.6, label=f'NBA (n={len(nba_ts)})', edgecolor='black')
                axes[0].hist(intl_ts, bins=20, alpha=0.6, label=f'International (n={len(intl_ts)})', edgecolor='black')
                axes[0].set_xlabel('True Shooting %')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('Shooting Efficiency (2021)')
                axes[0].legend()
                axes[0].grid(alpha=0.3)

            if 'player_id' in intl_2021.columns:
                intl_with_age = intl_2021.merge(self.player_df[['player_id', 'age_2021']], on='player_id', how='left')
                leagues = intl_with_age['league'].value_counts().head(4).index
                for league in leagues:
                    league_data = intl_with_age[intl_with_age['league'] == league]
                    if len(league_data) > 0:
                        axes[1].scatter(
                            league_data['age_2021'],
                            league_data.get('mpg', league_data['minutes'] / league_data['games']),
                            alpha=0.6, s=50, label=league, edgecolors='black', linewidth=0.5
                        )
                axes[1].set_xlabel('Age (2021)')
                axes[1].set_ylabel('Minutes Per Game')
                axes[1].set_title('Age vs MPG (International 2021)')
                axes[1].legend(loc='best', fontsize=8)
                axes[1].grid(alpha=0.3)

            if 'three_point_attempt_rate' in nba_2021.columns and 'three_point_attempt_rate' in intl_2021.columns:
                nba_3par = nba_2021['three_point_attempt_rate'].dropna()
                intl_3par = intl_2021['three_point_attempt_rate'].dropna()
                axes[2].hist(nba_3par, bins=20, alpha=0.6, label=f'NBA (n={len(nba_3par)})', edgecolor='black')
                axes[2].hist(intl_3par, bins=20, alpha=0.6, label=f'International (n={len(intl_3par)})', edgecolor='black')
                axes[2].set_xlabel('3PA Rate')
                axes[2].set_ylabel('Frequency')
                axes[2].set_title('3PA Rate (2021)')
                axes[2].legend()
                axes[2].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig('eda_visualizations.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: eda_visualizations.png")
        except Exception as e:
            print(f"EDA visualization failed: {e}")

        return self

    def calculate_statistics(self):
        """Vectorized per-game, per-36, and efficiency metrics."""
        print("\nCALCULATING STATISTICS")

        for df, name in [(self.nba_df, 'NBA'), (self.intl_df, 'International')]:
            games = df['games'].replace(0, np.nan)
            minutes = df['minutes'].replace(0, np.nan)

            df['ppg'] = self._safe_div(df['points'], games)
            df['apg'] = self._safe_div(df['assists'], games)
            df['rpg'] = self._safe_div(
                df['offensive_rebounds'].fillna(0) + df['defensive_rebounds'].fillna(0), games
            )
            df['spg'] = self._safe_div(df['steals'], games)
            df['bpg'] = self._safe_div(df['blocked_shots'], games)
            df['mpg'] = self._safe_div(df['minutes'], games)
            df['topg'] = self._safe_div(df['turnovers'], games)
            df['fpg'] = self._safe_div(df.get('personal_fouls', 0), games)

            df['fg_pct'] = self._safe_div(df['two_points_made'], df['two_points_attempted'])
            df['three_pt_pct'] = self._safe_div(df['three_points_made'], df['three_points_attempted'])
            df['ft_pct'] = self._safe_div(df['free_throws_made'], df['free_throws_attempted'])

            total_fga = df['two_points_attempted'].fillna(0) + df['three_points_attempted'].fillna(0)
            df['efg_pct'] = self._safe_div(
                df['two_points_made'].fillna(0) + 1.5 * df['three_points_made'].fillna(0),
                total_fga
            )

            misses = (
                (df['two_points_attempted'].fillna(0) - df['two_points_made'].fillna(0)) +
                (df['three_points_attempted'].fillna(0) - df['three_points_made'].fillna(0)) +
                (df['free_throws_attempted'].fillna(0) - df['free_throws_made'].fillna(0))
            )
            positive = (
                df['points'].fillna(0) +
                df['offensive_rebounds'].fillna(0) + df['defensive_rebounds'].fillna(0) +
                df['assists'].fillna(0) + df['steals'].fillna(0) + df['blocked_shots'].fillna(0)
            )
            negative = misses + df['turnovers'].fillna(0)
            df['efficiency'] = self._safe_div(positive - negative, games)

            df['pts_per_36'] = self._safe_div(df['points'], minutes) * 36
            df['ast_per_36'] = self._safe_div(df['assists'], minutes) * 36
            df['reb_per_36'] = self._safe_div(
                df['offensive_rebounds'].fillna(0) + df['defensive_rebounds'].fillna(0), minutes
            ) * 36

            if 'usage_percentage' not in df.columns or df['usage_percentage'].isna().all():
                team_poss = df.get('team_possessions', total_fga + 0.44 * df['free_throws_attempted'].fillna(0) + df['turnovers'].fillna(0))
                player_poss = total_fga + 0.44 * df['free_throws_attempted'].fillna(0) + df['turnovers'].fillna(0)
                df['usage_percentage'] = self._safe_div(player_poss, team_poss)

            if 'plus_minus' in df.columns:
                df['plus_minus_per_game'] = self._safe_div(df['plus_minus'], games)
            else:
                df['plus_minus_per_game'] = np.nan

        print("Metric calculation complete.")
        return self

    def analyze_performance_patterns(self):
        """Compute simple player development trajectories on international data."""
        print("\nPERFORMANCE PATTERNS")

        intl_sorted = self.intl_df.sort_values('season')
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

        traj_df = traj_df[traj_df['seasons_played'] >= 2]
        print(f"Players with 2+ intl seasons: {len(traj_df):,}")
        print(f"Avg PPG change: {traj_df['ppg_change'].mean():+.2f}")

        self.traj_df = traj_df
        return self

    def train_nba_success_model(self, min_games_nba=10):
        """Train a small calibrated classifier for NBA success probability."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
            from sklearn.metrics import (
                roc_auc_score, classification_report, precision_recall_curve,
                average_precision_score, brier_score_loss, roc_curve
            )
            from sklearn.preprocessing import StandardScaler
            from sklearn.calibration import CalibratedClassifierCV, calibration_curve
            from sklearn.inspection import permutation_importance
        except ImportError:
            print("scikit-learn not available; skipping model.")
            self.ml_artifacts['enabled'] = False
            return self

        try:
            both = set(self.nba_df['player_id']).intersection(set(self.intl_df['player_id']))
            if len(both) < 50:
                print(f"Insufficient overlap ({len(both)} players); skipping model.")
                self.ml_artifacts['enabled'] = False
                return self

            rows = []
            for pid in both:
                nba_seasons = sorted(self.nba_df.loc[self.nba_df['player_id'] == pid, 'season'].unique())
                intl_hist = self.intl_df.loc[self.intl_df['player_id'] == pid].copy()

                if not len(nba_seasons) or intl_hist.empty:
                    continue

                first_nba = nba_seasons[0]
                intl_before = intl_hist[intl_hist['season'] < first_nba]
                if intl_before.empty:
                    continue

                intl_last = intl_before.sort_values('season').iloc[-1]
                nba_first_df = self.nba_df[
                    (self.nba_df['player_id'] == pid) & (self.nba_df['season'] == first_nba)
                ]
                if nba_first_df.empty:
                    continue
                nba_first = nba_first_df.iloc[0]

                mpg = float(nba_first.get('mpg', 0) or 0)
                ppg = float(nba_first.get('ppg', 0) or 0)
                ts = float(nba_first.get('true_shooting_percentage', 0) or 0)
                success = mpg >= 15 and (ppg >= 8 or (ppg >= 5 and ts >= 0.58))

                ppg_trend = 0
                if len(intl_before) >= 2:
                    ppg_trend = intl_before.iloc[-1]['ppg'] - intl_before.iloc[0]['ppg']

                feats = {
                    'ppg': float(intl_last.get('ppg', 0) or 0),
                    'apg': float(intl_last.get('apg', 0) or 0),
                    'rpg': float(intl_last.get('rpg', 0) or 0),
                    'spg': float(intl_last.get('spg', 0) or 0),
                    'bpg': float(intl_last.get('bpg', 0) or 0),
                    'mpg': float(intl_last.get('mpg', 0) or 0),
                    'three_pt_pct': float(intl_last.get('three_pt_pct', 0) or 0),
                    'ft_pct': float(intl_last.get('ft_pct', 0) or 0),
                    'efficiency': float(intl_last.get('efficiency', 0) or 0),
                    'ts_pct': float(intl_last.get('true_shooting_percentage', 0) or 0),
                    'ppg_trend': ppg_trend,
                    'seasons_played': len(intl_before),
                    'ast_to_tov': float(intl_last.get('apg', 0) / max(intl_last.get('topg', 1), 0.1)),
                    'pts_per_min': float(intl_last.get('ppg', 0) / max(intl_last.get('mpg', 1), 1)),
                    'usage_pct': float(intl_last.get('usage_percentage', 0) or 0),
                    'efg_pct': float(intl_last.get('efg_pct', 0) or 0),
                    'games': float(intl_last.get('games', 0) or 0),
                }
                rows.append((pid, feats, int(success)))

            if len(rows) < 50:
                print(f"Insufficient samples ({len(rows)}); skipping model.")
                self.ml_artifacts['enabled'] = False
                return self

            X = pd.DataFrame([r[1] for r in rows])
            y = pd.Series([r[2] for r in rows])
            X = X.fillna(0).replace([np.inf, -np.inf], 0)

            print(f"Training set: {len(X)} players | positives: {y.sum()} ({y.mean()*100:.1f}%) | features: {len(X.columns)}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            base_clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            base_clf.fit(X_train_scaled, y_train)

            clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv='prefit')
            clf.fit(X_train_scaled, y_train)

            y_prob_uncal = base_clf.predict_proba(X_test_scaled)[:, 1]
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

            test_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)
            brier_uncal = brier_score_loss(y_test, y_prob_uncal)

            sorted_indices = np.argsort(y_prob)[::-1]
            precision_at = {}
            for k in [10, 20, 30]:
                if k <= len(y_test):
                    precision_at[k] = y_test.iloc[sorted_indices[:k]].mean()

            cv_results = cross_validate(
                base_clf, X_train_scaled, y_train,
                cv=5,
                scoring=['roc_auc', 'average_precision'],
                return_train_score=True
            )

            n_bootstrap = 100
            bootstrap_aucs, bootstrap_pr_aucs = [], []
            np.random.seed(42)
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
                if len(np.unique(y_test.iloc[indices])) < 2:
                    continue
                bootstrap_aucs.append(roc_auc_score(y_test.iloc[indices], y_prob[indices]))
                bootstrap_pr_aucs.append(average_precision_score(y_test.iloc[indices], y_prob[indices]))
            auc_ci = np.percentile(bootstrap_aucs, [25, 75]) if bootstrap_aucs else [test_auc, test_auc]
            pr_auc_ci = np.percentile(bootstrap_pr_aucs, [25, 75]) if bootstrap_pr_aucs else [pr_auc, pr_auc]

            perm_importance = permutation_importance(
                clf, X_test_scaled, y_test,
                n_repeats=10, random_state=42, scoring='average_precision'
            )

            feat_imp = pd.DataFrame({
                'feature': X.columns,
                'importance_gini': base_clf.feature_importances_,
                'importance_perm': perm_importance.importances_mean,
                'importance_perm_std': perm_importance.importances_std
            }).sort_values('importance_perm', ascending=False)

            trend_features = ['ppg_trend', 'seasons_played']
            X_train_no_trend = X_train.drop(columns=trend_features, errors='ignore')
            X_test_no_trend = X_test.drop(columns=trend_features, errors='ignore')
            scaler_ablation = StandardScaler()
            X_train_no_trend_scaled = scaler_ablation.fit_transform(X_train_no_trend)
            X_test_no_trend_scaled = scaler_ablation.transform(X_test_no_trend)
            clf_ablation = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
            clf_ablation.fit(X_train_no_trend_scaled, y_train)
            y_prob_ablation = clf_ablation.predict_proba(X_test_no_trend_scaled)[:, 1]
            pr_auc_ablation = average_precision_score(y_test, y_prob_ablation)

            self.ml_artifacts.update({
                'enabled': True,
                'model': clf,
                'base_model': base_clf,
                'scaler': scaler,
                'features': list(X.columns),
                'metrics': {
                    'test_auc': float(test_auc),
                    'test_auc_ci_25': float(auc_ci[0]),
                    'test_auc_ci_75': float(auc_ci[1]),
                    'cv_auc_median': float(np.median(cv_results['test_roc_auc'])),
                    'cv_auc_q25': float(np.percentile(cv_results['test_roc_auc'], 25)),
                    'cv_auc_q75': float(np.percentile(cv_results['test_roc_auc'], 75)),
                    'test_pr_auc': float(pr_auc),
                    'test_pr_auc_ci_25': float(pr_auc_ci[0]),
                    'test_pr_auc_ci_75': float(pr_auc_ci[1]),
                    'cv_pr_auc_median': float(np.median(cv_results['test_average_precision'])),
                    'cv_pr_auc_q25': float(np.percentile(cv_results['test_average_precision'], 25)),
                    'cv_pr_auc_q75': float(np.percentile(cv_results['test_average_precision'], 75)),
                    'brier_score_calibrated': float(brier),
                    'brier_score_uncalibrated': float(brier_uncal),
                    'brier_improvement': float(brier_uncal - brier),
                    'precision_at_10': float(precision_at.get(10, np.nan)),
                    'precision_at_20': float(precision_at.get(20, np.nan)),
                    'precision_at_30': float(precision_at.get(30, np.nan)),
                    'pr_auc_without_trends': float(pr_auc_ablation),
                    'pr_auc_trend_delta': float(pr_auc - pr_auc_ablation),
                    'n_train': int(len(X_train)),
                    'n_test': int(len(X_test)),
                    'success_rate': float(y.mean())
                },
                'feature_importance': feat_imp,
                'cv_fold_scores': {
                    'roc_auc': cv_results['test_roc_auc'].tolist(),
                    'pr_auc': cv_results['test_average_precision'].tolist()
                }
            })

            print("Model trained.")
            print(f"ROC-AUC (test): {test_auc:.3f} | PR-AUC (test): {pr_auc:.3f} | Brier (cal): {brier:.3f}")
            feat_imp.to_csv('feature_importance.csv', index=False)
            print("Saved: feature_importance.csv")

            # Diagnostic plots
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                y_test_arr = np.asarray(y_test).ravel()
                y_prob_arr = np.asarray(y_prob).ravel()
                y_prob_uncal_arr = np.asarray(y_prob_uncal).ravel()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                n_bins = min(10, max(3, int(y_test_arr.sum()) or 3))
                prob_true_cal, prob_pred_cal = calibration_curve(y_test_arr, y_prob_arr, n_bins=n_bins)
                prob_true_uncal, prob_pred_uncal = calibration_curve(y_test_arr, y_prob_uncal_arr, n_bins=n_bins)

                axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect')
                axes[0].plot(prob_pred_uncal, prob_true_uncal, 's-', label='Uncal')
                axes[0].plot(prob_pred_cal, prob_true_cal, 'o-', label='Cal')
                axes[0].set_xlabel('Mean predicted')
                axes[0].set_ylabel('Fraction positives')
                axes[0].set_title('Calibration')
                axes[0].legend(loc='best')
                axes[0].grid(alpha=0.3)

                precision, recall, _ = precision_recall_curve(y_test_arr, y_prob_arr)
                axes[1].plot(recall, precision, linewidth=2, label=f'PR-AUC={pr_auc:.3f}')
                axes[1].axhline(y=y_test_arr.mean(), color='r', linestyle='--', linewidth=2, label='Baseline')
                axes[1].set_xlabel('Recall')
                axes[1].set_ylabel('Precision')
                axes[1].set_title('PR Curve')
                axes[1].legend(loc='best')
                axes[1].grid(alpha=0.3)
                axes[1].set_xlim([0, 1])
                axes[1].set_ylim([0, 1])

                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test_arr, y_prob_arr)
                axes[2].plot(fpr, tpr, linewidth=2, label=f'AUC={test_auc:.3f}')
                axes[2].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
                axes[2].set_xlabel('FPR')
                axes[2].set_ylabel('TPR')
                axes[2].set_title('ROC')
                axes[2].legend(loc='best')
                axes[2].grid(alpha=0.3)
                axes[2].set_xlim([0, 1])
                axes[2].set_ylim([0, 1])

                plt.tight_layout()
                plt.savefig('ml_diagnostic_plots.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("Saved: ml_diagnostic_plots.png")
            except Exception as e:
                print(f"Could not generate diagnostics: {e}")

            # Current-year predictions for international players
            current_intl = self.intl_df[self.intl_df['season'] == 2021].copy()
            if not current_intl.empty:
                feats_current = pd.DataFrame()
                for feat in self.ml_artifacts['features']:
                    if feat in current_intl.columns:
                        feats_current[feat] = current_intl[feat].fillna(0)
                    elif feat == 'ppg_trend':
                        trends = []
                        for pid in current_intl['player_id']:
                            hist = self.intl_df[self.intl_df['player_id'] == pid].sort_values('season')
                            if len(hist) >= 2:
                                trends.append(hist.iloc[-1]['ppg'] - hist.iloc[0]['ppg'])
                            else:
                                trends.append(0)
                        feats_current[feat] = trends
                    elif feat == 'ast_to_tov':
                        feats_current[feat] = current_intl['apg'] / current_intl['topg'].replace(0, 0.1)
                    elif feat == 'pts_per_min':
                        feats_current[feat] = current_intl['ppg'] / current_intl['mpg'].replace(0, 1)
                    else:
                        feats_current[feat] = 0

                feats_current = feats_current[self.ml_artifacts['features']]
                feats_current = feats_current.fillna(0).replace([np.inf, -np.inf], 0)
                feats_current_scaled = scaler.transform(feats_current)

                current_intl['nba_success_prob'] = clf.predict_proba(feats_current_scaled)[:, 1]
                self._ml_current_year_preds = current_intl[['player_id', 'nba_success_prob']]
                print(f"Predictions generated for {len(current_intl)} current players.")

        except Exception as e:
            print(f"Model training failed: {e}")
            import traceback
            traceback.print_exc()
            self.ml_artifacts['enabled'] = False

        return self

    def calculate_team_weights(self):
        """Compute simple team weights based on successful NBA rotation players."""
        print("\nTEAM WEIGHTS")

        print("Definition: NBA seasons with MPG >= 20. Weights reflect prevalence above ~75th percentile in key categories.")
        successful_nba = self.nba_df[self.nba_df['mpg'] >= 20].copy()

        if len(successful_nba) < 50:
            print("Not enough NBA data; using conservative defaults.")
            self.team_weights = {
                'shooting_3pt': 1.15,
                'defense': 1.10,
                'playmaking': 1.10,
                'rebounding': 1.05,
                'youth': 1.05
            }
            return self

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

        print("Weights:")
        for k, v in self.team_weights.items():
            print(f"  {k}: {v:.3f}")
        return self

    def identify_scouting_targets(self):
        """Score and rank international players for 2021 season."""
        print("\nSCOUTING TARGETS")
        self.calculate_team_weights()

        prospects = self.intl_df[self.intl_df['season'] == 2021].copy()
        prospects = prospects.merge(self.player_df[['player_id', 'age_2021']], on='player_id', how='left')

        print("Criteria: season=2021, games>=10, mpg>=20, age<30")
        prospects = prospects[
            (prospects['games'] >= 10) &
            (prospects['mpg'] >= 20) &
            (prospects['age_2021'] < 30)
        ].copy()
        print(f"Eligible players: {len(prospects)}")

        prospects['has_nba_exp'] = prospects['player_id'].isin(self.nba_df['player_id'].unique())

        if getattr(self, 'traj_df', None) is not None:
            traj_merge = self.traj_df[['ppg_change', 'efficiency_change', 'seasons_played']].reset_index()
            prospects = prospects.merge(traj_merge, on='player_id', how='left')
        else:
            prospects['ppg_change'] = np.nan
            prospects['efficiency_change'] = np.nan

        if hasattr(self, '_ml_current_year_preds'):
            prospects = prospects.merge(self._ml_current_year_preds, on='player_id', how='left')
        else:
            prospects['nba_success_prob'] = np.nan

        def _norm(s):
            s = s.astype(float)
            mn, mx = s.min(), s.max()
            if pd.isna(mn) or pd.isna(mx) or mx <= mn:
                return pd.Series(np.nan, index=s.index)
            return (s - mn) / (mx - mn) * 100.0

        for metric in ['ppg', 'apg', 'rpg', 'efficiency', 'true_shooting_percentage']:
            if metric in prospects.columns:
                prospects[f'{metric}_norm'] = _norm(prospects[metric])

        prospects['performance_score'] = (
            prospects.get('ppg_norm', np.nan).fillna(50)*0.30 +
            prospects.get('efficiency_norm', np.nan).fillna(50)*0.25 +
            prospects.get('true_shooting_percentage_norm', np.nan).fillna(50)*0.20 +
            prospects.get('apg_norm', np.nan).fillna(50)*0.15 +
            prospects.get('rpg_norm', np.nan).fillna(50)*0.10
        )

        prospects['age_bonus'] = prospects['age_2021'].apply(
            lambda x: 1.3 if x < 24 else (1.2 if x < 26 else (1.1 if x < 28 else 1.0))
        )
        prospects['improvement_bonus'] = prospects['ppg_change'].fillna(0).apply(
            lambda x: 1.2 if x > 5 else (1.15 if x > 3 else (1.1 if x > 1 else 1.0))
        )

        print("Team fit weights:")
        for key, val in self.team_weights.items():
            print(f"  {key}: {val:.3f}")

        print("Scout score = performance_score × age_bonus × improvement_bonus × fit_multiplier × ml_multiplier")

        prospects['def_events_pg'] = prospects.get('spg', 0).fillna(0) + prospects.get('bpg', 0).fillna(0)
        fit_components = {
            'shooting_3pt': _norm(prospects.get('three_pt_pct', 0).fillna(0)),
            'defense': _norm(prospects['def_events_pg']),
            'playmaking': _norm(prospects.get('apg', 0).fillna(0)),
            'rebounding': _norm(prospects.get('rpg', 0).fillna(0))
        }

        fit_multiplier = (
            (1.0 + (fit_components['shooting_3pt'].fillna(50)/100)*(self.team_weights.get('shooting_3pt',1.0)-1.0)) *
            (1.0 + (fit_components['defense'].fillna(50)/100)*(self.team_weights.get('defense',1.0)-1.0)) *
            (1.0 + (fit_components['playmaking'].fillna(50)/100)*(self.team_weights.get('playmaking',1.0)-1.0)) *
            (1.0 + (fit_components['rebounding'].fillna(50)/100)*(self.team_weights.get('rebounding',1.0)-1.0)) *
            (prospects['age_bonus'] ** (self.team_weights.get('youth',1.0)-1.0))
        )

        ml_mult = 1.0
        if 'nba_success_prob' in prospects.columns and prospects['nba_success_prob'].notna().any():
            ml_mult = 1.0 + (prospects['nba_success_prob'].fillna(0.5) - 0.5) * (self.nba_prob_weight - 1.0) * 2.0
            print(f"NBA success probability weight: {self.nba_prob_weight:.2f}x")

        prospects['scout_score'] = (
            prospects['performance_score'] *
            prospects['age_bonus'] *
            prospects['improvement_bonus'] *
            fit_multiplier *
            ml_mult
        )

        prospects = prospects.sort_values('scout_score', ascending=False).drop_duplicates(subset='player_id', keep='first')
        self.top_prospects = prospects.nlargest(30, 'scout_score').copy()

        print("Top 30 prospects identified.")
        print(f"With NBA experience: {int(self.top_prospects['has_nba_exp'].sum())}")
        print(f"International only: {int((~self.top_prospects['has_nba_exp']).sum())}")
        return self

    def _format_stat(self, value, format_type='float'):
        """Format value with basic NaN handling for printing."""
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
        else:
            return str(value)

    def _format_probability_range(self, prob, ci_low=None, ci_high=None):
        """Format probability with optional CI; clamp extreme displays."""
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

    def generate_comprehensive_report(self):
        """Print a plain-text report to stdout."""
        print("\n" + "=" * 100)
        print("SCOUTING RECOMMENDATIONS")
        print("=" * 100)

        print("\nNote on data: player identities are anonymized per assignment.")

        print("\n" + "-" * 100)
        print("TOP 20 SCOUTING TARGETS")
        print("-" * 100 + "\n")

        top_20 = self.top_prospects.head(20)

        for idx, (_, player) in enumerate(top_20.iterrows(), 1):
            print(f"{'-' * 100}")
            print(f"RANK #{idx} | {player['first_name'].upper()} {player['last_name'].upper()}")
            print(f"{'-' * 100}")

            nba_status = "NBA experience" if player['has_nba_exp'] else "No NBA experience"
            age_str = self._format_stat(player.get('age_2021'), 'int')
            league_str = self._format_stat(player.get('league'), 'str') if pd.notna(player.get('league')) else "—"
            team_str = self._format_stat(player.get('team'), 'str') if pd.notna(player.get('team')) else "—"

            print(f"Age {age_str} | {league_str} | {team_str} | {nba_status}")

            print("\n2021 Season:")
            print(f"  Games: {self._format_stat(player.get('games'), 'int')} | MPG: {self._format_stat(player.get('mpg'), 'float1')}")
            print(f"  PPG: {self._format_stat(player.get('ppg'), 'float1')} | APG: {self._format_stat(player.get('apg'), 'float1')} | RPG: {self._format_stat(player.get('rpg'), 'float1')}")
            print(f"  3P%: {self._format_stat(player.get('three_pt_pct'), 'pct')} | FT%: {self._format_stat(player.get('ft_pct'), 'pct')}")
            print(f"  EFF: {self._format_stat(player.get('efficiency'), 'float1')} | TS%: {self._format_stat(player.get('true_shooting_percentage'), 'pct')}")

            if pd.notna(player.get('ppg_change')) and player['ppg_change'] != 0:
                arrow = "↑" if player['ppg_change'] > 0 else "↓"
                print(f"\nTrend: {arrow} {abs(player['ppg_change']):.1f} PPG over career")

            if pd.notna(player.get('nba_success_prob')):
                prob_display = self._format_probability_range(player['nba_success_prob'])
                print(f"ML NBA success probability: {prob_display}")

            print(f"\nScout score: {player['scout_score']:.1f}\n")

        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)

        if self.ml_artifacts.get('enabled'):
            m = self.ml_artifacts['metrics']
            print(f"Model: Calibrated Gradient Boosting")
            print(f"ROC-AUC (test): {m['test_auc']:.3f}  | PR-AUC (test): {m['test_pr_auc']:.3f}")
            print(f"Brier (cal/uncal): {m['brier_score_calibrated']:.3f} / {m['brier_score_uncalibrated']:.3f}")
            print(f"Precision@10: {m.get('precision_at_10', np.nan):.1%}")

        print(f"\nVectorized metrics computed; database constraints applied; basic EDA saved to file.")
        return self

    def save_outputs(self):
        """Persist CSV/JSON/DB artifacts; close DB."""
        print("\nSAVING OUTPUTS")

        output_cols = [
            'first_name','last_name','age_2021','league','team',
            'games','mpg','ppg','apg','rpg','spg','bpg',
            'efficiency','true_shooting_percentage','fg_pct','three_pt_pct','ft_pct',
            'has_nba_exp','scout_score'
        ]

        if 'ppg_change' in self.top_prospects.columns:
            output_cols += ['ppg_change','seasons_played']
        if 'nba_success_prob' in self.top_prospects.columns:
            output_cols += ['nba_success_prob']

        self.top_prospects[output_cols].to_csv('final_scouting_report.csv', index=False)
        print("Saved: final_scouting_report.csv")

        if self.ml_artifacts.get('enabled'):
            with open('ml_metrics.json', 'w') as f:
                json.dump(self.ml_artifacts['metrics'], f, indent=2)
            print("Saved: ml_metrics.json")

        if self.conn is not None:
            print(f"Database file: {self.db_path}")
            self.conn.close()

        print("\nDONE")
        return self

    def run_full_analysis(self):
        """Run the complete pipeline."""
        self.load_data()
        self.setup_database()
        self.etl_to_database()
        self.run_sql_examples()
        self.profile_data_structure()
        self.calculate_statistics()
        self.analyze_performance_patterns()
        self.train_nba_success_model()
        self.identify_scouting_targets()
        self.generate_comprehensive_report()
        self.save_outputs()


def main():
    analyzer = ImprovedBasketballAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
