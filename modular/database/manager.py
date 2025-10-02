"""
Database operations module for basketball scouting analysis.
Handles SQLite database setup, schema creation, and ETL operations.
"""

import sqlite3
import pandas as pd
from typing import List, Optional
from pathlib import Path

from modular.config import Config



class DatabaseManager:
    """Manages SQLite database operations for scouting data."""

    def __init__(self, db_path: str = None):
        """
        Initialize DatabaseManager.

        Args:
            db_path: Path to SQLite database file (default from Config)
        """
        self.db_path = db_path or Config.DB_PATH
        self.conn = None

    def connect(self) -> 'DatabaseManager':
        """Create database connection and apply pragmas."""
        self.conn = sqlite3.connect(self.db_path)

        # Apply pragmas for performance and reliability
        for pragma, value in Config.DB_PRAGMAS.items():
            self.conn.execute(f"PRAGMA {pragma}={value};")

        print(f"SQLite database connected: {self.db_path}")
        return self

    def create_schema(self) -> 'DatabaseManager':
        """Create database schema with constraints."""
        assert self.conn is not None, "Database not connected"

        print("Creating database schema...")

        # Disable foreign keys temporarily for schema creation
        self.conn.execute("PRAGMA foreign_keys=OFF;")

        # Create tables
        self.conn.executescript("""
            -- Players table
            DROP TABLE IF EXISTS players;
            CREATE TABLE players (
                player_id TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                birth_date TEXT,
                birth_year INTEGER,
                age_2021 INTEGER CHECK(age_2021 >= 15 AND age_2021 <= 60)
            );

            -- NBA stats table
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

            -- International stats table
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

            -- Data quality log table
            DROP TABLE IF EXISTS data_quality_log;
            CREATE TABLE data_quality_log (
                issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_description TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Re-enable foreign keys
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.commit()

        print("Schema created: players, nba_stats, intl_stats, data_quality_log")
        return self

    def create_indexes(self) -> 'DatabaseManager':
        """Create indexes for improved query performance."""
        assert self.conn is not None, "Database not connected"

        print("Creating indexes...")

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_season ON nba_stats(season);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intl_season ON intl_stats(season);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intl_league ON intl_stats(league);"
        )
        self.conn.commit()

        print("Indexes created")
        return self

    def filter_orphaned_stats(self, player_df: pd.DataFrame, 
                             nba_df: pd.DataFrame,
                             intl_df: pd.DataFrame,
                             quality_issues: List[str]) -> tuple:
        """
        Remove stats records without corresponding player demographics.

        Args:
            player_df: Player demographics DataFrame
            nba_df: NBA stats DataFrame
            intl_df: International stats DataFrame
            quality_issues: List to append quality issues to

        Returns:
            Tuple of (filtered_nba_df, filtered_intl_df, updated_quality_issues)
        """
        player_ids = set(player_df['player_id'].unique())

        # Check NBA orphans
        nba_orphans = set(nba_df['player_id'].unique()) - player_ids
        if len(nba_orphans) > 0:
            before = len(nba_df)
            orphan_names = nba_df[nba_df['player_id'].isin(nba_orphans)][
                ['first_name', 'last_name']
            ].drop_duplicates()

            issue = (f"Filtered {before - len(nba_df[nba_df['player_id'].isin(player_ids)])} "
                    f"NBA rows without demographics: "
                    f"{list(orphan_names.itertuples(index=False, name=None))}")
            quality_issues.append(issue)
            print(issue)

            nba_df = nba_df[nba_df['player_id'].isin(player_ids)].copy()

        # Check International orphans
        intl_orphans = set(intl_df['player_id'].unique()) - player_ids
        if len(intl_orphans) > 0:
            before = len(intl_df)
            orphan_names = intl_df[intl_df['player_id'].isin(intl_orphans)][
                ['first_name', 'last_name']
            ].drop_duplicates()

            issue = (f"Filtered {before - len(intl_df[intl_df['player_id'].isin(player_ids)])} "
                    f"International rows without demographics: "
                    f"{list(orphan_names.itertuples(index=False, name=None))}")
            quality_issues.append(issue)
            print(issue)

            intl_df = intl_df[intl_df['player_id'].isin(player_ids)].copy()

        return nba_df, intl_df, quality_issues

    def load_data(self, player_df: pd.DataFrame,
                  nba_df: pd.DataFrame,
                  intl_df: pd.DataFrame,
                  quality_issues: List[str]) -> tuple:
        """
        Load data into database tables.

        DEBUG ENHANCEMENT: Now returns filtered DataFrames for pipeline consistency.

        Args:
            player_df: Player demographics DataFrame
            nba_df: NBA stats DataFrame
            intl_df: International stats DataFrame
            quality_issues: List of data quality issues to log

        Returns:
            Tuple of (filtered_nba_df, filtered_intl_df) for downstream use
        """
        assert self.conn is not None, "Database not connected"

        print("\nLoading data into database...")

        # DEBUG: Show input counts
        print(f"[DEBUG ISSUE #1] Input counts to database:")
        print(f"  player_df: {len(player_df)} rows")
        print(f"  nba_df: {len(nba_df)} rows")
        print(f"  intl_df: {len(intl_df)} rows")

        # Filter orphaned records
        nba_df, intl_df, quality_issues = self.filter_orphaned_stats(
            player_df, nba_df, intl_df, quality_issues
        )

        # DEBUG: Show filtered counts
        print(f"[DEBUG ISSUE #1] After orphan filtering:")
        print(f"  nba_df: {len(nba_df)} rows (filtered)")
        print(f"  intl_df: {len(intl_df)} rows (filtered)")

        # Insert players
        player_cols = ['player_id', 'first_name', 'last_name', 
                      'birth_date', 'birth_year', 'age_2021']
        player_df[player_cols].drop_duplicates(subset='player_id').to_sql(
            'players', self.conn, if_exists='append', index=False
        )
        print(f"Loaded {len(player_df)} players")

        # Insert NBA stats
        nba_cols = ['player_id', 'season', 'team', 'games', 'minutes', 'points', 
                   'assists', 'offensive_rebounds', 'defensive_rebounds', 'steals', 
                   'blocked_shots', 'turnovers', 'personal_fouls', 'two_points_made', 
                   'two_points_attempted', 'three_points_made', 'three_points_attempted',
                   'free_throws_made', 'free_throws_attempted', 'true_shooting_percentage', 
                   'usage_percentage', 'plus_minus']
        nba_df[[c for c in nba_cols if c in nba_df.columns]].to_sql(
            'nba_stats', self.conn, if_exists='append', index=False
        )
        print(f"Loaded {len(nba_df)} NBA records")

        # Insert International stats
        intl_cols = ['player_id', 'season', 'league', 'team', 'games', 'starts', 
                    'minutes', 'points', 'assists', 'offensive_rebounds', 
                    'defensive_rebounds', 'steals', 'blocked_shots', 'turnovers', 
                    'personal_fouls', 'two_points_made', 'two_points_attempted',
                    'three_points_made', 'three_points_attempted', 'free_throws_made',
                    'free_throws_attempted', 'true_shooting_percentage', 'usage_percentage']
        intl_df[[c for c in intl_cols if c in intl_df.columns]].to_sql(
            'intl_stats', self.conn, if_exists='append', index=False
        )
        print(f"Loaded {len(intl_df)} International records")

        # Log quality issues
        for issue in quality_issues:
            self.conn.execute(
                "INSERT INTO data_quality_log (issue_description) VALUES (?)",
                (issue,)
            )
        print(f"Logged {len(quality_issues)} data quality issues")

        self.conn.commit()

        # DEBUG: Return filtered DataFrames for pipeline consistency
        print(f"[DEBUG ISSUE #1] Returning filtered DataFrames to pipeline")
        return nba_df, intl_df

    def run_sanity_checks(self) -> 'DatabaseManager':
        """Run SQL queries to verify data integrity."""
        assert self.conn is not None, "Database not connected"

        print("\n" + "=" * 100)
        print("SQL SANITY CHECKS")
        print("=" * 100)

        # Data completeness check
        query = """
        SELECT
            'NBA' as dataset,
            COUNT(*) as total_records,
            SUM(CASE WHEN games IS NULL OR games <= 0 THEN 1 ELSE 0 END) as invalid_games,
            SUM(CASE WHEN minutes IS NULL THEN 1 ELSE 0 END) as missing_minutes,
            SUM(CASE WHEN points IS NULL THEN 1 ELSE 0 END) as missing_points
        FROM nba_stats
        UNION ALL
        SELECT
            'International' as dataset,
            COUNT(*) as total_records,
            SUM(CASE WHEN games IS NULL OR games <= 0 THEN 1 ELSE 0 END) as invalid_games,
            SUM(CASE WHEN minutes IS NULL THEN 1 ELSE 0 END) as missing_minutes,
            SUM(CASE WHEN points IS NULL THEN 1 ELSE 0 END) as missing_points
        FROM intl_stats;
        """
        result = pd.read_sql_query(query, self.conn)
        print("\nData completeness:")
        print(result.to_string(index=False))

        # Top scorers by league
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
        print("\nTop scorers by league (2021, min 10 games):")
        print(result.to_string(index=False))

        # Quality issues count
        query = "SELECT COUNT(*) as total_issues FROM data_quality_log;"
        result = pd.read_sql_query(query, self.conn)
        print("\nData quality issues logged:")
        print(result.to_string(index=False))

        return self

    def close(self) -> None:
        """Close database connection."""
        if self.conn is not None:
            self.conn.close()
            print(f"\nDatabase connection closed: {self.db_path}")


if __name__ == "__main__":
    """Smoke test with synthetic data and in-memory database."""
    print("=" * 100)
    print("DATABASE MODULE SMOKE TEST")
    print("=" * 100)

    try:
        # Create synthetic test data
        print("\nCreating synthetic test data...")

        # Test player data
        player_df = pd.DataFrame({
            'player_id': ['test_player_1', 'test_player_2'],
            'first_name': ['Test', 'Test'],
            'last_name': ['Player1', 'Player2'],
            'birth_date': ['1995-01-15', '1993-06-20'],
            'birth_year': [1995, 1993],
            'age_2021': [26, 28]
        })

        # Test NBA data
        nba_df = pd.DataFrame({
            'player_id': ['test_player_1'],
            'season': [2021],
            'team': ['TestTeam'],
            'games': [50],
            'minutes': [1200.0],
            'points': [600.0],
            'assists': [200.0],
            'offensive_rebounds': [50.0],
            'defensive_rebounds': [150.0],
            'steals': [30.0],
            'blocked_shots': [20.0],
            'turnovers': [80.0],
            'personal_fouls': [100.0],
            'two_points_made': [150.0],
            'two_points_attempted': [300.0],
            'three_points_made': [50.0],
            'three_points_attempted': [150.0],
            'free_throws_made': [100.0],
            'free_throws_attempted': [120.0],
            'true_shooting_percentage': [0.55],
            'usage_percentage': [0.25],
            'plus_minus': [50.0]
        })

        # Test international data
        intl_df = pd.DataFrame({
            'player_id': ['test_player_2'],
            'season': [2021],
            'league': ['EuroLeague'],
            'team': ['TestTeam'],
            'games': [30],
            'starts': [25],
            'minutes': [800.0],
            'points': [400.0],
            'assists': [120.0],
            'offensive_rebounds': [30.0],
            'defensive_rebounds': [100.0],
            'steals': [20.0],
            'blocked_shots': [15.0],
            'turnovers': [60.0],
            'personal_fouls': [80.0],
            'two_points_made': [100.0],
            'two_points_attempted': [200.0],
            'three_points_made': [40.0],
            'three_points_attempted': [120.0],
            'free_throws_made': [80.0],
            'free_throws_attempted': [100.0],
            'true_shooting_percentage': [0.52],
            'usage_percentage': [0.22]
        })

        # Test quality issues
        quality_issues = ["Test issue 1", "Test issue 2"]

        print("✓ Synthetic data created")

        # Initialize database with in-memory SQLite
        db = DatabaseManager(db_path=':memory:')
        print("\n✓ DatabaseManager initialized")

        # Connect and setup
        db.connect()
        print("✓ Database connected")

        db.create_schema()
        print("✓ Schema created")

        db.create_indexes()
        print("✓ Indexes created")

        # Load data
        db.load_data(player_df, nba_df, intl_df, quality_issues)
        print("✓ Data loaded")

        # Run sanity checks
        db.run_sanity_checks()
        print("✓ Sanity checks passed")

        # Verify table counts
        query = "SELECT COUNT(*) FROM players"
        count = pd.read_sql_query(query, db.conn).iloc[0, 0]
        print(f"\nPlayers in DB: {count}")
        assert count == 2, f"Expected 2 players, got {count}"

        query = "SELECT COUNT(*) FROM nba_stats"
        count = pd.read_sql_query(query, db.conn).iloc[0, 0]
        print(f"NBA records in DB: {count}")
        assert count == 1, f"Expected 1 NBA record, got {count}"

        query = "SELECT COUNT(*) FROM intl_stats"
        count = pd.read_sql_query(query, db.conn).iloc[0, 0]
        print(f"International records in DB: {count}")
        assert count == 1, f"Expected 1 international record, got {count}"

        query = "SELECT COUNT(*) FROM data_quality_log"
        count = pd.read_sql_query(query, db.conn).iloc[0, 0]
        print(f"Quality issues logged: {count}")
        assert count == 2, f"Expected 2 quality issues, got {count}"

        # Close connection
        db.close()
        print("✓ Database closed")

        print("\n" + "=" * 100)
        print("DATABASE MODULE: PASSED ✓")
        print("=" * 100)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 100)
        print("DATABASE MODULE: FAILED ✗")
        print("=" * 100)

