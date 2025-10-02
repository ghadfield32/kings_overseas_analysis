"""
 configuration for basketball scouting analysis.
Includes column schemas for automated preprocessing and EDA.
"""

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


class ColumnSchemas:
    """Column schemas for different datasets and purposes."""

    # ==================== PLAYER DEMOGRAPHICS ====================
    PLAYER_ID_COLS = [
        'player_id',
        'first_name',
        'last_name'
    ]

    PLAYER_DATE_COLS = [
        'birth_date'
    ]

    PLAYER_NUMERICAL_DISCRETE = [
        'birth_year',
        'age_2021'
    ]

    # ==================== GAME STATISTICS (RAW) ====================
    # Identity columns
    STATS_ID_COLS = [
        'player_id',
        'season',
        'team',
        'league'  # International only
    ]

    # Discrete numerical (counts)
    STATS_NUMERICAL_DISCRETE = [
        'games',
        'starts',
        'two_points_made',
        'two_points_attempted',
        'three_points_made',
        'three_points_attempted',
        'free_throws_made',
        'free_throws_attempted',
        'offensive_rebounds',
        'defensive_rebounds',
        'assists',
        'steals',
        'blocked_shots',
        'turnovers',
        'personal_fouls',
        'blocked_shot_attempts',
        'screen_assists',
        'deflections',
        'loose_balls_recovered',
        'personal_fouls_drawn',
        'offensive_fouls',
        'charges_drawn',
        'technical_fouls',
        'flagrant_fouls',
        'ejections',
        'points_off_turnovers',
        'points_in_paint',
        'second_chance_points',
        'fast_break_points'
    ]

    # Continuous numerical (measurements)
    STATS_NUMERICAL_CONTINUOUS = [
        'minutes',
        'points',
        'plus_minus',
        'possessions',
        'estimated_possessions',
        'calculated_possessions',
        'plays_used',
        'team_possessions'
    ]

    # Percentage/rate columns (0-1 scale)
    STATS_PERCENTAGES = [
        'usage_percentage',
        'true_shooting_percentage',
        'three_point_attempt_rate',
        'free_throw_rate',
        'offensive_rebounding_percentage',
        'defensive_rebounding_percentage',
        'total_rebounding_percentage',
        'assist_percentage',
        'steal_percentage',
        'block_percentage',
        'turnover_percentage'
    ]

    # Advanced metrics
    STATS_ADVANCED = [
        'internal_box_plus_minus'
    ]

    # Categorical columns
    STATS_CATEGORICAL_NOMINAL = [
        'season_type',  # e.g., "Full Season", "Playoffs"
        'league',       # e.g., "NBA", "EuroLeague", "ACB"
        'team'
    ]

    # ==================== CALCULATED STATISTICS ====================
    # Per-game statistics
    CALCULATED_PER_GAME = [
        'ppg',   # points per game
        'apg',   # assists per game
        'rpg',   # rebounds per game
        'spg',   # steals per game
        'bpg',   # blocks per game
        'mpg',   # minutes per game
        'topg',  # turnovers per game
        'fpg'    # fouls per game
    ]

    # Per-36 minute statistics
    CALCULATED_PER_36 = [
        'pts_per_36',
        'ast_per_36',
        'reb_per_36'
    ]

    # Shooting percentages (calculated)
    CALCULATED_SHOOTING = [
        'fg_pct',        # field goal percentage
        'three_pt_pct',  # three-point percentage
        'ft_pct',        # free throw percentage
        'efg_pct'        # effective field goal percentage
    ]

    # Efficiency metrics
    CALCULATED_EFFICIENCY = [
        'efficiency',
        'plus_minus_per_game'
    ]

    # All calculated statistics
    CALCULATED_ALL = (
        CALCULATED_PER_GAME + 
        CALCULATED_PER_36 + 
        CALCULATED_SHOOTING + 
        CALCULATED_EFFICIENCY
    )

    # ==================== TARGET VARIABLES ====================
    # For ML modeling - NBA success prediction
    ML_TARGET_BINARY = [
        'nba_success'  # Binary: successful NBA career or not
    ]

    # For scouting - continuous targets
    SCOUTING_TARGETS = [
        'scout_score',
        'nba_success_prob'
    ]

    # ==================== FEATURE GROUPS FOR ML ====================
    # Basic features (readily available)
    ML_FEATURES_BASIC = [
        'ppg', 'apg', 'rpg', 'spg', 'bpg', 'mpg',
        'three_pt_pct', 'ft_pct', 'efficiency'
    ]

    # Advanced features (need calculation)
    ML_FEATURES_ADVANCED = [
        'true_shooting_percentage',
        'usage_percentage',
        'efg_pct',
        'ast_to_tov',
        'pts_per_min'
    ]

    # Trajectory features (derived from career data)
    ML_FEATURES_TRAJECTORY = [
        'ppg_trend',
        'seasons_played',
        'ppg_change',
        'efficiency_change'
    ]

    # All ML features
    ML_FEATURES_ALL = (
        ML_FEATURES_BASIC + 
        ML_FEATURES_ADVANCED + 
        ML_FEATURES_TRAJECTORY + 
        ['games']  # Sample size indicator
    )

    # ==================== EDA GROUPINGS ====================
    # For correlation analysis with target
    EDA_NUMERICAL_FOR_CORRELATION = (
        CALCULATED_PER_GAME + 
        CALCULATED_SHOOTING + 
        ['efficiency', 'usage_percentage', 'true_shooting_percentage']
    )

    # For distribution analysis
    EDA_DISTRIBUTIONS = CALCULATED_PER_GAME + ['efficiency']

    # For comparison across leagues
    EDA_LEAGUE_COMPARISON = [
        'ppg', 'apg', 'rpg', 'efficiency', 
        'three_pt_pct', 'fg_pct', 'mpg'
    ]

    @classmethod
    def get_numerical_cols(cls, dataset='stats'):
        """Get all numerical columns for a dataset."""
        if dataset == 'stats':
            return (
                cls.STATS_NUMERICAL_DISCRETE + 
                cls.STATS_NUMERICAL_CONTINUOUS + 
                cls.STATS_PERCENTAGES + 
                cls.STATS_ADVANCED
            )
        elif dataset == 'player':
            return cls.PLAYER_NUMERICAL_DISCRETE
        elif dataset == 'calculated':
            return cls.CALCULATED_ALL
        else:
            return []

    @classmethod
    def get_categorical_cols(cls, dataset='stats'):
        """Get all categorical columns for a dataset."""
        if dataset == 'stats':
            return cls.STATS_CATEGORICAL_NOMINAL
        else:
            return []

    @classmethod
    def get_id_cols(cls, dataset='stats'):
        """Get ID columns for a dataset."""
        if dataset == 'stats':
            return cls.STATS_ID_COLS
        elif dataset == 'player':
            return cls.PLAYER_ID_COLS
        else:
            return []


class Config:
    """Central configuration for the scouting analysis pipeline."""

    # ==================== FILE PATHS ====================
    DATA_DIR = 'api_data_files'
    DB_PATH = 'kings_scouting.db'

    # Data files
    PLAYER_FILE = 'player.json'
    NBA_FILE = 'nba_box_player_season.json'
    INTL_FILE = 'international_box_player_season.json'

    # Output files
    OUTPUT_DIR = 'outputs'
    SCOUTING_REPORT_CSV = 'final_scouting_report.csv'
    FEATURE_IMPORTANCE_CSV = 'feature_importance.csv'
    ML_METRICS_JSON = 'ml_metrics.json'
    DATA_QUALITY_REPORT = 'data_quality_report.txt'
    EDA_VISUALIZATION = 'eda_visualizations_.png'
    ML_DIAGNOSTIC_PLOTS = 'ml_diagnostic_plots.png'

    # ==================== ANALYSIS PARAMETERS ====================
    CURRENT_SEASON = 2021
    MIN_GAMES_THRESHOLD = 10
    MIN_MPG_THRESHOLD = 20
    MAX_AGE_THRESHOLD = 30
    NBA_MIN_GAMES = 10

    # NBA success criteria
    NBA_SUCCESS_MIN_MPG = 15
    NBA_SUCCESS_MIN_PPG = 8
    NBA_SUCCESS_MIN_PPG_ALT = 5
    NBA_SUCCESS_MIN_TS = 0.58

    # ==================== MODEL PARAMETERS ====================
    ML_TEST_SIZE = 0.25
    ML_RANDOM_STATE = 42
    ML_N_ESTIMATORS = 100
    ML_MAX_DEPTH = 4
    ML_LEARNING_RATE = 0.1
    ML_CV_FOLDS = 5
    ML_N_BOOTSTRAP = 100
    ML_PERM_IMPORTANCE_REPEATS = 10

    # ==================== SCOUTING PARAMETERS ====================
    TOP_N_PROSPECTS = 30
    TOP_N_REPORT = 20
    NBA_PROB_WEIGHT = 1.10

    # ==================== VALIDATION THRESHOLDS ====================
    MAX_GAMES_PER_SEASON = 100
    MAX_MINUTES_PER_SEASON = 4500
    MIN_AGE = 15
    MAX_AGE_ABSOLUTE = 60

    # ==================== DATABASE PRAGMAS ====================
    DB_PRAGMAS = {
        'journal_mode': 'WAL',
        'synchronous': 'NORMAL',
        'foreign_keys': 'ON'
    }

    # ==================== REQUIRED COLUMNS ====================
    REQUIRED_PLAYER_COLS = ['first_name', 'last_name', 'birth_date']
    REQUIRED_NBA_COLS = ['season', 'games', 'minutes', 'points', 'assists', 
                         'offensive_rebounds', 'defensive_rebounds']
    REQUIRED_INTL_COLS = ['season', 'games', 'minutes', 'points', 'assists',
                          'offensive_rebounds', 'defensive_rebounds']

    # ==================== VALIDATION FIELD GROUPS ====================
    NUMERIC_POSITIVE_FIELDS = ['games', 'minutes', 'points', 'assists', 
                               'steals', 'blocked_shots']
    PERCENTAGE_FIELDS = ['true_shooting_percentage', 'usage_percentage']

    SHOT_PAIRS = [
        ('two_points_made', 'two_points_attempted'),
        ('three_points_made', 'three_points_attempted'),
        ('free_throws_made', 'free_throws_attempted')
    ]

    # ==================== TEAM WEIGHTS ====================
    DEFAULT_TEAM_WEIGHTS = {
        'shooting_3pt': 1.15,
        'defense': 1.10,
        'playmaking': 1.10,
        'rebounding': 1.05,
        'youth': 1.05
    }

    # ==================== PERFORMANCE SCORE WEIGHTS ====================
    PERFORMANCE_WEIGHTS = {
        'ppg': 0.30,
        'efficiency': 0.25,
        'true_shooting_percentage': 0.20,
        'apg': 0.15,
        'rpg': 0.10
    }

    # ==================== AGE BONUSES ====================
    AGE_BONUSES = [
        (24, 1.3),
        (26, 1.2),
        (28, 1.1),
        (float('inf'), 1.0)
    ]

    # ==================== IMPROVEMENT BONUSES ====================
    IMPROVEMENT_BONUSES = [
        (5, 1.2),
        (3, 1.15),
        (1, 1.1),
        (float('-inf'), 1.0)
    ]

    # ==================== COLUMN SCHEMAS ====================
    # Import column schemas
    SCHEMAS = ColumnSchemas


if __name__ == "__main__":
    """Test configuration and schemas."""
    print("=" * 100)
    print("CONFIGURATION & SCHEMA TEST")
    print("=" * 100)

    # Test basic config access
    print(f"\n1. Basic Configuration:")
    print(f"   Data Directory: {Config.DATA_DIR}")
    print(f"   Current Season: {Config.CURRENT_SEASON}")
    print(f"   ML Random State: {Config.ML_RANDOM_STATE}")

    # Test column schemas
    print(f"\n2. Column Schemas:")
    print(f"   Player ID columns: {len(Config.SCHEMAS.PLAYER_ID_COLS)}")
    print(f"   Stats numerical discrete: {len(Config.SCHEMAS.STATS_NUMERICAL_DISCRETE)}")
    print(f"   Stats numerical continuous: {len(Config.SCHEMAS.STATS_NUMERICAL_CONTINUOUS)}")
    print(f"   Calculated statistics: {len(Config.SCHEMAS.CALCULATED_ALL)}")
    print(f"   ML features: {len(Config.SCHEMAS.ML_FEATURES_ALL)}")

    # Test schema methods
    print(f"\n3. Schema Methods:")
    num_cols = Config.SCHEMAS.get_numerical_cols('stats')
    print(f"   All numerical stats columns: {len(num_cols)}")
    cat_cols = Config.SCHEMAS.get_categorical_cols('stats')
    print(f"   All categorical stats columns: {len(cat_cols)}")

    # Display some column groups
    print(f"\n4. Key Column Groups:")
    print(f"   Per-game stats: {Config.SCHEMAS.CALCULATED_PER_GAME}")
    print(f"   Shooting percentages: {Config.SCHEMAS.CALCULATED_SHOOTING}")
    print(f"   ML basic features: {Config.SCHEMAS.ML_FEATURES_BASIC[:5]}...")
    print(f"   EDA correlation columns: {len(Config.SCHEMAS.EDA_NUMERICAL_FOR_CORRELATION)}")

    # Verify no duplicates in column lists
    print(f"\n5. Validation:")
    all_calc = Config.SCHEMAS.CALCULATED_ALL
    if len(all_calc) == len(set(all_calc)):
        print(f"   ✓ No duplicates in calculated columns")
    else:
        print(f"   ✗ Duplicates found in calculated columns")

    ml_features = Config.SCHEMAS.ML_FEATURES_ALL
    if len(ml_features) == len(set(ml_features)):
        print(f"   ✓ No duplicates in ML features")
    else:
        print(f"   ✗ Duplicates found in ML features")

    print("\n" + "=" * 100)
    print("CONFIGURATION TEST: PASSED")
    print("=" * 100)
