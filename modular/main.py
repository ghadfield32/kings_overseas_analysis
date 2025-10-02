"""
main.py - FULLY CORRECTED VERSION

Main pipeline orchestration for basketball scouting analysis.
Coordinates all modules in the modular structure.

FIXES APPLIED:
1. Added ml_predictions storage to __init__
2. Fixed run_ml_pipeline() to extract predictions properly
3. Fixed run_scouting_pipeline() argument order and type
"""

# --- path bootstrap: works for "python modular/main.py" AND notebooks ---
import sys
from pathlib import Path

if __package__ in (None, ""):
    try:
        # When running as a file (script/module), __file__ exists
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        # When pasted/executed inside a notebook/REPL, __file__ is missing
        PROJECT_ROOT = Path.cwd().resolve()
        # If CWD is ".../modular", go up one level to project root
        if PROJECT_ROOT.name == "modular":
            PROJECT_ROOT = PROJECT_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
# --- end bootstrap ---


from modular.config import Config
from modular.data.loader import DataLoader
from modular.database.manager import DatabaseManager
from modular.features.statistics import calculate_statistics
from modular.analysis.performance import analyze_performance_patterns
from modular.models.predictor import train_nba_success_model, NBASuccessPredictor
from modular.scouting.targets import identify_scouting_targets
from modular.reporting.generator import generate_comprehensive_report


class ScoutingPipeline:
    """Orchestrates the complete scouting analysis pipeline."""

    def __init__(self, data_dir: str = None, db_path: str = None):
        """
        Initialize the pipeline.

        Args:
            data_dir: Directory containing data files
            db_path: Path to database file
        """
        self.data_dir = data_dir or Config.DATA_DIR
        self.db_path = db_path or Config.DB_PATH

        # Components
        self.loader = DataLoader(self.data_dir)
        self.db = DatabaseManager(self.db_path)

        # Data holders
        self.player_df = None
        self.nba_df = None
        self.intl_df = None
        self.quality_issues = []
        self.traj_df = None
        self.ml_artifacts = None
        self.ml_predictions = None  # ✓ FIXED: Added predictions storage
        self.top_prospects = None

    def run_data_pipeline(self):
        """Step 1: Load and validate data."""
        print("\n" + "="*100)
        print("STEP 1: DATA LOADING AND VALIDATION")
        print("="*100)

        self.player_df, self.nba_df, self.intl_df, self.quality_issues = \
            self.loader.load_and_validate()

        print(f"\n✓ Loaded {len(self.player_df)} players")
        print(f"✓ Loaded {len(self.nba_df)} NBA records")
        print(f"✓ Loaded {len(self.intl_df)} International records")
        print(f"✓ Identified {len(self.quality_issues)} data quality issues")

        return self

    def run_database_pipeline(self):
        """Step 2: Setup database and load data."""
        print("\n" + "="*100)
        print("STEP 2: DATABASE SETUP AND ETL")
        print("="*100)

        # DEBUG ISSUE #1: Track data drift before/after database
        print(f"\n[DEBUG ISSUE #1] Pre-database counts:")
        print(f"  player_df: {len(self.player_df)} rows")
        print(f"  nba_df: {len(self.nba_df)} rows")
        print(f"  intl_df: {len(self.intl_df)} rows")

        self.db.connect()
        self.db.create_schema()
        self.db.create_indexes()

        # FIX ISSUE #1: Capture filtered DataFrames
        filtered_nba, filtered_intl = self.db.load_data(
            self.player_df,
            self.nba_df,
            self.intl_df,
            self.quality_issues
        )

        # DEBUG ISSUE #1: Check what was filtered
        nba_removed = len(self.nba_df) - len(filtered_nba)
        intl_removed = len(self.intl_df) - len(filtered_intl)
        print(f"\n[DEBUG ISSUE #1] Database filtering results:")
        print(f"  nba_df: {len(self.nba_df)} → {len(filtered_nba)} ({nba_removed} removed)")
        print(f"  intl_df: {len(self.intl_df)} → {len(filtered_intl)} ({intl_removed} removed)")

        # FIX ISSUE #1: Update pipeline DataFrames with filtered versions
        self.nba_df = filtered_nba
        self.intl_df = filtered_intl
        print(f"[DEBUG ISSUE #1] ✓ Pipeline DataFrames updated to filtered versions")

        self.db.run_sanity_checks()

        print(f"\n✓ Database created: {self.db_path}")

        return self

    def run_statistics_pipeline(self):
        """Step 3: Calculate statistics."""
        print("\n" + "="*100)
        print("STEP 3: STATISTICS CALCULATION")
        print("="*100)

        self.nba_df, self.intl_df = calculate_statistics(
            self.nba_df, 
            self.intl_df
        )

        # Display some calculated stats
        calc_cols = [col for col in Config.SCHEMAS.CALCULATED_ALL 
                     if col in self.intl_df.columns]
        print(f"\n✓ Calculated {len(calc_cols)} statistics")
        print(f"  Per-game: {', '.join(Config.SCHEMAS.CALCULATED_PER_GAME[:5])}")
        print(f"  Shooting: {', '.join(Config.SCHEMAS.CALCULATED_SHOOTING)}")
        print(f"  Efficiency: {', '.join(Config.SCHEMAS.CALCULATED_EFFICIENCY)}")

        return self

    def run_analysis_pipeline(self):
        """Step 4: Analyze performance patterns."""
        print("\n" + "="*100)
        print("STEP 4: PERFORMANCE ANALYSIS")
        print("="*100)

        self.traj_df = analyze_performance_patterns(
            self.intl_df,
            self.nba_df
        )

        if self.traj_df is not None and len(self.traj_df) > 0:
            print(f"\n✓ Analyzed {len(self.traj_df)} player trajectories")
            print(f"  Average PPG change: {self.traj_df['ppg_change'].mean():+.2f}")
            print(f"  Average efficiency change: {self.traj_df['efficiency_change'].mean():+.2f}")

        return self

    def run_ml_pipeline(self):
        """Step 5: Train ML model and extract predictions."""
        print("\n" + "="*100)
        print("STEP 5: MACHINE LEARNING MODEL")
        print("="*100)

        try:
            # Train model - returns ml_artifacts dict
            self.ml_artifacts = train_nba_success_model(
                self.player_df, self.nba_df, self.intl_df
            )

            if self.ml_artifacts['enabled']:
                metrics = self.ml_artifacts['metrics']
                print(f"\n✓ ML model trained successfully")
                print(f"  ROC-AUC: {metrics['test_auc']:.3f}")
                print(f"  PR-AUC: {metrics['test_pr_auc']:.3f}")
                print(f"  Brier Score: {metrics['brier_score_calibrated']:.3f}")

                # ✓ FIXED: Extract predictions from model artifacts
                # The predictor stores predictions in a separate location
                # We need to extract them or generate them fresh
                print("\n  Generating predictions for current prospects...")

                # Create predictor instance to access predictions
                predictor = NBASuccessPredictor()
                predictor.ml_artifacts = self.ml_artifacts

                # Generate predictions for current season
                predictions_df = predictor.predict_current_prospects(
                    self.intl_df, 
                    season=Config.CURRENT_SEASON
                )

                # Store predictions separately for scouting module
                self.ml_predictions = predictions_df
                print(f"  ✓ Generated {len(predictions_df)} predictions")

                # Debug output (optional - remove in production)
                print(f"\n  [DEBUG] Predictions DataFrame:")
                print(f"    Type: {type(self.ml_predictions)}")
                print(f"    Shape: {self.ml_predictions.shape}")
                print(f"    Columns: {list(self.ml_predictions.columns)}")
            else:
                self.ml_predictions = None

        except Exception as e:
            print(f"\n⚠ ML training error: {e}")
            import traceback
            traceback.print_exc()
            self.ml_artifacts = {'enabled': False}
            self.ml_predictions = None

        return self

    def run_scouting_pipeline(self):
        """Step 6: Identify scouting targets."""
        print("\n" + "="*100)
        print("STEP 6: SCOUTING TARGET IDENTIFICATION")
        print("="*100)

        try:
            # Debug output (optional - remove in production)
            print("\n[DEBUG] Inputs to scouting function:")
            print(f"  player_df: {type(self.player_df)} ({len(self.player_df)} rows)")
            print(f"  nba_df: {type(self.nba_df)} ({len(self.nba_df)} rows)")
            print(f"  intl_df: {type(self.intl_df)} ({len(self.intl_df)} rows)")
            if self.traj_df is not None:
                print(f"  traj_df: {type(self.traj_df)} ({len(self.traj_df)} rows)")
            if self.ml_predictions is not None:
                print(f"  ml_predictions: {type(self.ml_predictions)} ({len(self.ml_predictions)} rows)")

            # ✓ FIXED: Correct argument order and pass predictions DataFrame
            # Using named arguments to prevent order errors
            self.top_prospects = identify_scouting_targets(
                player_df=self.player_df,              # ✓ Correct
                nba_df=self.nba_df,                    # ✓ Correct order
                intl_df=self.intl_df,                  # ✓ Correct order
                traj_df=self.traj_df,                  # ✓ Correct
                ml_predictions=self.ml_predictions,    # ✓ DataFrame, not dict!
                season=Config.CURRENT_SEASON,
                top_n=Config.TOP_N_PROSPECTS
            )

            if self.top_prospects is not None and len(self.top_prospects) > 0:
                print(f"\n✓ Identified {len(self.top_prospects)} top prospects")
                print(f"  Top scout score: {self.top_prospects.iloc[0]['scout_score']:.1f}")
                print(f"  Average age: {self.top_prospects['age_2021'].mean():.1f} years")

        except Exception as e:
            print(f"\n⚠ Scouting error: {e}")
            import traceback
            traceback.print_exc()

        return self

    def run_reporting_pipeline(self):
        """Step 7: Generate reports."""
        print("\n" + "="*100)
        print("STEP 7: REPORT GENERATION")
        print("="*100)

        try:
            if self.top_prospects is not None and len(self.top_prospects) > 0:
                generate_comprehensive_report(
                    self.top_prospects, self.player_df, 
                    self.nba_df, self.intl_df,
                    self.ml_artifacts, self.quality_issues, self.db_path
                )

                print(f"\n✓ Generated scouting report")
                print(f"✓ Saved outputs to current directory")
            else:
                print("\n⚠ No prospects to report")

        except Exception as e:
            print(f"\n⚠ Reporting error: {e}")
            import traceback
            traceback.print_exc()

        return self

    def run_full_pipeline(self):
        """Execute the complete analysis pipeline."""
        print("\n" + "="*100)
        print("SACRAMENTO KINGS - INTERNATIONAL SCOUTING ANALYSIS")
        print("MODULAR PIPELINE v2.1 (FIXED)")
        print("="*100)

        print(f"\nConfiguration:")
        print(f"  Data Directory: {self.data_dir}")
        print(f"  Database: {self.db_path}")
        print(f"  Current Season: {Config.CURRENT_SEASON}")
        print(f"  Min Games: {Config.MIN_GAMES_THRESHOLD}")
        print(f"  Min MPG: {Config.MIN_MPG_THRESHOLD}")
        print(f"  Max Age: {Config.MAX_AGE_THRESHOLD}")

        try:
            self.run_data_pipeline()
            self.run_database_pipeline()
            self.run_statistics_pipeline()
            self.run_analysis_pipeline()
            self.run_ml_pipeline()
            self.run_scouting_pipeline()
            self.run_reporting_pipeline()

            print("\n" + "="*100)
            print("PIPELINE COMPLETE ✓")
            print("="*100)

            self._print_summary()

        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if self.db.conn is not None:
                self.db.close()

    def _print_summary(self):
        """Print pipeline execution summary."""
        print("\nPipeline Summary:")
        print("─" * 100)

        print(f"\n📊 Data Processing:")
        print(f"   Players: {len(self.player_df):,}")
        print(f"   NBA Records: {len(self.nba_df):,}")
        print(f"   International Records: {len(self.intl_df):,}")
        print(f"   Quality Issues: {len(self.quality_issues)}")

        if self.traj_df is not None and len(self.traj_df) > 0:
            print(f"\n📈 Performance Analysis:")
            print(f"   Players with Trajectories: {len(self.traj_df)}")
            improving = (self.traj_df['ppg_change'] > 3).sum()
            print(f"   Strongly Improving: {improving}")

        if self.ml_artifacts and self.ml_artifacts.get('enabled'):
            print(f"\n🤖 Machine Learning:")
            metrics = self.ml_artifacts['metrics']
            print(f"   ROC-AUC: {metrics['test_auc']:.3f}")
            print(f"   PR-AUC: {metrics['test_pr_auc']:.3f}")

        if self.top_prospects is not None and len(self.top_prospects) > 0:
            print(f"\n🎯 Scouting:")
            print(f"   Top Prospects: {len(self.top_prospects)}")
            print(f"   With NBA Experience: {int(self.top_prospects['has_nba_exp'].sum())}")

        print("\n" + "─" * 100)

    def cleanup(self):
        """Cleanup resources."""
        if self.db.conn is not None:
            self.db.close()


def main():
    """Main entry point for the scouting analysis."""
    pipeline = ScoutingPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    # Smoke test before running full pipeline
    print("Running pre-flight checks...")

    # Check that Config is accessible
    assert Config.CURRENT_SEASON == 2021, "Config not loaded correctly"
    print("✓ Config loaded")

    # Check data directory exists
    data_dir = Path(Config.DATA_DIR)
    if not data_dir.exists():
        print(f"⚠ Warning: Data directory not found: {Config.DATA_DIR}")
        print("  Please ensure data files are in the correct location")
    else:
        print(f"✓ Data directory found: {Config.DATA_DIR}")

    # Run main pipeline
    main()
