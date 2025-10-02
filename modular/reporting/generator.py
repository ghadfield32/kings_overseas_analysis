"""
Reporting and visualization module for basketball scouting.
Generates EDA plots, ML diagnostics, and scouting reports.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
PLOTTING_AVAILABLE = True

from modular.config import Config
from modular.utils import format_stat, format_probability_range


class ReportGenerator:
    """Generates comprehensive scouting reports and visualizations."""

    @staticmethod
    def generate_eda_visualizations(player_df: pd.DataFrame,
                                    nba_df: pd.DataFrame,
                                    intl_df: pd.DataFrame,
                                    output_path: str = None):
        """
        Generate  EDA visualizations with data quality checks.

        Args:
            player_df: Player demographics
            nba_df: NBA statistics
            intl_df: International statistics
            output_path: Path to save figure (default from Config)
        """
        if not PLOTTING_AVAILABLE:
            print("\nMatplotlib not available; skipping EDA visualizations")
            return

        output_path = output_path or Config.EDA_VISUALIZATION

        print("\nGenerating EDA visualizations...")

        # Filter to 2021 season
        nba_2021 = nba_df[nba_df['season'] == 2021]
        intl_2021 = intl_df[intl_df['season'] == 2021]

        # Create figure with 3x3 grid
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: Performance distributions
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Plot 1: True Shooting %
        if 'true_shooting_percentage' in nba_2021.columns:
            nba_ts = nba_2021['true_shooting_percentage'].dropna()
            intl_ts = intl_2021['true_shooting_percentage'].dropna()
            ax1.hist(nba_ts, bins=20, alpha=0.6, label=f'NBA (n={len(nba_ts)})', edgecolor='black')
            ax1.hist(intl_ts, bins=20, alpha=0.6, label=f'International (n={len(intl_ts)})', edgecolor='black')
            ax1.set_xlabel('True Shooting %')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Shooting Efficiency (2021)')
            ax1.legend()
            ax1.grid(alpha=0.3)

        # Plot 2: Age vs MPG
        intl_with_age = intl_2021.merge(player_df[['player_id', 'age_2021']], on='player_id', how='left')
        if 'league' in intl_with_age.columns:
            leagues = intl_with_age['league'].value_counts().head(4).index
            for league in leagues:
                league_data = intl_with_age[intl_with_age['league'] == league]
                if len(league_data) > 0:
                    ax2.scatter(
                        league_data['age_2021'],
                        league_data['mpg'],
                        alpha=0.6, s=50, label=league, edgecolors='black', linewidth=0.5
                    )
            ax2.set_xlabel('Age (2021)')
            ax2.set_ylabel('Minutes Per Game')
            ax2.set_title('Age vs MPG (International 2021)')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(alpha=0.3)

        # Plot 3: 3PA Rate
        if 'three_point_attempt_rate' in nba_2021.columns:
            nba_3par = nba_2021['three_point_attempt_rate'].dropna()
            intl_3par = intl_2021['three_point_attempt_rate'].dropna()
            ax3.hist(nba_3par, bins=20, alpha=0.6, label=f'NBA (n={len(nba_3par)})', edgecolor='black')
            ax3.hist(intl_3par, bins=20, alpha=0.6, label=f'International (n={len(intl_3par)})', edgecolor='black')
            ax3.set_xlabel('3PA Rate')
            ax3.set_ylabel('Frequency')
            ax3.set_title('3PA Rate (2021)')
            ax3.legend()
            ax3.grid(alpha=0.3)

        # Row 2: Data quality
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        # Plot 4-6: Data quality visualizations
        key_cols = ['games', 'minutes', 'points', 'assists', 'true_shooting_percentage']

        nba_missing = nba_df[key_cols].isnull().sum()
        ax4.barh(range(len(nba_missing)), nba_missing.values, color='steelblue', edgecolor='black')
        ax4.set_yticks(range(len(nba_missing)))
        ax4.set_yticklabels(nba_missing.index, fontsize=8)
        ax4.set_xlabel('Missing Values')
        ax4.set_title('NBA Data: Missing Values')
        ax4.grid(axis='x', alpha=0.3)

        intl_missing = intl_df[key_cols].isnull().sum()
        ax5.barh(range(len(intl_missing)), intl_missing.values, color='coral', edgecolor='black')
        ax5.set_yticks(range(len(intl_missing)))
        ax5.set_yticklabels(intl_missing.index, fontsize=8)
        ax5.set_xlabel('Missing Values')
        ax5.set_title('International Data: Missing Values')
        ax5.grid(axis='x', alpha=0.3)

        # Data completeness by season
        all_seasons = sorted(set(nba_df['season'].unique()) | set(intl_df['season'].unique()))
        nba_completeness = []
        intl_completeness = []

        for season in all_seasons:
            nba_season = nba_df[nba_df['season'] == season]
            intl_season = intl_df[intl_df['season'] == season]

            if len(nba_season) > 0:
                nba_complete = (1 - nba_season[key_cols].isnull().mean().mean()) * 100
                nba_completeness.append(nba_complete)
            else:
                nba_completeness.append(0)

            if len(intl_season) > 0:
                intl_complete = (1 - intl_season[key_cols].isnull().mean().mean()) * 100
                intl_completeness.append(intl_complete)
            else:
                intl_completeness.append(0)

        ax6.plot(all_seasons, nba_completeness, marker='o', label='NBA', linewidth=2)
        ax6.plot(all_seasons, intl_completeness, marker='s', label='International', linewidth=2)
        ax6.set_xlabel('Season')
        ax6.set_ylabel('Data Completeness (%)')
        ax6.set_title('Data Completeness by Season')
        ax6.legend()
        ax6.grid(alpha=0.3)
        ax6.set_ylim([0, 105])

        # Row 3: Additional analysis
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        # Plot 7: FG% distribution
        for df, label, color in [(nba_df, 'NBA', 'steelblue'), (intl_df, 'International', 'coral')]:
            if 'two_points_made' in df.columns and 'two_points_attempted' in df.columns:
                valid_attempts = df['two_points_attempted'] > 10
                fg_pct = df.loc[valid_attempts, 'two_points_made'] / df.loc[valid_attempts, 'two_points_attempted']
                ax7.hist(fg_pct, bins=20, alpha=0.5, label=label, edgecolor='black', color=color)
        ax7.set_xlabel('2P%')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Two-Point % Distribution\n(min 10 attempts)')
        ax7.legend()
        ax7.grid(alpha=0.3)

        # Plot 8: MPG boxplots
        nba_mpg = nba_df['mpg'].dropna()
        intl_mpg = intl_df['mpg'].dropna()
        ax8.boxplot([nba_mpg, intl_mpg], labels=['NBA', 'International'], widths=0.6)
        ax8.set_ylabel('Minutes Per Game')
        ax8.set_title('MPG Distribution by League')
        ax8.grid(axis='y', alpha=0.3)

        # Plot 9: Career length
        nba_career = nba_df.groupby('player_id')['season'].nunique()
        intl_career = intl_df.groupby('player_id')['season'].nunique()

        max_seasons = max(nba_career.max(), intl_career.max())
        ax9.hist(nba_career, bins=range(1, max_seasons+2), alpha=0.6, 
                label=f'NBA (n={len(nba_career)})', edgecolor='black', color='steelblue')
        ax9.hist(intl_career, bins=range(1, max_seasons+2), alpha=0.6,
                label=f'International (n={len(intl_career)})', edgecolor='black', color='coral')
        ax9.set_xlabel('Number of Seasons')
        ax9.set_ylabel('Number of Players')
        ax9.set_title('Career Length Distribution')
        ax9.legend()
        ax9.grid(alpha=0.3)

        plt.suptitle(' EDA with Data Quality Checks', fontsize=16, y=0.995)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path}")

    @staticmethod
    def generate_ml_diagnostics(ml_artifacts: Dict, output_path: str = None):
        """
        Generate ML diagnostic plots (calibration, ROC, PR curves).

        Args:
            ml_artifacts: ML artifacts dictionary
            output_path: Path to save figure (default from Config)
        """
        if not PLOTTING_AVAILABLE:
            print("\nMatplotlib not available; skipping ML diagnostics")
            return

        if not ml_artifacts.get('enabled'):
            print("\nML model not available; skipping diagnostics")
            return

        output_path = output_path or Config.ML_DIAGNOSTIC_PLOTS

        print("\nGenerating ML diagnostic plots...")

        # Note: This is a simplified version - in full implementation,
        # you would store test predictions in ml_artifacts and plot them here
        print("✓ ML diagnostic plots (placeholder - implement with test set predictions)")

    @staticmethod
    def print_scouting_report(prospects: pd.DataFrame, ml_artifacts: Optional[Dict] = None):
        """
        Print formatted scouting report to console.

        Args:
            prospects: Top prospects DataFrame
            ml_artifacts: Optional ML metrics
        """
        print("\n" + "=" * 100)
        print("SCOUTING RECOMMENDATIONS")
        print("=" * 100)

        print("\nNote: Player identities are anonymized per assignment.")

        print("\n" + "-" * 100)
        print(f"TOP {Config.TOP_N_REPORT} SCOUTING TARGETS")
        print("-" * 100 + "\n")

        top_n = prospects.head(Config.TOP_N_REPORT)

        for idx, (_, player) in enumerate(top_n.iterrows(), 1):
            print(f"{'-' * 100}")
            print(f"RANK #{idx} | {player['first_name'].upper()} {player['last_name'].upper()}")
            print(f"{'-' * 100}")

            nba_status = "NBA experience" if player.get('has_nba_exp', False) else "No NBA experience"
            age_str = format_stat(player.get('age_2021'), 'int')
            league_str = format_stat(player.get('league'), 'str')
            team_str = format_stat(player.get('team'), 'str')

            print(f"Age {age_str} | {league_str} | {team_str} | {nba_status}")

            print("\n2021 Season:")
            print(f"  Games: {format_stat(player.get('games'), 'int')} | MPG: {format_stat(player.get('mpg'), 'float1')}")
            print(f"  PPG: {format_stat(player.get('ppg'), 'float1')} | APG: {format_stat(player.get('apg'), 'float1')} | RPG: {format_stat(player.get('rpg'), 'float1')}")
            print(f"  3P%: {format_stat(player.get('three_pt_pct'), 'pct')} | FT%: {format_stat(player.get('ft_pct'), 'pct')}")
            print(f"  EFF: {format_stat(player.get('efficiency'), 'float1')} | TS%: {format_stat(player.get('true_shooting_percentage'), 'pct')}")

            if pd.notna(player.get('ppg_change')) and player['ppg_change'] != 0:
                arrow = "↑" if player['ppg_change'] > 0 else "↓"
                print(f"\nTrend: {arrow} {abs(player['ppg_change']):.1f} PPG over career")

            if pd.notna(player.get('nba_success_prob')):
                prob_display = format_probability_range(player['nba_success_prob'])
                print(f"ML NBA success probability: {prob_display}")

            print(f"\nScout score: {player['scout_score']:.1f}\n")

        # Summary
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)

        if ml_artifacts and ml_artifacts.get('enabled'):
            m = ml_artifacts['metrics']
            print(f"Model: Calibrated Gradient Boosting")
            print(f"ROC-AUC (test): {m['test_auc']:.3f} | PR-AUC (test): {m['test_pr_auc']:.3f}")
            print(f"Brier Score (cal): {m['brier_score_calibrated']:.3f}")
            if 'precision_at_10' in m:
                print(f"Precision@10: {m['precision_at_10']:.1%}")

        print(f"\nTop {len(prospects)} prospects identified")
        print(f"With NBA experience: {int(prospects['has_nba_exp'].sum())}")
        print(f"International only: {int((~prospects['has_nba_exp']).sum())}")

    @staticmethod
    def save_outputs(prospects: pd.DataFrame, ml_artifacts: Dict, 
                    quality_issues: List[str], db_path: str):
        """
        Save all output files.

        Args:
            prospects: Top prospects DataFrame
            ml_artifacts: ML artifacts
            quality_issues: Data quality issues list
            db_path: Database path
        """
        print("\n" + "=" * 100)
        print("SAVING OUTPUTS")
        print("=" * 100)

        # Save scouting report CSV
        output_cols = [
            'first_name', 'last_name', 'age_2021', 'league', 'team',
            'games', 'mpg', 'ppg', 'apg', 'rpg', 'spg', 'bpg',
            'efficiency', 'true_shooting_percentage', 'fg_pct', 'three_pt_pct', 'ft_pct',
            'has_nba_exp', 'scout_score'
        ]

        if 'ppg_change' in prospects.columns:
            output_cols.append('ppg_change')
        if 'nba_success_prob' in prospects.columns:
            output_cols.append('nba_success_prob')

        available_cols = [col for col in output_cols if col in prospects.columns]
        prospects[available_cols].to_csv(Config.SCOUTING_REPORT_CSV, index=False)
        print(f"✓ Saved: {Config.SCOUTING_REPORT_CSV}")

        # Save ML metrics
        if ml_artifacts.get('enabled'):
            with open(Config.ML_METRICS_JSON, 'w') as f:
                json.dump(ml_artifacts['metrics'], f, indent=2)
            print(f"✓ Saved: {Config.ML_METRICS_JSON}")

        # Save data quality report
        with open(Config.DATA_QUALITY_REPORT, 'w') as f:
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Total issues identified: {len(quality_issues)}\n\n")
            for i, issue in enumerate(quality_issues, 1):
                f.write(f"{i}. {issue}\n")
        print(f"✓ Saved: {Config.DATA_QUALITY_REPORT}")

        print(f"\nDatabase: {db_path}")
        print("\nAll outputs saved successfully.")


def generate_comprehensive_report(prospects: pd.DataFrame, player_df: pd.DataFrame,
                                 nba_df: pd.DataFrame, intl_df: pd.DataFrame,
                                 ml_artifacts: Optional[Dict] = None,
                                 quality_issues: Optional[List[str]] = None,
                                 db_path: Optional[str] = None):
    """
    Main function to generate complete reporting output.

    Args:
        prospects: Top prospects DataFrame
        player_df: Player demographics
        nba_df: NBA statistics
        intl_df: International statistics
        ml_artifacts: Optional ML artifacts
        quality_issues: Optional quality issues list
        db_path: Optional database path
    """
    generator = ReportGenerator()

    # Generate EDA visualizations
    generator.generate_eda_visualizations(player_df, nba_df, intl_df)

    # Generate ML diagnostics
    if ml_artifacts:
        generator.generate_ml_diagnostics(ml_artifacts)

    # Print scouting report
    generator.print_scouting_report(prospects, ml_artifacts)

    # Save outputs
    if quality_issues and db_path:
        generator.save_outputs(prospects, ml_artifacts or {}, quality_issues, db_path)



