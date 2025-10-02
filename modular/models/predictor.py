"""
Machine learning module for NBA success prediction.
Trains a calibrated classifier to predict NBA success from international stats.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, precision_recall_curve,
    average_precision_score, brier_score_loss, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
SKLEARN_AVAILABLE = True


# When run as part of package
from modular.config import Config
from modular.utils import safe_div



class NBASuccessPredictor:
    """Predicts NBA success probability from international performance."""

    def __init__(self):
        """Initialize the predictor."""
        self.ml_artifacts = {
            'enabled': False,
            'model': None,
            'scaler': None,
            'features': None,
            'metrics': None,
            'feature_importance': None
        }
        self.current_year_predictions = None

    def define_nba_success(self, nba_df: pd.DataFrame, min_games: int = 10) -> pd.Series:
        """
        Define NBA success criteria.

        Success = MPG >= 15 AND (PPG >= 8 OR (PPG >= 5 AND TS% >= 0.58))

        Args:
            nba_df: NBA statistics DataFrame
            min_games: Minimum games threshold

        Returns:
            Series of player_id -> success boolean
        """
        # Filter to meaningful NBA seasons
        nba_filtered = nba_df[nba_df['games'] >= min_games].copy()

        # Calculate success per season
        nba_filtered['success'] = (
            (nba_filtered['mpg'] >= Config.NBA_SUCCESS_MIN_MPG) &
            (
                (nba_filtered['ppg'] >= Config.NBA_SUCCESS_MIN_PPG) |
                (
                    (nba_filtered['ppg'] >= Config.NBA_SUCCESS_MIN_PPG_ALT) &
                    (nba_filtered['true_shooting_percentage'] >= Config.NBA_SUCCESS_MIN_TS)
                )
            )
        )

        # A player is successful if they had ANY successful season
        player_success = nba_filtered.groupby('player_id')['success'].max()

        return player_success

    def extract_features(self, player_id: str, intl_df: pd.DataFrame, 
                        first_nba_season: int) -> Optional[Dict]:
        """
        Extract features from last international season before NBA.

        Args:
            player_id: Player identifier
            intl_df: International statistics DataFrame
            first_nba_season: First NBA season for this player

        Returns:
            Dictionary of features or None if insufficient data
        """
        # Get international history before NBA
        intl_hist = intl_df[
            (intl_df['player_id'] == player_id) & 
            (intl_df['season'] < first_nba_season)
        ].copy()

        if intl_hist.empty:
            return None

        # Get last season stats
        intl_last = intl_hist.sort_values('season').iloc[-1]

        # Calculate trajectory features
        ppg_trend = 0
        if len(intl_hist) >= 2:
            ppg_trend = intl_hist.iloc[-1]['ppg'] - intl_hist.iloc[0]['ppg']

        # Build feature dictionary
        features = {
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
            'seasons_played': len(intl_hist),
            'ast_to_tov': float(intl_last.get('apg', 0) / max(intl_last.get('topg', 1), 0.1)),
            'pts_per_min': float(intl_last.get('ppg', 0) / max(intl_last.get('mpg', 1), 1)),
            'usage_pct': float(intl_last.get('usage_percentage', 0) or 0),
            'efg_pct': float(intl_last.get('efg_pct', 0) or 0),
            'games': float(intl_last.get('games', 0) or 0),
        }

        return features

    def build_training_data(self, player_df: pd.DataFrame, nba_df: pd.DataFrame,
                           intl_df: pd.DataFrame) -> tuple:
        """
        Build training dataset from players who played in both leagues.

        Args:
            player_df: Player demographics DataFrame
            nba_df: NBA statistics DataFrame
            intl_df: International statistics DataFrame

        Returns:
            Tuple of (X, y, player_ids) or (None, None, None) if insufficient data
        """
        # Find players with both NBA and international experience
        both_leagues = set(nba_df['player_id']).intersection(set(intl_df['player_id']))

        if len(both_leagues) < 50:
            print(f"Insufficient overlap ({len(both_leagues)} players); need at least 50")
            return None, None, None

        print(f"\n[DEBUG ISSUE #3] Training data construction:")
        print(f"  Players with both NBA + Intl: {len(both_leagues)}")

        # Define success
        player_success = self.define_nba_success(nba_df)

        print(f"  Successful players: {player_success.sum()} ({player_success.mean():.1%})")
        print(f"  Failed players: {(~player_success).sum()}")

        # Build training examples
        rows = []
        leakage_count = 0
        leakage_samples = []

        for pid in both_leagues:
            # Get first NBA season
            nba_seasons = sorted(nba_df[nba_df['player_id'] == pid]['season'].unique())
            if not len(nba_seasons):
                continue

            first_nba = nba_seasons[0]

            # DEBUG ISSUE #3: Check for temporal leakage
            intl_seasons = sorted(intl_df[intl_df['player_id'] == pid]['season'].unique())
            if intl_seasons and intl_seasons[-1] >= first_nba:
                leakage_count += 1
                if len(leakage_samples) < 5:
                    leakage_samples.append({
                        'player_id': pid,
                        'first_nba': first_nba,
                        'last_intl': intl_seasons[-1]
                    })

            # Extract features
            features = self.extract_features(pid, intl_df, first_nba)
            if features is None:
                continue

            # Get success label
            success = player_success.get(pid, False)

            rows.append((pid, features, int(success)))

        # DEBUG ISSUE #3: Report temporal leakage
        if leakage_count > 0:
            print(f"\n[DEBUG ISSUE #3] ⚠️ TEMPORAL LEAKAGE DETECTED:")
            print(f"  {leakage_count} players have Intl data >= their first NBA season")
            print(f"  Sample cases:")
            for sample in leakage_samples:
                print(f"    {sample['player_id']}: NBA start {sample['first_nba']}, Intl last {sample['last_intl']}")

        if len(rows) < 50:
            print(f"Insufficient training examples ({len(rows)}); need at least 50")
            return None, None, None

        # Convert to DataFrames
        player_ids = [r[0] for r in rows]
        X = pd.DataFrame([r[1] for r in rows])
        y = pd.Series([r[2] for r in rows])

        # Handle missing/invalid values
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # DEBUG ISSUE #3: Feature distribution analysis
        print(f"\n[DEBUG ISSUE #3] Feature distributions:")
        for col in ['ppg', 'efficiency', 'three_pt_pct', 'ppg_trend', 'mpg']:
            if col in X.columns:
                print(f"  {col}: mean={X[col].mean():.2f}, std={X[col].std():.2f}, "
                      f"min={X[col].min():.2f}, max={X[col].max():.2f}")

        # DEBUG ISSUE #3: Class separation analysis
        print(f"\n[DEBUG ISSUE #3] Feature means by outcome class:")
        X_with_y = X.copy()
        X_with_y['target'] = y
        for col in ['ppg', 'efficiency', 'three_pt_pct', 'mpg']:
            if col in X.columns:
                mean_success = X_with_y[X_with_y['target']==1][col].mean()
                mean_fail = X_with_y[X_with_y['target']==0][col].mean()
                separation = mean_success - mean_fail
                print(f"  {col}: Success={mean_success:.2f}, Failure={mean_fail:.2f}, Δ={separation:.2f}")

        return X, y, player_ids

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Train calibrated gradient boosting classifier.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            True if training successful, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available; skipping model training")
            return False

        print(f"\nTraining model on {len(X)} examples...")
        print(f"  Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Features: {len(X.columns)}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.ML_TEST_SIZE,
            stratify=y,
            random_state=Config.ML_RANDOM_STATE
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train base model
        base_clf = GradientBoostingClassifier(
            n_estimators=Config.ML_N_ESTIMATORS,
            max_depth=Config.ML_MAX_DEPTH,
            learning_rate=Config.ML_LEARNING_RATE,
            random_state=Config.ML_RANDOM_STATE
        )
        base_clf.fit(X_train_scaled, y_train)

        # Calibrate probabilities
        clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv='prefit')
        clf.fit(X_train_scaled, y_train)

        # Evaluate
        y_prob_uncal = base_clf.predict_proba(X_test_scaled)[:, 1]
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_prob, y_prob_uncal, X_train_scaled, y_train, base_clf)

        # Feature importance
        feat_imp = self._calculate_feature_importance(
            base_clf, clf, X, X_test_scaled, y_test
        )

        # Store artifacts
        self.ml_artifacts = {
            'enabled': True,
            'model': clf,
            'base_model': base_clf,
            'scaler': scaler,
            'features': list(X.columns),
            'metrics': metrics,
            'feature_importance': feat_imp
        }

        print(f"\n✓ Model trained successfully")
        print(f"  ROC-AUC: {metrics['test_auc']:.3f}")
        print(f"  PR-AUC: {metrics['test_pr_auc']:.3f}")
        print(f"  Brier Score (calibrated): {metrics['brier_score_calibrated']:.3f}")

        # Save feature importance
        feat_imp.to_csv('feature_importance.csv', index=False)
        print(f"  Saved: feature_importance.csv")

        return True

    def _calculate_metrics(self, y_test, y_prob, y_prob_uncal, 
                          X_train_scaled, y_train, base_clf) -> Dict:
        """Calculate comprehensive model metrics."""
        # Basic metrics
        test_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        brier_uncal = brier_score_loss(y_test, y_prob_uncal)

        # Precision at K
        sorted_indices = np.argsort(y_prob)[::-1]
        precision_at = {}
        for k in [10, 20, 30]:
            if k <= len(y_test):
                precision_at[k] = y_test.iloc[sorted_indices[:k]].mean()

        # Cross-validation
        cv_results = cross_validate(
            base_clf, X_train_scaled, y_train,
            cv=Config.ML_CV_FOLDS,
            scoring=['roc_auc', 'average_precision'],
            return_train_score=True
        )

        # Bootstrap confidence intervals
        np.random.seed(Config.ML_RANDOM_STATE)
        bootstrap_aucs = []
        for _ in range(Config.ML_N_BOOTSTRAP):
            indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
            if len(np.unique(y_test.iloc[indices])) < 2:
                continue
            bootstrap_aucs.append(roc_auc_score(y_test.iloc[indices], y_prob[indices]))

        auc_ci = np.percentile(bootstrap_aucs, [2.5, 97.5]) if bootstrap_aucs else [test_auc, test_auc]

        return {
            'test_auc': float(test_auc),
            'test_auc_ci_low': float(auc_ci[0]),
            'test_auc_ci_high': float(auc_ci[1]),
            'test_pr_auc': float(pr_auc),
            'cv_auc_median': float(np.median(cv_results['test_roc_auc'])),
            'cv_pr_auc_median': float(np.median(cv_results['test_average_precision'])),
            'brier_score_calibrated': float(brier),
            'brier_score_uncalibrated': float(brier_uncal),
            'brier_improvement': float(brier_uncal - brier),
            'precision_at_10': float(precision_at.get(10, np.nan)),
            'precision_at_20': float(precision_at.get(20, np.nan)),
            'precision_at_30': float(precision_at.get(30, np.nan)),
            'n_train': int(len(X_train_scaled)),
            'n_test': int(len(y_test))
        }

    def _calculate_feature_importance(self, base_clf, clf, X, X_test_scaled, y_test) -> pd.DataFrame:
        """Calculate feature importance using multiple methods."""
        # Gini importance from base model
        gini_imp = base_clf.feature_importances_

        # Permutation importance
        perm_imp = permutation_importance(
            clf, X_test_scaled, y_test,
            n_repeats=Config.ML_PERM_IMPORTANCE_REPEATS,
            random_state=Config.ML_RANDOM_STATE,
            scoring='average_precision'
        )

        feat_imp = pd.DataFrame({
            'feature': X.columns,
            'importance_gini': gini_imp,
            'importance_perm': perm_imp.importances_mean,
            'importance_perm_std': perm_imp.importances_std
        }).sort_values('importance_perm', ascending=False)

        return feat_imp

    def predict_current_prospects(self, intl_df: pd.DataFrame, 
                                 season: int = 2021) -> pd.DataFrame:
        """
        Generate predictions for current international prospects.

        Args:
            intl_df: International statistics DataFrame
            season: Season to predict for

        Returns:
            DataFrame with player_id and nba_success_prob
        """
        if not self.ml_artifacts['enabled']:
            return pd.DataFrame(columns=['player_id', 'nba_success_prob'])

        current_intl = intl_df[intl_df['season'] == season].copy()
        if current_intl.empty:
            return pd.DataFrame(columns=['player_id', 'nba_success_prob'])

        # Extract features for current season
        feats_current = pd.DataFrame()
        for feat in self.ml_artifacts['features']:
            if feat in current_intl.columns:
                feats_current[feat] = current_intl[feat].fillna(0)
            elif feat == 'ppg_trend':
                # Calculate trend for each player
                trends = []
                for pid in current_intl['player_id']:
                    hist = intl_df[intl_df['player_id'] == pid].sort_values('season')
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

        # Ensure correct column order
        feats_current = feats_current[self.ml_artifacts['features']]
        feats_current = feats_current.fillna(0).replace([np.inf, -np.inf], 0)

        # Scale and predict
        feats_scaled = self.ml_artifacts['scaler'].transform(feats_current)
        probs = self.ml_artifacts['model'].predict_proba(feats_scaled)[:, 1]

        result = pd.DataFrame({
            'player_id': current_intl['player_id'],
            'nba_success_prob': probs
        })

        self.current_year_predictions = result
        print(f"\n✓ Generated predictions for {len(result)} current prospects")

        return result

    def get_artifacts(self) -> Dict:
        """Get ML artifacts for use in downstream modules."""
        return self.ml_artifacts


def train_nba_success_model(player_df: pd.DataFrame, nba_df: pd.DataFrame, 
                            intl_df: pd.DataFrame) -> Dict:
    """
    Main function to train NBA success prediction model.

    Args:
        player_df: Player demographics DataFrame
        nba_df: NBA statistics DataFrame  
        intl_df: International statistics DataFrame

    Returns:
        Dictionary of ML artifacts
    """
    print("\n" + "=" * 100)
    print("TRAINING NBA SUCCESS MODEL")
    print("=" * 100)

    if not SKLEARN_AVAILABLE:
        print("\nscikit-learn not available; skipping model training")
        return {'enabled': False}

    predictor = NBASuccessPredictor()

    # Build training data
    X, y, player_ids = predictor.build_training_data(player_df, nba_df, intl_df)

    if X is None:
        print("\nInsufficient data for model training")
        return {'enabled': False}

    # Train model
    success = predictor.train_model(X, y)

    if not success:
        return {'enabled': False}

    # Generate predictions for current prospects
    predictor.predict_current_prospects(intl_df, season=Config.CURRENT_SEASON)

    return predictor.get_artifacts()
