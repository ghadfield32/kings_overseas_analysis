# Sacramento Kings Data Science Assessment

**Candidate Submission** - Geoffrey Hadfield - ghadfield32@gmail.com

## Executive Summary

This repository contains a comprehensive analysis of international basketball players for Sacramento Kings scouting recommendations. The analysis combines NBA and European league data (2010-2021) to identify top prospects using advanced machine learning models and data-driven team fit scoring.

## 🏀 Analysis Overview

### Data Coverage
- **NBA Statistics**: 465 players across 12 seasons (2010-2021)
- **International Statistics**: 1,473 players across 11 seasons (2010-2021) 
- **Overlap**: 433 players with both NBA and international experience
- **Leagues Analyzed**: EuroLeague, EuroCup, Spain - ACB, Italy - Liga A

### Key Deliverables
- **Top 20 Scouting Targets** with detailed player profiles
- **Machine Learning Model** for NBA success prediction (ROC-AUC: 0.548, PR-AUC: 0.262)
- **Data-driven team fit weights** calculated from 485 successful NBA players
- **Comprehensive EDA** with performance trend analysis

## 🎯 Top 5 Scouting Recommendations

### 1. **LA TORRE MCCASKILL** - Scout Score: 115.1
- **Age**: 25 | **League**: EuroCup | **Team**: RedHawks
- **2021 Stats**: 17.9 PPG, 2.8 APG, 5.3 RPG, 52.0% 3P%, 68.5% TS%
- **NBA Success Probability**: 35.9%
- **Key Strength**: Elite 3-point shooting (52.0%)

### 2. **BATISTA SHOUTVIN** - Scout Score: 114.1
- **Age**: 23 | **League**: EuroCup | **Team**: Fighting Irish
- **2021 Stats**: 14.7 PPG, 4.2 APG, 1.8 RPG, 35.5% 3P%, 64.5% TS%
- **NBA Success Probability**: 48.4%
- **Key Strength**: Strong development trajectory (+6.9 PPG over career)

### 3. **XABI NWABA** - Scout Score: 99.5
- **Age**: 27 | **League**: Italy - Liga A | **Team**: Bisons
- **2021 Stats**: 16.9 PPG, 1.2 APG, 7.4 RPG, 62.9% FT%, 59.8% TS%
- **NBA Success Probability**: 45.7%
- **Key Strength**: Excellent rebounding (7.4 RPG)

### 4. **POYTHRESS WIGGINTON** - Scout Score: 99.1
- **Age**: 25 | **League**: Italy - Liga A | **Team**: Mustangs
- **2021 Stats**: 18.5 PPG, 1.6 APG, 4.2 RPG, 44.3% 3P%, 64.9% TS%
- **NBA Success Probability**: 33.1%
- **Key Strength**: High-volume scoring with efficiency

### 5. **FARLEY ATKINS** - Scout Score: 97.9
- **Age**: 24 | **League**: EuroCup | **Team**: Chippewas
- **2021 Stats**: 15.3 PPG, 2.8 APG, 5.5 RPG, 43.2% 3P%, 64.4% TS%
- **NBA Success Probability**: 8.9%
- **Key Strength**: Versatile forward with good shooting

## 🔬 Methodology

### Machine Learning Model
- **Algorithm**: Calibrated Gradient Boosting with Platt scaling
- **Features**: 17 statistical metrics including trends, ratios, and advanced stats
- **Performance**: 
  - Test ROC-AUC: 0.548 (IQR: 0.370-0.642)
  - Test PR-AUC: 0.262 (IQR: 0.168-0.414)
  - Brier Score: 0.133 (improved calibration)
  - Precision@10: 20.0%

### Scout Score Formula
```
SCOUT_SCORE = performance_score × age_bonus × improvement_bonus × fit_multiplier × ml_multiplier
```

**Components:**
- **Performance Score**: Weighted combination of PPG, APG, RPG, EFF, TS%
- **Age Bonus**: 1.3 (<24), 1.2 (24-25), 1.1 (26-27), 1.0 (28+)
- **Improvement Bonus**: 1.2 (>5 PPG growth), 1.15 (3-5), 1.1 (1-3), 1.0 (<1)
- **Team Fit Multiplier**: Data-driven weights for 3PT (1.120), defense (1.101), playmaking (1.101), rebounding (1.075), youth (1.050)
- **ML Multiplier**: Calibrated NBA success probability adjustment

### Data-Driven Team Weights
Calculated from 485 successful NBA players (MPG ≥ 20):
- **3-Point Shooting**: 1.120
- **Defense**: 1.101  
- **Playmaking**: 1.101
- **Rebounding**: 1.075
- **Youth**: 1.050

## 📊 Key Findings

### Player Development Trends
- **624 players tracked** across multiple seasons
- **184 players** showing significant improvement (+2.0+ PPG)
- **Top Improver**: rod_haslem (+14.4 PPG over 7 seasons)

### Scouting Criteria Applied
- Currently playing internationally (2021)
- Minimum 10 games played
- Minimum 20 MPG
- Age under 30
- **217 players** met initial criteria

### Final Recommendations
- **30 elite prospects** identified
- **6 with NBA experience**, **24 international-only**
- **10 prospects under age 26**
- **0 prospects** with >70% predicted NBA success (realistic calibration)

## 📁 Repository Structure

```
├── overseas_analysis.py          # Main analysis script (1,498 lines)
├── final_scouting_report.csv     # Top 30 player recommendations
├── eda_visualizations.png        # Exploratory data analysis plots
├── ml_diagnostic_plots.png       # Machine learning model diagnostics
├── ml_metrics.json              # Model performance metrics
├── feature_importance.csv       # Feature importance rankings
├── kings_scouting.db            # SQLite database with processed data
├── api_data_files/              # Original JSON data files
│   ├── nba_box_player_season.json
│   ├── international_box_player_season.json
│   └── player.json
└── pyproject.toml               # Python dependencies
```

## 🛠 Technical Implementation

### Performance Optimizations
- **40% faster execution** through vectorized operations
- **Consolidated player_id creation** for data integrity
- **Optimized groupby operations** for trend analysis
- **Proper NaN handling** (no artificial zeros)

### Database Design
- **Primary keys and foreign keys** implemented
- **Check constraints** enforced
- **Referential integrity** maintained
- **192 NBA records excluded** due to demographic formatting issues

### Model Improvements
- **7 new features added** (trends, ratios, advanced stats)
- **Trend features impact**: +0.112 PR-AUC improvement
- **Calibrated probabilities** prevent unrealistic 100% predictions
- **Bootstrap confidence intervals** quantify uncertainty

## ⚠️ Important Limitations

### Data Anonymization
- **Player names are ANONYMIZED** per assignment requirements
- **External validation post-2021 is NOT possible** on these identities
- **Results reflect PROCESS QUALITY**, not real-world scouting accuracy

### Sample Size Considerations
- **Training samples**: 89, **Test samples**: 30
- **Success rate**: 12.6% (highly imbalanced dataset)
- **PR-AUC more reliable** than ROC-AUC for this dataset
- **Small test set increases variance** in performance metrics

## 🚀 Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   python overseas_analysis.py
   ```

3. **View Results**:
   - Check `final_scouting_report.csv` for player rankings
   - Review `eda_visualizations.png` for data exploration
   - Examine `ml_diagnostic_plots.png` for model validation

## 📈 Business Impact

This analysis provides the Sacramento Kings with:
- **Data-driven scouting priorities** based on NBA success patterns
- **Quantified player valuations** using calibrated ML models
- **Age-appropriate development curves** for international prospects
- **Team-specific fit scoring** aligned with successful NBA players

The methodology can be extended to future draft classes and international leagues, providing a scalable framework for international player evaluation.

---

**Note**: This analysis was completed as part of the Sacramento Kings Data Science Assessment. All player names have been anonymized per assignment requirements.
