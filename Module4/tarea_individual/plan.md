# Player Rating Prediction Project Plan

## Project Overview

Build a regression model to predict player ratings in football matches based on performance statistics and historical context.

## Project Structure

```
Module4/tarea_individual/
├── data/                           # Raw and processed data
├── notebooks/                      # Existing sofascore_scraping.ipynb
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data cleaning, feature engineering
│   ├── feature_engineering.py     # Rolling averages, context features
│   ├── model_training.py          # Model training and validation
│   ├── evaluation.py              # Model evaluation and metrics
│   └── utils.py                   # Helper functions
├── models/                         # Trained models
├── results/                        # Model results and analysis
└── requirements.txt                # Dependencies
```

## Phase 1: Data Collection & Infrastructure Setup

### Task 1.1: Create Project Structure

- [X] Create `src/` directory with `__init__.py`
- [X] Create `models/` directory for trained models
- [X] Create `results/` directory for outputs
- [X] Create `requirements.txt` with necessary packages

### Task 1.2: Data Collection Functions

- [X] Complete `extract_round_events()` function for getting all matches
- [X] Complete `extract_match_players_stats()` function for player statistics
- [X] Complete `save_players_stats_to_csv()` function for data persistence
- [X] Test data collection pipeline end-to-end

### Task 1.3: Data Quality Assessment

- [ ] Analyze data completeness and missing values
- [ ] Check rating distribution and identify any outliers
- [ ] Determine minimum minutes threshold for reliable ratings
- [ ] Document data schema and field descriptions

## Phase 2: Data Preprocessing & Feature Engineering

### Task 2.1: Data Preprocessing (`data_preprocessing.py`)

- [ ] Filter records with valid ratings
- [ ] Separate goalkeepers from field players
- [ ] Handle missing values (fill with 0)
- [ ] Data type conversions and validation
- [ ] Create clean, structured dataset

### Task 2.2: Feature Engineering (`feature_engineering.py`)

- [ ] Extract performance statistics features
- [ ] Calculate rolling averages from previous matches (configurable window)
- [ ] Create match context features (minutes, substitute status, etc.)
- [ ] Generate position-specific features
- [ ] Feature scaling and normalization

### Task 2.3: Data Validation

- [ ] Ensure rating values are between 0-10
- [ ] Validate feature ranges and distributions
- [ ] Check for data leakage between train/validation sets
- [ ] Create data quality reports

## Phase 3: Model Development

### Task 3.1: Baseline Models (`model_training.py`)

- [ ] Implement simple baseline (mean rating prediction)
- [ ] Linear regression model
- [ ] Random Forest model
- [ ] XGBoost model (if available)
- [ ] Separate models for goalkeepers vs field players

### Task 3.2: Model Training Pipeline

- [ ] Implement cross-validation strategy
- [ ] Time-aware train/validation/test splits
- [ ] Hyperparameter tuning (GridSearch/RandomizedSearch)
- [ ] Model persistence and loading

### Task 3.3: Feature Importance Analysis

- [ ] Extract feature importance from tree-based models
- [ ] Analyze correlation between features and target
- [ ] Identify most impactful performance statistics
- [ ] Create feature importance visualizations

## Phase 4: Model Evaluation & Analysis

### Task 4.1: Performance Metrics (`evaluation.py`)

- [ ] Implement RMSE, MAE, R² metrics
- [ ] Cross-validation performance analysis
- [ ] Train vs validation performance comparison
- [ ] Model comparison and ranking

### Task 4.2: Error Analysis

- [ ] Analyze prediction errors by position
- [ ] Identify cases where model performs poorly
- [ ] Error distribution analysis
- [ ] Outlier detection in predictions

### Task 4.3: Model Interpretability

- [ ] Feature importance ranking
- [ ] Partial dependence plots for key features
- [ ] SHAP values analysis (if applicable)
- [ ] Model decision explanations

## Phase 5: Iteration & Improvement

### Task 5.1: Model Refinement

- [ ] Test position-specific models
- [ ] Experiment with different feature combinations
- [ ] Optimize hyperparameters
- [ ] Ensemble methods (if beneficial)

### Task 5.2: Feature Engineering Improvements

- [ ] Test different rolling window sizes
- [ ] Create interaction features
- [ ] Add team-level context features
- [ ] Implement feature selection methods

### Task 5.3: Validation Strategy

- [ ] Test different validation approaches
- [ ] Time-based validation splits
- [ ] Leave-one-match-out validation
- [ ] Robustness testing

## Phase 6: Documentation & Deployment

### Task 6.1: Code Documentation

- [ ] Add docstrings to all functions
- [ ] Create README with usage examples
- [ ] Document model performance and limitations
- [ ] Create user guide for model usage

### Task 6.2: Results Documentation

- [ ] Create comprehensive results report
- [ ] Save model artifacts and metadata
- [ ] Document feature importance findings
- [ ] Create model comparison summary

### Task 6.3: Future Improvements

- [ ] Identify areas for model enhancement
- [ ] Plan for real-time prediction pipeline
- [ ] Consider additional data sources
- [ ] Performance optimization opportunities

## Technical Requirements

### Dependencies

- pandas, numpy, scikit-learn
- matplotlib, seaborn for visualization
- requests for API calls
- joblib for model persistence

### Data Requirements

- Minimum dataset size: 1000+ player-match records
- Rating distribution should be roughly normal
- Sufficient feature coverage across positions

### Performance Targets

- RMSE < 1.0 (on 0-10 scale)
- R² > 0.3 (baseline improvement)
- Feature importance insights for actionable improvements

## Success Criteria

1. [ ] Functional data collection pipeline
2. [ ] Clean, processed dataset with engineered features
3. [ ] Trained models with cross-validation performance
4. [ ] Feature importance analysis and insights
5. [ ] Model comparison and recommendations
6. [ ] Documentation and reproducibility

## Timeline Estimate

- **Phase 1-2**: 2-3 days (data infrastructure and preprocessing)
- **Phase 3**: 2-3 days (model development)
- **Phase 4**: 1-2 days (evaluation and analysis)
- **Phase 5**: 2-3 days (iteration and improvement)
- **Phase 6**: 1 day (documentation)

**Total Estimated Time**: 8-12 days
