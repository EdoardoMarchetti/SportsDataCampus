# Player Rating Prediction Project

A machine learning project to predict football player ratings based on performance statistics and historical context.

## Project Overview

This project aims to build regression models that can predict player ratings in football matches using:
- Performance statistics (passes, goals, tackles, etc.)
- Historical context (rolling averages from previous matches)
- Match context (minutes played, substitute status, position)
- Position-specific features

## Project Structure

```
Module4/tarea_individual/
├── data/                           # Raw and processed data
├── notebooks/                      # Jupyter notebooks
│   └── sofascore_scraping.ipynb   # Data collection notebook
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── data_preprocessing.py      # Data cleaning and validation
│   ├── feature_engineering.py     # Feature creation and engineering
│   ├── model_training.py          # Model training and validation
│   ├── evaluation.py              # Model evaluation and analysis
│   └── utils.py                   # Utility functions
├── models/                         # Trained models
├── results/                        # Model results and analysis
├── plan.md                         # Project plan and tasks
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Features

### Data Preprocessing (`data_preprocessing.py`)
- Filter records with valid ratings
- Separate goalkeepers from field players
- Handle missing values
- Data quality validation
- Clean dataset creation

### Feature Engineering (`feature_engineering.py`)
- Performance statistics extraction
- Rolling averages from previous matches
- Match context features
- Position-specific features
- Feature normalization

### Model Training (`model_training.py`)
- Multiple regression models (Linear, Ridge, Lasso, Random Forest)
- Cross-validation
- Hyperparameter tuning
- Model persistence
- Feature importance analysis

### Evaluation (`evaluation.py`)
- Comprehensive performance metrics
- Error analysis by position
- Visualization of results
- Outlier detection
- Model comparison

### Utilities (`utils.py`)
- Data loading/saving functions
- Experiment logging
- Data validation
- Feature summaries

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection

Use the notebook `notebooks/sofascore_scraping.ipynb` to:
- Extract match events for each round
- Collect player statistics for each match
- Save data to CSV format

### 2. Data Preprocessing

```python
from src.data_preprocessing import create_clean_dataset, separate_goalkeepers

# Load and clean data
df = pd.read_csv('data/your_data.csv')
df_clean = create_clean_dataset(df)

# Separate goalkeepers and field players
gk_df, field_df = separate_goalkeepers(df_clean)
```

### 3. Feature Engineering

```python
from src.feature_engineering import engineer_all_features

# Create all features
df_features = engineer_all_features(df_clean, rolling_window=3, normalize=True)
```

### 4. Model Training

```python
from src.model_training import PlayerRatingModel, train_multiple_models

# Train a single model
model = PlayerRatingModel('random_forest')
X, y = model.prepare_features(df_features)
metrics = model.train(X, y)

# Train multiple models for comparison
results = train_multiple_models(df_features)
```

### 5. Model Evaluation

```python
from src.evaluation import generate_evaluation_report

# Generate comprehensive evaluation report
report = generate_evaluation_report(df_with_predictions, 'rating', 'rating_pred')
```

## Model Types

### Baseline Model
- Simple mean prediction for comparison

### Linear Models
- **Linear Regression**: Basic linear model
- **Ridge Regression**: Linear model with L2 regularization
- **Lasso Regression**: Linear model with L1 regularization

### Tree-based Models
- **Random Forest**: Ensemble of decision trees

## Performance Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## Data Requirements

- Minimum dataset size: 1000+ player-match records
- Rating values between 0-10
- Sufficient feature coverage across positions
- Historical data for rolling averages

## Configuration

Key parameters that can be adjusted:
- Rolling window size for historical context (default: 3 matches)
- Feature normalization method (standard, minmax, robust)
- Cross-validation folds (default: 5)
- Train/test split ratio (default: 80/20)

## Output Files

The project generates:
- **Processed datasets**: Clean, feature-engineered data
- **Trained models**: Saved model artifacts
- **Evaluation reports**: Performance metrics and analysis
- **Visualizations**: Error analysis plots, feature importance charts
- **Experiment logs**: Configuration and results tracking

## Future Improvements

- Position-specific models
- Advanced feature engineering
- Ensemble methods
- Real-time prediction pipeline
- Additional data sources

## Contributing

1. Follow the project structure
2. Add proper documentation and type hints
3. Include error handling and logging
4. Test functions before committing

## License

This project is part of the Sports Data Campus curriculum.

## Contact

For questions or issues, refer to the project plan in `plan.md`.
