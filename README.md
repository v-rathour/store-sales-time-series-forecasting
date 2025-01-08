# Time Series Sales Forecasting

This project involves forecasting sales using time series models.

## File Overview
1. `data_preprocessing.py`: Loads and preprocesses data (merging, missing values).
2. `exploratory_analysis.py`: Performs exploratory data analysis (EDA) and visualizations.
3. `feature_engineering.py`: Generates features for modeling (lags, rolling averages, etc.).
4. `modeling.py`: Trains the forecasting model (Random Forest).
5. `forecasting.py`: Uses the trained model to make predictions on the test data.
6. `evaluation.py`: Evaluates the model's performance based on Mean Absolute Error.
7. `config.py`: Stores configuration settings (e.g., file paths).
8. `requirements.txt`: Lists required Python packages.

## How to Run
1. Preprocess data: `python src/data_preprocessing.py`
2. Perform EDA: `python src/exploratory_analysis.py`
3. Generate features: `python src/feature_engineering.py`
4. Train the model: `python src/modeling.py`
5. Make predictions: `python src/forecasting.py`
6. Evaluate the model: `python src/evaluation.py`
