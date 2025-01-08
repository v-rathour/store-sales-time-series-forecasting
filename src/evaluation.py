import pandas as pd
from sklearn.metrics import mean_absolute_error

# import pandas as pd

# Generate synthetic actual sales data for testing
data = {
    'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
    'sales': [100 + i * 5 for i in range(100)]  # Example sales data
}

# Create DataFrame
actual_sales = pd.DataFrame(data)

# Save to CSV
actual_sales.to_csv('data/actual_sales.csv', index=False)


# def evaluate_model():
#     # Load actual and predicted sales data
#     test = pd.read_csv('data/test_forecasted.csv')

#     # Assuming actual sales are available in 'test.csv'
#     actual = pd.read_csv('data/test.csv', parse_dates=['date'])

#     # Merge actual and predicted sales
#     merged = test.merge(actual[['date', 'sales']], on='date', how='left')
#     merged['actual_sales'] = merged['sales']
#     merged['predicted_sales'] = merged['predicted_sales']

#     # Calculate Mean Absolute Error
#     mae = mean_absolute_error(merged['actual_sales'], merged['predicted_sales'])
#     print(f'Mean Absolute Error: {mae}')

# if __name__ == "__main__":
#     evaluate_model()

def evaluate_model():
    # Assuming 'test_forecasted.csv' contains predictions and 'actual_sales.csv' contains actual values
    test = pd.read_csv('data/test_forecasted.csv')
    actual = pd.read_csv('data/actual_sales.csv')  # Ensure this file has the 'sales' column

    # Check column names (for debugging)
    print(actual.columns)  # Ensure 'sales' is in the columns of actual
    print(test.columns)    # Ensure 'date' and 'forecasted_sales' are in the columns of test

    # Merge on 'date' column
    merged = test.merge(actual[['date', 'sales']], on='date', how='left')

    # Evaluate model (e.g., calculating RMSE or other metrics)
    merged['error'] = merged['sales'] - merged['forecasted_sales']
    merged['abs_error'] = merged['error'].abs()
    rmse = (merged['error'] ** 2).mean() ** 0.5
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    evaluate_model()