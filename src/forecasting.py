# import pandas as pd
# import pickle


# def forecast_sales():
#     # Load preprocessed data
#     test = pd.read_csv('data/test_preprocessed.csv', parse_dates=['date'])

#     # Load the trained model
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     # Prepare the features
#     test['year'] = test['date'].dt.year
#     test['month'] = test['date'].dt.month
#     test['day'] = test['date'].dt.day
#     test['weekday'] = test['date'].dt.weekday
#     test['is_weekend'] = test['weekday'].isin([5, 6])
#     test['lag_1'] = test['sales'].shift(1)
#     test['lag_7'] = test['sales'].shift(7)
#     test['rolling_mean_7'] = test['sales'].rolling(window=7).mean()

#     # Select features for prediction
#     features = ['year', 'month', 'day', 'weekday', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']
#     X_test = test[features].dropna()

#     # Forecast sales
#     test['predicted_sales'] = model.predict(X_test)

#     # Save forecasted results
#     test.to_csv('data/test_forecasted.csv', columns=['date', 'predicted_sales'], index=False)

# if __name__ == "__main__":
#     forecast_sales()


import pandas as pd
import numpy as np  # Ensure numpy is imported

def forecast_sales():
    # Load preprocessed data
    train = pd.read_csv('data/train_preprocessed.csv', parse_dates=['date'])
    test = pd.read_csv('data/test_preprocessed.csv', parse_dates=['date'])

    # Feature Engineering (e.g., lag features, rolling means)
    test['lag_1'] = train['sales'].shift(1)  # Use 'train' data for creating lag features
    test['lag_7'] = train['sales'].shift(7)
    test['lag_30'] = train['sales'].shift(30)
    test['rolling_mean_7'] = train['sales'].rolling(window=7).mean()
    test['rolling_mean_30'] = train['sales'].rolling(window=30).mean()

    # Replace this with your model's prediction (for now we're generating random predictions)
    # predictions = model.predict(X_test)  # Use your trained model for actual predictions

    # Generate random sales predictions for testing purposes
    predictions = np.random.randint(100, 500, size=len(test))

    # Add the predictions to the test dataset
    test['forecasted_sales'] = predictions

    # Save predictions into 'test_forecasted.csv' file
    test.to_csv('data/test_forecasted.csv', index=False)
    print("Predictions saved to 'data/test_forecasted.csv'")

# Run the forecasting function
forecast_sales()
