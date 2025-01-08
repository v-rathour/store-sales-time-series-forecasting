import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

def train_model():
    # Load featured data
    train = pd.read_csv('data/train_featured.csv', parse_dates=['date'])

    # Select features and target
    features = ['year', 'month', 'day', 'weekday', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']
    target = 'sales'

    X = train[features].dropna()
    y = train[target].loc[X.index]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')

    # Save the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()
