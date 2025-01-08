import pandas as pd

def generate_features(data):
    # Extract time-based features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday
    data['is_weekend'] = data['weekday'].isin([5, 6])

    # Lag features for previous sales (e.g., previous 1 day)
    data['lag_1'] = data['sales'].shift(1)
    data['lag_7'] = data['sales'].shift(7)

    # Rolling mean for past 7 days
    data['rolling_mean_7'] = data['sales'].rolling(window=7).mean()

    return data

if __name__ == "__main__":
    train = pd.read_csv('data/train_preprocessed.csv', parse_dates=['date'])
    train = generate_features(train)
    train.to_csv('data/train_featured.csv', index=False)
