import pandas as pd

def preprocess_data():
    # Load datasets
    train = pd.read_csv('data/train.csv', parse_dates=['date'])
    test = pd.read_csv('data/test.csv', parse_dates=['date'])
    stores = pd.read_csv('data/stores.csv')
    oil = pd.read_csv('data/oil.csv', parse_dates=['date'])
    holidays = pd.read_csv('data/holidays_events.csv', parse_dates=['date'])

    # Merge datasets
    train = train.merge(stores, on='store_nbr', how='left')
    test = test.merge(stores, on='store_nbr', how='left')
    train = train.merge(oil, on='date', how='left')
    test = test.merge(oil, on='date', how='left')
    train = train.merge(holidays, on='date', how='left', suffixes=('', '_holiday'))
    test = test.merge(holidays, on='date', how='left', suffixes=('', '_holiday'))

    # Fill missing values
    train['dcoilwtico'] = train['dcoilwtico'].fillna(method='ffill')
    test['dcoilwtico'] = test['dcoilwtico'].fillna(method='ffill')
    train['type'] = train['type'].fillna('None')
    test['type'] = test['type'].fillna('None')

    return train, test

if __name__ == "__main__":
    train, test = preprocess_data()
    train.to_csv('data/train_preprocessed.csv', index=False)
    test.to_csv('data/test_preprocessed.csv', index=False)
