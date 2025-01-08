import pandas as pd
import matplotlib.pyplot as plt

def perform_eda():
    # Load preprocessed data
    train = pd.read_csv('data/train_preprocessed.csv', parse_dates=['date'])

    # Visualize sales trends over time
    plt.figure(figsize=(10, 6))
    plt.plot(train['date'], train['sales'], label='Sales')
    plt.title('Sales Trends')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Check for missing values
    print(f'Missing values in train data:\n{train.isnull().sum()}')

    # Sales distribution by store type
    plt.figure(figsize=(10, 6))
    train.groupby('type')['sales'].mean().plot(kind='bar')
    plt.title('Average Sales by Store Type')
    plt.ylabel('Average Sales')
    plt.show()

if __name__ == "__main__":
    perform_eda()
