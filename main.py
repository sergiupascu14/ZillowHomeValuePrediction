import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(properties_path, train_path):
    print("Loading data...")
    properties = pd.read_csv(properties_path)
    train = pd.read_csv(train_path)

    data = pd.merge(train, properties, on='parcelid', how='left')
    return data


def basic_eda(df):
    print(f"Dataset Shape: {df.shape}")

    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_values / len(df)) * 100

    print("\nTop 10 features with most missing values (%):")
    print(missing_percent.head(10))

    plt.figure(figsize=(10, 6))
    sns.histplot(df['logerror'], bins=100, kde=True, color='blue')
    plt.title('Distribution of Log Error (Target Variable)')
    plt.xlabel('Log Error')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    PROPS_FILE = "data/properties_2016.csv"
    TRAIN_FILE = "data/train_2016_v2.csv"

    try:
        df_final = load_data(PROPS_FILE, TRAIN_FILE)
        basic_eda(df_final)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the CSV files are in the 'data' folder.")