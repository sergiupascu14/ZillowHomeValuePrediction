import pandas as pd
import numpy as np


def clean_data(file_path):
    df = pd.read_csv(file_path)

    missing_stats = df.isnull().mean() * 100
    df = df.drop(columns=missing_stats[missing_stats > 50].index.tolist())

    if 'parcelid' in df.columns:
        df = df.drop(columns=['parcelid'])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if 'yearbuilt' in df.columns:
        df['property_age'] = 2016 - df['yearbuilt']

    if 'bathroomcnt' in df.columns and 'bedroomcnt' in df.columns:
        df['total_rooms'] = df['bathroomcnt'] + df['bedroomcnt']

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    df = df.drop(columns=to_drop)

    df = df[(df['target'] > df['target'].quantile(0.01)) & (df['target'] < df['target'].quantile(0.99))]

    return df


if __name__ == "__main__":
    cleaned_df = clean_data('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/data/Zillow.csv')
    cleaned_df.to_csv('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/data/Zillow_Cleaned.csv', index=False)