import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor


def generate_visuals(file_path):
    df = pd.read_csv(file_path)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df['target'], bins=100, kde=True, color='royalblue')
    plt.title('Distribution of Log Error', fontsize=15)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/plots/target_distribution.png', dpi=300)

    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['target'].abs().sort_values(ascending=False)
    top_features = correlations.head(15).index
    sns.heatmap(numeric_df[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap', fontsize=15)
    plt.tight_layout()
    plt.savefig('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/plots/correlation_heatmap.png', dpi=300)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=['target'])
    y = df['target']
    model = XGBRegressor(n_estimators=100, random_state=42).fit(X, y)

    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='seagreen')
    plt.title('Top 10 Feature Importance (XGBoost)', fontsize=15)
    plt.tight_layout()
    plt.savefig('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/plots/feature_importance.png', dpi=300)


if __name__ == "__main__":
    generate_visuals('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/data/Zillow_Cleaned.csv')