import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBRegressor


def train_ensemble(data_path):
    df = pd.read_csv(data_path)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)

    ensemble = VotingRegressor(estimators=[('xgb', xgb), ('rf', rf)])
    ensemble.fit(X_train, y_train)

    with open('final_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)

    preds = ensemble.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    y_test_binary = (y_test > 0).astype(int)
    preds_binary = (preds > 0).astype(int)
    class_acc = accuracy_score(y_test_binary, preds_binary)

    tolerance_acc = (np.abs(y_test - preds) < 0.05).mean()

    print(f"Ensemble MAE: {mae:.5f}")
    print(f"Directional Accuracy: {class_acc:.2%}")
    print(f"Tolerance Accuracy (Error < 0.05): {tolerance_acc:.2%}")

    cm = confusion_matrix(y_test_binary, preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Over', 'Under'])
    disp.plot(cmap='magma')
    plt.title('Final Ensemble Confusion Matrix')
    plt.savefig('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/plots/confusion_matrix_final.png', dpi=300)


if __name__ == "__main__":
    train_ensemble('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/data/Zillow_Cleaned.csv')