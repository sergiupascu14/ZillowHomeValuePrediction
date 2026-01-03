import pandas as pd
import pickle
import numpy as np


def predict_new_data(data_path):
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path).iloc[:5]
    X_new = df.drop(columns=['target'])

    for col in X_new.select_dtypes(include=['object']).columns:
        X_new[col] = X_new[col].astype('category').cat.codes

    predictions = model.predict(X_new)

    print("--- New Predictions (Log Error) ---")
    for i, pred in enumerate(predictions):
        status = "Underestimated" if pred > 0 else "Overestimated"
        print(f"Property {i + 1}: Predicted LogError = {pred:.5f} ({status})")


if __name__ == "__main__":
    predict_new_data('/Users/pascusergiu/PycharmProjects/ZillowHomeValuePrediction/data/Zillow_Cleaned.csv')