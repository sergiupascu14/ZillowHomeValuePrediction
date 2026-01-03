# Zillow Home Value Prediction: A Residual Analysis Approach

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a Machine Learning solution designed to predict the **Log Error** of Zillow’s proprietary "Zestimate" algorithm. 

Unlike traditional real estate models that predict absolute property prices, this project focuses on **Residual Analysis**. The goal is to identify systematic patterns and biases that the primary valuation model failed to capture. By modeling the error itself, we create a corrective layer that enhances overall valuation precision.

### The Objective
The target variable is defined as:
$$logerror = \log(Zestimate) - \log(SalePrice)$$

## 🛠️ Methodology

### 1. Data Preprocessing & Heuristics
The dataset was subjected to a rigorous cleaning pipeline to handle high sparsity and multicollinearity:
- **Feature Pruning:** Removed columns with >50% missing values to prevent bias injection.
- **Robust Imputation:** Utilized median filling for numerical features to mitigate the influence of outliers.
- **Feature Engineering:** Derived `Property Age` (calculated relative to 2016) and `Total Rooms`.
- **Filtering:** Applied a 0.98 correlation threshold and clipped outliers at the 1st and 99th percentiles.

### 2. Ensemble Architecture
We implemented an **Ensemble Voting Regressor** to balance precision and generalization:
- **XGBoost Regressor:** Optimized for capturing non-linear interactions.
- **Random Forest Regressor:** Integrated to reduce individual model variance.



## 📊 Performance Results

The model was evaluated on a 20% hold-out test set using both regression and directional classification metrics:

| Metric | Value | Significance |
| :--- | :--- | :--- |
| **MAE** | **0.05323** | Precision of the corrective log-error estimate. |
| **Directional Accuracy** | **57.38%** | Success rate in predicting over vs. under-valuation. |
| **Tolerance Accuracy** | **67.28%** | Predictions within a strict ±0.05 error margin. |

## 🚀 Installation & Usage

### 1. Clone the repository
`git clone https://github.com/sergiupascu14/ZillowHomeValuePrediction.git`  
`cd ZillowHomeValuePrediction`

### 2. Install dependencies
`pip install pandas numpy matplotlib seaborn scikit-learn xgboost`

### 3. Execution Flow
To reproduce the results, run the scripts in the following order:
1. **Preprocess:** `python src/data_preprocessing.py` (Cleans the data and creates `data/Zillow_Cleaned.csv`)
2. **Train:** `python src/model_training.py` (Trains the Ensemble model and saves `.pkl` files in `src/`)
3. **Visualize:** `python src/visualization.py` (Generates performance charts in the `plots/` folder)
4. **Predict:** `python src/predict.py` (Runs inference on sample data)

## 📁 Project Structure

Based on the current implementation, the repository is organized as follows:

```text
├── data/
│   ├── Zillow.csv              # Original dataset
│   └── Zillow_Cleaned.csv      # Processed and engineered data
├── plots/                      # Generated evaluation charts
│   ├── confusion_matrix_final.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── target_distribution.png
├── src/                        # Core source code and saved models
│   ├── data_preprocessing.py
│   ├── model_training.py       # Main ensemble training logic
│   ├── visualization.py        # Chart generation logic
│   ├── predict.py              # Inference script
│   ├── final_model.pkl         # Trained Ensemble model
│   └── scaler.pkl              # Saved StandardScaler
├── LICENSE                     # MIT License file
├── main.py                     # Project entry point
└── README.md                   # Project documentation
```


## 🎓 Academic Context
Developed as a final project for the **Machine Learning** course.
- **Institution:** Technical University of Cluj-Napoca (UTCN)
- **Faculty:** Cyber-Physical Systems
- **Author:** Pascu Sergiu-Andrei
- **Academic Year:** 2025-2026

## 📄 License
This project is licensed under the **MIT License**.

```text
MIT License
Copyright (c) 2026 Pascu Sergiu-Andrei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...
```

---
*Technical University of Cluj-Napoca - 2026*