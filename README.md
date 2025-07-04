# FYP-Cellular Network Traffic Prediction Using CNN-LSTM and Boosting
 Cellular traffic prediction, deep learning, CNN-LSTM, Boosting, Kolmogorov-Arnold Network (KAN)
This project aims to design and implement a cellular network traffic prediction model based on deep learning. It uses CNN-LSTM hybrid structure to extract spatio-temporal features, and introduces Boosting  technology to improve prediction accuracy. The Kolmogorov-Arnold network (KAN) will also be attempted to demonstrate the potential of the new neural network architecture. The goal is to achieve better prediction results than traditional methods on public data sets, and have certain generalization ability.

## Project Overview

This project aims to predict cellular network traffic throughput based on time series data using a hybrid CNN-LSTM deep learning model combined with XGBoost for boosting prediction accuracy. The model processes multi-feature data including network type, throughput, and locality (area) information, with thorough data preprocessing including anomaly detection and feature scaling.

## Features

* Data cleaning with Isolation Forest for anomaly detection
* One-hot encoding of categorical features (Network Type, Locality)
* Feature scaling with MinMaxScaler
* Sliding window method to generate time series sequences for training
* Hybrid CNN-LSTM architecture for temporal feature extraction and sequence modeling
* Boosting with XGBoost to correct residual errors and improve performance
* Model training with early stopping and learning rate scheduling
* Visualization tools to compare true throughput and predictions over time
* Interactive plotting for exploring predictions over user-selected time intervals

## Requirements

* Python 3.8+
* pandas
* numpy
* scikit-learn
* tensorflow (>=2.0)
* xgboost
* matplotlib
* seaborn

Install dependencies with:

```bash
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn
```

## Usage

1. **Prepare Dataset:**
   Place the `signal_metrics.csv` dataset in the project directory. The dataset should include columns: `Timestamp`, `Network Type`, `Data Throughput (Mbps)`, and `Locality`(part).The full dataset is from Kaggle.com
   
3. **Run Data Preprocessing:**
   The script will load data, encode categorical features, detect and remove anomalies, normalize features, and generate training sequences with a sliding window.

4. **Train Model:**
   The CNN-LSTM model is trained on preprocessed data with early stopping to prevent overfitting.

5. **Boost Predictions:**
   An XGBoost model trains on residual errors to improve prediction accuracy.

6. **Evaluate and Visualize:**
   The script outputs performance metrics (MSE, MAE, RMSE, R²) and generates plots comparing actual and predicted throughput.

7. **Interactive Visualization:**
   (Test) Launch the UI to visualize predictions vs. real data over arbitrary time intervals.

## Notes

* Ensure the environment has GPU support for faster training (optional).
* Adjust hyperparameters like window size, batch size, learning rate in `main.py` as needed.
* The locality features are scaled down to reduce their impact relative to numerical throughput values.
