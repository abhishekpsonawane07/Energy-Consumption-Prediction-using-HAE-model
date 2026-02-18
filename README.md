# Energy-Consumption-Prediction-using-HAE-model
To compare traditional statistical models and machine learning models with a deep learning hierarchical attention-based model for electricity demand forecasting and evaluate which performs best.


---

# ⚡ Energy Consumption Forecasting

## Comparative Study of SARIMA, XGBoost and Hierarchical Attention-Enhanced LSTM (HAE-LSTM)

---

## 📌 Project Overview

This project presents a **comparative analysis of three different forecasting approaches** for electricity demand prediction:

* 📊 SARIMA (Statistical Model)
* 🌲 XGBoost (Machine Learning Model)
* 🧠 Hierarchical Attention-Enhanced LSTM (Deep Learning Model)

The goal is to evaluate which model provides the most accurate electricity demand forecasting using real-world historical data.

---

## 🎯 Objective

* To forecast electricity demand using historical time-series data.
* To compare statistical, machine learning, and deep learning approaches.
* To evaluate model performance using standard regression metrics.
* To determine the most reliable model for short-term and long-term forecasting.

---

## 📂 Dataset

* Historical Electricity Demand Data (2009–2024)
* Half-hourly demand readings (48 readings per day)
* Target variable: `tsd` (Total System Demand)
* No missing values

---

## 🧠 Models Implemented

### 1️⃣ SARIMA Model

* Captures seasonality and trend
* Suitable for linear time-series forecasting
* Traditional statistical forecasting method

---

### 2️⃣ XGBoost Model

* Gradient Boosting Decision Tree model
* Captures nonlinear relationships
* Strong performance in structured data

---

### 3️⃣ HAE-LSTM Model (Proposed Model)

Hierarchical Attention-Enhanced Long Short-Term Memory Network

#### Architecture:

* **Multi-scale Inputs:**

  * Daily (48 timesteps)
  * Weekly (7 days)
  * Monthly (30 days)
  * Yearly (365 days)

* For each level:

  * Bidirectional LSTM (32 units)
  * Batch Normalization
  * Attention Mechanism

* Feature Fusion:

  * Concatenation of all attention outputs

* Fully Connected Layers:

  * Dense (32 units, ReLU)
  * Dense (1 unit, Linear)

This hierarchical structure allows the model to:

* Capture short-term fluctuations
* Capture seasonal patterns
* Capture long-term yearly trends
* Focus on important time steps using attention

---

## 🏗 HAE-LSTM Model Architecture

```
Daily Input  → BiLSTM → BatchNorm → Attention
Weekly Input → BiLSTM → BatchNorm → Attention
Monthly Input → BiLSTM → BatchNorm → Attention
Yearly Input → BiLSTM → BatchNorm → Attention

Concatenate → Dense(32) → Dense(1) → Final Prediction
```

---

## 📊 Model Evaluation Metrics

The following performance metrics were used:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)

Lower values indicate better performance.

---

Perfect 👌 I’ll now update your **GitHub report performance section** properly formatted and professional, including your actual results.

You can directly paste this inside your README under **Model Performance**.

---

# 📊 Model Performance Evaluation

The models were evaluated using:

* **RMSE (Root Mean Squared Error)** → Measures magnitude of prediction error in MW
* **MAPE (Mean Absolute Percentage Error)** → Measures average percentage error

Lower values indicate better forecasting performance.

---

## 📈 Performance Comparison

| Model                     | RMSE (MW)  | MAPE (%)  |
| SARIMA MODEL              | ---------- | 15.44%    |
| XGBoost                   | 3786.88    | 11.29%    |
| XGBoost (GridSearch + CV) | 2744.98    | 7.70%     |
| HAE-LSTM (Proposed Model) | 489.45     | 1.36%     |

---

## 🏆 Best Performing Model

The **HAE-LSTM model significantly outperformed all other models**, achieving:

* 🔹 **489.45 MW RMSE**
* 🔹 **1.36% MAPE**

This indicates:

* Extremely low forecasting error
* High prediction stability
* Strong ability to capture multi-scale temporal dependencies

Compared to tuned XGBoost:

* RMSE reduced by **~82%**
* MAPE reduced by **~82%**

This demonstrates the effectiveness of:

* Hierarchical input structure
* Bidirectional LSTM
* Attention mechanism

---

## 📌 Interpretation of Results

### 🔹 XGBoost

* Performs reasonably well on structured data.
* Improvement observed after hyperparameter tuning (GridSearch + Cross-Validation).
* Still struggles with long-term temporal dependencies.

### 🔹 HAE-LSTM

* Captures short-term and long-term patterns simultaneously.
* Attention mechanism improves feature importance learning.
* Achieves very high forecasting accuracy (≈ 98.64% accuracy implied by MAPE).

---


## 📊 Visualization

* Predicted vs Actual demand graph
* Error distribution analysis
* Model performance comparison charts

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* XGBoost
* Statsmodels (SARIMA)
* NumPy, Pandas
* Matplotlib
* Scikit-learn

---

## 🚀 Key Contributions

* Implementation of a multi-scale hierarchical forecasting architecture
* Integration of attention mechanism for improved temporal feature extraction
* Comparative evaluation of statistical, machine learning, and deep learning models
* Performance benchmarking using multiple regression metrics

---

## 📌 Conclusion

* SARIMA performs well for linear seasonal data.
* XGBoost captures nonlinear relationships efficiently.
* HAE-LSTM outperforms other models due to:

  * Hierarchical multi-scale learning
  * Bidirectional LSTM
  * Attention mechanism

The proposed HAE-LSTM model provides a robust and scalable approach for real-world electricity demand forecasting.

---

## 🔮 Future Work

* Hyperparameter tuning for deeper optimization
* Integration with real-time dashboard
* Deployment using Flask/FastAPI
* Adding exogenous variables (weather, holidays, economic indicators)

---

# 🎓 Academic Value

This project demonstrates:

* Time-series modeling expertise
* Deep learning architecture design
* Model evaluation and benchmarking
* Practical energy forecasting application


