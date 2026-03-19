# AI-Enabled Smart Grid Monitoring System

A complete end-to-end AI platform for real-time smart grid management, featuring anomaly detection, load forecasting, predictive maintenance, and a live web dashboard.

Built by **Reuben Abraham Jacob**

---
## Project Overview

Power grids are continuously monitored for faults and abnormal conditions. Traditional monitoring systems raise alarms only when predefined thresholds are crossed — they react to problems but cannot predict them.

This project builds an AI-enabled smart grid monitoring system that goes beyond reactive monitoring. Using machine learning, the system:

- Detects electrical faults and anomalies in real time using unsupervised learning
- Forecasts reactive and active power demand 15 minutes ahead, enabling proactive power factor correction
- Predicts equipment degradation before failure occurs, enabling condition-based maintenance instead of scheduled maintenance
- Presents everything on a live web dashboard with real time gauges, alerts, and trend graphs

The system is currently built on synthetic smart grid data modeled after realistic electrical load patterns. The next phase involves building a real sensor circuit using Arduino with CT and PT transducers, collecting real electrical data, and retraining all models on real data to compare performance against the synthetic baseline. The complete system will then be deployed on an NVIDIA Jetson Orin Nano for standalone edge inference — enabling the system to run independently without a laptop, directly alongside the electrical panel.

## System Architecture
```
Sensor Data (Voltage, Current, PF, Active/Reactive Power, Frequency)
                            |
            ----------------+----------------
            |               |               |
    Anomaly Detection  Load Forecasting  Predictive
    (Isolation Forest) (XGBoost/ANN/LSTM) Maintenance
            |               |          (Random Forest)
            |               |               |
            ----------------+----------------
                            |
                  Streamlit Dashboard
                  (Live Web Interface)
```

---

## Modules

### Module 1 - Data Generation
Synthetic smart grid data simulating 30 days of electrical sensor readings at 15-minute intervals. Includes realistic day/night load patterns, injected fault anomalies (2%), and gradual equipment degradation events.

### Module 2 - Anomaly Detection
Isolation Forest model detecting voltage faults and power factor degradation in real time.

| Metric    | Score |
|-----------|-------|
| Precision | 0.93  |
| Recall    | 0.95  |
| F1 Score  | 0.94  |

### Module 3 - Load Forecasting
Three models benchmarked for short-term reactive and active power forecasting (15 minutes ahead).

| Model    | R² (Reactive) | MAE (Reactive) | R² (Active) | MAE (Active) |
|----------|---------------|----------------|-------------|--------------|
| XGBoost  | 0.997         | 0.191 kVAR     | 1.000       | 0.073 kW     |
| ANN      | 0.987         | 0.706 kVAR     | 0.991       | 1.260 kW     |
| LSTM     | 0.847         | 2.506 kVAR     | 0.958       | 2.694 kW     |

Key finding: XGBoost outperforms ANN and LSTM on tabular smart grid data with explicit feature engineering, consistent with existing literature on structured time series forecasting.

### Module 4 - Predictive Maintenance
Random Forest Regressor predicting equipment degradation level (0 to 1) from rolling window features. Key finding: 4-hour rolling average of power factor accounts for 91% of feature importance.

| Metric | Score |
|--------|-------|
| MAE    | 0.069 |
| RMSE   | 0.113 |
| R²     | 0.908 |

Health thresholds:
- 0.0 - 0.3: Healthy, no action required
- 0.3 - 0.7: Early degradation, schedule inspection
- 0.7 - 1.0: Critical, immediate maintenance required

### Module 5 - Streamlit Dashboard
Live web dashboard with three tabs:
- Live Monitor: Real time gauges, anomaly alerts, scrolling graphs
- Load Forecasting: Predicted vs actual graphs, APFC recommendations
- Equipment Health: Degradation gauge, health score, trend graphs

---

## Tech Stack

| Category       | Tools                                   |
|----------------|-----------------------------------------|
| Language       | Python 3.11                             |
| ML Framework   | Scikit-learn, XGBoost, TensorFlow/Keras |
| Dashboard      | Streamlit, Plotly                       |
| Data           | Pandas, NumPy                           |
| Visualization  | Matplotlib, Plotly                      |
| Model Storage  | Joblib                                  |

---

## How to Run

1. Clone the repository:
```
git clone https://github.com/ReubenAbrahamJacob/Ai-Smart-Grid.git
cd Ai-Smart-Grid/smart-grid-ai
```

2. Create virtual environment and install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Generate dataset and train models:
```
python data_generation.py
python anomaly_detection.py
python load_forecasting.py
python predictive_maintenance.py
```

4. Launch dashboard:
```
streamlit run dashboard.py
```

---

## Research Findings

1. XGBoost outperforms ANN and LSTM for short-term load forecasting on tabular smart grid data with explicit feature engineering.
2. The 4-hour rolling average of power factor is the strongest predictor of equipment degradation (91% feature importance).
3. Isolation Forest achieves 95% recall for fault detection — appropriate for power systems where missing a fault is more costly than a false alarm.

---

## Author

**Reuben Abraham Jacob**
