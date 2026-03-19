# AI-Enabled Smart Grid Monitoring System

A complete end-to-end AI platform for real-time smart grid management, featuring anomaly detection, load forecasting, predictive maintenance, and a live web dashboard.

<img width="1512" height="861" alt="Screenshot 2026-03-19 at 12 50 27 PM" src="https://github.com/user-attachments/assets/653a091c-292c-4ac8-8bbb-37e6ac645bf3" />


## Project Overview

Power grids are continuously monitored for faults and abnormal conditions. Traditional monitoring systems raise alarms only when predefined thresholds are crossed — they react to problems but cannot predict them.

This project builds an AI-enabled smart grid monitoring system that goes beyond reactive monitoring. Using machine learning, the system:

- Detects electrical faults and anomalies in real time using unsupervised learning
- Forecasts reactive and active power demand 15 minutes ahead, enabling proactive power factor correction
- Predicts equipment degradation before failure occurs, enabling condition-based maintenance instead of scheduled maintenance
- Presents everything on a live web dashboard with real time gauges, alerts, and trend graphs

The system is currently built on synthetic smart grid data modeled after realistic electrical load patterns. The next phase involves building a real sensor circuit using Arduino with CT and PT transducers, collecting real electrical data, and retraining all models on real data to compare performance against the synthetic baseline. The complete system will then be deployed on an NVIDIA Jetson Orin Nano for standalone edge inference — enabling the system to run independently without a laptop, directly alongside the electrical panel.


<img width="1494" height="863" alt="Screenshot 2026-03-19 at 12 50 37 PM" src="https://github.com/user-attachments/assets/e3c58720-7b84-4aa3-91e8-0890db378126" />



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
<img width="2084" height="1182" alt="anomaly_predictions" src="https://github.com/user-attachments/assets/43688313-64a0-4f7c-b243-2f1f1633efeb" />

### Module 3 - Load Forecasting
Three models benchmarked for short-term reactive and active power forecasting (15 minutes ahead).

| Model    | R² (Reactive) | MAE (Reactive) | R² (Active) | MAE (Active) |
|----------|---------------|----------------|-------------|--------------|
| XGBoost  | 0.997         | 0.191 kVAR     | 1.000       | 0.073 kW     |
| ANN      | 0.987         | 0.706 kVAR     | 0.991       | 1.260 kW     |
| LSTM     | 0.847         | 2.506 kVAR     | 0.958       | 2.694 kW     |

Key finding: XGBoost outperforms ANN and LSTM on tabular smart grid data with explicit feature engineering, consistent with existing literature on structured time series forecasting.

<img width="2084" height="1476" alt="forecasting_results" src="https://github.com/user-attachments/assets/0d10cbbd-7cbe-416f-a10d-e9f02f65a9e3" />


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

<img width="2080" height="1476" alt="degradation_overview" src="https://github.com/user-attachments/assets/904eaa66-7565-4e0a-93ef-545f0a522568" />


### Module 5 - Streamlit Dashboard
Live web dashboard with three tabs:
- Live Monitor: Real time gauges, anomaly alerts, scrolling graphs
- Load Forecasting: Predicted vs actual graphs, APFC recommendations
- Equipment Health: Degradation gauge, health score, trend graphs

<img width="1499" height="859" alt="Screenshot 2026-03-19 at 12 50 05 PM" src="https://github.com/user-attachments/assets/24c899bc-0e12-48ac-8005-e8a27db7f8e6" />
<img width="1502" height="859" alt="Screenshot 2026-03-19 at 12 51 01 PM" src="https://github.com/user-attachments/assets/ac1f995d-b517-403b-9b5e-3983f13e3f3a" />
<img width="1502" height="858" alt="Screenshot 2026-03-19 at 12 51 24 PM" src="https://github.com/user-attachments/assets/0e77017a-443e-4835-9df5-1aac5a7eb2f1" />
<img width="1504" height="835" alt="Screenshot 2026-03-19 at 12 51 53 PM" src="https://github.com/user-attachments/assets/fbedca86-dabd-4e28-bcbd-d3d1e4edab1f" />
<img width="1504" height="835" alt="Screenshot 2026-03-19 at 12 52 04 PM" src="https://github.com/user-attachments/assets/6e465107-8b26-4977-bc01-92cf97d76c66" />

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
