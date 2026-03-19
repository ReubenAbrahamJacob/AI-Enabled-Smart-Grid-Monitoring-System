"""
anomaly_detection.py
AI-Enabled Smart Grid Monitoring System
Author: Reuben Abraham Jacob

Module 2 - Anomaly Detection
Uses Isolation Forest to detect voltage faults and power factor
degradation in smart grid sensor readings.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


FEATURES = [
    'voltage',
    'current',
    'active_power',
    'reactive_power',
    'power_factor',
    'frequency'
]


def load_data(filepath='smart_grid_data.csv'):
    """Load smart grid dataset from CSV."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def train_anomaly_model(df, contamination=0.02):
    """
    Train Isolation Forest anomaly detection model.

    Parameters:
        df            : smart grid dataframe
        contamination : expected proportion of anomalies (default 2%)

    Returns:
        model  : trained Isolation Forest model
        scaler : fitted StandardScaler
    """
    X        = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_scaled)

    return model, scaler


def predict_anomalies(model, scaler, df):
    """
    Predict anomalies on new data.

    Parameters:
        model  : trained Isolation Forest model
        scaler : fitted StandardScaler
        df     : dataframe with sensor readings

    Returns:
        predictions : array of 0 (normal) or 1 (anomaly)
    """
    X           = df[FEATURES].values
    X_scaled    = scaler.transform(X)
    predictions = model.predict(X_scaled)
    predictions = np.where(predictions == -1, 1, 0)
    return predictions


def evaluate_model(actual, predicted):
    """Print classification report and confusion matrix."""
    print("Model Evaluation")
    print("=" * 50)
    print(classification_report(
        actual, predicted, target_names=['Normal', 'Anomaly']))
    print("Confusion Matrix:")
    cm = confusion_matrix(actual, predicted)
    print(cm)
    print()
    tn, fp, fn, tp = cm.ravel()
    print("True Positives (caught faults):  ", tp)
    print("False Positives (false alarms):  ", fp)
    print("False Negatives (missed faults): ", fn)
    print("True Negatives (correct normal): ", tn)


def plot_anomalies(df, predictions, save_path='anomaly_predictions.png'):
    """
    Plot predicted vs actual anomalies over first 7 days.

    Parameters:
        df          : smart grid dataframe
        predictions : array of predicted anomaly labels
        save_path   : path to save the plot
    """
    df = df.copy()
    df['predicted_anomaly'] = predictions
    week = df.iloc[:7*24*4]

    actual_anomalies    = week[week['anomaly'] == 1]
    predicted_anomalies = week[week['predicted_anomaly'] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Isolation Forest - Predicted vs Actual Anomalies', fontsize=16)

    axes[0].plot(week['timestamp'], week['voltage'],
                 color='orange', linewidth=0.8, label='Voltage')
    axes[0].scatter(actual_anomalies['timestamp'], actual_anomalies['voltage'],
                    color='red', s=80, zorder=5, label='Actual Anomaly')
    axes[0].set_title('Actual Anomalies (Ground Truth)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(week['timestamp'], week['voltage'],
                 color='orange', linewidth=0.8, label='Voltage')
    axes[1].scatter(predicted_anomalies['timestamp'], predicted_anomalies['voltage'],
                    color='blue', s=80, zorder=5, label='Predicted Anomaly')
    axes[1].set_title('Predicted Anomalies (Model Output)')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as", save_path)


def save_model(model, scaler,
               model_path='anomaly_model.pkl',
               scaler_path='scaler.pkl'):
    """Save trained model and scaler to disk."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model saved as", model_path)
    print("Scaler saved as", scaler_path)


def load_model(model_path='anomaly_model.pkl',
               scaler_path='scaler.pkl'):
    """Load saved model and scaler from disk."""
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_single_reading(model, scaler, reading):
    """
    Predict anomaly for a single new sensor reading.

    Parameters:
        model   : trained Isolation Forest model
        scaler  : fitted StandardScaler
        reading : dict with keys matching FEATURES

    Returns:
        result : 'ANOMALY' or 'NORMAL'
    """
    x        = np.array([[reading[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)
    pred     = model.predict(x_scaled)
    return 'ANOMALY' if pred[0] == -1 else 'NORMAL'


if __name__ == '__main__':
    print("Loading data...")
    df = load_data('smart_grid_data.csv')

    print("Training Isolation Forest...")
    model, scaler = train_anomaly_model(df)

    print("Predicting anomalies...")
    predictions = predict_anomalies(model, scaler, df)

    print("Evaluating model...")
    evaluate_model(df['anomaly'], predictions)

    print("Plotting results...")
    plot_anomalies(df, predictions)

    print("Saving model...")
    save_model(model, scaler)

    print("Testing on new reading...")
    test_reading = {
        'voltage'       : 261.0,
        'current'       : 48.0,
        'active_power'  : 24.0,
        'reactive_power': 7.5,
        'power_factor'  : 0.680,
        'frequency'     : 49.98
    }
    result = predict_single_reading(model, scaler, test_reading)
    print("Test reading result:", result)
