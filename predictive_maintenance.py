"""
predictive_maintenance.py
AI-Enabled Smart Grid Monitoring System
Author: Reuben Abraham Jacob

Module 4 - Predictive Maintenance
Random Forest Regressor predicting equipment degradation level (0 to 1).
Key finding: pf_rolling_mean_16 accounts for 91% of feature importance.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PM_FEATURES = [
    'voltage', 'current', 'active_power', 'reactive_power',
    'power_factor', 'frequency',
    'pf_rolling_mean_4', 'pf_rolling_mean_16', 'pf_rolling_std_16',
    'rp_rolling_mean_4', 'rp_rolling_mean_16', 'rp_rolling_std_16',
    'current_rolling_mean_16', 'current_rolling_std_16',
    'pf_lag_4', 'pf_lag_16', 'rp_lag_4', 'rp_lag_16'
]


def load_data(filepath='smart_grid_data.csv'):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['pf_rolling_mean_4']       = df['power_factor'].rolling(window=4).mean()
    df['pf_rolling_mean_16']      = df['power_factor'].rolling(window=16).mean()
    df['pf_rolling_std_16']       = df['power_factor'].rolling(window=16).std()
    df['rp_rolling_mean_4']       = df['reactive_power'].rolling(window=4).mean()
    df['rp_rolling_mean_16']      = df['reactive_power'].rolling(window=16).mean()
    df['rp_rolling_std_16']       = df['reactive_power'].rolling(window=16).std()
    df['current_rolling_mean_16'] = df['current'].rolling(window=16).mean()
    df['current_rolling_std_16']  = df['current'].rolling(window=16).std()
    df['pf_lag_4']                = df['power_factor'].shift(4)
    df['pf_lag_16']               = df['power_factor'].shift(16)
    df['rp_lag_4']                = df['reactive_power'].shift(4)
    df['rp_lag_16']               = df['reactive_power'].shift(16)

    return df.dropna().reset_index(drop=True)


def split_data(df, ratio=0.80):
    split = int(len(df) * ratio)
    X     = df[PM_FEATURES].values
    y     = df['degradation_level'].values
    return X[:split], X[split:], y[:split], y[split:]


def train_model(X_train_scaled, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    return model


def evaluate(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")


def plot_results(df, predictions, split,
                 save_path='predictive_maintenance_results.png'):
    test_timestamps = df['timestamp'].iloc[split:].reset_index(drop=True)
    actual          = df['degradation_level'].values[split:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Predictive Maintenance - Random Forest Results', fontsize=16)

    axes[0].plot(test_timestamps, actual,
                 color='red', linewidth=1.0, label='Actual')
    axes[0].plot(test_timestamps, predictions,
                 color='blue', linewidth=1.0,
                 label='Predicted', linestyle='--')
    axes[0].axhline(y=0.3, color='orange', linestyle='--',
                    label='Early threshold (0.3)')
    axes[0].axhline(y=0.7, color='red', linestyle='--',
                    label='Critical threshold (0.7)')
    axes[0].set_ylabel('Degradation Level')
    axes[0].set_title('Predicted vs Actual Degradation Level')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:10]
    axes[1].bar(range(10), importances[indices], color='steelblue')
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels(
        [PM_FEATURES[i] for i in indices], rotation=45, ha='right')
    axes[1].set_ylabel('Feature Importance')
    axes[1].set_title('Top 10 Features for Degradation Detection')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as", save_path)


if __name__ == '__main__':
    print("Loading data...")
    df = load_data('smart_grid_data.csv')

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Scaling features...")
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("Training Random Forest...")
    model = train_model(X_train_scaled, y_train)

    print("Evaluating...")
    predictions = np.clip(model.predict(X_test_scaled), 0, 1)
    evaluate(y_test, predictions)

    print("Plotting results...")
    split = int(len(df) * 0.80)
    plot_results(df, predictions, split)

    print("Saving model...")
    joblib.dump(model,  'predictive_maintenance_model.pkl')
    joblib.dump(scaler, 'scaler_pm.pkl')
    print("Model saved.")

    print("\nHealth Status:")
    healthy  = (predictions < 0.3).sum()
    early    = ((predictions >= 0.3) & (predictions < 0.7)).sum()
    critical = (predictions >= 0.7).sum()
    total    = len(predictions)
    print(f"Healthy:           {healthy}  ({round(healthy/total*100, 1)}%)")
    print(f"Early Degradation: {early}    ({round(early/total*100, 1)}%)")
    print(f"Critical:          {critical} ({round(critical/total*100, 1)}%)")