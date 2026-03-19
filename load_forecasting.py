"""
load_forecasting.py
AI-Enabled Smart Grid Monitoring System
Author: Reuben Abraham Jacob

Module 3 - Load Forecasting
XGBoost, ANN and LSTM benchmarked for short term power forecasting.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping


FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'quarter_hour',
    'voltage', 'current', 'active_power', 'power_factor', 'frequency',
    'lag_1', 'lag_4', 'lag_8', 'lag_96', 'rolling_mean_4', 'rolling_mean_96'
]


def load_data(filepath='smart_grid_data.csv'):
    df = pd.read_csv(filepath)
    df['timestamp']       = pd.to_datetime(df['timestamp'])
    df['hour']            = df['timestamp'].dt.hour
    df['day_of_week']     = df['timestamp'].dt.dayofweek
    df['month']           = df['timestamp'].dt.month
    df['is_weekend']      = (df['day_of_week'] >= 5).astype(int)
    df['quarter_hour']    = df['timestamp'].dt.minute // 15
    df['lag_1']           = df['reactive_power'].shift(1)
    df['lag_4']           = df['reactive_power'].shift(4)
    df['lag_8']           = df['reactive_power'].shift(8)
    df['lag_96']          = df['reactive_power'].shift(96)
    df['rolling_mean_4']  = df['reactive_power'].rolling(window=4).mean()
    df['rolling_mean_96'] = df['reactive_power'].rolling(window=96).mean()
    return df.dropna().reset_index(drop=True)


def split_data(df, ratio=0.80):
    split      = int(len(df) * ratio)
    X          = df[FEATURE_COLS]
    y_reactive = df['reactive_power']
    y_active   = df['active_power']
    return (X.iloc[:split], X.iloc[split:],
            y_reactive.iloc[:split], y_reactive.iloc[split:],
            y_active.iloc[:split], y_active.iloc[split:])


def evaluate(actual, predicted, label):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    print(f"{label} -> MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}")


def train_xgboost(X_train, y_reactive_train, y_active_train):
    xgb_r = XGBRegressor(n_estimators=200, learning_rate=0.05,
                          max_depth=6, random_state=42)
    xgb_a = XGBRegressor(n_estimators=200, learning_rate=0.05,
                          max_depth=6, random_state=42)
    xgb_r.fit(X_train, y_reactive_train)
    xgb_a.fit(X_train, y_active_train)
    return xgb_r, xgb_a


def build_ann(input_dim=16):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_ann(X_train_scaled, y_r_scaled, y_a_scaled):
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    ann_r = build_ann()
    ann_r.fit(X_train_scaled, y_r_scaled, epochs=100, batch_size=32,
              validation_split=0.1, callbacks=[early_stop], verbose=0)
    ann_a = build_ann()
    ann_a.fit(X_train_scaled, y_a_scaled, epochs=100, batch_size=32,
              validation_split=0.1, callbacks=[early_stop], verbose=0)
    return ann_r, ann_a


def create_sequences(X, y, lookback=96):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_lstm(lookback=96, n_features=16):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm(X_scaled, y_r_scaled, y_a_scaled, lookback=96):
    X_seq, y_r_seq = create_sequences(X_scaled, y_r_scaled, lookback)
    _,     y_a_seq = create_sequences(X_scaled, y_a_scaled, lookback)
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    lstm_r = build_lstm(lookback, X_scaled.shape[1])
    lstm_r.fit(X_seq, y_r_seq, epochs=100, batch_size=32,
               validation_split=0.1, callbacks=[early_stop], verbose=0)
    lstm_a = build_lstm(lookback, X_scaled.shape[1])
    lstm_a.fit(X_seq, y_a_seq, epochs=100, batch_size=32,
               validation_split=0.1, callbacks=[early_stop], verbose=0)
    return lstm_r, lstm_a


if __name__ == '__main__':
    df = load_data('smart_grid_data.csv')
    X_train, X_test, y_r_train, y_r_test, y_a_train, y_a_test = split_data(df)

    # XGBOOST
    print("Training XGBoost...")
    xgb_r, xgb_a = train_xgboost(X_train, y_r_train, y_a_train)
    evaluate(y_r_test, xgb_r.predict(X_test), 'XGBoost Reactive')
    evaluate(y_a_test, xgb_a.predict(X_test), 'XGBoost Active')
    joblib.dump(xgb_r, 'xgb_reactive_model.pkl')
    joblib.dump(xgb_a, 'xgb_active_model.pkl')

    # ANN
    print("Training ANN...")
    scaler_X = StandardScaler()
    scaler_r = StandardScaler()
    scaler_a = StandardScaler()
    X_tr_sc  = scaler_X.fit_transform(X_train)
    X_te_sc  = scaler_X.transform(X_test)
    y_r_sc   = scaler_r.fit_transform(y_r_train.values.reshape(-1,1))
    y_a_sc   = scaler_a.fit_transform(y_a_train.values.reshape(-1,1))
    ann_r, ann_a = train_ann(X_tr_sc, y_r_sc, y_a_sc)
    pred_ann_r = scaler_r.inverse_transform(ann_r.predict(X_te_sc)).flatten()
    pred_ann_a = scaler_a.inverse_transform(ann_a.predict(X_te_sc)).flatten()
    evaluate(y_r_test, pred_ann_r, 'ANN Reactive')
    evaluate(y_a_test, pred_ann_a, 'ANN Active')
    ann_r.save('ann_reactive_model.keras')
    ann_a.save('ann_active_model.keras')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_r, 'scaler_y_reactive.pkl')
    joblib.dump(scaler_a, 'scaler_y_active.pkl')

    # LSTM
    print("Training LSTM...")
    lstm_sc_X = StandardScaler()
    lstm_sc_r = StandardScaler()
    lstm_sc_a = StandardScaler()
    X_all_sc  = lstm_sc_X.fit_transform(df[FEATURE_COLS].values)
    y_r_all   = lstm_sc_r.fit_transform(
                df['reactive_power'].values.reshape(-1,1)).flatten()
    y_a_all   = lstm_sc_a.fit_transform(
                df['active_power'].values.reshape(-1,1)).flatten()
    lstm_r, lstm_a = train_lstm(X_all_sc, y_r_all, y_a_all)
    lstm_r.save('lstm_reactive_model.keras')
    lstm_a.save('lstm_active_model.keras')

    print("\nAll models trained and saved.")