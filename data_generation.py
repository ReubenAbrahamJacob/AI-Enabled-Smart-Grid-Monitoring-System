"""
data_generation.py
AI-Enabled Smart Grid Monitoring System
Author: Reuben Abraham Jacob

Module 1 - Synthetic Smart Grid Data Generation
Simulates 30 days of electrical sensor readings at 15-minute intervals
including realistic load patterns and injected fault anomalies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_smart_grid_data(days=30, freq='15min', seed=42):
    """
    Generate synthetic smart grid sensor data.

    Parameters:
        days : number of days to simulate (default 30)
        freq : sampling frequency (default 15 minutes)
        seed : random seed for reproducibility

    Returns:
        df : pandas DataFrame with smart grid readings
    """
    np.random.seed(seed)

    # TIMESTAMPS
    periods    = days * 24 * 4
    timestamps = pd.date_range(start='2026-01-01', periods=periods, freq=freq)
    hours      = timestamps.hour

    # BASE LOAD PATTERN
    # Realistic day/night cycle using sine waves
    # Low at night (12am-6am), peaks during day (10am-6pm)
    base_load = (
        20 +
        30 * np.sin(np.pi * (hours - 6) / 12) +
        10 * np.sin(np.pi * (hours - 18) / 6)
    )
    base_load = np.clip(base_load, 10, 80)

    # ELECTRICAL MEASUREMENTS
    noise          = np.random.normal(0, 3, periods)
    active_power   = base_load + noise
    power_factor   = 0.75 + 0.20 * np.random.beta(5, 2, periods)
    reactive_power = active_power * np.sqrt(1 - power_factor**2) / power_factor
    voltage        = np.random.normal(230, 2, periods)
    current        = (active_power * 1000) / (voltage * power_factor)
    frequency      = np.random.normal(50, 0.05, periods)

    # INJECT ANOMALIES
    # 2% of readings are faults - voltage spike/drop + power factor degradation
    anomaly         = np.zeros(periods)
    anomaly_indices = np.random.choice(
                      periods, size=int(0.02 * periods), replace=False)
    voltage[anomaly_indices]      += np.random.choice(
                                     [-30, 30], size=len(anomaly_indices))
    power_factor[anomaly_indices] -= 0.2
    anomaly[anomaly_indices]       = 1

    # BUILD DATAFRAME
    df = pd.DataFrame({
        'timestamp'     : timestamps,
        'voltage'       : np.round(voltage, 2),
        'current'       : np.round(current, 2),
        'active_power'  : np.round(active_power, 2),
        'reactive_power': np.round(reactive_power, 2),
        'power_factor'  : np.round(np.clip(power_factor, 0.5, 1.0), 3),
        'frequency'     : np.round(frequency, 3),
        'anomaly'       : anomaly.astype(int)
    })

    return df


def visualize_data(df, save_path='smart_grid_overview.png'):
    """
    Visualize key smart grid parameters over first 7 days.

    Parameters:
        df        : smart grid dataframe
        save_path : path to save the plot
    """
    week      = df.iloc[:7*24*4]
    anomalies = week[week['anomaly'] == 1]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Smart Grid Data - First 7 Days', fontsize=16)

    axes[0].plot(week['timestamp'], week['active_power'],
                 color='blue', linewidth=0.8)
    axes[0].set_ylabel('Active Power (kW)')
    axes[0].set_title('Load Pattern')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(week['timestamp'], week['power_factor'],
                 color='green', linewidth=0.8)
    axes[1].axhline(y=0.9, color='red', linestyle='--',
                    label='Min acceptable PF (0.9)')
    axes[1].set_ylabel('Power Factor')
    axes[1].set_title('Power Factor')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(week['timestamp'], week['voltage'],
                 color='orange', linewidth=0.8)
    axes[2].axhline(y=230, color='red', linestyle='--',
                    label='Nominal 230V')
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_title('Voltage')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(week['timestamp'], week['active_power'],
                 color='blue', linewidth=0.8, label='Normal')
    axes[3].scatter(anomalies['timestamp'], anomalies['active_power'],
                    color='red', s=50, zorder=5, label='Anomaly')
    axes[3].set_ylabel('Active Power (kW)')
    axes[3].set_title('Anomaly Locations')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as", save_path)


if __name__ == '__main__':
    print("Generating smart grid dataset...")
    df = generate_smart_grid_data(days=30)
    df.to_csv('smart_grid_data.csv', index=False)

    print("Dataset saved as smart_grid_data.csv")
    print("Shape:", df.shape)
    print("Date range:", df['timestamp'].min(), "to", df['timestamp'].max())
    print("Anomalies:", df['anomaly'].sum(),
          "(", round(df['anomaly'].mean()*100, 1), "%)")

    print("Visualizing data...")
    visualize_data(df)