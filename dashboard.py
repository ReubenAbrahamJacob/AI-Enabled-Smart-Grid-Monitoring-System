
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Smart Grid Monitor",
    page_icon="⚡",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .alert-red {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    .alert-green {
        background-color: #00cc44;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    .alert-orange {
        background-color: #ff8800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# LOAD DATA AND MODELS
@st.cache_data
def load_data():
    df = pd.read_csv('smart_grid_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_resource
def load_models():
    anomaly_model  = joblib.load('anomaly_model.pkl')
    anomaly_scaler = joblib.load('scaler.pkl')
    xgb_reactive   = joblib.load('xgb_reactive_model.pkl')
    xgb_active     = joblib.load('xgb_active_model.pkl')
    pm_model       = joblib.load('predictive_maintenance_model.pkl')
    pm_scaler      = joblib.load('scaler_pm.pkl')
    return anomaly_model, anomaly_scaler, xgb_reactive, xgb_active, pm_model, pm_scaler


# FEATURE ENGINEERING FOR LOAD FORECASTING
def engineer_features(df):
    df = df.copy()
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


# FEATURE ENGINEERING FOR PREDICTIVE MAINTENANCE
def engineer_pm_features(df):
    df = df.copy()
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


def make_gauge(value, title, min_val, max_val, threshold_low, threshold_high):
    if value < threshold_low:
        color = "red"
    elif value < threshold_high:
        color = "orange"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, threshold_low],    'color': '#ffcccc'},
                {'range': [threshold_low, threshold_high], 'color': '#fff3cc'},
                {'range': [threshold_high, max_val],   'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_low
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig


# MAIN DASHBOARD
def main():
    # HEADER
    st.title("AI-Enabled Smart Grid Monitoring System")
    st.markdown("**Reuben Abraham Jacob**")
    st.markdown("---")

    # LOAD DATA AND MODELS
    df_raw = load_data()
    anomaly_model, anomaly_scaler, xgb_reactive, xgb_active, pm_model, pm_scaler = load_models()

    # ENGINEER FEATURES
    df_feat = engineer_features(df_raw)
    df_pm   = engineer_pm_features(df_raw)

    # SIDEBAR CONTROLS
    st.sidebar.title("Simulation Controls")
    st.sidebar.markdown("---")

    # Initialize session state
    if 'live_idx' not in st.session_state:
        st.session_state.live_idx = 500

    auto_refresh = st.sidebar.checkbox("Auto Refresh (Live Mode)", value=False)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 3)

    if auto_refresh:
        st.session_state.live_idx += 1
        if st.session_state.live_idx >= len(df_feat) - 100:
            st.session_state.live_idx = 100
        start_idx = st.session_state.live_idx
    else:
        start_idx = st.sidebar.slider(
            "Start Time Index",
            min_value=100,
            max_value=len(df_feat) - 100,
            value=st.session_state.live_idx
        )
        st.session_state.live_idx = start_idx

    window_size = st.sidebar.slider(
        "History Window (readings)",
        min_value=24,
        max_value=200,
        value=96
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Status**")
    st.sidebar.success("Anomaly Detection: Active")
    st.sidebar.success("Load Forecasting: Active")
    st.sidebar.success("Predictive Maintenance: Active")

    # CURRENT READING
    current_raw = df_raw.iloc[start_idx]

    # TABS
    tab1, tab2, tab3 = st.tabs([
        "Live Monitor",
        "Load Forecasting",
        "Equipment Health"
    ])

    # TAB 1 - LIVE MONITOR
    with tab1:
        st.subheader("Real Time Grid Parameters")

        # ANOMALY DETECTION
        ANOMALY_FEATURES = ['voltage', 'current', 'active_power',
                            'reactive_power', 'power_factor', 'frequency']
        x_anomaly    = np.array([[current_raw[f] for f in ANOMALY_FEATURES]])
        x_scaled     = anomaly_scaler.transform(x_anomaly)
        anomaly_pred = anomaly_model.predict(x_scaled)[0]
        is_anomaly   = anomaly_pred == -1

        if is_anomaly:
            st.markdown(
                '<div class="alert-red">ANOMALY DETECTED - Check Grid Immediately</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="alert-green">System Normal - All Parameters Within Range</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # GAUGES
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                make_gauge(current_raw['voltage'], 'Voltage (V)', 180, 260, 207, 240),
                use_container_width=True)
        with col2:
            st.plotly_chart(
                make_gauge(current_raw['power_factor'], 'Power Factor', 0.5, 1.0, 0.75, 0.9),
                use_container_width=True)
        with col3:
            st.plotly_chart(
                make_gauge(current_raw['active_power'], 'Active Power (kW)', 0, 80, 20, 60),
                use_container_width=True)
        with col4:
            st.plotly_chart(
                make_gauge(current_raw['reactive_power'], 'Reactive Power (kVAR)', 0, 40, 10, 25),
                use_container_width=True)

        st.markdown("---")

        # LIVE GRAPHS
        history = df_raw.iloc[start_idx - window_size:start_idx]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(history, x='timestamp', y='voltage',
                         title='Voltage (V)',
                         color_discrete_sequence=['orange'])
            fig.add_hline(y=230, line_dash="dash", line_color="red",
                         annotation_text="Nominal 230V")
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(history, x='timestamp', y='power_factor',
                         title='Power Factor',
                         color_discrete_sequence=['green'])
            fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                         annotation_text="Min 0.9")
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.line(history, x='timestamp', y='active_power',
                         title='Active Power (kW)',
                         color_discrete_sequence=['blue'])
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.line(history, x='timestamp', y='reactive_power',
                         title='Reactive Power (kVAR)',
                         color_discrete_sequence=['purple'])
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)

        # CURRENT METRICS
        st.markdown("---")
        st.subheader("Current Readings")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Voltage",         f"{current_raw['voltage']:.1f} V")
        col2.metric("Current",         f"{current_raw['current']:.1f} A")
        col3.metric("Active Power",    f"{current_raw['active_power']:.1f} kW")
        col4.metric("Reactive Power",  f"{current_raw['reactive_power']:.1f} kVAR")
        col5.metric("Power Factor",    f"{current_raw['power_factor']:.3f}")
        col6.metric("Frequency",       f"{current_raw['frequency']:.2f} Hz")

    # TAB 2 - LOAD FORECASTING
    with tab2:
        st.subheader("Load Forecasting - Next 15 Minutes")

        FORECAST_FEATURES = ['hour', 'day_of_week', 'month', 'is_weekend',
                             'quarter_hour', 'voltage', 'current', 'active_power',
                             'power_factor', 'frequency', 'lag_1', 'lag_4',
                             'lag_8', 'lag_96', 'rolling_mean_4', 'rolling_mean_96']

        feat_idx   = min(start_idx, len(df_feat) - 1)
        x_forecast = df_feat[FORECAST_FEATURES].iloc[feat_idx].values.reshape(1, -1)

        pred_reactive = xgb_reactive.predict(x_forecast)[0]
        pred_active   = xgb_active.predict(x_forecast)[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Reactive Power",
                    f"{current_raw['reactive_power']:.2f} kVAR")
        col2.metric("Predicted Reactive Power",
                    f"{pred_reactive:.2f} kVAR",
                    delta=f"{pred_reactive - current_raw['reactive_power']:.2f} kVAR")
        col3.metric("Current Active Power",
                    f"{current_raw['active_power']:.2f} kW")
        col4.metric("Predicted Active Power",
                    f"{pred_active:.2f} kW",
                    delta=f"{pred_active - current_raw['active_power']:.2f} kW")

        st.markdown("---")

        if pred_reactive > 15:
            st.warning("APFC Recommendation: Switch in Capacitor Banks 1 and 2 - High reactive power predicted")
        elif pred_reactive > 8:
            st.info("APFC Recommendation: Switch in Capacitor Bank 1 - Moderate reactive power predicted")
        else:
            st.success("APFC Recommendation: No capacitor switching needed - Low reactive power predicted")

        st.markdown("---")

        history_feat          = df_feat.iloc[feat_idx - window_size:feat_idx]
        forecast_preds_reactive = xgb_reactive.predict(history_feat[FORECAST_FEATURES].values)
        forecast_preds_active   = xgb_active.predict(history_feat[FORECAST_FEATURES].values)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_feat['timestamp'],
                y=history_feat['reactive_power'],
                name='Actual',
                line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(
                x=history_feat['timestamp'],
                y=forecast_preds_reactive,
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')))
            fig.update_layout(
                title='Reactive Power - Actual vs Predicted (kVAR)',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_feat['timestamp'],
                y=history_feat['active_power'],
                name='Actual',
                line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(
                x=history_feat['timestamp'],
                y=forecast_preds_active,
                name='Predicted',
                line=dict(color='orange', width=2, dash='dash')))
            fig.update_layout(
                title='Active Power - Actual vs Predicted (kW)',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3 - EQUIPMENT HEALTH
    with tab3:
        st.subheader("Predictive Maintenance - Equipment Health Monitor")

        PM_FEATURES = ['voltage', 'current', 'active_power', 'reactive_power',
                      'power_factor', 'frequency', 'pf_rolling_mean_4',
                      'pf_rolling_mean_16', 'pf_rolling_std_16',
                      'rp_rolling_mean_4', 'rp_rolling_mean_16',
                      'rp_rolling_std_16', 'current_rolling_mean_16',
                      'current_rolling_std_16', 'pf_lag_4', 'pf_lag_16',
                      'rp_lag_4', 'rp_lag_16']

        pm_idx      = min(start_idx, len(df_pm) - 1)
        x_pm        = df_pm[PM_FEATURES].iloc[pm_idx].values.reshape(1, -1)
        x_pm_scaled = pm_scaler.transform(x_pm)
        deg_level   = float(np.clip(pm_model.predict(x_pm_scaled)[0], 0, 1))

        if deg_level < 0.3:
            health_status = "HEALTHY"
            health_color  = "alert-green"
            health_msg    = "Equipment operating normally. No action required."
        elif deg_level < 0.7:
            health_status = "EARLY DEGRADATION"
            health_color  = "alert-orange"
            health_msg    = "Equipment showing signs of wear. Schedule inspection soon."
        else:
            health_status = "CRITICAL"
            health_color  = "alert-red"
            health_msg    = "Immediate maintenance required. Risk of failure."

        st.markdown(
            f'<div class="{health_color}">{health_status} - {health_msg}</div>',
            unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=deg_level,
                title={'text': "Equipment Degradation Level", 'font': {'size': 16}},
                delta={'reference': 0.3},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "red" if deg_level > 0.7 else
                                     "orange" if deg_level > 0.3 else "green"},
                    'steps': [
                        {'range': [0, 0.3],   'color': '#ccffcc'},
                        {'range': [0.3, 0.7], 'color': '#fff3cc'},
                        {'range': [0.7, 1.0], 'color': '#ffcccc'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.7
                    }
                }
            ))
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Degradation Level", f"{deg_level:.3f}")
            st.metric("Health Score",      f"{(1-deg_level)*100:.1f} / 100")
            st.metric("Power Factor Trend",
                      f"{df_pm['pf_rolling_mean_16'].iloc[pm_idx]:.3f}")
            st.metric("Reactive Power Trend",
                      f"{df_pm['rp_rolling_mean_16'].iloc[pm_idx]:.2f} kVAR")

        st.markdown("---")

        pm_history  = df_pm.iloc[max(0, pm_idx - window_size):pm_idx]
        deg_history = np.clip(
            pm_model.predict(pm_scaler.transform(
                pm_history[PM_FEATURES].values)), 0, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pm_history['timestamp'],
            y=deg_history,
            name='Degradation Level',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                     annotation_text="Early Warning (0.3)")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                     annotation_text="Critical (0.7)")
        fig.update_layout(
            title='Equipment Degradation Trend',
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    # AUTO REFRESH
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == '__main__':
    main()