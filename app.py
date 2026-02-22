# =====================================================
# CPU TEMPERATURE PREDICTION – RANDOM FOREST INTERFACE
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="CPU Temperature Predictor", layout="centered")

st.title("🔥 CPU Temperature Prediction System")
st.write("Random Forest–based intelligent server temperature predictor")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("server_data1.csv")

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Feature Engineering
    df['CPU_Workload'] = df['CPU_Utilization_Pct'] * df['Clock_Speed_GHz']
    df['Thermal_Load'] = df['Power_Consumption_W'] / (df['Fan_Speed_RPM'] + 1)
    df['Cooling_Efficiency'] = df['Airflow_CFM'] / (df['Ambient_Temp_C'] + 1)
    df['Power_per_Current'] = (
        df['Power_Consumption_W'] / (df['Current_Load_Amps'] + 1e-6)
    )

    return df

df = load_data()

# ----------------------------
# Prepare Features & Target
# ----------------------------
X = df.drop("CPU_Temperature_C", axis=1)
y = df["CPU_Temperature_C"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Random Forest Model
# ----------------------------
@st.cache_resource
def train_model():
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

st.success("✅ Random Forest model trained successfully")

# ----------------------------
# User Input Section
# ----------------------------
st.subheader("🖥️ Enter Server Parameters")

cpu_util = st.slider("CPU Utilization (%)", 0, 100, 60)
memory_util = st.slider("Memory Usage (%)", 0, 100, 65)
clock_speed = st.number_input("Clock Speed (GHz)", 1.0, 5.0, 3.2)
ambient_temp = st.number_input("Ambient Temperature (°C)", 10.0, 50.0, 28.0)
voltage = st.number_input("Voltage (V)", 0.8, 1.5, 1.25)
current = st.number_input("Current Load (Amps)", 10.0, 300.0, 120.0)
power = st.number_input("Power Consumption (W)", 50.0, 500.0, 150.0)
fan_speed = st.number_input("Fan Speed (RPM)", 1000.0, 10000.0, 5000.0)
airflow = st.number_input("Airflow (CFM)", 50.0, 500.0, 220.0)

# ----------------------------
# Feature Engineering (Input)
# ----------------------------
cpu_workload = cpu_util * clock_speed
thermal_load = power / (fan_speed + 1)
cooling_eff = airflow / (ambient_temp + 1)
power_per_current = power / (current + 1e-6)

input_data = pd.DataFrame([[
    cpu_util,
    memory_util,
    clock_speed,
    ambient_temp,
    voltage,
    current,
    power,
    fan_speed,
    airflow,
    cpu_workload,
    thermal_load,
    cooling_eff,
    power_per_current
]], columns=X.columns)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict CPU Temperature"):
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")
    st.metric(
        label="Predicted CPU Temperature",
        value=f"{prediction:.2f} °C"
    )

    if prediction >= 80:
        st.error("⚠️ Warning: Overheating Risk Detected!")
    else:
        st.success("✅ CPU Temperature within Safe Range")

# =====================================================
# END OF APP
# =====================================================