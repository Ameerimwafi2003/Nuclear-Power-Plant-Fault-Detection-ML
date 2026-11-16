
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Nuclear Power Plant Fault Detection",
    page_icon="⚛️",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD MODEL AND DATA
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(
        BASE_DIR / "models" / "npp_xgboost_model.joblib"
    )

@st.cache_data
def load_data():
    return pd.read_csv(
        BASE_DIR / "prepared_data" / "npp_prepared.csv"
    )

model = load_model()
data = load_data()

with open(
    BASE_DIR / "models" / "feature_names.json",
    "r",
    encoding="utf-8"
) as file:
    feature_names = json.load(file)

# --------------------------------------------------
# 3. DASHBOARD TITLE
# --------------------------------------------------

st.title("⚛️ Nuclear Power Plant Fault Detection")

st.write(
    "An interactive machine learning system for detecting "
    "abnormal operating conditions using simulated "
    "nuclear power plant sensor measurements."
)

st.info(
    "Educational research prototype only. "
    "Not validated for real nuclear power plant operation."
)

st.divider()

# --------------------------------------------------
# 4. SCENARIO SELECTION
# --------------------------------------------------

st.sidebar.header("Simulation Controls")

scenario_names = sorted(data["Scenario"].unique())

selected_scenario = st.sidebar.selectbox(
    "Select Nuclear Power Plant Scenario",
    scenario_names
)

scenario_data = data[
    data["Scenario"] == selected_scenario
].sort_values("TIME")

selected_time = st.sidebar.select_slider(
    "Select Simulation Time",
    options=scenario_data["TIME"].tolist()
)

selected_record = scenario_data[
    scenario_data["TIME"] == selected_time
].iloc[0]

# --------------------------------------------------
# 5. DISPLAY SENSOR MEASUREMENTS
# --------------------------------------------------

st.subheader("Current Sensor Measurements")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Pressure (P)",
    f"{selected_record['P']:.2f}"
)

col2.metric(
    "Average Temperature (TAVG)",
    f"{selected_record['TAVG']:.2f}"
)

col3.metric(
    "Feedwater Flow (WFWA)",
    f"{selected_record['WFWA']:.2f}"
)

col4.metric(
    "Simulation Time",
    int(selected_time)
)

st.divider()

# --------------------------------------------------
# 6. MODEL PREDICTION
# --------------------------------------------------

st.subheader("Machine Learning Prediction")

input_data = pd.DataFrame(
    [selected_record[feature_names].to_dict()],
    columns=feature_names
)

prediction = int(model.predict(input_data)[0])

fault_score = float(
    model.predict_proba(input_data)[0, 1]
)

predicted_condition = (
    "FAULT DETECTED"
    if prediction == 1
    else "NORMAL OPERATION"
)

if prediction == 1:
    st.error(f"⚠️ Prediction: {predicted_condition}")
else:
    st.success(f"✅ Prediction: {predicted_condition}")

st.metric(
    "Model Fault Score",
    f"{fault_score * 100:.2f}%"
)

# --------------------------------------------------
# 7. COMPARE WITH SCENARIO LABEL
# --------------------------------------------------

st.subheader("Prediction Verification")

actual_label = int(selected_record["Label"])

actual_condition = (
    "Fault Scenario"
    if actual_label == 1
    else "Normal Operation"
)

st.write("**Selected Scenario:**", selected_scenario)
st.write("**Dataset Label:**", actual_condition)
st.write("**Model Prediction:**", predicted_condition)

if prediction == actual_label:
    st.success("Prediction matches the dataset label.")
else:
    st.warning(
        "Prediction differs from the dataset label. "
        "This example can be used to investigate "
        "model limitations."
    )

# --------------------------------------------------
# 8. DISPLAY ALL SENSOR DATA
# --------------------------------------------------

with st.expander("View All 96 Sensor Measurements"):
    sensor_table = pd.DataFrame({
        "Sensor": feature_names,
        "Value": [
            selected_record[name]
            for name in feature_names
        ]
    })

    st.dataframe(
        sensor_table,
        use_container_width=True
    )

# --------------------------------------------------
# 9. ADDITIONAL INFORMATION
# --------------------------------------------------

st.divider()

st.caption(
    "Model: XGBoost | "
    "Dataset: NPPmalfunctions (PCTRAN Simulation) | "
    "Purpose: Educational Machine Learning Research"
)


# --------------------------------------------------
# 10. STRESS TESTING AND MODEL LIMITATIONS
# --------------------------------------------------

st.divider()

st.subheader("Model Stress Testing")

st.write(
    "The following malfunction scenarios were excluded "
    "from model training because their recorded sensor "
    "measurements matched normal operation."
)

stress_scenarios = {
    5: "Spark Presence for Hydrogen Burn",
    8: "Anticipated Transient without Scram",
    14: "Moderator Dilution"
}

stress_results = []

for scenario_id, scenario_name in stress_scenarios.items():

    scenario_records = data[
        data["Scenario_ID"] == scenario_id
    ].sort_values("TIME")

    # Use a representative middle record
    record = scenario_records.iloc[
        len(scenario_records) // 2
    ]

    sample = pd.DataFrame(
        [record[feature_names].to_dict()],
        columns=feature_names
    )

    predicted_label = int(model.predict(sample)[0])

    fault_probability = float(
        model.predict_proba(sample)[0, 1]
    )

    stress_results.append({
        "Scenario": scenario_name,
        "Dataset Label": "Fault",
        "Model Prediction": (
            "Fault" if predicted_label == 1 else "Normal"
        ),
        "Fault Score (%)": round(
            fault_probability * 100, 2
        ),
        "Match": (
            "Yes" if predicted_label == 1 else "No"
        )
    })

stress_df = pd.DataFrame(stress_results)

st.dataframe(
    stress_df,
    use_container_width=True
)

st.warning(
    "These scenarios contain measurements that are "
    "indistinguishable from normal operation in the "
    "available dataset. Predictions should not be "
    "interpreted as evidence of real-world nuclear safety."
)
