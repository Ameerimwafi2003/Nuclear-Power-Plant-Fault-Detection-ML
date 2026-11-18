"""NPP Sentinel AI — light-theme educational PCTRAN simulation dashboard.

Run from the project directory with:
    python -m streamlit run app.py

Requires the advanced model artifacts saved by 03_Advanced_Model_Experiments.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "prepared_data"
RESULTS_DIR = BASE_DIR / "results"
PAGE_BG = "#F7F9FC"
PANEL = "#FFFFFF"
TEAL = "#087F8C"
AMBER = "#D97732"
RED = "#D64550"
TEXT = "#172B4D"
MUTED = "#66758A"
GRID = "#E6ECF2"

st.set_page_config(
    page_title="NPP Sentinel AI | Simulation Control Room",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    .stApp, [data-testid="stAppViewContainer"] {background:#F7F9FC;color:#172B4D;}
    [data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E3EAF1;}
    [data-testid="stHeader"] {background:rgba(247,249,252,.94);}
    .block-container {max-width:1550px;padding-top:1.35rem;padding-bottom:3rem;}
    h1,h2,h3,h4 {letter-spacing:-.025em;color:#172B4D !important;}
    p,li {color:#45566E;}
    [data-testid="stMetric"] {background:#FFFFFF;border:1px solid #E1E8EF;
        border-radius:15px;padding:16px 18px;box-shadow:0 3px 14px rgba(23,43,77,.035);}
    [data-testid="stMetricLabel"] {color:#64748B;}
    [data-testid="stMetricValue"] {color:#172B4D;}
    div[data-testid="stAlert"] {border-radius:12px;}
    div[data-testid="stTabs"] button {font-weight:600;}
    .hero {border:1px solid #D9E6E8;border-radius:20px;padding:22px 26px;
           background:linear-gradient(110deg,#FFFFFF 12%,#E9F7F6 100%);margin-bottom:14px;}
    .eyebrow {color:#087F8C;font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;}
    .hero-title {font-size:2.15rem;letter-spacing:-.045em;font-weight:700;color:#172B4D;margin:6px 0;}
    .subtle {color:#64748B;font-size:.93rem;}
    .status-good {color:#14805D;font-weight:700;}
    .status-bad {color:#C83D49;font-weight:700;}
    .pill {display:inline-block;padding:5px 11px;border:1px solid #BFDEDB;
           border-radius:100px;color:#087F8C;font-size:12px;margin-top:12px;}
    .small-note {color:#64748B;font-size:.84rem;}
    hr {border-color:#E2E8F0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Apply one consistent theme to Plotly charts, regardless of the Streamlit UI theme.
# All charts use white paper and plotting areas, dark text and light gridlines.
LIGHT_CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(family="DM Sans, Arial, sans-serif", color=TEXT, size=13),
    title_font=dict(color=TEXT),
    legend=dict(font=dict(color=TEXT), bgcolor="rgba(255,255,255,0)"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    margin=dict(l=14, r=14, t=28, b=14),
)


def style_chart(fig, height=360):
    fig.update_layout(**LIGHT_CHART_LAYOUT)
    fig.update_layout(height=height, title_text="")
    return fig


# -----------------------------------------------------------------------------
# Artifact loading: use the newer revised dataset for normal exploration.
# The original dataset is only needed for the three excluded stress scenarios.
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_joblib(path: str):
    return joblib.load(path)


if nn is not None:
    class NuclearAutoencoder(nn.Module):
        """Same architecture used in the advanced experiment notebook."""

        def __init__(self, input_dim: int = 96):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16),
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))
else:
    NuclearAutoencoder = None


@st.cache_resource(show_spinner=False)
def load_autoencoder(weights_path: str, input_dim: int):
    if torch is None or NuclearAutoencoder is None:
        raise RuntimeError("PyTorch is not installed. Install torch to use Autoencoder predictions.")
    net = NuclearAutoencoder(input_dim=input_dim)
    # Load only your own trusted model files; never accept uploaded weights from strangers.
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net


REQUIRED = {
    "Revised simulation dataset": DATA_DIR / "npp_revised.csv",
    "Advanced Random Forest": MODEL_DIR / "advanced_random_forest.pkl",
    "Advanced XGBoost": MODEL_DIR / "advanced_xgboost.pkl",
    "Advanced Autoencoder": MODEL_DIR / "advanced_autoencoder.pth",
    "Autoencoder scaler": MODEL_DIR / "advanced_autoencoder_robust_scaler.pkl",
    "Feature names": MODEL_DIR / "advanced_feature_names.pkl",
    "Autoencoder configuration": MODEL_DIR / "advanced_autoencoder_metadata.json",
}
missing = [f"{name}: {path.relative_to(BASE_DIR)}" for name, path in REQUIRED.items() if not path.exists()]
if missing:
    st.error("Some advanced experiment files are missing. Please place this app.py in your project root.")
    st.code("\n".join(missing))
    st.stop()

try:
    df = load_csv(str(REQUIRED["Revised simulation dataset"]))
    rf = load_joblib(str(REQUIRED["Advanced Random Forest"]))
    xgb = load_joblib(str(REQUIRED["Advanced XGBoost"]))
    scaler = load_joblib(str(REQUIRED["Autoencoder scaler"]))
    features = list(load_joblib(str(REQUIRED["Feature names"])))
    ae_config = load_json(str(REQUIRED["Autoencoder configuration"]))
    ae = load_autoencoder(str(REQUIRED["Advanced Autoencoder"]), len(features))
except Exception as exc:
    st.error("Could not load one of the trained artifacts. Ensure you are using files from the same notebook run.")
    st.exception(exc)
    st.stop()

if any(col not in df.columns for col in ["TIME", "Scenario", "Scenario_ID", "Label", *features]):
    st.error("The revised CSV columns do not match the trained model feature list.")
    st.stop()

# Experimental pipeline intentionally fits the log/robust transformation on training data.
# Reproduce its order EXACTLY at inference time.
clip_limit = float(ae_config["clip_limit"])
ae_threshold = float(ae_config["threshold"])


def signed_log_transform(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def ae_errors(feature_frame: pd.DataFrame) -> np.ndarray:
    values = feature_frame[features].to_numpy(dtype=np.float64)
    transformed = signed_log_transform(values)
    transformed = scaler.transform(transformed)
    transformed = np.clip(transformed, -clip_limit, clip_limit).astype(np.float32)
    with torch.no_grad():
        reconstructed = ae(torch.from_numpy(transformed)).numpy()
    return np.mean((transformed - reconstructed) ** 2, axis=1)


def predict_batch(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[features]
    fault_score = xgb.predict_proba(x)[:, 1]
    anomalies = ae_errors(x)
    return pd.DataFrame({
        "RF": rf.predict(x).astype(int),
        "XGBoost": xgb.predict(x).astype(int),
        "Autoencoder": (anomalies > ae_threshold).astype(int),
        "XGB_Score": fault_score,
        "AE_Error": anomalies,
    }, index=frame.index)


@st.cache_data(show_spinner=False)
def load_comparison(path: str):
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Layout and scenario controls
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI • Industrial Simulation • Research Prototype</div>
      <div class="hero-title">⚛️ NPP SENTINEL AI</div>
      <div class="subtle">Fault detection, anomaly analysis, and sensor intelligence for simulated nuclear plant scenarios.</div>
      <div class="pill">● OFFLINE SIMULATION DATA &nbsp;·&nbsp; NOT A REAL PLANT MONITOR</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Educational ML research prototype. Not intended or validated for actual nuclear plant operations or safety decisions.")

scenarios = (
    df[["Scenario_ID", "Scenario"]]
    .drop_duplicates()
    .sort_values("Scenario_ID")
    .reset_index(drop=True)
)
scenario_lookup = {int(row.Scenario_ID): str(row.Scenario) for row in scenarios.itertuples()}

# Two dashboard modes: manual exploration and a scripted educational walkthrough.
GUIDED_STAGES = [
    {
        "name": "1 · Normal Operation", "scenario": 0, "record": 227,
        "model": "XGBoost", "sensors": ["P", "TAVG"],
        "goal": "Establish what recorded normal operation looks like.",
        "talk": "We start with recorded normal-operation measurements. The three models analyze the same sensor record. We compare their predictions with its known dataset label.",
    },
    {
        "name": "2 · Loss of Coolant Accident", "scenario": 1, "record": 300,
        "model": "XGBoost", "sensors": ["P", "DNBR"],
        "goal": "Observe a simulated fault and the corresponding AI response.",
        "talk": "This is a held-out hot-leg loss-of-coolant accident. The sensor curves and XGBoost scores replay recorded data; their patterns can be compared with the normal scenario. Different sensors have different units.",
    },
    {
        "name": "3 · Compare Three Models", "scenario": 16, "record": 300,
        "model": "Random Forest", "sensors": ["DNBR", "WSTB"],
        "goal": "Show that model performance varies across fault types.",
        "talk": "In the held-out containment failure / hydrogen-detonation scenario, Random Forest correctly classified only about 40.75% of the records, while XGBoost and Autoencoder classified all records in that test split. We can inspect model disagreements across time.",
    },
    {
        "name": "4 · Explain Anomaly Detection", "scenario": 18, "record": 400,
        "model": "Autoencoder", "sensors": ["DWB", "DTHY"],
        "goal": "Explain how the Autoencoder detects unusual sensor patterns.",
        "talk": "The Autoencoder reconstructs transformed sensor measurements learned from the normal training run. It flags a record when reconstruction MSE exceeds the fixed training-derived threshold. Its performance is sensitive to several sensors.",
    },
    {
        "name": "5 · Research Findings & Limitations", "scenario": 0, "record": 227,
        "model": "XGBoost", "sensors": ["P"],
        "goal": "Summarize the held-out tests and be transparent about limitations.",
        "talk": "XGBoost and Autoencoder scored 100% balanced accuracy on the selected held-out fault test split, but there is only one normal operating simulation, and individual records within scenarios are correlated. This does not establish real-world safety.",
    },
]

if "guided_stage" not in st.session_state:
    st.session_state.guided_stage = 0


def apply_demo_stage(stage_num):
    """Update sidebar widget state before the widgets are created in a rerun."""
    stage_num = max(0, min(stage_num, len(GUIDED_STAGES) - 1))
    st.session_state.guided_stage = stage_num
    stage = GUIDED_STAGES[stage_num]
    if stage["scenario"] in scenario_lookup:
        st.session_state.scenario_select = stage["scenario"]
    st.session_state.model_select = stage["model"]
    st.session_state.demo_record_requested = stage["record"]
    st.session_state.trend_select = [x for x in stage["sensors"] if x in features]


def reset_record_on_scenario_change():
    st.session_state.simulation_record = 0


st.sidebar.markdown("## 🎛️ Simulation Controls")
def on_mode_change():
    if st.session_state.dashboard_mode == "Guided Demo Mode":
        apply_demo_stage(0)


st.sidebar.radio(
    "Dashboard mode",
    ["Explore Mode", "Guided Demo Mode"],
    key="dashboard_mode",
    on_change=on_mode_change,
    help="Explore freely, or follow a five-stage demonstration for your professor.",
)

if st.session_state.dashboard_mode == "Guided Demo Mode":
    st.sidebar.caption(f"Demo stage {st.session_state.guided_stage + 1} of {len(GUIDED_STAGES)}")
    st.sidebar.button("↺ Load current demo stage", use_container_width=True,
                      on_click=apply_demo_stage, args=(st.session_state.guided_stage,))

if "scenario_select" not in st.session_state:
    st.session_state.scenario_select = int(next(iter(scenario_lookup)))
if st.session_state.scenario_select not in scenario_lookup:
    st.session_state.scenario_select = int(next(iter(scenario_lookup)))

chosen_id = st.sidebar.selectbox(
    "Plant scenario",
    options=list(scenario_lookup),
    format_func=lambda item: scenario_lookup[item],
    key="scenario_select",
    on_change=reset_record_on_scenario_change,
)
selected = df.loc[df["Scenario_ID"] == chosen_id].sort_values("TIME").reset_index(drop=True)
if selected.empty:
    st.error("No measurements exist for this scenario.")
    st.stop()

if "demo_record_requested" in st.session_state:
    st.session_state.simulation_record = min(int(st.session_state.pop("demo_record_requested")), len(selected)-1)
if "simulation_record" not in st.session_state:
    st.session_state.simulation_record = min(120, len(selected) - 1)
if st.session_state.simulation_record >= len(selected):
    st.session_state.simulation_record = len(selected)-1

time_index = st.sidebar.slider(
    "Simulation record",
    min_value=0,
    max_value=len(selected) - 1,
    step=1,
    key="simulation_record",
    help="Move through recorded measurements. TIME is the simulation's recorded time field.",
)
row = selected.iloc[time_index]
selected_time = row["TIME"]
if "model_select" not in st.session_state:
    st.session_state.model_select = "XGBoost"
model_choice = st.sidebar.radio(
    "Primary AI model", ["XGBoost", "Random Forest", "Autoencoder"],
    key="model_select",
)
model_col = {"XGBoost": "XGBoost", "Random Forest": "RF", "Autoencoder": "Autoencoder"}[model_choice]
st.sidebar.divider()
st.sidebar.caption(f"{len(df):,} revised records · {len(features)} sensors · {len(scenario_lookup)} scenarios")
st.sidebar.info("Predictions replay historical simulated measurements; they are not live sensor feeds.")

# Inference for entire selected scenario is cached through Streamlit data cache.
@st.cache_data(show_spinner=False)
def predictions_for_scenario(scenario_id: int, frame: pd.DataFrame):
    return predict_batch(frame)


with st.spinner("Analyzing simulation records..."):
    pred = predictions_for_scenario(chosen_id, selected)
current = pred.iloc[time_index]
actual = int(row["Label"])
chosen_prediction = int(current[model_col])
all_agree = len(set(int(current[x]) for x in ("RF", "XGBoost", "Autoencoder"))) == 1
is_fault = chosen_prediction == 1

st.markdown("### Control Room Overview")
a, b, c, d, e = st.columns(5)
a.metric("AI Status", "FAULT ALERT" if is_fault else "NORMAL")
b.metric("Selected Model", model_choice)
c.metric("Simulation Time", f"{selected_time:g}")
d.metric("Model Agreement", "3 / 3" if all_agree else "Mixed")
e.metric("Recorded Label", "FAULT" if actual else "NORMAL")

if is_fault:
    st.error("⚠️ Selected model detects an abnormal pattern in this simulated record.")
else:
    st.success("✅ Selected model classifies this simulated record as normal.")

if chosen_prediction != actual:
    st.warning("The selected model prediction differs from this record's dataset label.")

# -----------------------------------------------------------------------------
# Guided demo: short, interactive walkthrough using the same predictions and data.
# The existing four-tab dashboard remains available in Explore Mode.
# -----------------------------------------------------------------------------
if st.session_state.dashboard_mode == "Guided Demo Mode":
    stage = GUIDED_STAGES[st.session_state.guided_stage]
    st.divider()
    st.markdown("### 🎓 Guided Project Demonstration")
    st.progress((st.session_state.guided_stage + 1) / len(GUIDED_STAGES))
    st.caption(f"Stage {st.session_state.guided_stage + 1} of {len(GUIDED_STAGES)} · {stage['name']}")
    st.markdown(f"#### {stage['goal']}")
    st.info(stage["talk"])
    st.caption("The scenario/model/record controls remain adjustable. Click 'Load current demo stage' in the sidebar to restore the suggested settings.")
    stage_prev, stage_next = st.columns(2)
    with stage_prev:
        st.button("← Previous stage", disabled=st.session_state.guided_stage == 0,
                  use_container_width=True, on_click=apply_demo_stage,
                  args=(st.session_state.guided_stage - 1,))
    with stage_next:
        st.button("Next stage →" if st.session_state.guided_stage < 4 else "↺ Restart demo",
                  use_container_width=True, type="primary", on_click=apply_demo_stage,
                  args=(st.session_state.guided_stage + 1 if st.session_state.guided_stage < 4 else 0,))

    if st.session_state.guided_stage in (0, 1):
        st.markdown("#### Recorded Sensor Trends")
        selected_sensors = [s for s in st.session_state.get("trend_select", []) if s in features]
        if selected_sensors:
            frame = selected[["TIME", *selected_sensors]].melt("TIME", var_name="Sensor", value_name="Reading")
            fig = px.line(frame, x="TIME", y="Reading", color="Sensor", color_discrete_sequence=[TEAL, AMBER])
            fig.add_vline(x=float(selected_time), line_dash="dash", line_color=MUTED)
            st.plotly_chart(style_chart(fig, 350), use_container_width=True)
            st.caption("The vertical line marks the selected record. Different sensors can use different units.")
        st.markdown("#### XGBoost Fault Score Across This Scenario")
        fig = px.line(x=selected["TIME"], y=pred["XGB_Score"], labels={"x": "Simulation TIME", "y": "Fault score"})
        fig.update_traces(line_color=TEAL)
        fig.add_hline(y=0.5, line_dash="dash", line_color=AMBER)
        fig.add_vline(x=float(selected_time), line_dash="dot", line_color=MUTED)
        st.plotly_chart(style_chart(fig, 280), use_container_width=True)
        st.caption("Dashed horizontal line = XGBoost's 0.5 decision threshold. Values above it produce a fault prediction.")
    elif st.session_state.guided_stage == 2:
        st.markdown("#### Compare Model Predictions")
        names = {"RF": "Random Forest", "XGBoost": "XGBoost", "Autoencoder": "Autoencoder"}
        for col, (key, name) in zip(st.columns(3), names.items()):
            col.metric(name, "FAULT" if int(current[key]) else "NORMAL")
        view = pd.DataFrame({"TIME": selected["TIME"],
                             "Random Forest": pred["RF"],
                             "XGBoost": pred["XGBoost"],
                             "Autoencoder": pred["Autoencoder"]})
        melted = view.melt(id_vars="TIME", var_name="Model", value_name="Fault prediction")
        fig = px.line(melted, x="TIME", y="Fault prediction", color="Model", line_shape="hv",
                      color_discrete_sequence=[AMBER, TEAL, RED])
        fig.update_yaxes(tickvals=[0, 1], ticktext=["Normal", "Fault"])
        fig.add_vline(x=float(selected_time), line_dash="dash", line_color=MUTED)
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
        st.caption("The chart shows recorded-time predictions, not a real-time reactor monitor.")
    elif st.session_state.guided_stage == 3:
        st.markdown("#### Reconstruction Error vs Anomaly Threshold")
        st.metric("Selected reconstruction MSE", f"{float(current['AE_Error']):.6g}")
        st.caption(f"Anomaly threshold = {ae_threshold:.8g}; readings exceeding the threshold are flagged.")
        fig = px.line(x=selected["TIME"], y=pred["AE_Error"],
                      labels={"x": "Simulation TIME", "y": "Reconstruction MSE"})
        fig.update_traces(line_color=AMBER)
        fig.add_hline(y=ae_threshold, line_dash="dash", line_color=RED)
        fig.add_vline(x=float(selected_time), line_dash="dot", line_color=MUTED)
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)
        st.warning("The Autoencoder's scoring is sensitive to selected sensors. Its perfect held-out test score does not establish real-world reliability.")
    else:
        st.markdown("#### Selected Validation and Test Results")
        summary = RESULTS_DIR / "final_advanced_experiment_summary.csv"
        if summary.exists():
            results = load_comparison(str(summary))
            st.dataframe(results, use_container_width=True, hide_index=True)
            chart = results[["Model", "Validation_Balanced_Accuracy", "Test_Balanced_Accuracy"]].copy()
            chart[["Validation_Balanced_Accuracy", "Test_Balanced_Accuracy"]] *= 100
            chart = chart.melt("Model", var_name="Split", value_name="Balanced accuracy (%)")
            fig = px.bar(chart, x="Model", y="Balanced accuracy (%)", color="Split", barmode="group",
                         color_discrete_sequence=[TEAL, AMBER])
            fig.update_yaxes(range=[0, 105])
            st.plotly_chart(style_chart(fig, 350), use_container_width=True)
        st.warning("One normal run, correlated time-series measurements, simulator-specific patterns, and sensor sensitivity limit the conclusions. These are educational results only.")
        st.markdown("**Stress-test takeaway:** Three excluded fault-labelled scenarios have sensor patterns indistinguishable from the normal run. Sensor-only models cannot reliably distinguish identical inputs.")
else:
    # -------------------------------------------------------------------------
    # Full explore dashboard: existing four tabs
    # -------------------------------------------------------------------------
    monitor_tab, ai_tab, research_tab, stress_tab = st.tabs([
        "📈 Plant Monitor", "🧠 AI Diagnostics", "📊 Research Results", "🧪 Stress Testing"
    ])

    with monitor_tab:
        st.markdown("#### Instrument Readings")
        main_sensors = [("P", "Pressure (P)"), ("TAVG", "Average Temp (TAVG)"),
                        ("WFWA", "Feedwater Flow (WFWA)"), ("DNBR", "DNBR")]
        sensor_cols = st.columns(4)
        for container, (sensor, label) in zip(sensor_cols, main_sensors):
            if sensor in selected.columns:
                container.metric(label, f"{float(row[sensor]):,.3f}")
            else:
                container.metric(label, "N/A")

        available_sensors = [name for name in ["P", "TAVG", "WFWA", "DNBR", "WSTB", "DWB", "RM1", "RM4"] if name in features]
        defaults = [name for name in ["P", "TAVG"] if name in available_sensors]
        if "trend_select" not in st.session_state:
            st.session_state.trend_select = defaults
        chart_sensors = st.multiselect("Trend sensors", features, max_selections=5, key="trend_select",
                                        help="Select up to five measurements to compare over the simulation.")
        if chart_sensors:
            trends = selected[["TIME", *chart_sensors]].melt("TIME", var_name="Sensor", value_name="Reading")
            fig = px.line(trends, x="TIME", y="Reading", color="Sensor", template="plotly_white",
                          color_discrete_sequence=[TEAL, AMBER, "#8d9bff", RED, "#c9a5ef"])
            fig.add_vline(x=float(selected_time), line_dash="dash", line_color="#8091A4", opacity=0.7)
            fig.update_layout(height=385, margin=dict(l=10, r=10, t=18, b=10),
                              legend_title_text="Sensors",
                              title=None)
            st.plotly_chart(style_chart(fig, height=385), use_container_width=True)
            if len(chart_sensors) > 1:
                st.caption("Sensors may use different physical units; use one sensor at a time to assess its absolute magnitude.")
        st.markdown("#### Scenario Replay")
        timeline = pd.DataFrame({"TIME": selected["TIME"].to_numpy(),
                                 "Prediction": pred[model_col].to_numpy()})
        timeline["Prediction"] = timeline["Prediction"].map({0: "Normal", 1: "Fault"})
        fig_timeline = px.scatter(timeline, x="TIME", y="Prediction", color="Prediction",
                                  color_discrete_map={"Normal": TEAL, "Fault": RED}, template="plotly_white")
        fig_timeline.add_vline(x=float(selected_time), line_dash="dash", line_color="#8091A4", opacity=0.8)
        fig_timeline.update_traces(marker=dict(size=6))
        fig_timeline.update_layout(height=200, showlegend=False,                                margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(style_chart(fig_timeline, height=215), use_container_width=True)
        with st.expander(f"View all {len(features)} sensor measurements"):
            sensor_table = pd.DataFrame({"Sensor": features, "Value": [row[feature] for feature in features]})
            st.dataframe(sensor_table, use_container_width=True, hide_index=True)

    with ai_tab:
        st.markdown("#### Model-by-Model Predictions")
        result_cols = st.columns(3)
        for container, label, key in zip(result_cols,
                                          ["Random Forest", "XGBoost", "Autoencoder"],
                                          ["RF", "XGBoost", "Autoencoder"]):
            prediction = int(current[key])
            container.metric(label, "FAULT" if prediction else "NORMAL")
        left, right = st.columns(2)
        with left:
            st.markdown("##### XGBoost Fault Score")
            score = float(current["XGB_Score"])
            st.metric("Model score", f"{score * 100:.2f}%")
            st.progress(min(max(score, 0), 1))
            st.caption("Uncalibrated model score, not a verified probability of an actual nuclear incident.")
        with right:
            st.markdown("##### Autoencoder Reconstruction Error")
            error = float(current["AE_Error"])
            st.metric("Reconstruction MSE", f"{error:,.6g}")
            st.caption(f"Training-derived anomaly threshold: {ae_threshold:.8g}. Values above it are flagged.")
        st.markdown("##### AI Scores Throughout the Scenario")
        plot_data = pd.DataFrame({"TIME": selected["TIME"].to_numpy(),
                                  "XGBoost fault score": pred["XGB_Score"].to_numpy()})
        score_fig = px.line(plot_data, x="TIME", y="XGBoost fault score", template="plotly_white")
        score_fig.update_traces(line_color=TEAL)
        score_fig.add_hline(y=0.5, line_dash="dash", line_color=AMBER)
        score_fig.add_vline(x=float(selected_time), line_dash="dot", line_color="#8091A4")
        score_fig.update_layout(height=300,                             margin=dict(l=10, r=10, t=15, b=10), yaxis_range=[0, 1])
        st.plotly_chart(style_chart(score_fig, height=310), use_container_width=True)
        anomaly_plot = pd.DataFrame({"TIME": selected["TIME"].to_numpy(),
                                     "Reconstruction MSE": pred["AE_Error"].to_numpy()})
        anomaly_fig = px.line(anomaly_plot, x="TIME", y="Reconstruction MSE", template="plotly_white")
        anomaly_fig.update_traces(line_color=AMBER)
        anomaly_fig.add_hline(y=ae_threshold, line_dash="dash", line_color=RED)
        anomaly_fig.add_vline(x=float(selected_time), line_dash="dot", line_color="#8091A4")
        anomaly_fig.update_layout(height=300,                               margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(style_chart(anomaly_fig, height=310), use_container_width=True)
        st.caption("Autoencoder used a signed-log transform, a training-fitted RobustScaler, and clipping to ±10. Its detection is sensitive to some sensors.")
        feature_path = RESULTS_DIR / "advanced_xgb_feature_importance.csv"
        if feature_path.exists():
            feats = load_comparison(str(feature_path)).head(15)
            if {"Sensor", "Importance"}.issubset(feats.columns):
                fig = px.bar(feats.sort_values("Importance"), x="Importance", y="Sensor", orientation="h",
                             template="plotly_white", title="XGBoost — Top 15 Feature Importances")
                fig.update_traces(marker_color=TEAL)
                fig.update_layout(height=460,                               margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(style_chart(fig, height=460), use_container_width=True)
                st.caption("Feature importance reflects model split usage; it does not establish physical causality.")

    with research_tab:
        st.markdown("#### Reproducible Model Evaluation")
        comparison_path = RESULTS_DIR / "final_advanced_experiment_summary.csv"
        if comparison_path.exists():
            comparison = load_comparison(str(comparison_path))
            required_columns = {"Model", "Validation_Balanced_Accuracy", "Test_Balanced_Accuracy"}
            if required_columns.issubset(comparison.columns):
                display_table = comparison[["Model", "Validation_Balanced_Accuracy", "Test_Balanced_Accuracy"]].copy()
                display_table["Validation_Balanced_Accuracy"] *= 100
                display_table["Test_Balanced_Accuracy"] *= 100
                st.dataframe(display_table.rename(columns={
                    "Validation_Balanced_Accuracy": "Validation balanced accuracy (%)",
                    "Test_Balanced_Accuracy": "Test balanced accuracy (%)"
                }).round(2), use_container_width=True, hide_index=True)
                chart_data = display_table.melt("Model", var_name="Dataset", value_name="Balanced accuracy (%)")
                comp_fig = px.bar(chart_data, x="Model", y="Balanced accuracy (%)", color="Dataset",
                                  barmode="group", template="plotly_white", color_discrete_sequence=[TEAL, AMBER])
                comp_fig.update_layout(height=360,                                    margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 105])
                st.plotly_chart(style_chart(comp_fig, height=365), use_container_width=True)
        else:
            st.info("Run Step 31 of the advanced notebook to create the experiment summary CSV.")
        st.markdown("#### Research Caveats")
        st.warning("The dataset contains one normal-operation simulation. Fault scenarios are held out by scenario ID, but normal train/validation/test data are time-separated portions of the same run. Individual records within a scenario are correlated.")
        st.info("The Autoencoder reached perfect classification on the selected test set, but an exploratory five-sensor scoring ablation reduced its validation balanced accuracy from 100% to 66.81%. These findings do not establish deployment safety.")
        scenario_path = RESULTS_DIR / "advanced_test_scenario_results.csv"
        if scenario_path.exists():
            st.markdown("#### Performance by Held-Out Test Scenario")
            st.dataframe(load_comparison(str(scenario_path)), use_container_width=True, hide_index=True)

    with stress_tab:
        st.markdown("#### Excluded Fault Scenarios: Detectability Challenge")
        st.write("Three simulator scenarios were excluded from the revised dataset because their sensor time series matched the normal run. This is a known observability limitation, not a model-quality certificate.")
        original_path = DATA_DIR / "npp_prepared.csv"
        stress_scenarios = {5: "Spark Presence for Hydrogen Burn",
                            8: "Anticipated Transient without Scram",
                            14: "Moderator Dilution"}
        if original_path.exists():
            original = load_csv(str(original_path))
            if all(f in original.columns for f in ["Scenario_ID", *features]):
                stress_records = []
                for scenario_id, name in stress_scenarios.items():
                    scenario = original.loc[original["Scenario_ID"] == scenario_id]
                    if scenario.empty:
                        stress_records.append({"Scenario": name, "Status": "Not found in original dataset"})
                        continue
                    sample = scenario.iloc[[len(scenario) // 2]]
                    sample_pred = predict_batch(sample).iloc[0]
                    stress_records.append({
                        "Scenario": name,
                        "Dataset Label": "Fault",
                        "Random Forest": "Fault" if int(sample_pred["RF"]) else "Normal",
                        "XGBoost": "Fault" if int(sample_pred["XGBoost"]) else "Normal",
                        "Autoencoder": "Fault" if int(sample_pred["Autoencoder"]) else "Normal",
                        "XGBoost score (%)": round(float(sample_pred["XGB_Score"]) * 100, 2),
                        "Autoencoder error": round(float(sample_pred["AE_Error"]), 8),
                    })
                st.dataframe(pd.DataFrame(stress_records), use_container_width=True, hide_index=True)
                st.caption("One representative middle measurement per excluded scenario; not an exhaustive evaluation.")
            else:
                st.error("Original dataset is present but missing expected scenario or sensor columns.")
        else:
            st.info("To run excluded-scenario stress tests, retain prepared_data/npp_prepared.csv from your earlier project. The main dashboard uses npp_revised.csv.")
        st.warning("If fault and normal measurements are identical, sensor-only classifiers cannot distinguish them reliably. These simulations are educational and do not verify real plant safety.")

st.divider()
st.caption("NPP Sentinel AI • PCTRAN simulated nuclear plant data • Educational ML project • No operational safety certification")
