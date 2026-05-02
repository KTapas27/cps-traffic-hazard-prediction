import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="UAE Traffic Hazard Prediction",
    page_icon="🚗",
    layout="wide"
)

# ── LOAD MODEL ──
@st.cache_resource
def load_model():
    xgb = joblib.load(r'C:\Users\tapas\Desktop\cps_traffic_project\outputs\xgb_balanced_model.pkl')
    return xgb

model = load_model()

# ── HEADER ──
st.title("🇦🇪 UAE Traffic Accident Severity Predictor")
st.markdown("**AI-Driven Traffic & Climate Hazard Prediction System** — BITS Pilani Dubai Campus")
st.divider()

# ── SIDEBAR INPUTS ──
st.sidebar.header("Input Conditions")
st.sidebar.subheader("Location")
location = st.sidebar.selectbox("UAE Location", [
    "Sheikh Zayed Road, Dubai",
    "Abu Dhabi - Dubai Highway (E11)",
    "Emirates Road (E611)",
    "Al Ain Road (E66)",
    "Dubai - Sharjah Road (E11)"
])

location_coords = {
    "Sheikh Zayed Road, Dubai": (25.2048, 55.2708),
    "Abu Dhabi - Dubai Highway (E11)": (24.4539, 54.3773),
    "Emirates Road (E611)": (25.1, 55.5),
    "Al Ain Road (E66)": (24.8, 55.8),
    "Dubai - Sharjah Road (E11)": (25.3, 55.4)
}
lat, lng = location_coords[location]

st.sidebar.subheader("Weather Conditions")
visibility = st.sidebar.slider("Visibility (miles)", 0.1, 10.0, 5.0, 0.1)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0.0, 60.0, 10.0, 0.5)
temperature = st.sidebar.slider("Temperature (°F)", 60.0, 120.0, 95.0, 1.0)
humidity = st.sidebar.slider("Humidity (%)", 5.0, 80.0, 20.0, 1.0)
pressure = st.sidebar.slider("Pressure (in)", 29.0, 31.0, 29.92, 0.01)
precipitation = st.sidebar.slider("Precipitation (in)", 0.0, 1.0, 0.0, 0.01)

st.sidebar.subheader("Time")
hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
month = st.sidebar.slider("Month", 1, 12, 7)
day_of_week = st.sidebar.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
is_night = 1 if (hour < 6 or hour > 19) else 0

st.sidebar.subheader("Road")
distance = st.sidebar.slider("Accident Distance (mi)", 0.0, 5.0, 0.5, 0.1)

# ── SANDSTORM DETECTION ──
sandstorm_risk = 1 if (visibility < 1.0 and wind_speed > 25) else 0

# ── FEATURE VECTOR ──
# Weather encoded - simplified mapping
weather_map = {
    (True, True): 15,   # sandstorm
    (True, False): 8,   # low vis
    (False, True): 12,  # high wind
    (False, False): 3   # clear
}
weather_encoded = weather_map[(visibility < 2, wind_speed > 20)]
state_encoded = 5  # approximate UAE equivalent

features = np.array([[
    lat, lng, distance,
    temperature, temperature - 5,
    humidity, pressure,
    visibility, wind_speed,
    precipitation, weather_encoded, state_encoded,
    hour, month, day_map[day_of_week],
    is_night, sandstorm_risk
]])

# ── PREDICTION ──
pred = model.predict(features)[0] + 1
proba = model.predict_proba(features)[0]

# ── MAIN LAYOUT ──
col1, col2, col3 = st.columns(3)

severity_colors = {1: "green", 2: "yellow", 3: "orange", 4: "red"}
severity_labels = {
    1: "Minor",
    2: "Moderate",
    3: "Serious",
    4: "Fatal"
}
severity_desc = {
    1: "Low risk — normal conditions",
    2: "Moderate risk — drive carefully",
    3: "High risk — reduce speed",
    4: "Extreme risk — avoid if possible"
}

with col1:
    st.subheader("Predicted Severity")
    color = severity_colors[pred]
    st.markdown(f"""
    <div style='background-color:{color};padding:20px;border-radius:10px;text-align:center'>
        <h1 style='color:black;margin:0'>Severity {pred}</h1>
        <h3 style='color:black;margin:0'>{severity_labels[pred]}</h3>
        <p style='color:black;margin:0'>{severity_desc[pred]}</p>
    </div>
    """, unsafe_allow_html=True)

    if sandstorm_risk:
        st.error("⚠️ SANDSTORM CONDITIONS DETECTED — Visibility critically low + High winds")
    else:
        st.success("✅ No sandstorm risk detected")

with col2:
    st.subheader("Severity Probabilities")
    fig = go.Figure(go.Bar(
        x=[f"S{i+1}" for i in range(4)],
        y=[p*100 for p in proba],
        marker_color=['green','yellow','orange','red'],
        text=[f"{p*100:.1f}%" for p in proba],
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(t=20, b=20),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Risk Meter")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [1, 4]},
            'bar': {'color': color},
            'steps': [
                {'range': [1, 2], 'color': "lightgreen"},
                {'range': [2, 3], 'color': "yellow"},
                {'range': [3, 4], 'color': "orange"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 3.5
            }
        }
    ))
    fig2.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── SANDSTORM COMPARISON ──
st.subheader("🌪️ Sandstorm Impact Simulation")
col4, col5 = st.columns(2)

sandstorm_features = features.copy()
sandstorm_features[0][7] = 0.3   # visibility
sandstorm_features[0][8] = 35    # wind speed
sandstorm_features[0][16] = 1    # sandstorm flag
sandstorm_features[0][5] = 15    # humidity

sandstorm_pred = model.predict(sandstorm_features)[0] + 1
sandstorm_proba = model.predict_proba(sandstorm_features)[0]

with col4:
    st.markdown("**Normal conditions (your input)**")
    fig3 = go.Figure(go.Bar(
        x=[f"S{i+1}" for i in range(4)],
        y=[p*100 for p in proba],
        marker_color=['green','yellow','orange','red'],
        text=[f"{p*100:.1f}%" for p in proba],
        textposition='outside'
    ))
    fig3.update_layout(height=250, margin=dict(t=10,b=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.info(f"Predicted: **Severity {pred} ({severity_labels[pred]})**")

with col5:
    st.markdown("**UAE sandstorm conditions (0.3mi vis, 35mph wind)**")
    fig4 = go.Figure(go.Bar(
        x=[f"S{i+1}" for i in range(4)],
        y=[p*100 for p in sandstorm_proba],
        marker_color=['green','yellow','orange','red'],
        text=[f"{p*100:.1f}%" for p in sandstorm_proba],
        textposition='outside'
    ))
    fig4.update_layout(height=250, margin=dict(t=10,b=10), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    sc = severity_colors[sandstorm_pred]
    st.warning(f"Predicted: **Severity {sandstorm_pred} ({severity_labels[sandstorm_pred]})**")

st.divider()

# ── UAE MAP ──
st.subheader("🗺️ UAE Highway Risk Map")

highway_points = pd.DataFrame({
    'lat': [25.2048, 24.4539, 25.1, 24.8, 25.3, 24.2, 25.05, 24.6],
    'lon': [55.2708, 54.3773, 55.5, 55.8, 55.4, 54.5, 55.15, 54.9],
    'location': [
        'Sheikh Zayed Rd', 'AD-DXB Highway',
        'Emirates Road', 'Al Ain Road',
        'DXB-SHJ Road', 'Abu Dhabi Ring',
        'Al Khail Road', 'E311 MBZ'
    ],
    'risk': [pred, 2, 2, 1, 3, 2, 1, 2]
})

color_map = {1:'green', 2:'yellow', 3:'orange', 4:'red'}
highway_points['color'] = highway_points['risk'].map(color_map)

fig5 = px.scatter_mapbox(
    highway_points,
    lat='lat', lon='lon',
    color='risk',
    size=[20]*len(highway_points),
    hover_name='location',
    hover_data={'risk': True, 'lat': False, 'lon': False},
    color_continuous_scale=['green','yellow','orange','red'],
    range_color=[1,4],
    zoom=7,
    center={"lat": 24.8, "lon": 55.0},
    mapbox_style="open-street-map",
    height=400
)
fig5.update_layout(margin=dict(t=0,b=0))
st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.subheader("📊 Batch Simulation — Population Level Impact")

if st.button("Run 1000 Scenario Simulations"):
    normal_preds = []
    storm_preds = []
    
    for i in range(1000):
        noise = np.random.normal(0, 0.5, features.shape)
        f_normal = features + noise
        f_storm = sandstorm_features + noise
        normal_preds.append(model.predict(f_normal)[0] + 1)
        storm_preds.append(model.predict(f_storm)[0] + 1)
    
    col6, col7 = st.columns(2)
    
    normal_counts = pd.Series(normal_preds).value_counts().sort_index()
    storm_counts = pd.Series(storm_preds).value_counts().sort_index()
    
    with col6:
        st.markdown("**Normal conditions — 1000 simulations**")
        fig6 = go.Figure(go.Bar(
            x=[f"S{i}" for i in normal_counts.index],
            y=normal_counts.values,
            marker_color=['green','yellow','orange','red'][:len(normal_counts)]
        ))
        fig6.update_layout(height=250, margin=dict(t=10,b=10))
        st.plotly_chart(fig6, use_container_width=True)
    
    with col7:
        st.markdown("**Sandstorm conditions — 1000 simulations**")
        fig7 = go.Figure(go.Bar(
            x=[f"S{i}" for i in storm_counts.index],
            y=storm_counts.values,
            marker_color=['green','yellow','orange','red'][:len(storm_counts)]
        ))
        fig7.update_layout(height=250, margin=dict(t=10,b=10))
        st.plotly_chart(fig7, use_container_width=True)
    
    normal_avg = np.mean(normal_preds)
    storm_avg = np.mean(storm_preds)
    pct_change = (storm_avg - normal_avg) / normal_avg * 100
    st.metric(
        label="Severity increase under sandstorm",
        value=f"{pct_change:.1f}%",
        delta=f"avg {normal_avg:.2f} → {storm_avg:.2f}"
    )
st.markdown("*Model: XGBoost trained on 1.5M accident records with SMOTE balancing | BITS Pilani Dubai Campus*")