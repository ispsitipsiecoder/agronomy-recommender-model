import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import requests
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import (
    NANO_UREA_SPECS, RAIN_PROBABILITY_THRESHOLD,
    PEST_RISK_THRESHOLD_HIGH, PEST_RISK_THRESHOLD_LOW,
    BIO_CONTROL_OPTIONS, PILOT_LAT, PILOT_LNG
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Precision Agronomy Recommender",
    page_icon  = "🌾",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fdf9; }
    .stApp { font-family: Arial, sans-serif; }
    .header-box {
        background: linear-gradient(135deg, #1D6A3E, #2E8B57);
        padding: 24px 32px; border-radius: 12px;
        margin-bottom: 24px; color: white;
    }
    .header-box h1 { color: white; margin: 0; font-size: 26px; }
    .header-box p  { color: #c8f0d8; margin: 6px 0 0; font-size: 14px; }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 18px 20px; border: 1px solid #e0f0e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-label { font-size: 12px; color: #666; font-weight: 500;
                    text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
    .metric-sub   { font-size: 13px; color: #555; }
    .spray  { color: #1D6A3E; }
    .delay  { color: #E67E22; }
    .basal  { color: #7F8C8D; }
    .pest   { color: #C0392B; }
    .bio    { color: #2980B9; }
    .monitor{ color: #27AE60; }
    .shap-box {
        background: #f0f8f2; border-radius: 8px;
        padding: 14px 18px; border-left: 4px solid #1D6A3E;
        margin-top: 12px;
    }
    .shap-title { font-size: 13px; font-weight: 600;
                  color: #1D6A3E; margin-bottom: 8px; }
    .shap-row   { font-size: 13px; color: #333;
                  padding: 3px 0; border-bottom: 1px solid #e0ede4; }
    .weather-box {
        background: #e8f4fd; border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #2980B9;
        margin-bottom: 16px; font-size: 13px;
    }
    .section-title {
        font-size: 13px; font-weight: 600; color: #1D6A3E;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-bottom: 8px; margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    spray_model = joblib.load(os.path.join(base, 'models/spray_model.pkl'))
    pest_model  = joblib.load(os.path.join(base, 'models/pest_model.pkl'))
    le_spray    = joblib.load(os.path.join(base, 'models/le_spray.pkl'))
    le_pest     = joblib.load(os.path.join(base, 'models/le_pest.pkl'))
    return spray_model, pest_model, le_spray, le_pest

spray_model, pest_model, le_spray, le_pest = load_models()

FEATURE_COLS = [
    'N','P','K','temperature','humidity','ph','rainfall',
    'days_after_transplanting','rain_prob_8h','ndvi_stress','pest_history',
    'zinc_ppm','boron_ppm','sulphur_ppm','iron_ppm','manganese_ppm','copper_ppm'
]
PEST_FEATURE_COLS = [
    'pest_history','ndvi_stress','temperature',
    'humidity','rainfall','rain_prob_8h','ph','zinc_ppm'
]

# ── Live weather ──────────────────────────────────────────────────────────────

    # ── District coordinates map ──────────────────────────────────────────────────
INDIA_DISTRICTS = {
    "Gautam Buddha Nagar, UP": (28.4595, 77.5022),
    "Lucknow, UP":             (26.8467, 80.9462),
    "Varanasi, UP":            (25.3176, 82.9739),
    "Patna, Bihar":            (25.5941, 85.1376),
    "Jaipur, Rajasthan":       (26.9124, 75.7873),
    "Bhopal, MP":              (23.2599, 77.4126),
    "Nagpur, Maharashtra":     (21.1458, 79.0882),
    "Pune, Maharashtra":       (18.5204, 73.8567),
    "Hyderabad, Telangana":    (17.3850, 78.4867),
    "Chennai, Tamil Nadu":     (13.0827, 80.2707),
    "Bengaluru, Karnataka":    (12.9716, 77.5946),
    "Kolkata, West Bengal":    (22.5726, 88.3639),
    "Bhubaneswar, Odisha":     (20.2961, 85.8245),
    "Raipur, Chhattisgarh":    (21.2514, 81.6296),
    "Ranchi, Jharkhand":       (23.3441, 85.3096),
    "Amritsar, Punjab":        (31.6340, 74.8723),
    "Chandigarh, Punjab":      (30.7333, 76.7794),
    "Ahmedabad, Gujarat":      (23.0225, 72.5714),
    "Guwahati, Assam":         (26.1445, 91.7362),
    "Thiruvananthapuram, Kerala": (8.5241, 76.9366),
}

OWM_API_KEY = "b8efc77c00b3709ffdee0d0f0372544b"

@st.cache_data(ttl=1800)
def get_weather_owm(district: str):
    """Fetch live weather from OpenWeatherMap for selected district."""
    lat, lng = INDIA_DISTRICTS[district]
    try:
        # Current weather
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": lat, "lon": lng,
                "appid": OWM_API_KEY,
                "units": "metric"
            }, timeout=5
        )
        d = r.json()

        if d.get("cod") != 200:
            raise ValueError("API not ready")

        # Forecast for 8-hour rain check
        f = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={
                "lat": lat, "lon": lng,
                "appid": OWM_API_KEY,
                "units": "metric", "cnt": 3   # next 3 x 3hr slots = 9 hours
            }, timeout=5
        )
        forecast = f.json()
        rain_probs = [
            item.get("pop", 0)             # pop = probability of precipitation
            for item in forecast.get("list", [])
        ]
        max_rain = max(rain_probs) if rain_probs else 0.10

        return {
            "rain_prob":   round(max_rain, 2),
            "temp":        round(d["main"]["temp"], 1),
            "humidity":    round(d["main"]["humidity"], 1),
            "description": d["weather"][0]["description"].title(),
            "wind_kmh":    round(d["wind"]["speed"] * 3.6, 1),
            "safe":        max_rain < 0.20,
            "source":      "OpenWeatherMap"
        }

    except Exception:
        # Fallback to Open-Meteo if OWM key not yet active
        r2 = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lng,
                "hourly": "precipitation_probability,temperature_2m,relative_humidity_2m",
                "forecast_days": 1, "timezone": "Asia/Kolkata"
            }, timeout=5
        )
        d2 = r2.json()["hourly"]
        max_rain2 = max(d2["precipitation_probability"][:8]) / 100
        return {
            "rain_prob":   round(max_rain2, 2),
            "temp":        round(d2["temperature_2m"][0], 1),
            "humidity":    round(d2["relative_humidity_2m"][0], 1),
            "description": "Live forecast",
            "wind_kmh":    0,
            "safe":        max_rain2 < 0.20,
            "source":      "Open-Meteo (fallback)"
        }

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🌾 Precision Agronomy Recommender</h1>
    <p>IFFCO Nano-Fertiliser Optimization & Low-Pesticide Advisory System &nbsp;|&nbsp;
       Pilot Crop: Paddy &nbsp;|&nbsp; District: Gautam Buddha Nagar, UP</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown('<div class="section-title">🌱 Soil Profile</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        N  = st.slider("Nitrogen (N) kg/ha",  40, 120, 75)
        P  = st.slider("Phosphorus (P) kg/ha", 20, 70, 45)
        K  = st.slider("Potassium (K) kg/ha",  30, 55, 40)
        ph = st.slider("Soil pH",              5.0, 8.5, 6.5, 0.1)
    with c2:
        temp     = st.slider("Temperature °C",   18, 38, 27)
        humidity = st.slider("Humidity %",        50, 95, 80)
        rainfall = st.slider("Seasonal rainfall mm", 100, 350, 220)
        oc       = st.slider("Organic Carbon %",  0.1, 1.5, 0.5, 0.1)

    st.markdown('<div class="section-title">🌾 Crop Information</div>',
                unsafe_allow_html=True)

    dat = st.slider("Days After Transplanting (DAT)", 0, 120, 32)

    windows = NANO_UREA_SPECS['application_windows']['paddy']
    w1 = windows[0]
    w2 = windows[1]
    if w1['dat_min'] <= dat <= w1['dat_max']:
        st.success(f"✅ In Spray Window 1 — {w1['stage'].replace('_',' ').title()} (DAT {w1['dat_min']}–{w1['dat_max']})")
    elif w2['dat_min'] <= dat <= w2['dat_max']:
        st.success(f"✅ In Spray Window 2 — {w2['stage'].replace('_',' ').title()} (DAT {w2['dat_min']}–{w2['dat_max']})")
    else:
        st.info(f"ℹ️ Outside spray windows — next window at DAT {w1['dat_min'] if dat < w1['dat_min'] else w2['dat_min'] if dat < w2['dat_min'] else 'completed'}")

    st.markdown('<div class="section-title">🐛 Pest Context</div>',
                unsafe_allow_html=True)

    pest_history = st.slider("District Pest Pressure (0 = low, 1 = high)", 0.0, 1.0, 0.35, 0.05)
    ndvi_stress  = st.slider("Crop Health / NDVI (0 = stressed, 1 = healthy)", 0.3, 1.0, 0.72, 0.05)

    st.markdown('<div class="section-title">⚗️ Micronutrients</div>',
                unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        zinc    = st.number_input("Zinc (ppm)",     0.1, 3.0, 0.55, 0.05)
        boron   = st.number_input("Boron (ppm)",    0.1, 2.0, 0.42, 0.05)
        sulphur = st.number_input("Sulphur (ppm)", 2.0, 30.0, 10.8, 0.5)
    with mc2:
        iron    = st.number_input("Iron (ppm)",     2.0, 25.0, 7.9, 0.5)
        mangan  = st.number_input("Manganese (ppm)",0.5, 10.0, 3.5, 0.5)
        copper  = st.number_input("Copper (ppm)",   0.2, 4.0,  1.1, 0.1)

    get_rec = st.button("🔍 Get Recommendation", use_container_width=True, type="primary")

# ── Right panel ───────────────────────────────────────────────────────────────
with right:

    # Live weather
    district = st.sidebar.selectbox(
    "📍 Select your district",
    list(INDIA_DISTRICTS.keys()),
    index=0
)
weather = get_weather_owm(district)
spray_safe = "✅ Clear — Safe to spray" if weather['safe'] else "⚠️ Rain risk — Delay spray"
    st.markdown(f"""
    🌤️ <strong>Live Weather — {district}</strong> &nbsp;|&nbsp;
    {weather['description']} &nbsp;|&nbsp;
    Temp: {weather['temp']}°C &nbsp;|&nbsp;
    Humidity: {weather['humidity']}% &nbsp;|&nbsp;
    Wind: {weather['wind_kmh']} km/h &nbsp;|&nbsp;
    Rain (8h): {weather['rain_prob']*100:.0f}% &nbsp;|&nbsp;
    {spray_safe} &nbsp;|&nbsp;
    <span style='font-size:10px;color:#aaa'>via {weather['source']}</span>

    if get_rec:
        # Build input
        farmer = pd.DataFrame([{
            'N': N, 'P': P, 'K': K,
            'temperature': temp, 'humidity': humidity,
            'ph': ph, 'rainfall': rainfall,
            'days_after_transplanting': dat,
            'rain_prob_8h': weather['rain_prob'],
            'ndvi_stress':  ndvi_stress,
            'pest_history': pest_history,
            'zinc_ppm': zinc, 'boron_ppm': boron,
            'sulphur_ppm': sulphur, 'iron_ppm': iron,
            'manganese_ppm': mangan, 'copper_ppm': copper
        }])

        # Predictions
        spray_pred  = spray_model.predict(farmer[FEATURE_COLS])[0]
        spray_proba = spray_model.predict_proba(farmer[FEATURE_COLS])[0]
        spray_label = le_spray.inverse_transform([spray_pred])[0]

        pest_pred   = pest_model.predict(farmer[PEST_FEATURE_COLS])[0]
        pest_proba  = pest_model.predict_proba(farmer[PEST_FEATURE_COLS])[0]
        pest_label  = le_pest.inverse_transform([pest_pred])[0]

        # SHAP
        explainer   = shap.TreeExplainer(spray_model)
        shap_vals   = explainer.shap_values(farmer[FEATURE_COLS])
        shap_for_pred = shap_vals[0, :, spray_pred]
        top3 = pd.Series(shap_for_pred, index=FEATURE_COLS)\
                 .abs().sort_values(ascending=False).head(3)

        # ── Module 1 output ───────────────────────────────────────────────────
        st.markdown('<div class="section-title">🌿 Module 1 — Fertiliser Recommendation</div>',
                    unsafe_allow_html=True)

        css_class = {'Spray':'spray','Delay':'delay','Basal Only':'basal'}.get(spray_label,'basal')
        icon      = {'Spray':'💧','Delay':'⏳','Basal Only':'🌱'}.get(spray_label,'🌱')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Action</div>
                <div class="metric-value {css_class}">{icon} {spray_label}</div>
                <div class="metric-sub">Confidence: {max(spray_proba):.0%}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            if spray_label == 'Spray':
                dose = 4.0 if N < 60 else 3.0 if N < 80 else 2.0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Nano Urea Plus Dose</div>
                    <div class="metric-value spray">{dose} ml/L</div>
                    <div class="metric-sub">Mix in water — foliar spray</div>
                </div>""", unsafe_allow_html=True)
            elif spray_label == 'Delay':
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Reason</div>
                    <div class="metric-value delay">🌧️ Rain Risk</div>
                    <div class="metric-sub">Check again in 8 hours</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Note</div>
                    <div class="metric-value basal">Outside Window</div>
                    <div class="metric-sub">Apply basal dose at transplanting only</div>
                </div>""", unsafe_allow_html=True)

        # ── Module 2 output ───────────────────────────────────────────────────
        st.markdown('<div class="section-title">🐛 Module 2 — Pest Risk Advisory</div>',
                    unsafe_allow_html=True)

        pest_css  = {'Pesticide':'pest','Bio Control':'bio','Monitor':'monitor'}.get(pest_label,'monitor')
        pest_icon = {'Pesticide':'🚨','Bio Control':'🌿','Monitor':'👀'}.get(pest_label,'👀')

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Intervention</div>
                <div class="metric-value {pest_css}">{pest_icon} {pest_label}</div>
                <div class="metric-sub">Confidence: {max(pest_proba):.0%}</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            if pest_label == 'Bio Control':
                options = BIO_CONTROL_OPTIONS.get('paddy', [])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Suggested Action</div>
                    <div class="metric-value bio" style="font-size:14px">{options[0] if options else 'Neem-based spray'}</div>
                </div>""", unsafe_allow_html=True)
            elif pest_label == 'Pesticide':
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Action Required</div>
                    <div class="metric-value pest" style="font-size:14px">
                        Contact local IFFCO agronomist for pesticide recommendation
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Action</div>
                    <div class="metric-value monitor" style="font-size:14px">
                        Low risk — monitor field every 3 days
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── SHAP explanation ──────────────────────────────────────────────────
        label_map = {
            'rain_prob_8h':              'Rain probability (next 8h)',
            'days_after_transplanting':  'Days after transplanting',
            'N':                         'Soil nitrogen level',
            'ndvi_stress':               'Crop health (NDVI)',
            'pest_history':              'District pest history',
            'humidity':                  'Humidity',
            'temperature':               'Temperature',
            'ph':                        'Soil pH',
            'sulphur_ppm':               'Sulphur level',
            'zinc_ppm':                  'Zinc level',
        }

        st.markdown("""
        <div class="shap-box">
            <div class="shap-title">🔍 Why this recommendation?</div>
        """, unsafe_allow_html=True)

        for feat, val in top3.items():
            nice = label_map.get(feat, feat.replace('_',' ').title())
            bar_width = min(int(val * 80), 100)
            direction = "increased" if shap_for_pred[FEATURE_COLS.index(feat)] > 0 else "reduced"
            st.markdown(f"""
            <div class="shap-row">
                <strong>{nice}</strong>
                &nbsp;—&nbsp; {direction} spray confidence
                <div style="background:#c8e6d4;border-radius:4px;height:6px;
                            width:{bar_width}%;margin-top:3px;"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── IFFCO rule reminder ───────────────────────────────────────────────
        st.markdown(f"""
        <div style="margin-top:14px;padding:10px 14px;background:#fff8e1;
                    border-radius:8px;border-left:4px solid #F39C12;
                    font-size:12px;color:#7A4A00;">
            ⚠️ <strong>IFFCO Rule:</strong>
            {NANO_UREA_SPECS['basal_rule']}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#aaa;">
            <div style="font-size:48px;">🌾</div>
            <div style="font-size:16px;margin-top:12px;">
                Adjust the field parameters on the left<br>and click
                <strong>Get Recommendation</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:1px solid #e0ede4;margin-top:32px">
<div style="text-align:center;font-size:12px;color:#aaa;padding:8px">
    Precision Agronomy Recommender &nbsp;|&nbsp;
    IFFCO Internship Project &nbsp;|&nbsp;
    Kartikey Negi — Amity University &nbsp;|&nbsp;
    Mentor: Mr. Mayur Gupta
</div>
""", unsafe_allow_html=True)