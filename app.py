import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import requests
import os
import sys
from datetime import date, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import (
    NANO_UREA_SPECS, RAIN_PROBABILITY_THRESHOLD,
    PEST_RISK_THRESHOLD_HIGH, PEST_RISK_THRESHOLD_LOW,
    BIO_CONTROL_OPTIONS, SOIL_BENCHMARKS_PADDY
)

st.set_page_config(
    page_title="IFFCO Kisan — Agronomy AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --green-900: #0a2e1a;
    --green-800: #0f3d22;
    --green-700: #145230;
    --green-600: #1a6b3e;
    --green-500: #1D9E75;
    --green-400: #2ECC8A;
    --green-300: #6EDDB0;
    --green-100: #e0f5eb;
    --green-50:  #f0faf5;
    --amber:     #F59E0B;
    --amber-light: #FEF3C7;
    --red:       #DC2626;
    --red-light: #FEE2E2;
    --blue:      #2563EB;
    --blue-light: #DBEAFE;
    --cream:     #FAFAF7;
    --ink:       #0f1a12;
    --muted:     #5a7a62;
    --border:    #d4e8da;
    --shadow:    0 4px 24px rgba(10,46,26,0.08);
    --shadow-lg: 0 8px 48px rgba(10,46,26,0.14);
}
* { font-family: 'Sora', sans-serif !important; box-sizing: border-box; }
.stApp {
    background: var(--cream) !important;
    background-image: radial-gradient(ellipse at 0% 0%, rgba(29,158,117,0.06) 0%, transparent 60%), radial-gradient(ellipse at 100% 100%, rgba(29,158,117,0.04) 0%, transparent 50%) !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { background: var(--green-900) !important; padding: 0 !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: rgba(255,255,255,0.85) !important; font-size: 12px !important; }
.sidebar-brand { background: linear-gradient(160deg, var(--green-800), var(--green-900)); padding: 28px 20px 20px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px; }
.sidebar-brand-title { font-size: 20px; font-weight: 700; color: white; letter-spacing: -0.5px; }
.sidebar-brand-sub { font-size: 11px; color: var(--green-300); margin-top: 4px; }
.sidebar-section { padding: 0 16px; margin-bottom: 20px; }
.sidebar-section-label { font-size: 9px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--green-300); margin-bottom: 10px; opacity: 0.7; }
.main-header { background: linear-gradient(135deg, var(--green-900) 0%, var(--green-700) 50%, var(--green-600) 100%); padding: 32px 40px; position: relative; overflow: hidden; }
.main-header::before { content: ''; position: absolute; top: -40px; right: -40px; width: 200px; height: 200px; background: radial-gradient(circle, rgba(46,204,138,0.15) 0%, transparent 70%); border-radius: 50%; }
.header-eyebrow { font-size: 10px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--green-300); margin-bottom: 8px; }
.header-title { font-size: 32px; font-weight: 700; color: white; letter-spacing: -0.8px; line-height: 1.1; margin-bottom: 6px; }
.header-title span { color: var(--green-400); }
.header-subtitle { font-size: 14px; color: rgba(255,255,255,0.65); }
.header-badges { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.header-badge { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); color: rgba(255,255,255,0.8); font-size: 11px; font-weight: 500; padding: 4px 12px; border-radius: 20px; }
.weather-strip { background: white; border-bottom: 1px solid var(--border); padding: 12px 40px; display: flex; align-items: center; gap: 24px; flex-wrap: wrap; box-shadow: 0 2px 8px rgba(10,46,26,0.04); }
.weather-item { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--ink); }
.weather-item span.label { color: var(--muted); font-size: 11px; }
.weather-safe { color: var(--green-600); font-weight: 600; }
.weather-risk { color: var(--red); font-weight: 600; }
.weather-source { font-size: 10px; color: var(--muted); margin-left: auto; }
.content-area { padding: 28px 40px; }
.module-card { background: white; border-radius: 16px; border: 1px solid var(--border); box-shadow: var(--shadow); margin-bottom: 20px; overflow: hidden; }
.module-card-header { padding: 16px 20px; border-bottom: 1px solid var(--green-50); display: flex; align-items: center; gap: 10px; background: linear-gradient(to right, var(--green-50), white); }
.module-num { width: 28px; height: 28px; border-radius: 8px; background: var(--green-900); color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.module-title { font-size: 13px; font-weight: 600; color: var(--green-900); flex: 1; }
.module-subtitle { font-size: 11px; color: var(--muted); }
.module-body { padding: 20px; }
.decision-hero { border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; display: flex; align-items: center; gap: 16px; }
.decision-hero.spray { background: linear-gradient(135deg, #052e16, #166534); }
.decision-hero.delay { background: linear-gradient(135deg, #78350f, #b45309); }
.decision-hero.basal { background: linear-gradient(135deg, #1e3a5f, #1d4ed8); }
.decision-icon { font-size: 36px; flex-shrink: 0; }
.decision-text { flex: 1; }
.decision-action { font-size: 22px; font-weight: 700; color: white; letter-spacing: -0.3px; }
.decision-detail { font-size: 13px; color: rgba(255,255,255,0.75); margin-top: 4px; }
.decision-confidence { background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2); color: white; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 20px; flex-shrink: 0; }
.metric-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
.metric-mini { background: var(--green-50); border-radius: 10px; padding: 12px 14px; border: 1px solid var(--green-100); }
.metric-mini-label { font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.metric-mini-val { font-size: 20px; font-weight: 700; color: var(--green-700); margin-top: 2px; }
.metric-mini-sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
.saving-hero { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1px solid #86efac; border-radius: 12px; padding: 14px 18px; display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.saving-amount { font-size: 28px; font-weight: 700; color: #166534; }
.saving-label { font-size: 12px; color: #166534; opacity: 0.8; }
.saving-detail { font-size: 11px; color: var(--muted); margin-top: 2px; }
.shap-section { background: var(--green-50); border-radius: 10px; padding: 14px 16px; border-left: 3px solid var(--green-500); margin-top: 14px; }
.shap-label { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.shap-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.shap-name { font-size: 12px; color: var(--ink); width: 150px; flex-shrink: 0; }
.shap-track { flex: 1; height: 5px; background: var(--border); border-radius: 3px; }
.shap-fill { height: 100%; border-radius: 3px; background: linear-gradient(to right, var(--green-600), var(--green-400)); }
.shap-dir { font-size: 10px; font-weight: 500; width: 60px; text-align: right; }
.pest-hero { border-radius: 12px; padding: 16px 20px; display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
.pest-hero.pesticide { background: linear-gradient(135deg, #450a0a, #991b1b); }
.pest-hero.bio { background: linear-gradient(135deg, #052e16, #15803d); }
.pest-hero.monitor { background: linear-gradient(135deg, #172554, #1d4ed8); }
.pest-icon-large { font-size: 32px; }
.pest-action { font-size: 18px; font-weight: 700; color: white; }
.pest-detail { font-size: 12px; color: rgba(255,255,255,0.75); margin-top: 3px; }
.suitability-score-wrap { display: flex; align-items: center; gap: 20px; margin-bottom: 18px; }
.suit-verdict { font-size: 16px; font-weight: 600; color: var(--green-900); }
.suit-advice { font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.5; }
.nutrient-item { margin-bottom: 10px; }
.nutrient-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.nutrient-name { font-size: 12px; font-weight: 500; color: var(--ink); }
.nutrient-badge { font-size: 10px; font-weight: 600; padding: 1px 8px; border-radius: 10px; }
.n-good { background: var(--green-100); color: var(--green-700); }
.n-low  { background: var(--amber-light); color: #92400E; }
.n-high { background: var(--red-light); color: var(--red); }
.nutrient-track-outer { height: 8px; background: var(--green-50); border-radius: 4px; position: relative; border: 1px solid var(--border); }
.nutrient-bar { height: 100%; border-radius: 4px; }
.ideal-line { position: absolute; top: -2px; width: 2px; height: 12px; background: var(--ink); border-radius: 1px; opacity: 0.3; }
.crop-table { width: 100%; border-collapse: collapse; margin-top: 4px; }
.crop-table th { font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
.crop-table td { font-size: 12px; padding: 8px 10px; border-bottom: 1px solid var(--green-50); color: var(--ink); }
.crop-table tr:last-child td { border-bottom: none; }
.crop-table tr.current-crop td { background: var(--green-50); font-weight: 600; }
.suit-pill { font-size: 10px; font-weight: 600; padding: 2px 9px; border-radius: 10px; }
.suit-exc  { background: #dcfce7; color: #166534; }
.suit-good { background: var(--green-100); color: var(--green-700); }
.suit-med  { background: var(--amber-light); color: #92400E; }
.suit-low  { background: var(--red-light); color: var(--red); }
.cal-wrap { margin: 14px 0; overflow-x: auto; }
.cal-row { display: flex; gap: 3px; min-width: 500px; }
.cal-stage { flex: 1; height: 24px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 9px; font-weight: 600; color: white; position: relative; }
.cal-stage.current::after { content: '📍'; position: absolute; top: -18px; left: 50%; transform: translateX(-50%); font-size: 12px; }
.cal-labels { display: flex; gap: 3px; min-width: 500px; margin-top: 4px; }
.cal-label-item { flex: 1; font-size: 9px; color: var(--muted); text-align: center; }
.hindi-box { background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; padding: 16px 20px; margin-top: 14px; }
.hindi-toggle-label { font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(255,255,255,0.5); margin-bottom: 8px; }
.hindi-text { font-size: 15px; color: white; line-height: 1.7; }
.rule-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 12px 16px; font-size: 12px; color: #92400E; margin-top: 14px; display: flex; gap: 8px; align-items: flex-start; }
.empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty-icon { font-size: 56px; margin-bottom: 16px; }
.empty-title { font-size: 18px; font-weight: 600; color: var(--green-800); margin-bottom: 6px; }
.empty-sub { font-size: 13px; line-height: 1.5; }
.stButton button { background: linear-gradient(135deg, var(--green-700), var(--green-600)) !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: 'Sora', sans-serif !important; font-weight: 600 !important; font-size: 14px !important; padding: 10px 20px !important; width: 100% !important; box-shadow: 0 4px 14px rgba(29,106,62,0.35) !important; }
</style>
""", unsafe_allow_html=True)

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
    "Amritsar, Punjab":        (31.6340, 74.8723),
    "Ahmedabad, Gujarat":      (23.0225, 72.5714),
    "Guwahati, Assam":         (26.1445, 91.7362),
    "Ranchi, Jharkhand":       (23.3441, 85.3096),
}


# Average soil profiles per district based on ICAR/SHC published data
DISTRICT_SOIL = {
    "Gautam Buddha Nagar, UP": {"N":82, "P":44, "K":41, "ph":7.2, "temp":27, "humidity":78, "rainfall":220, "pest":0.45, "ndvi":0.68},
    "Lucknow, UP":             {"N":78, "P":40, "K":39, "ph":7.4, "temp":28, "humidity":76, "rainfall":198, "pest":0.40, "ndvi":0.70},
    "Varanasi, UP":            {"N":75, "P":38, "K":40, "ph":7.6, "temp":29, "humidity":74, "rainfall":185, "pest":0.42, "ndvi":0.65},
    "Patna, Bihar":            {"N":88, "P":46, "K":42, "ph":6.8, "temp":28, "humidity":82, "rainfall":245, "pest":0.50, "ndvi":0.72},
    "Jaipur, Rajasthan":       {"N":62, "P":32, "K":35, "ph":7.8, "temp":32, "humidity":48, "rainfall":120, "pest":0.28, "ndvi":0.55},
    "Bhopal, MP":              {"N":72, "P":36, "K":38, "ph":7.0, "temp":28, "humidity":68, "rainfall":195, "pest":0.38, "ndvi":0.66},
    "Nagpur, Maharashtra":     {"N":68, "P":34, "K":36, "ph":7.2, "temp":31, "humidity":62, "rainfall":165, "pest":0.42, "ndvi":0.60},
    "Pune, Maharashtra":       {"N":70, "P":38, "K":37, "ph":6.8, "temp":27, "humidity":65, "rainfall":180, "pest":0.35, "ndvi":0.64},
    "Hyderabad, Telangana":    {"N":65, "P":33, "K":36, "ph":7.1, "temp":30, "humidity":60, "rainfall":155, "pest":0.38, "ndvi":0.62},
    "Chennai, Tamil Nadu":     {"N":74, "P":40, "K":38, "ph":6.5, "temp":30, "humidity":78, "rainfall":210, "pest":0.44, "ndvi":0.68},
    "Bengaluru, Karnataka":    {"N":76, "P":42, "K":40, "ph":6.2, "temp":24, "humidity":70, "rainfall":220, "pest":0.36, "ndvi":0.70},
    "Kolkata, West Bengal":    {"N":90, "P":48, "K":44, "ph":6.4, "temp":29, "humidity":85, "rainfall":280, "pest":0.55, "ndvi":0.74},
    "Bhubaneswar, Odisha":     {"N":85, "P":44, "K":42, "ph":6.6, "temp":29, "humidity":80, "rainfall":260, "pest":0.50, "ndvi":0.72},
    "Amritsar, Punjab":        {"N":95, "P":52, "K":45, "ph":7.8, "temp":25, "humidity":68, "rainfall":160, "pest":0.35, "ndvi":0.75},
    "Ahmedabad, Gujarat":      {"N":60, "P":30, "K":34, "ph":7.6, "temp":32, "humidity":55, "rainfall":130, "pest":0.30, "ndvi":0.55},
    "Guwahati, Assam":         {"N":92, "P":48, "K":44, "ph":5.8, "temp":26, "humidity":88, "rainfall":320, "pest":0.58, "ndvi":0.76},
    "Ranchi, Jharkhand":       {"N":80, "P":42, "K":40, "ph":6.0, "temp":25, "humidity":75, "rainfall":240, "pest":0.45, "ndvi":0.70},
}

OWM_API_KEY = "b8efc77c00b3709ffdee0d0f0372544b"

CROP_IDEAL = {
    "Paddy":   {"N":(80,120),"P":(40,60),"K":(38,45),"ph":(5.5,7.0)},
    "Wheat":   {"N":(100,140),"P":(40,60),"K":(35,45),"ph":(6.0,7.5)},
    "Maize":   {"N":(80,120),"P":(35,55),"K":(35,50),"ph":(5.8,7.0)},
    "Mustard": {"N":(80,100),"P":(30,50),"K":(30,45),"ph":(6.0,7.5)},
    "Soybean": {"N":(40,70), "P":(35,55),"K":(35,50),"ph":(6.0,6.8)},
}

HINDI = {
    "Spray":      "आज छिड़काव करें",
    "Delay":      "प्रतीक्षा करें — बारिश की संभावना",
    "Basal Only": "केवल बेसल उर्वरक — छिड़काव की आवश्यकता नहीं",
    "Pesticide":  "कीटनाशक की आवश्यकता — कृषि विशेषज्ञ से संपर्क करें",
    "Bio Control":"जैव नियंत्रण — नीम आधारित छिड़काव करें",
    "Monitor":    "निगरानी करें — खेत की जांच 3 दिन में करें",
}

FEATURE_COLS = [
    'N','P','K','temperature','humidity','ph','rainfall',
    'days_after_transplanting','rain_prob_8h','ndvi_stress','pest_history',
    'zinc_ppm','boron_ppm','sulphur_ppm','iron_ppm','manganese_ppm','copper_ppm'
]
PEST_FEATURE_COLS = [
    'pest_history','ndvi_stress','temperature',
    'humidity','rainfall','rain_prob_8h','ph','zinc_ppm'
]
LABEL_MAP = {
    'rain_prob_8h':'Rain probability (8h)',
    'days_after_transplanting':'Crop stage (DAT)',
    'N':'Soil nitrogen','ndvi_stress':'Crop health',
    'pest_history':'Pest history','humidity':'Humidity',
    'temperature':'Temperature','ph':'Soil pH',
    'sulphur_ppm':'Sulphur','zinc_ppm':'Zinc',
}

@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    return (
        joblib.load(os.path.join(base,'models/spray_model.pkl')),
        joblib.load(os.path.join(base,'models/pest_model.pkl')),
        joblib.load(os.path.join(base,'models/le_spray.pkl')),
        joblib.load(os.path.join(base,'models/le_pest.pkl')),
    )

@st.cache_data(ttl=1800)
def get_weather(district):
    lat, lng = INDIA_DISTRICTS[district]
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat":lat,"lon":lng,"appid":OWM_API_KEY,"units":"metric"},timeout=5)
        d = r.json()
        if str(d.get("cod")) != "200":
            raise ValueError("OWM not ready")
        f = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat":lat,"lon":lng,"appid":OWM_API_KEY,"units":"metric","cnt":3},timeout=5)
        fc = f.json()
        pops = [i.get("pop",0) for i in fc.get("list",[])]
        mr = max(pops) if pops else 0.10
        return {"rain_prob":round(mr,2),"temp":round(d["main"]["temp"],1),
                "humidity":round(d["main"]["humidity"],1),
                "description":d["weather"][0]["description"].title(),
                "wind_kmh":round(d["wind"]["speed"]*3.6,1),
                "safe":mr<0.20,"source":"OpenWeatherMap"}
    except Exception:
        r2 = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude":lat,"longitude":lng,
                    "hourly":"precipitation_probability,temperature_2m,relative_humidity_2m",
                    "forecast_days":1,"timezone":"Asia/Kolkata"},timeout=5)
        d2 = r2.json()["hourly"]
        mr2 = max(d2["precipitation_probability"][:8])/100
        return {"rain_prob":round(mr2,2),"temp":round(d2["temperature_2m"][0],1),
                "humidity":round(d2["relative_humidity_2m"][0],1),
                "description":"Live forecast","wind_kmh":0,
                "safe":mr2<0.20,"source":"Open-Meteo"}


@st.cache_data(ttl=3600)
def get_forecast_7day(district):
    """Fetch 7-day hourly forecast and compute daily spray windows."""
    lat, lng = INDIA_DISTRICTS[district]
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lng,
                "daily": "precipitation_probability_max,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
                "hourly": "precipitation_probability,temperature_2m",
                "forecast_days": 7,
                "timezone": "Asia/Kolkata"
            }, timeout=5
        )
        d = r.json()
        daily = d.get("daily", {})
        dates       = daily.get("time", [])
        rain_probs  = daily.get("precipitation_probability_max", [])
        temp_max    = daily.get("temperature_2m_max", [])
        temp_min    = daily.get("temperature_2m_min", [])
        wind        = daily.get("windspeed_10m_max", [])

        days = []
        for i, dt in enumerate(dates):
            rain = rain_probs[i] if i < len(rain_probs) else 50
            tmax = temp_max[i]   if i < len(temp_max)  else 28
            tmin = temp_min[i]   if i < len(temp_min)  else 22
            wnd  = wind[i]       if i < len(wind)       else 10

            # Spray suitability logic
            if rain < 20 and wnd < 25:
                status = "GO"
                reason = "Clear — ideal spray conditions"
                color  = "#16a34a"
            elif rain < 35 and wnd < 30:
                status = "POSSIBLE"
                reason = "Monitor — check morning forecast"
                color  = "#F59E0B"
            else:
                status = "DELAY"
                reason = "Rain expected — do not spray"
                color  = "#DC2626"

            days.append({
                "date": dt, "rain": rain, "tmax": tmax,
                "tmin": tmin, "wind": wnd,
                "status": status, "reason": reason, "color": color
            })
        return days
    except Exception as e:
        return []

def get_crop_stage(dat):
    if dat<15:   return "Establishment"
    if dat<30:   return "Tillering"
    if dat<=35:  return "Active Tillering"
    if dat<=44:  return "Panicle Initiation"
    if dat<=55:  return "Pre-Flowering"
    if dat<=70:  return "Flowering"
    if dat<=100: return "Grain Filling"
    return "Ripening"

def soil_suitability(N, P, K, ph, crop="Paddy"):
    ideal = CROP_IDEAL[crop]
    scores = {}
    for feat, val, key in [("Nitrogen",N,"N"),("Phosphorus",P,"P"),("Potassium",K,"K"),("pH",ph,"ph")]:
        lo, hi = ideal[key]
        mid = (lo+hi)/2
        rng = max((hi-lo)/2, 1)
        scores[feat] = round(max(0, 1 - abs(val-mid)/rng) * 100)
    return round(np.mean(list(scores.values()))), scores

def nutrient_status(val, lo, hi):
    if val < lo: return "Low",  "n-low",  "#F59E0B"
    if val > hi: return "High", "n-high", "#DC2626"
    return "Good", "n-good", "#16a34a"


import sqlite3

def init_db():
    conn = sqlite3.connect('recommendations.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            district TEXT,
            crop TEXT,
            dat INTEGER,
            spray_action TEXT,
            dose REAL,
            pest_action TEXT,
            soil_score INTEGER,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def log_recommendation(district, crop, dat, spray_label, dose, pest_label, soil_score, confidence):
    from datetime import datetime
    conn = sqlite3.connect('recommendations.db')
    conn.execute("""
        INSERT INTO logs (timestamp, district, crop, dat, spray_action, dose, pest_action, soil_score, confidence)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M"), district, crop, dat,
          spray_label, dose, pest_label, soil_score, round(confidence,2)))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('recommendations.db')
    df = pd.read_sql("SELECT * FROM logs ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    return df

init_db()

spray_model, pest_model, le_spray, le_pest = load_models()

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="sidebar-brand-title">🌾 IFFCO Kisan AI</div><div class="sidebar-brand-sub">Precision Agronomy Recommender</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">📍 Location</div>', unsafe_allow_html=True)
    district = st.selectbox("District", list(INDIA_DISTRICTS.keys()), label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🌾 Crop & Stage</div>', unsafe_allow_html=True)
    crop = st.selectbox("Crop", ["Paddy","Wheat","Maize"], label_visibility="collapsed")
    transplant_date = st.date_input("Transplanting date", value=date.today()-timedelta(days=32), max_value=date.today())
    dat = max(0, (date.today() - transplant_date).days)
    st.markdown(f'<div style="font-size:11px;color:var(--green-300);margin-top:4px">📅 Day {dat} of crop cycle</div></div>', unsafe_allow_html=True)
    # Auto-populate soil values from district profile
    soil = DISTRICT_SOIL.get(district, {"N":80,"P":45,"K":40,"ph":6.5,"temp":27,"humidity":80,"rainfall":220,"pest":0.35,"ndvi":0.70})

    st.markdown('''<div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:8px 12px;margin:0 0 12px;font-size:11px;color:var(--green-300);border-left:2px solid var(--green-400)">
        📊 Soil values auto-filled from district average.<br>Adjust if you have your soil test card.
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🌱 Soil Nutrients</div>', unsafe_allow_html=True)
    N  = st.slider("Nitrogen (N) kg/ha",   40, 140, soil["N"])
    P  = st.slider("Phosphorus (P) kg/ha", 20, 70,  soil["P"])
    K  = st.slider("Potassium (K) kg/ha",  30, 55,  min(soil["K"], 55))
    ph = st.slider("Soil pH", 5.0, 8.5, soil["ph"], 0.1)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🌡️ Field Conditions</div>', unsafe_allow_html=True)
    temp      = st.slider("Temperature °C",        18, 40,  soil["temp"])
    humidity  = st.slider("Humidity %",            40, 95,  soil["humidity"])
    rainfall  = st.slider("Seasonal Rainfall mm", 100, 400, soil["rainfall"])
    ndvi      = st.slider("Crop health (0=stressed, 1=healthy)", 0.3, 1.0, soil["ndvi"], 0.05)
    pest_hist = st.slider("Pest pressure (0=low, 1=high)",       0.0, 1.0, soil["pest"], 0.05)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">⚗️ Micronutrients (ppm)</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        zinc    = st.number_input("Zinc",      0.1, 3.0,  0.55, 0.05)
        boron   = st.number_input("Boron",     0.1, 2.0,  0.42, 0.05)
        sulphur = st.number_input("Sulphur",   2.0, 30.0, 10.8, 0.5)
    with cb:
        iron   = st.number_input("Iron",       2.0, 25.0, 7.9,  0.5)
        mangan = st.number_input("Manganese",  0.5, 10.0, 3.5,  0.5)
        copper = st.number_input("Copper",     0.2, 4.0,  1.1,  0.1)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    lang    = st.toggle("🇮🇳 हिंदी में दिखाएं", value=False)
    get_rec = st.button("🔍 Get Field Recommendation", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

weather = get_weather(district)
stage_name = get_crop_stage(dat)
forecast_days = get_forecast_7day(district)

st.markdown(f"""
<div class="main-header">
    <div class="header-eyebrow">IFFCO · Precision Agronomy AI</div>
    <div class="header-title">Smart Farm <span>Advisor</span></div>
    <div class="header-subtitle">Real-time nano-fertiliser and pest risk recommendations for {district}</div>
    <div class="header-badges">
        <span class="header-badge">🌾 {crop}</span>
        <span class="header-badge">📅 Day {dat} · {stage_name}</span>
        <span class="header-badge">🌡️ {weather['temp']}°C · {weather['humidity']}% humidity</span>
        <span class="header-badge">{'✅ Safe to spray' if weather['safe'] else '⚠️ Rain risk'}</span>
    </div>
</div>
""", unsafe_allow_html=True)

safe_cls = "weather-safe" if weather['safe'] else "weather-risk"
safe_txt = "✅ Safe to spray today" if weather['safe'] else "⚠️ Rain expected — delay spray"
st.markdown(f"""
<div class="weather-strip">
    <div class="weather-item"><span class="label">Condition</span>&nbsp;{weather['description']}</div>
    <div class="weather-item"><span class="label">Temperature</span>&nbsp;{weather['temp']}°C</div>
    <div class="weather-item"><span class="label">Humidity</span>&nbsp;{weather['humidity']}%</div>
    <div class="weather-item"><span class="label">Wind</span>&nbsp;{weather['wind_kmh']} km/h</div>
    <div class="weather-item"><span class="label">Rain (8h)</span>&nbsp;{weather['rain_prob']*100:.0f}%</div>
    <div class="weather-item {safe_cls}">{safe_txt}</div>
    <div class="weather-source">via {weather['source']}</div>
</div>
""", unsafe_allow_html=True)

stages = [("Establish",0,15,"#94a3b8"),("Tillering",15,30,"#64748b"),
          ("Spray 1",30,36,"#16a34a"),("Panicle",36,45,"#64748b"),
          ("Spray 2",45,56,"#16a34a"),("Flowering",56,70,"#64748b"),
          ("Grain Fill",70,100,"#64748b"),("Harvest",100,121,"#94a3b8")]
cal_html = '<div class="cal-wrap"><div class="cal-row">'
lbl_html = '<div class="cal-labels">'
for name,s,e,color in stages:
    cur = s <= dat < e
    bd  = "3px solid white" if cur else "none"
    cal_html += f'<div class="cal-stage{"  current" if cur else ""}" style="background:{color};border:{bd};flex:{e-s}">{name}</div>'
    lbl_html += f'<div class="cal-label-item" style="flex:{e-s};{"color:#166534;font-weight:600" if cur else ""}">{s}–{e}</div>'
cal_html += '</div>'
lbl_html += '</div></div>'

st.markdown(f"""
<div class="content-area" style="padding-bottom:0">
    <div class="module-card">
        <div class="module-card-header">
            <div class="module-num">📅</div>
            <div><div class="module-title">Crop Growth Calendar</div>
            <div class="module-subtitle">Current stage: <strong>{stage_name}</strong> · Day {dat}</div></div>
        </div>
        <div class="module-body">{cal_html}{lbl_html}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="content-area" style="padding-top:0">', unsafe_allow_html=True)

# ── 7-Day Weather Forecast & Spray Predictor ─────────────────────────────────
if forecast_days:
    from datetime import datetime
    import plotly.graph_objects as go

    best_days     = [d for d in forecast_days if d["status"] == "GO"]
    possible_days = [d for d in forecast_days if d["status"] == "POSSIBLE"]

    st.markdown("""
    <div class="module-card">
        <div class="module-card-header">
            <div class="module-num">🌦️</div>
            <div>
                <div class="module-title">7-Day Rain Forecast & Spray Window Predictor</div>
                <div class="module-subtitle">Live forecast · Best spray days automatically detected</div>
            </div>
        </div>
    </div>
    """.replace("</div>\n    </div>", "</div></div>"), unsafe_allow_html=True)

    # Best spray days alert
    if best_days:
        best_dates = ", ".join([
            datetime.strptime(d["date"],"%Y-%m-%d").strftime("%a %d %b")
            for d in best_days[:3]
        ])
        st.success(f"✅ Best spray days this week: **{best_dates}** — rain below 20%, ideal for Nano Urea foliar spray")
    elif possible_days:
        pos = datetime.strptime(possible_days[0]["date"],"%Y-%m-%d").strftime("%a %d %b")
        st.warning(f"⚠️ No ideal spray days — possible window: **{pos}**. Monitor morning forecast before spraying.")
    else:
        st.error("🌧️ Rain expected all week — delay all foliar sprays. Check next week.")

    # Build plotly bar chart
    labels  = [datetime.strptime(d["date"],"%Y-%m-%d").strftime("%a\n%d %b") for d in forecast_days]
    rains   = [d["rain"] for d in forecast_days]
    colors  = [d["color"] for d in forecast_days]
    statuses= [d["status"] for d in forecast_days]
    tmaxs   = [d["tmax"] for d in forecast_days]
    tmins   = [d["tmin"] for d in forecast_days]
    texts   = [f"{r}%<br>{s}<br>{mx:.0f}°/{mn:.0f}°C"
               for r,s,mx,mn in zip(rains,statuses,tmaxs,tmins)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=rains,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{r}%" for r in rains],
        textposition="outside",
        textfont=dict(size=11, family="Sora"),
        hovertext=texts,
        hoverinfo="text",
        width=0.55,
    ))

    # Threshold line at 20%
    fig.add_hline(
        y=20,
        line_dash="dash",
        line_color="rgba(22,163,74,0.6)",
        line_width=1.5,
        annotation_text="Safe spray threshold (20%)",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="#16a34a",
    )

    fig.update_layout(
        height=280,
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,250,245,0.5)",
        font=dict(family="Sora", size=11, color="#0f1a12"),
        yaxis=dict(
            title=dict(text="Rain probability (%)", font=dict(size=10)),
            range=[0, 110],
            gridcolor="rgba(212,232,218,0.5)",
            tickfont=dict(size=10),
            zeroline=False,
        ),
        xaxis=dict(
            tickfont=dict(size=10),
            gridcolor="rgba(0,0,0,0)",
        ),
        bargap=0.3,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Spray calendar table
    cols = st.columns(len(forecast_days))
    for i, (col, day) in enumerate(zip(cols, forecast_days)):
        dt  = datetime.strptime(day["date"], "%Y-%m-%d")
        lbl = dt.strftime("%a")
        num = dt.strftime("%d")
        if day["status"] == "GO":
            col.markdown(f"""
            <div style="text-align:center;background:#f0fdf4;border:1px solid #86efac;
                        border-radius:10px;padding:8px 4px;">
                <div style="font-size:11px;font-weight:600;color:#166534">{lbl}</div>
                <div style="font-size:16px;font-weight:700;color:#166534">{num}</div>
                <div style="font-size:14px;margin:4px 0">✅</div>
                <div style="font-size:9px;font-weight:600;color:#16a34a">SPRAY</div>
            </div>""", unsafe_allow_html=True)
        elif day["status"] == "POSSIBLE":
            col.markdown(f"""
            <div style="text-align:center;background:#fffbeb;border:1px solid #fde68a;
                        border-radius:10px;padding:8px 4px;">
                <div style="font-size:11px;font-weight:600;color:#92400E">{lbl}</div>
                <div style="font-size:16px;font-weight:700;color:#92400E">{num}</div>
                <div style="font-size:14px;margin:4px 0">⚠️</div>
                <div style="font-size:9px;font-weight:600;color:#b45309">CHECK</div>
            </div>""", unsafe_allow_html=True)
        else:
            col.markdown(f"""
            <div style="text-align:center;background:#fef2f2;border:1px solid #fecaca;
                        border-radius:10px;padding:8px 4px;">
                <div style="font-size:11px;font-weight:600;color:#991b1b">{lbl}</div>
                <div style="font-size:16px;font-weight:700;color:#991b1b">{num}</div>
                <div style="font-size:14px;margin:4px 0">🌧️</div>
                <div style="font-size:9px;font-weight:600;color:#DC2626">DELAY</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

if get_rec:
    farmer = pd.DataFrame([{
        'N':N,'P':P,'K':K,'temperature':temp,'humidity':humidity,
        'ph':ph,'rainfall':rainfall,'days_after_transplanting':dat,
        'rain_prob_8h':weather['rain_prob'],'ndvi_stress':ndvi,
        'pest_history':pest_hist,'zinc_ppm':zinc,'boron_ppm':boron,
        'sulphur_ppm':sulphur,'iron_ppm':iron,'manganese_ppm':mangan,'copper_ppm':copper
    }])

    spray_pred  = spray_model.predict(farmer[FEATURE_COLS])[0]
    spray_proba = spray_model.predict_proba(farmer[FEATURE_COLS])[0]
    spray_label = le_spray.inverse_transform([spray_pred])[0]
    pest_pred   = pest_model.predict(farmer[PEST_FEATURE_COLS])[0]
    pest_proba  = pest_model.predict_proba(farmer[PEST_FEATURE_COLS])[0]
    pest_label  = le_pest.inverse_transform([pest_pred])[0]

    # Log recommendation to SQLite
    overall_score, _ = soil_suitability(N, P, K, ph, crop)
    log_recommendation(
        district, crop, dat, spray_label,
        4.0 if N < 60 else 3.0 if N < 80 else 2.0,
        pest_label, overall_score, float(max(spray_proba))
    )

    explainer     = shap.TreeExplainer(spray_model)
    shap_vals     = explainer.shap_values(farmer[FEATURE_COLS])
    shap_for_pred = shap_vals[0, :, spray_pred]
    top3 = pd.Series(shap_for_pred, index=FEATURE_COLS).abs().sort_values(ascending=False).head(3)

    dose = 4.0 if N < 60 else 3.0 if N < 80 else 2.0

    dec_cls  = {'Spray':'spray','Delay':'delay','Basal Only':'basal'}.get(spray_label,'basal')
    dec_icon = {'Spray':'💧','Delay':'⏳','Basal Only':'🌱'}.get(spray_label,'🌱')
    dec_det  = {
        'Spray':      f'Apply {dose} ml Nano Urea Plus per litre of water · foliar spray',
        'Delay':      f'Rain probability {weather["rain_prob"]*100:.0f}% — wait 8 hours and recheck',
        'Basal Only': f'Crop is at {stage_name} — outside the IFFCO spray window'
    }.get(spray_label,'')

    st.markdown(f"""
    <div class="module-card">
        <div class="module-card-header">
            <div class="module-num">1</div>
            <div><div class="module-title">Module 1 — Nano-Fertiliser Recommendation</div>
            <div class="module-subtitle">XGBoost · {max(spray_proba):.0%} confidence</div></div>
        </div>
        <div class="module-body">
            <div class="decision-hero {dec_cls}">
                <div class="decision-icon">{dec_icon}</div>
                <div class="decision-text">
                    <div class="decision-action">{'Spray today — ' + str(dose) + ' ml/L' if spray_label=='Spray' else spray_label}</div>
                    <div class="decision-detail">{dec_det}</div>
                </div>
                <div class="decision-confidence">{max(spray_proba):.0%}</div>
            </div>
    """, unsafe_allow_html=True)

    if spray_label == 'Spray':
        saving = round(max(0, 266 - dose * 0.5 * 150 * 0.34))
        st.markdown(f"""
            <div class="saving-hero">
                <div>
                    <div class="saving-amount">₹{saving}</div>
                    <div class="saving-label">saved per acre this spray</div>
                </div>
                <div style="border-left:1px solid #86efac;padding-left:14px">
                    <div class="saving-detail">Conventional urea bag ≈ ₹266/acre</div>
                    <div class="saving-detail">Nano Urea {dose}ml/L ≈ ₹{266-saving}/acre</div>
                    <div class="saving-detail" style="margin-top:4px;color:#166534;font-weight:600">
                        Replaces 1 full 45kg urea bag with {dose*0.5:.0f}ml of Nano Urea Plus
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    shap_rows = ""
    for feat, val in top3.items():
        nice = LABEL_MAP.get(feat, feat.replace('_',' ').title())
        w    = min(int(abs(val)*60), 100)
        direction = "↑ Increased spray confidence" if shap_for_pred[FEATURE_COLS.index(feat)] > 0 else "↓ Reduced spray confidence"
        shap_rows += f'<div class="shap-row"><div class="shap-name">{nice}</div><div class="shap-track"><div class="shap-fill" style="width:{w}%"></div></div><div class="shap-dir">{direction[:2]}</div></div>'

    st.markdown(f"""
            <div class="shap-section">
                <div class="shap-label">Why this recommendation?</div>
                {shap_rows}
            </div>
    """, unsafe_allow_html=True)

    if lang:
        hindi_spray = HINDI.get(spray_label, '')
        st.markdown(f"""
            <div class="hindi-box">
                <div class="hindi-toggle-label">हिंदी में सलाह</div>
                <div class="hindi-text">{hindi_spray} — {dose} मिलीलीटर नैनो यूरिया प्रति लीटर पानी।</div>
                <div class="hindi-sub">मौसम {'साफ है' if weather['safe'] else 'खराब है — प्रतीक्षा करें'}।</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
            <div class="rule-box">
                <div class="rule-icon">⚠️</div>
                <div>{NANO_UREA_SPECS['basal_rule']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pest_cls  = {'Pesticide':'pesticide','Bio Control':'bio','Monitor':'monitor'}.get(pest_label,'monitor')
    pest_icon = {'Pesticide':'🚨','Bio Control':'🌿','Monitor':'👀'}.get(pest_label,'👀')
    pest_det  = {
        'Pesticide':  'Risk above 35% — contact IFFCO agronomist for pesticide guidance',
        'Bio Control':'Use neem-based spray (5% NSKE) · Trichoderma soil application',
        'Monitor':    'Low outbreak risk — check field every 3 days'
    }.get(pest_label,'')

    st.markdown(f"""
    <div class="module-card">
        <div class="module-card-header">
            <div class="module-num">2</div>
            <div><div class="module-title">Module 2 — Pest Risk Advisory</div>
            <div class="module-subtitle">LightGBM · {max(pest_proba):.0%} confidence · Outbreak probability: {pest_hist*100:.0f}%</div></div>
        </div>
        <div class="module-body">
            <div class="pest-hero {pest_cls}">
                <div class="pest-icon-large">{pest_icon}</div>
                <div>
                    <div class="pest-action">{pest_label}</div>
                    <div class="pest-detail">{pest_det}</div>
                </div>
                <div class="decision-confidence" style="margin-left:auto">{max(pest_proba):.0%}</div>
            </div>
    """, unsafe_allow_html=True)

    if lang:
        st.markdown(f"""
            <div class="hindi-box">
                <div class="hindi-toggle-label">हिंदी में सलाह</div>
                <div class="hindi-text">{HINDI.get(pest_label,'')}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="module-card">
        <div class="module-body">
            <div class="empty-state">
                <div class="empty-icon">🌾</div>
                <div class="empty-title">Ready for your field data</div>
                <div class="empty-sub">Adjust the parameters in the sidebar and click<br><strong>Get Field Recommendation</strong> to see your results.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if get_rec:
    overall, scores = soil_suitability(N, P, K, ph, crop)

    verdict = "Excellent" if overall>=85 else "Good" if overall>=70 else "Moderate" if overall>=55 else "Poor"
    verdict_color = "#166534" if overall>=85 else "#1D6A3E" if overall>=70 else "#92400E" if overall>=55 else "#DC2626"

    crop_suits = {}
    for c in CROP_IDEAL:
        s, _ = soil_suitability(N, P, K, ph, c)
        crop_suits[c] = s
    crop_suits = dict(sorted(crop_suits.items(), key=lambda x: x[1], reverse=True))

    def suit_pill(s):
        if s>=85: return "Excellent","suit-exc"
        if s>=70: return "Good","suit-good"
        if s>=55: return "Moderate","suit-med"
        return "Low","suit-low"

    ideal_p = CROP_IDEAL[crop]
    def make_bar(val, lo, hi, max_val):
        pct = min(val/max_val*100, 100)
        ideal_pct = min((lo+hi)/2/max_val*100, 100)
        status, badge_cls, color = nutrient_status(val, lo, hi)
        return (f'<div class="nutrient-item"><div class="nutrient-header">' +
                f'<span class="nutrient-name">{val:.1f}</span>' +
                f'<span class="nutrient-badge {badge_cls}">{status}</span></div>' +
                f'<div class="nutrient-track-outer">' +
                f'<div class="nutrient-bar" style="width:{pct:.0f}%;background:{color}"></div>' +
                f'<div class="ideal-line" style="left:{ideal_pct:.0f}%"></div></div>' +
                f'<div style="display:flex;justify-content:space-between;margin-top:2px">' +
                f'<span style="font-size:9px;color:var(--muted)">0</span>' +
                f'<span style="font-size:9px;color:var(--muted)">Ideal: {lo}-{hi}</span>' +
                f'<span style="font-size:9px;color:var(--muted)">{max_val}</span></div></div>')

    table_rows = ""
    for c, s in crop_suits.items():
        lbl, pill_cls = suit_pill(s)
        cur = ' class="current-crop"' if c == crop else ''
        icon = {"Paddy":"🌾","Wheat":"🌿","Maize":"🌽","Mustard":"🌱","Soybean":"🫘"}.get(c,"🌾")
        bar_w = s
        bar_color = "#16a34a" if s>=70 else "#F59E0B" if s>=55 else "#DC2626"
        table_rows += f"""<tr{cur}>
            <td>{icon} {'<strong>' if c==crop else ''}{c}{'</strong>' if c==crop else ''}</td>
            <td><div style="height:6px;width:{bar_w}px;background:{bar_color};border-radius:3px;max-width:100px"></div></td>
            <td><strong>{s}%</strong></td>
            <td><span class="suit-pill {pill_cls}">{lbl}</span></td>
        </tr>"""

    advice_map = {
        "Paddy":   "Add DAP if phosphorus is low. Ensure proper water management for optimal yield.",
        "Wheat":   "Your soil suits wheat well for rabi season. Consider wheat as next crop.",
        "Maize":   "Moderate suitability. Improve phosphorus levels before sowing.",
        "Mustard": "Moderate. Ensure good drainage and adequate sulphur application.",
        "Soybean": "Low suitability. Soil pH may need adjustment for soybean cultivation.",
    }

    # Build module 3 in pieces to avoid f-string rendering issues
    n_bar = make_bar(N, *ideal_p['N'], 160)
    p_bar = make_bar(P, *ideal_p['P'], 80)
    k_bar = make_bar(K, *ideal_p['K'], 60)
    ph_bar = make_bar(ph, *ideal_p['ph'], 10)
    dash = round(overall * 2.39)

    st.markdown(f'''
    <div class="module-card">
        <div class="module-card-header">
            <div class="module-num">3</div>
            <div><div class="module-title">Module 3 — Soil Suitability & Crop Advisory</div>
            <div class="module-subtitle">ICAR norms for {district} · Based on your soil profile</div></div>
        </div>
        <div class="module-body">
            <div class="suitability-score-wrap">
                <svg width="90" height="90" viewBox="0 0 90 90">
                    <circle cx="45" cy="45" r="38" fill="none" stroke="#e0f5eb" stroke-width="8"/>
                    <circle cx="45" cy="45" r="38" fill="none" stroke="{verdict_color}" stroke-width="8"
                        stroke-dasharray="{dash} 239" stroke-linecap="round"
                        transform="rotate(-90 45 45)"/>
                    <text x="45" y="42" text-anchor="middle" font-size="20" font-weight="700" fill="{verdict_color}" font-family="Sora">{overall}%</text>
                    <text x="45" y="57" text-anchor="middle" font-size="10" fill="#5a7a62" font-family="Sora">suitable</text>
                </svg>
                <div>
                    <div class="suit-verdict" style="color:{verdict_color}">{verdict} suitability for {crop}</div>
                    <div class="suit-advice">{advice_map.get(crop,"")}</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
                <div>
                    <div style="font-size:10px;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:10px">Nitrogen (kg/ha)</div>
                    {n_bar}
                    <div style="font-size:10px;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:10px;margin-top:8px">Phosphorus (kg/ha)</div>
                    {p_bar}
                </div>
                <div>
                    <div style="font-size:10px;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:10px">Potassium (kg/ha)</div>
                    {k_bar}
                    <div style="font-size:10px;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:10px;margin-top:8px">Soil pH</div>
                    {ph_bar}
                </div>
            </div>
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:8px">Crop suitability comparison</div>
            <table class="crop-table">
                <tr><th>Crop</th><th>Suitability</th><th>Score</th><th>Status</th></tr>
                {table_rows}
            </table>
        </div>
    </div>
    ''', unsafe_allow_html=True)


# ── Recommendation History ────────────────────────────────────────────────────
st.markdown("""
<div class="module-card" style="margin-top:20px">
    <div class="module-card-header">
        <div class="module-num">📋</div>
        <div>
            <div class="module-title">Recommendation History</div>
            <div class="module-subtitle">Last 20 recommendations logged this session</div>
        </div>
    </div>
</div>
""".replace("</div>\n</div>","</div></div>"), unsafe_allow_html=True)

history_df = get_history()
if len(history_df) > 0:
    history_df = history_df.rename(columns={
        "timestamp":"Time","district":"District","crop":"Crop",
        "dat":"DAT","spray_action":"Spray","dose":"Dose ml/L",
        "pest_action":"Pest","soil_score":"Soil %","confidence":"Confidence"
    })
    history_df = history_df.drop(columns=["id"], errors="ignore")
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No recommendations yet — click Get Recommendation to start logging.")

st.markdown("""
<div style="text-align:center;padding:20px 40px;font-size:11px;color:var(--muted);border-top:1px solid var(--border);margin-top:8px">
    IFFCO Kisan Precision Agronomy Recommender &nbsp;·&nbsp;
    Kartikey Negi — Amity University &nbsp;·&nbsp;
    Mentor: Mr. Mayur Gupta &nbsp;·&nbsp; 2026
</div>
</div>
""", unsafe_allow_html=True)
