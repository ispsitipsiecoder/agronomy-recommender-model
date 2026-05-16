# 🌾 Precision Agronomy Recommender
### Hyper-Localized AI Advisory for Nano-Fertilizer Optimization & Low-Pesticide Agriculture

> **Intern Project — IFFCO x Data Science**  
> Built as part of a data science internship to bridge the behavioral gap between traditional farming and nano-fertilizer technology.

---

## 🎯 What This Does

This AI recommendation engine acts as a digital agronomist for individual farm plots. It combines soil health data, real-time weather, satellite imagery, and crop stage intelligence to give farmers two critical, hyper-local recommendations:

**Module 1 — Nano-Fertilizer Optimizer**  
Tells the farmer exactly when and how much Nano Urea Plus / Nano DAP to spray, based on their soil NPK levels, crop growth stage (Days After Transplanting), and an 8-hour rain-free window check.

**Module 2 — Low-Pesticide Pest Risk Advisor**  
Predicts outbreak probability using district pest history (KCC data) and satellite NDVI crop stress scores. Only recommends chemical pesticides when risk > 35% — otherwise recommends biological controls.

---

## 💡 Problem Statement

Indian smallholder farmers face a dual chemical overuse crisis:
- Indiscriminate broadcasting of conventional urea degrades soil and costs ₹3,000–5,000/acre in inputs
- Blanket pesticide schedules waste 30–40% of spend and build pest resistance

The root cause: **no hyper-local, real-time, crop-stage-aware decision support exists at the farm level.**

---

## 🏗️ Architecture

```
Data Sources                 AI Engine                    Delivery
─────────────────────────    ──────────────────────────   ──────────────────
Soil Health Cards (SHC)  ──▶                             
IMD Weather API          ──▶  Module 1: XGBoost          
KCC Pest Data (NCIPM)   ──▶  + SHAP Explainability  ──▶  FastAPI ──▶ IFFCO Kisan App
Satellite NDVI           ──▶                              (11 languages)
GPS Field Mapping        ──▶  Module 2: LightGBM         
                              + Pest Risk Threshold   
```

---

## 📊 Expected Impact

| Metric | Conventional | This System | Improvement |
|--------|-------------|-------------|-------------|
| Nitrogen Use Efficiency | 30–40% | 85–90% | +50% |
| Input Cost / Acre | ₹3,000–5,000 | ₹2,300–3,800 | 20–25% saving |
| N₂O Emissions | Baseline | — | 45% reduction |
| Pesticide Spend | Baseline | — | 30–40% saving |
| Application Accuracy | Manual broadcast | GPS-guided spray | Eliminates overuse |

---

## 📁 Project Structure

```
precision-agronomy-recommender/
├── data/
│   ├── raw/              # Unmodified source datasets (gitignored)
│   │   ├── README.md     # Download instructions for all datasets
│   │   └── iffco_nano_urea_specs.json
│   └── processed/        # ETL pipeline outputs
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Phase 1 ← START HERE
│   ├── 02_etl_pipeline.ipynb       # Phase 2
│   └── 03_model_training.ipynb     # Phase 3
├── src/
│   ├── config.py          # Rules engine — IFFCO hard constraints
│   ├── feature_schema.py  # Data contracts for entire pipeline
│   ├── etl.py             # Phase 2
│   └── models.py          # Phase 3
├── api/                   # FastAPI app — Phase 4
├── requirements.txt
└── .env                   # API keys (never commit this)
```

---

## 🚀 Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/precision-agronomy-recommender.git
cd precision-agronomy-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Download datasets (see data/raw/README.md for instructions)

# 5. Run Phase 1 notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 📦 Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| Crop Recommendation (NPK) | [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | Base soil-crop training data |
| Soil Health Cards (SHC) | [soilhealth.dac.gov.in](https://soilhealth.dac.gov.in) | District-level soil micronutrients |
| KCC Pest Queries | [KCC-CHAKSHU (ICAR-IASRI)](https://kcc-chakshu.icar-web.com) | District pest outbreak history |
| Weather Forecast | [Open-Meteo](https://open-meteo.com) (dev) / IMD (prod) | 8-hour rain-free spray window |
| Satellite NDVI | NASA MODIS / Sentinel Hub | Crop stress detection |

---

## 🔬 Tech Stack

- **ML Models:** XGBoost (fertilizer), LightGBM (pest risk)
- **Explainability:** SHAP values
- **Geospatial:** GeoPandas
- **API:** FastAPI + Uvicorn
- **Cloud:** Oracle Cloud Infrastructure (OCI)
- **Mobile:** TensorFlow Lite (offline inference)
- **Chatbot:** Rasa / Dialogflow (Hindi + English + 9 regional languages)

---

## 🗺️ Roadmap

| Phase | Weeks | Status |
|-------|-------|--------|
| Phase 1: Foundation & Data Scoping | 1–3 | 🟡 In Progress |
| Phase 2: Data Kitchen (ETL Pipeline) | 4–7 | ⬜ Planned |
| Phase 3: Model Development + SHAP | 8–12 | ⬜ Planned |
| Phase 4: Cloud Backend + FastAPI | 13–16 | ⬜ Planned |
| Phase 5: Chatbot Integration | 17–20 | ⬜ Planned |
| Phase 6: Pilot A/B Test | 21+ | ⬜ Planned |

---

## 📝 License
Built for internship and academic purposes. IFFCO product data sourced from publicly available FCO-approved specifications.
