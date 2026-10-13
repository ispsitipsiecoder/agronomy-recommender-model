# Precision Agronomy Recommender

A tool that tells a farmer *when* to spray nano-fertilizer, *how much*, and whether a pest problem is serious enough to justify chemical pesticides at all.

I built this during my data science internship (IFFCO x Data Science). It's still a work in progress, and I've tried to be upfront below about what works and what doesn't yet.

## Why I built this

Most smallholder farmers in India apply urea by broadcasting it across the field and spray pesticides on a fixed calendar, whether or not pests are actually around. That costs money, wears down the soil, and helps pests build resistance. Nano-fertilizers can fix a lot of this, but only if they're applied at the right growth stage and not right before rain. Farmers rarely have anyone telling them that for their own plot.

I wanted to see if a model could fill that gap.

## What it does

**1. Fertilizer advice (XGBoost).** Given the soil's NPK levels, how many days since transplanting, and the weather forecast, it recommends how much Nano Urea Plus or Nano DAP to apply, and checks that there's an 8-hour rain-free window so the spray isn't wasted.

**2. Pest risk (LightGBM).** Using past pest reports for the district (from KCC query data) and satellite crop-stress scores (NDVI), it estimates how likely an outbreak is. Above 35% risk it suggests chemical control. Below that, it points to biological options first.

I used SHAP so each recommendation comes with a reason. A farmer or an agronomist is far more likely to trust "your nitrogen is low and the crop is at tillering" than a bare number.

## Where things stand

| Part | Status |
|---|---|
| Data exploration | Done |
| ETL pipeline | CONFIRM |
| Fertilizer model | CONFIRM |
| Pest risk model | CONFIRM |
| FastAPI backend (`app.py`) | CONFIRM |
| Cloud deployment | Not yet |
| Multilingual chatbot, offline mobile mode | Not yet |
| Field pilot | Not yet |

*(Edit the CONFIRM rows to Done, In progress, or Not yet before publishing.)*

### Results

| Model | Data | Metric | Score |
|---|---|---|---|
| Fertilizer (XGBoost) | [rows] | [metric] | [value] |
| Pest risk (LightGBM) | [rows] | [metric] | [value] |

*(Fill in from your training notebook, or delete this table if the models aren't trained yet.)*

### A note on the impact numbers

Nano-fertilizer research suggests big gains in nitrogen use efficiency and 20-25% lower input costs. Those are the goals this project is designed around. **I haven't measured them**, since that would need a real field trial.

## How it fits together

```
Soil Health Cards ─┐
Weather (Open-Meteo) ─┤
KCC pest data ────────┼──> XGBoost (fertilizer) ─┐
Satellite NDVI ───────┤    LightGBM (pest risk) ─┼──> FastAPI ──> SQLite log
Crop NPK dataset ─────┘    SHAP explanations ────┘
```

## Running it yourself

```bash
git clone https://github.com/ispsitipsiecoder/agronomy-recommender-model.git
cd agronomy-recommender-model

python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # add your own API keys
```

Download the datasets using the instructions in `data/raw/README.md`, then start with `notebooks/01_data_exploration.ipynb`. To try the API:

```bash
uvicorn app:app --reload
```

and open http://127.0.0.1:8000/docs.

## What's in the repo

- `notebooks/` holds the exploration, ETL, and model training notebooks, in order.
- `src/` has the config (IFFCO product rules), the feature schema, and the ETL and model code.
- `models/` stores trained model files.
- `app.py` is the API.
- `data/` is where datasets go. Raw data isn't committed.

## Data sources

- [Crop Recommendation dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) (Kaggle) for base soil and crop data
- [Soil Health Cards](https://soilhealth.dac.gov.in) for district soil micronutrients
- [KCC-CHAKSHU](https://kcc-chakshu.icar-web.com) for pest query history
- [Open-Meteo](https://open-meteo.com) for weather forecasts
- NASA MODIS / Sentinel Hub for NDVI

## Tech

Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, GeoPandas, FastAPI, SQLite.

## What I'd do next

1. Finish and validate both models on held-out data.
2. Deploy the API to the cloud.
3. Add a simple Hindi and English chatbot front end.
4. Test with real farm plots, which is the only way to know if the recommendations help.

## About

Kartikey Negi, B.Tech CSE (Data Science), Amity University.
[LinkedIn](https://linkedin.com/in/kartikey-negi-7b265a279) | kartikeynegii10@gmail.com

Built for internship and academic purposes. IFFCO product details come from publicly available FCO-approved specifications.
