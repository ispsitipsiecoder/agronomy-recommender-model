# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# RULES ENGINE — All hard constraints sourced from IFFCO FCO-approved spec
# sheets. These values are NOT tunable by the ML model. They are guardrails.
# ─────────────────────────────────────────────────────────────────────────────

# ── Pilot Crop ────────────────────────────────────────────────────────────────
PILOT_CROP = "paddy"

# ── IFFCO Nano Urea Plus — FCO Approved Specifications ───────────────────────
NANO_UREA_SPECS = {
    "product_name": "Nano Urea Plus",
    "manufacturer": "IFFCO",
    "nitrogen_content_pct": 20,          # 20% Nitrogen (w/v)
    "particle_size_nm": 20,              # <100nm, typically 20–50nm
    "formulation": "liquid",

    # Hard dosage limits — model must NEVER go outside these bounds
    "min_dose_ml_per_litre": 2.0,
    "max_dose_ml_per_litre": 4.0,

    # Spray timing windows per crop (Days After Transplanting / Sowing)
    "application_windows": {
        "paddy": [
            {
                "spray_number": 1,
                "stage": "active_tillering",
                "dat_min": 30,           # Days After Transplanting
                "dat_max": 35,
                "description": "First spray at active tillering stage"
            },
            {
                "spray_number": 2,
                "stage": "pre_flowering",
                "dat_min": 45,
                "dat_max": 55,
                "description": "Second spray at pre-flowering / panicle initiation"
            },
        ],
        "wheat": [
            {
                "spray_number": 1,
                "stage": "crown_root_initiation",
                "das_min": 21,           # Days After Sowing
                "das_max": 25,
                "description": "First spray at crown root initiation"
            },
            {
                "spray_number": 2,
                "stage": "flag_leaf",
                "das_min": 45,
                "das_max": 50,
                "description": "Second spray at flag leaf emergence"
            },
        ]
    },

    # Environmental impact vs conventional urea (for report metrics)
    "emission_reduction": {
        "nitrous_oxide_pct": 45,         # 45% N₂O reduction vs conventional
        "ammonia_volatilization_pct": 70  # 70% NH₃ reduction vs conventional
    },

    # CRITICAL AGRONOMIC RULE — hardcoded, not ML-tunable
    "basal_rule": (
        "Nano Urea Plus ONLY replaces top-dressed urea. "
        "It does NOT replace basal nitrogen applied at sowing. "
        "Farmers must still apply recommended basal fertilizer at transplanting."
    ),
}

# ── IFFCO Nano DAP — FCO Approved Specifications ─────────────────────────────
NANO_DAP_SPECS = {
    "product_name": "Nano DAP",
    "manufacturer": "IFFCO",
    "phosphorus_content_pct": 8,         # 8% P₂O₅ (w/v)
    "nitrogen_content_pct": 8,           # 8% N (w/v)
    "particle_size_nm": 100,             # <100nm
    "formulation": "liquid",

    "min_dose_ml_per_litre": 2.0,
    "max_dose_ml_per_litre": 4.0,

    "application_windows": {
        "paddy": [
            {
                "spray_number": 1,
                "stage": "tillering",
                "dat_min": 20,
                "dat_max": 30,
            },
            {
                "spray_number": 2,
                "stage": "panicle_initiation",
                "dat_min": 40,
                "dat_max": 50,
            }
        ]
    }
}

# ── Weather Constraints ───────────────────────────────────────────────────────
# Rain-free window required AFTER foliar spray for absorption
RAIN_FREE_WINDOW_HOURS = 8

# If rain probability exceeds this threshold in next 8h → DELAY SPRAY
RAIN_PROBABILITY_THRESHOLD = 0.20       # 20% chance of rain = delay

# ── Pest Risk Thresholds (Module 2) ──────────────────────────────────────────
# Above this → recommend chemical pesticide intervention
PEST_RISK_THRESHOLD_HIGH = 0.35         # 35% outbreak probability

# Below this → recommend biological/cultural controls only
PEST_RISK_THRESHOLD_LOW = 0.15          # 15% — monitor only

# Biological control options ordered by effectiveness for paddy
BIO_CONTROL_OPTIONS = {
    "paddy": [
        "Neem-based spray — 5% NSKE (Neem Seed Kernel Extract)",
        "Trichoderma viride soil application (2.5 kg/ha)",
        "Pheromone traps for Yellow Stem Borer (5 traps/ha)",
        "Yellow sticky traps for BPH (Brown Plant Hopper) monitoring",
        "Light traps at 1 per 2 acres during evening hours",
    ]
}

# ── Soil Nutrient Benchmark Ranges for Paddy (kg/ha) ─────────────────────────
# Used by SHAP to explain why a recommendation was made
SOIL_BENCHMARKS_PADDY = {
    "nitrogen": {"low": 0, "medium": 280, "high": 560},       # kg/ha
    "phosphorus": {"low": 0, "medium": 12, "high": 25},       # kg/ha
    "potassium": {"low": 0, "medium": 110, "high": 280},      # kg/ha
    "ph": {"acidic": 5.5, "optimal_min": 5.5, "optimal_max": 7.0, "alkaline": 7.0},
    "organic_carbon": {"low": 0, "medium": 0.5, "high": 0.75}  # %
}

# ── External API Endpoints ────────────────────────────────────────────────────
# Open-Meteo (no API key — use for dev and Phase 1/2)
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# IMD official (requires IP whitelisting — use for production Phase 4+)
IMD_API_BASE = "https://api.imd.gov.in"

# KCC-CHAKSHU pest data portal
NCIPM_PORTAL = "https://kcc-chakshu.icar-web.com"

# ── Pilot Geography ───────────────────────────────────────────────────────────
PILOT_DISTRICT = "Gautam Buddha Nagar"
PILOT_STATE = "Uttar Pradesh"
PILOT_LAT = 28.4595
PILOT_LNG = 77.5022

# ── Oracle Cloud Config (loaded from .env in production) ─────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

OCI_DB_NAME     = os.getenv("OCI_DB_NAME", "agronomy_dev_db")
OCI_DB_USER     = os.getenv("OCI_DB_USER", "")
OCI_DB_PASSWORD = os.getenv("OCI_DB_PASSWORD", "")
IMD_API_KEY     = os.getenv("IMD_API_KEY", "")
