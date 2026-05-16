# src/feature_schema.py
# ─────────────────────────────────────────────────────────────────────────────
# Data contracts for the entire pipeline.
# Every module (ETL, model, API, chatbot) imports from here.
# Changing a field here = changing it everywhere. Single source of truth.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class CropStage(str, Enum):
    """Growth stages for Paddy — determines which spray window is active."""
    PRE_TRANSPLANT      = "pre_transplant"       # < Day 0
    ESTABLISHMENT       = "establishment"         # Day 0–15
    TILLERING           = "tillering"             # Day 15–35  ← Spray 1 window
    ACTIVE_TILLERING    = "active_tillering"      # Day 30–35  ← Nano Urea Spray 1
    PANICLE_INITIATION  = "panicle_initiation"    # Day 35–50
    PRE_FLOWERING       = "pre_flowering"         # Day 45–55  ← Nano Urea Spray 2
    FLOWERING           = "flowering"             # Day 55–70
    GRAIN_FILLING       = "grain_filling"         # Day 70–100
    RIPENING            = "ripening"              # Day 100–120
    HARVEST             = "harvest"               # Day 120+


class SprayAction(str, Enum):
    """Module 1 output — what to do with fertilizer today."""
    SPRAY           = "Spray"           # Conditions met — spray now
    DELAY           = "Delay"           # Rain risk — wait
    BASAL_ONLY      = "Basal Only"      # Not in spray window — no foliar needed
    NOT_APPLICABLE  = "Not Applicable"  # Crop stage not reached yet


class PestIntervention(str, Enum):
    """Module 2 output — level of pest response needed."""
    MONITOR         = "Monitor"         # < 15% risk — observe only
    BIO_CONTROL     = "Bio Control"     # 15–35% risk — biological measures
    PESTICIDE       = "Pesticide"       # > 35% risk — chemical intervention
    URGENT          = "Urgent"          # > 70% risk — immediate action


class NutrientLevel(str, Enum):
    """Soil nutrient status classification — used in SHAP explanations."""
    LOW             = "Low"
    MEDIUM          = "Medium"
    HIGH            = "High"
    DEFICIENT       = "Deficient"       # Below critical threshold


# ── Input Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class SoilProfile:
    """
    Sourced from Soil Health Card (SHC).
    Major nutrients are required. Micronutrients may be missing → KNN imputed in ETL.
    All units in kg/ha unless noted.
    """
    # Major nutrients (required — always on SHC)
    nitrogen_kg_ha: float
    phosphorus_kg_ha: float
    potassium_kg_ha: float
    ph: float                               # 1.0 – 14.0
    organic_carbon_pct: float              # percentage

    # Micronutrients (optional — often missing, imputed in Phase 2 ETL)
    zinc_ppm: Optional[float]     = None
    boron_ppm: Optional[float]    = None
    sulphur_ppm: Optional[float]  = None
    iron_ppm: Optional[float]     = None
    manganese_ppm: Optional[float] = None
    copper_ppm: Optional[float]   = None

    # Metadata
    shc_card_id: Optional[str]    = None   # Government SHC card number
    sample_date: Optional[str]    = None   # ISO date of soil sampling

    def has_micronutrients(self) -> bool:
        """Returns True if any micronutrient data is available."""
        return any([
            self.zinc_ppm, self.boron_ppm, self.sulphur_ppm,
            self.iron_ppm, self.manganese_ppm, self.copper_ppm
        ])


@dataclass
class WeatherSnapshot:
    """
    Sourced from Open-Meteo (dev) or IMD API (production).
    Captured at recommendation time for the field's exact GPS location.
    """
    # Core spray-decision fields
    rain_probability_next_8h_max: float     # 0.0 – 1.0 (max across 8 hourly readings)
    rain_probability_next_24h_max: float    # 0.0 – 1.0

    # Environmental conditions
    temperature_celsius: float
    humidity_pct: float                     # Relative humidity %
    wind_speed_kmh: float

    # Raw hourly readings (for the 8h window logic)
    hourly_rain_prob_8h: List[float] = field(default_factory=list)   # 8 values

    # Metadata
    fetched_at: Optional[str] = None        # ISO datetime of API call
    data_source: str = "open-meteo"         # "open-meteo" or "imd"

    def is_safe_to_spray(self, threshold: float = 0.20) -> bool:
        """Core spray-window check — returns True if rain risk is acceptable."""
        return self.rain_probability_next_8h_max < threshold


@dataclass
class FieldProfile:
    """
    The unified object representing ONE farmer's field.
    This is the single input to both ML modules.
    Built by the ETL pipeline, stored in Oracle DB, passed to the API.
    """
    # Identity
    farmer_id: str
    field_id: str

    # Location
    gps_lat: float
    gps_lng: float
    district: str
    state: str
    village: Optional[str] = None

    # Crop info
    crop: str = "paddy"
    variety: Optional[str] = None          # e.g. "IR-36", "Swarna"
    sowing_date: Optional[str] = None      # ISO date
    transplanting_date: Optional[str] = None

    # Derived crop stage (computed by ETL from transplanting_date)
    days_after_transplanting: Optional[int] = None
    growth_stage: Optional[CropStage] = None

    # Soil and weather (nested)
    soil: Optional[SoilProfile] = None
    weather: Optional[WeatherSnapshot] = None

    # NDVI crop stress index — from satellite imagery
    # 0.0 = severely stressed, 1.0 = healthy
    ndvi_stress_score: Optional[float] = None

    # Pest context
    district_pest_history_score: Optional[float] = None   # 0.0 – 1.0, from KCC data
    current_pest_alerts: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if field profile has minimum data for recommendations."""
        return all([
            self.soil is not None,
            self.weather is not None,
            self.days_after_transplanting is not None,
        ])


# ── Output Dataclasses ────────────────────────────────────────────────────────

@dataclass
class FertilizerRecommendation:
    """
    Output of Module 1 — Nano-fertilizer optimizer.
    Fed into the chatbot NLU layer for natural language delivery.
    """
    # Core recommendation
    spray_action: SprayAction
    nano_urea_dose_ml_per_litre: float      # 0 if not applicable
    nano_dap_dose_ml_per_litre: float       # 0 if not applicable

    # Spray window logic
    is_safe_to_spray: bool
    delay_reason: Optional[str] = None      # Why spray was delayed

    # Explainability (SHAP output — Phase 3)
    # Dict mapping feature name → contribution to recommendation
    shap_values: dict = field(default_factory=dict)

    # Human-readable explanations (localized in Phase 5)
    reason_en: str = ""                     # English explanation
    reason_hi: str = ""                     # Hindi explanation

    # Cost estimate
    estimated_cost_inr_per_acre: Optional[float] = None
    cost_saving_vs_conventional_pct: Optional[float] = None


@dataclass
class PestRiskRecommendation:
    """
    Output of Module 2 — Pest risk advisor.
    """
    # Core recommendation
    pest_risk_probability: float            # 0.0 – 1.0
    intervention: PestIntervention

    # Specific pest identified (if known)
    likely_pest: Optional[str] = None       # e.g. "Yellow Stem Borer"

    # What to do
    bio_control_options: List[str] = field(default_factory=list)
    pesticide_recommendation: Optional[str] = None   # Only if intervention == PESTICIDE

    # Explainability
    shap_values: dict = field(default_factory=dict)
    reason_en: str = ""
    reason_hi: str = ""


@dataclass
class RecommendationOutput:
    """
    Final combined output of both modules.
    This is what the FastAPI /recommend endpoint returns.
    Chatbot layer converts this into conversational language.
    """
    field_id: str
    farmer_id: str
    generated_at: str                       # ISO datetime

    fertilizer: FertilizerRecommendation
    pest_risk: PestRiskRecommendation

    # Overall confidence score (0.0 – 1.0)
    confidence: float = 0.0

    # Flag for offline TFLite inference
    inferred_offline: bool = False


# ── Feature Vector (for ML model input) ───────────────────────────────────────

def field_profile_to_feature_vector(fp: FieldProfile) -> dict:
    """
    Converts a FieldProfile into a flat feature dict for XGBoost/LightGBM.
    This is what gets passed to model.predict().

    Called by: src/models.py in Phase 3.
    """
    if not fp.is_complete():
        raise ValueError(f"FieldProfile {fp.field_id} is missing required data.")

    return {
        # Soil features
        "soil_n":           fp.soil.nitrogen_kg_ha,
        "soil_p":           fp.soil.phosphorus_kg_ha,
        "soil_k":           fp.soil.potassium_kg_ha,
        "soil_ph":          fp.soil.ph,
        "organic_carbon":   fp.soil.organic_carbon_pct,
        "zinc_ppm":         fp.soil.zinc_ppm or 0.0,
        "boron_ppm":        fp.soil.boron_ppm or 0.0,
        "sulphur_ppm":      fp.soil.sulphur_ppm or 0.0,

        # Crop stage
        "days_after_transplanting": fp.days_after_transplanting,

        # Weather (spray window logic)
        "rain_prob_8h_max": fp.weather.rain_probability_next_8h_max,
        "temperature":      fp.weather.temperature_celsius,
        "humidity":         fp.weather.humidity_pct,

        # Satellite + pest context
        "ndvi_stress":      fp.ndvi_stress_score or 0.5,
        "pest_history":     fp.district_pest_history_score or 0.0,
    }
