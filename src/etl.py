import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))
from config import (
    NANO_UREA_SPECS,
    RAIN_PROBABILITY_THRESHOLD,
    PEST_RISK_THRESHOLD_HIGH,
    PEST_RISK_THRESHOLD_LOW,
    PILOT_CROP
)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load and filter raw crop recommendation CSV to paddy rows only."""
    df = pd.read_csv(filepath)
    paddy = df[df["label"] == "rice"].copy().reset_index(drop=True)
    print(f"Loaded {len(df)} total rows — {len(paddy)} paddy rows extracted")
    return paddy


def expand_dataset(df: pd.DataFrame, target_size: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Expand a small dataset to target_size by sampling with replacement.
    Small Gaussian noise is added to continuous features to avoid duplicates.
    """
    np.random.seed(seed)
    expanded = df.sample(n=target_size, replace=True, random_state=seed).reset_index(drop=True)

    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    for col in numeric_cols:
        noise = np.random.normal(0, expanded[col].std() * 0.05, target_size)
        expanded[col] = (expanded[col] + noise).round(2)

    print(f"Dataset expanded to {len(expanded)} rows")
    return expanded


def add_field_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Add agronomic and environmental features not present in the base dataset.
    DAT distribution is biased toward spray windows so the model
    sees enough examples of each spray action to learn from.
    """
    np.random.seed(seed)
    n = len(df)

    # Days After Transplanting — biased toward IFFCO spray windows
    dats = np.concatenate([
        np.random.randint(30, 36, int(n * 0.20)),   # active tillering
        np.random.randint(45, 56, int(n * 0.20)),   # pre-flowering
        np.random.randint(0,  30, int(n * 0.15)),   # pre-window
        np.random.randint(36, 45, int(n * 0.15)),   # between windows
        np.random.randint(56, 121, int(n * 0.30)),  # post-window
    ])
    np.random.shuffle(dats)
    df["days_after_transplanting"] = dats[:n]

    # Rain probability — beta distribution skewed low (most crop-season days are clear)
    df["rain_prob_8h"]  = np.random.beta(1.5, 5, n).round(2)
    df["ndvi_stress"]   = np.random.uniform(0.3, 1.0, n).round(2)
    df["pest_history"]  = np.random.uniform(0.0, 0.8, n).round(2)

    return df


def add_micronutrients(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Add micronutrient columns based on ICAR published deficiency ranges
    for western Uttar Pradesh paddy soils. Missing values are introduced
    at realistic rates (~20-35%) to simulate real SHC card gaps.
    """
    np.random.seed(seed)
    n = len(df)

    specs = [
        ("zinc_ppm",      0.58, 0.22, 0.35),
        ("boron_ppm",     0.48, 0.18, 0.30),
        ("sulphur_ppm",   11.2,  4.5, 0.00),
        ("iron_ppm",      8.40,  3.2, 0.20),
        ("manganese_ppm", 3.80,  1.4, 0.00),
        ("copper_ppm",    1.20, 0.45, 0.00),
    ]

    for col, mean, std, missing_rate in specs:
        values = np.random.normal(mean, std, n).clip(0.1, mean * 3).round(3)
        if missing_rate > 0:
            values[np.random.random(n) < missing_rate] = np.nan
        df[col] = values

    return df


def impute_micronutrients(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Fill missing micronutrient values using K-Nearest Neighbours imputation."""
    micro_cols = [
        "zinc_ppm", "boron_ppm", "sulphur_ppm",
        "iron_ppm", "manganese_ppm", "copper_ppm"
    ]
    before = df[micro_cols].isnull().sum().sum()
    imputer = KNNImputer(n_neighbors=k)
    df[micro_cols] = imputer.fit_transform(df[micro_cols])
    after = df[micro_cols].isnull().sum().sum()
    print(f"KNN imputation: {before} missing values filled ({after} remaining)")
    return df


def get_crop_stage(dat: int) -> str:
    """Map Days After Transplanting to a named paddy growth stage."""
    if dat < 15:   return "establishment"
    if dat < 30:   return "tillering"
    if dat <= 35:  return "active_tillering"
    if dat <= 44:  return "panicle_initiation"
    if dat <= 55:  return "pre_flowering"
    if dat <= 70:  return "flowering"
    if dat <= 100: return "grain_filling"
    return "ripening"


def generate_labels(row: pd.Series) -> pd.Series:
    """
    Rule-based label generation using IFFCO FCO constraints from config.py.
    Converts raw agronomic features into training targets for both ML modules.

    Module 1 targets: spray_action, nano_urea_dose
    Module 2 targets: pest_intervention
    """
    stage = get_crop_stage(int(row["days_after_transplanting"]))
    in_spray_window = stage in ("active_tillering", "pre_flowering")
    rain_risk = row["rain_prob_8h"] > RAIN_PROBABILITY_THRESHOLD

    # Module 1 — fertiliser recommendation
    if not in_spray_window:
        spray_action = "Basal Only"
        dose = 0.0
    elif rain_risk:
        spray_action = "Delay"
        dose = 0.0
    else:
        spray_action = "Spray"
        n_level = row["N"]
        if n_level < 60:
            dose = 4.0
        elif n_level < 80:
            dose = 3.0
        else:
            dose = 2.0

    # Module 2 — pest risk recommendation
    pest = row["pest_history"]
    if pest > PEST_RISK_THRESHOLD_HIGH:
        pest_action = "Pesticide"
    elif pest > PEST_RISK_THRESHOLD_LOW:
        pest_action = "Bio Control"
    else:
        pest_action = "Monitor"

    return pd.Series({
        "crop_stage":        stage,
        "spray_action":      spray_action,
        "nano_urea_dose":    dose,
        "pest_intervention": pest_action
    })


def run_pipeline(
    raw_path:       str = "../data/raw/crop_recommendation.csv",
    output_path:    str = "../data/processed/paddy_features.csv",
    target_size:    int = 500,
    seed:           int = 42
) -> pd.DataFrame:
    """
    Full ETL pipeline — load, expand, enrich, impute, label, save.
    Returns the processed DataFrame.
    """
    print("=" * 50)
    print("  Precision Agronomy ETL Pipeline")
    print("=" * 50)

    df = load_raw_data(raw_path)
    df = expand_dataset(df, target_size, seed)
    df = add_field_features(df, seed)
    df = add_micronutrients(df, seed)
    df = impute_micronutrients(df)

    # Generate training labels
    labels = df.apply(generate_labels, axis=1)
    df = pd.concat([df.drop(columns=["label"], errors="ignore"), labels], axis=1)

    # Validate
    assert df.isnull().sum().sum() == 0, "Pipeline failed — missing values remain"
    assert "spray_action" in df.columns,  "Pipeline failed — labels missing"

    df.to_csv(output_path, index=False)

    print(f"\nPipeline complete")
    print(f"  Shape        : {df.shape}")
    print(f"  Output       : {output_path}")
    print(f"  Spray labels : {df['spray_action'].value_counts().to_dict()}")
    print(f"  Pest labels  : {df['pest_intervention'].value_counts().to_dict()}")
    print(f"  Missing vals : {df.isnull().sum().sum()}")
    print("=" * 50)

    return df


if __name__ == "__main__":
    run_pipeline()