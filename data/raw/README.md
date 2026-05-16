# data/raw/

Raw datasets live here. **These files are gitignored and must be downloaded manually.**
Never edit files in this folder — they are the unmodified source of truth.
All cleaned outputs go to `data/processed/`.

---

## Files required for Phase 1 & 2

| File | Source | Download Link | Status |
|------|--------|---------------|--------|
| `crop_recommendation.csv` | Kaggle (Atharva Ingle / ICFA) | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset | Download manually |
| `kcc_pest_queries.csv` | KCC-CHAKSHU (ICAR-IASRI) | https://kcc-chakshu.icar-web.com/6_data_extract.php | Filter: Crop=Rice, Type=Pest/Disease |
| `shc_district.csv` | Soil Health Card Portal | https://soilhealth.dac.gov.in | Select: UP → Gautam Buddha Nagar |
| `iffco_nano_urea_specs.json` | Created manually from FCO spec sheet | Already in this folder | ✅ Done |

---

## Dataset descriptions

### crop_recommendation.csv
- **Rows:** ~2200 across 22 crops
- **Columns:** N, P, K, temperature, humidity, ph, rainfall, label
- **Our use:** Filter `label == 'rice'` → base training data for Module 1 (fertilizer optimizer)
- **License:** CC0 Public Domain

### kcc_pest_queries.csv
- **Rows:** ~10.9M farmer pest/disease queries (2015–2020)
- **Columns:** state, district, block, crop, season, query_type, pest_description
- **Our use:** Aggregate by district+season → pest outbreak frequency score for Module 2 (pest risk advisor)
- **Source:** ICAR-IASRI Kisan Call Centre analytics

### shc_district.csv
- **Rows:** Varies by district — typically 5,000–50,000 soil samples
- **Columns:** N, P, K, pH, OC, Zn, B, S, Fe, Mn, Cu (may have missing micronutrients)
- **Our use:** Primary soil training data — merged with crop_recommendation.csv in ETL

### iffco_nano_urea_specs.json
- **Created from:** IFFCO FCO-approved product specification sheet
- **Our use:** Rules engine — hard dosage constraints and spray timing windows

---

## Download instructions

### 1. Kaggle dataset
```bash
# Option A — Kaggle CLI (if you have it set up)
kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data/raw/
unzip data/raw/crop-recommendation-dataset.zip -d data/raw/
mv data/raw/Crop_recommendation.csv data/raw/crop_recommendation.csv

# Option B — Manual
# Go to https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
# Click Download → rename to crop_recommendation.csv → drop in data/raw/
```

### 2. KCC-CHAKSHU pest data
```
1. Go to https://kcc-chakshu.icar-web.com/6_data_extract.php
2. Set filters: Crop = Rice | Query Type = Pest/Disease | Year = 2018–2020
3. Click Export → Download CSV
4. Save as data/raw/kcc_pest_queries.csv
```

### 3. Soil Health Card data
```
1. Go to https://soilhealth.dac.gov.in
2. Navigate: State Reports → Uttar Pradesh → Gautam Buddha Nagar
3. Download district soil summary CSV
4. Save as data/raw/shc_district.csv
```

---

## Notes for Phase 2 (ETL)
- `crop_recommendation.csv` is missing micronutrients (Zn, B, S) — apply KNN imputation
- `shc_district.csv` may have inconsistent column names across districts — normalize in ETL
- `kcc_pest_queries.csv` is large (~500MB) — filter to Rice rows immediately on load
