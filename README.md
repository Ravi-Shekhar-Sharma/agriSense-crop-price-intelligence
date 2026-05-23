# AgriSense (एग्रीसेंस) — Indian Crop Price Intelligence

**AI-powered crop price forecasting across 34 Indian states and 22 commodities**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://agrisense-crop-price-intelligence-mreb4edhyc2lagkvvtvgzi.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-189AB4?style=flat)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## Overview

AgriSense is a crop price intelligence dashboard built for the Indian agricultural market. It ingests daily retail and wholesale price data from the Government of India, engineers 84 predictive features per observation, and trains an XGBoost model for each of 22 essential commodities — from staple cereals and pulses to oils and vegetables.

The system addresses a real gap: farmers, traders, wholesalers, and policymakers have no unified tool to monitor price trends, forecast short-term movements, and identify regional disparities across India's fragmented agricultural supply chain. AgriSense provides all of this in a single, interactive dashboard.

---

## Key Results

| Metric | Value |
|---|---|
| Average MAPE (XGBoost) | **0.49%** (vs Baseline 2.51%) |
| Average R² | **0.994** |
| Models Trained | **66** (22 commodities × 3 types: Baseline, XGBoost, LightGBM) |
| Dataset Size | **507,263 observations** · 34 states · 22 commodities · 732 days |
| Forecast Horizon | **7 days** (autoregressive, day-by-day rollout) |

---

## Dashboard Features

| Tab | Description |
|---|---|
| **Overview** | Current price KPIs, retail vs wholesale trend chart, price volatility (CV%) across all 22 commodities, and stakeholder perspectives (Farmer / Trader / Wholesaler / Policymaker) |
| **Price Explorer** | Price distribution, 30-day rolling volatility, seasonal monthly patterns, and year-on-year comparison |
| **State Analysis** | State-wise price ranking, price risk zone classification (CV-based), and region-wise spread |
| **Trading Intelligence** | Margin & spread analysis by state, anomaly detection, and monthly procurement timing signals |
| **7-Day Forecast** | Autoregressive XGBoost forecast with confidence band, price labels, and downloadable CSV |
| **Model Performance** | MAPE comparison (Baseline vs XGBoost) across all 22 commodities, full RMSE / MAE / R² metrics table |

---

## Project Structure

```
agrisense/
├── app.py                        # Streamlit dashboard (8 tabs)
├── requirements.txt
├── packages.txt                  # System deps for Streamlit Cloud
├── .gitignore
├── src/
│   ├── preprocessing.py          # Raw data cleaning & merging
│   ├── feature_engineering.py    # 84-feature pipeline (lags, rolling, seasonal)
│   └── model_training.py         # XGBoost + LightGBM training loop
├── models/
│   ├── xgb_<commodity>.pkl       # 22 XGBoost models
│   ├── lgbm_<commodity>.pkl      # 22 LightGBM models
│   └── label_encoders.pkl
├── data/
│   └── processed/
│       └── merged_retail_wholesale.csv   # Source data (pushed to repo)
├── notebooks/
│   └── 01_eda.py
└── outputs/
    ├── model_results.csv         # RMSE, MAE, MAPE, R² for all models
    └── feature_report.txt
```

---

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first launch, `features.csv` is auto-generated from `merged_retail_wholesale.csv` (~60 seconds). Subsequent loads are instant.

---

## Methodology

| Step | Detail |
|---|---|
| **Data** | Daily retail and wholesale prices, Aug 2022 – Aug 2024, sourced from the Government of India price monitoring portal |
| **Features** | 84 engineered features per row: time-based (year, month, cyclical encoding), lag features (1, 7, 14, 30 days), rolling statistics (mean, std, z-score, min/max over 7/30-day windows), price spread & margin, harvest flags, festival proximity calendar (Diwali, Holi, Eid, Navratri, Pongal), commodity category and state region one-hot encodings |
| **Model** | XGBoost regressor trained independently per commodity; time-based train/test split (train: Aug 2022 – Jan 2024, test: Feb – Aug 2024) |
| **Evaluation** | RMSE, MAE, MAPE, R² on the held-out test set; compared against a naive last-value baseline |

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `XGBoost` · `LightGBM` · `Scikit-learn` · `Streamlit` · `Plotly`

---

*Built by **Ravi Shekhar Sharma** | C&A Winter Project 2025*
