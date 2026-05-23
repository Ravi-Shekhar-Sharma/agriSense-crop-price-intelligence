"""
AgriSense — Feature Engineering Pipeline
Day 2 | C&A Club Winter Project
Input  : data/processed/merged_retail_wholesale.csv
Output : data/processed/features.csv

Key fix: CSV is WIDE format (retail_tomato, wholesale_tomato ...)
         → reshape to LONG format (date, state, commodity, retail_price, wholesale_price)
         → then all lag/rolling/feature logic runs cleanly
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH  = "data/processed/merged_retail_wholesale.csv"
OUTPUT_PATH = "data/processed/features.csv"
REPORT_PATH = "outputs/feature_report.txt"

os.makedirs("outputs", exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)


# ─────────────────────────────────────────────
# STEP 0 — LOAD & RESHAPE WIDE → LONG
# ─────────────────────────────────────────────
log("\n" + "="*60)
log("AGRISENSE — FEATURE ENGINEERING PIPELINE")
log("="*60)

df_wide = pd.read_csv(INPUT_PATH)
df_wide.columns = df_wide.columns.str.strip().str.lower().str.replace(" ", "_")
df_wide["date"] = pd.to_datetime(df_wide["date"])

log(f"\n[LOAD] Wide shape: {df_wide.shape}")
log(f"[LOAD] Date range: {df_wide['date'].min().date()} → {df_wide['date'].max().date()}")
log(f"[LOAD] States: {df_wide['state'].nunique()}")

# Extract commodity names from retail_ prefix columns
commodities = [c.replace("retail_", "") for c in df_wide.columns if c.startswith("retail_")]
log(f"[LOAD] Commodities detected ({len(commodities)}): {commodities}")

# Melt retail prices
retail_long = df_wide[["date", "state"] + [f"retail_{c}" for c in commodities]].melt(
    id_vars=["date", "state"],
    value_vars=[f"retail_{c}" for c in commodities],
    var_name="commodity",
    value_name="retail_price"
)
retail_long["commodity"] = retail_long["commodity"].str.replace("retail_", "", regex=False)

# Melt wholesale prices
wholesale_long = df_wide[["date", "state"] + [f"wholesale_{c}" for c in commodities]].melt(
    id_vars=["date", "state"],
    value_vars=[f"wholesale_{c}" for c in commodities],
    var_name="commodity",
    value_name="wholesale_price"
)
wholesale_long["commodity"] = wholesale_long["commodity"].str.replace("wholesale_", "", regex=False)

# Merge into long format
df = retail_long.merge(wholesale_long, on=["date", "state", "commodity"], how="inner")

log(f"\n[RESHAPE] Long format shape: {df.shape}")
log(f"[RESHAPE] Columns: {list(df.columns)}")
log(f"[RESHAPE] Commodities: {df['commodity'].nunique()} | States: {df['state'].nunique()}")

# Sort — critical for lag/rolling
df = df.sort_values(["commodity", "state", "date"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 1 — MISSING VALUE IMPUTATION
# ─────────────────────────────────────────────
log("\n[IMPUTE] Missing values before imputation:")
log(f"  retail_price:    {df['retail_price'].isna().sum()}")
log(f"  wholesale_price: {df['wholesale_price'].isna().sum()}")

def impute_group(group):
    group["retail_price"]    = group["retail_price"].ffill().bfill().interpolate(method="linear", limit_direction="both")
    group["wholesale_price"] = group["wholesale_price"].ffill().bfill().interpolate(method="linear", limit_direction="both")
    return group

df = df.groupby(["commodity", "state"], group_keys=False).apply(impute_group)

log(f"[IMPUTE] After imputation:")
log(f"  retail_price:    {df['retail_price'].isna().sum()}")
log(f"  wholesale_price: {df['wholesale_price'].isna().sum()}")


# ─────────────────────────────────────────────
# STEP 2 — TIME-BASED FEATURES
# ─────────────────────────────────────────────
log("\n[TIME] Extracting calendar features...")

df["year"]           = df["date"].dt.year
df["month"]          = df["date"].dt.month
df["day_of_month"]   = df["date"].dt.day
df["day_of_week"]    = df["date"].dt.dayofweek
df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
df["quarter"]        = df["date"].dt.quarter
df["day_of_year"]    = df["date"].dt.dayofyear
df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

# Cyclical encoding — Dec(12) and Jan(1) should be close, not 11 apart
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

log("  Added: year, month, day_of_month, day_of_week, week_of_year, quarter,")
log("         day_of_year, is_weekend, is_month_end, month_sin/cos, dow_sin/cos")


# ─────────────────────────────────────────────
# STEP 3 — LAG FEATURES
# ─────────────────────────────────────────────
log("\n[LAG] Creating lag features...")

LAG_DAYS = [1, 7, 14, 30]

def add_lags(group):
    for lag in LAG_DAYS:
        group[f"retail_lag_{lag}d"]    = group["retail_price"].shift(lag)
        group[f"wholesale_lag_{lag}d"] = group["wholesale_price"].shift(lag)
    for lag in [7, 30]:
        group[f"retail_change_{lag}d"]     = group["retail_price"] - group[f"retail_lag_{lag}d"]
        group[f"retail_pct_chg_{lag}d"]    = group["retail_price"].pct_change(lag) * 100
        group[f"wholesale_pct_chg_{lag}d"] = group["wholesale_price"].pct_change(lag) * 100
    return group

df = df.groupby(["commodity", "state"], group_keys=False).apply(add_lags)
lag_cols = [c for c in df.columns if "lag" in c or "chg" in c]
log(f"  Added {len(lag_cols)} lag/momentum columns")


# ─────────────────────────────────────────────
# STEP 4 — ROLLING STATISTICS
# ─────────────────────────────────────────────
log("\n[ROLLING] Computing rolling statistics...")

def add_rolling(group):
    for w in [7, 30]:
        group[f"retail_roll_mean_{w}d"]    = group["retail_price"].rolling(w, min_periods=3).mean()
        group[f"retail_roll_std_{w}d"]     = group["retail_price"].rolling(w, min_periods=3).std()
        group[f"wholesale_roll_mean_{w}d"] = group["wholesale_price"].rolling(w, min_periods=3).mean()
        group[f"wholesale_roll_std_{w}d"]  = group["wholesale_price"].rolling(w, min_periods=3).std()
        group[f"retail_zscore_{w}d"]       = (
            (group["retail_price"] - group[f"retail_roll_mean_{w}d"])
            / (group[f"retail_roll_std_{w}d"] + 1e-8)
        )
    group["retail_roll_min_30d"] = group["retail_price"].rolling(30, min_periods=3).min()
    group["retail_roll_max_30d"] = group["retail_price"].rolling(30, min_periods=3).max()
    group["retail_30d_range"]    = group["retail_roll_max_30d"] - group["retail_roll_min_30d"]
    return group

df = df.groupby(["commodity", "state"], group_keys=False).apply(add_rolling)
roll_cols = [c for c in df.columns if "roll" in c or "zscore" in c]
log(f"  Added {len(roll_cols)} rolling columns")


# ─────────────────────────────────────────────
# STEP 5 — PRICE SPREAD FEATURES
# ─────────────────────────────────────────────
log("\n[SPREAD] Computing spread features...")

df["price_spread"] = df["retail_price"] - df["wholesale_price"]
df["spread_ratio"] = df["retail_price"] / (df["wholesale_price"] + 1e-8)
df["margin_pct"]   = (df["price_spread"] / (df["wholesale_price"] + 1e-8)) * 100

def add_spread_rolling(group):
    group["spread_roll_7d"]  = group["price_spread"].rolling(7,  min_periods=2).mean()
    group["spread_roll_30d"] = group["price_spread"].rolling(30, min_periods=5).mean()
    return group

df = df.groupby(["commodity", "state"], group_keys=False).apply(add_spread_rolling)
log("  Added: price_spread, spread_ratio, margin_pct, spread_roll_7d/30d")


# ─────────────────────────────────────────────
# STEP 6 — SEASONAL / HARVEST FLAGS
# ─────────────────────────────────────────────
log("\n[SEASONAL] Adding harvest/seasonal flags...")

HARVEST_MONTHS = {
    "tomato":       [11, 12, 1, 2, 5, 6],
    "onion":        [3, 4, 5, 10, 11],
    "potato":       [12, 1, 2, 3],
    "wheat":        [3, 4, 5],
    "atta":         [3, 4, 5],
    "rice":         [10, 11],
    "tur_dal":      [12, 1, 2],
    "urad_dal":     [10, 11, 12],
    "moong_dal":    [3, 4, 9, 10],
    "gram_dal":     [3, 4, 5],
    "masoor_dal":   [3, 4, 5],
    "mustard_oil":  [3, 4, 5],
    "groundnut_oil":[10, 11, 12],
    "soya_oil":     [10, 11],
    "sunflower_oil":[3, 4],
    "palm_oil":     [10, 11, 12, 1],
    "sugar":        [11, 12, 1, 2, 3],
    "gur":          [11, 12, 1, 2, 3],
}

def is_harvest_month(row):
    c = str(row["commodity"]).lower().strip()
    m = row["month"]
    if c in HARVEST_MONTHS:
        return int(m in HARVEST_MONTHS[c])
    # partial match fallback
    for key, months in HARVEST_MONTHS.items():
        if key in c or c in key:
            return int(m in months)
    return 0

df["is_harvest_month"] = df.apply(is_harvest_month, axis=1)
df["is_kharif_season"] = df["month"].isin([6, 7, 8, 9]).astype(int)
df["is_rabi_season"]   = df["month"].isin([11, 12, 1, 2]).astype(int)
df["is_monsoon"]       = df["month"].isin([6, 7, 8, 9]).astype(int)
df["is_summer"]        = df["month"].isin([3, 4, 5]).astype(int)
df["is_winter"]        = df["month"].isin([11, 12, 1, 2]).astype(int)

log(f"  Harvest flag: {df['is_harvest_month'].mean()*100:.1f}% of rows flagged")


# ─────────────────────────────────────────────
# STEP 7 — FESTIVAL CALENDAR
# ─────────────────────────────────────────────
log("\n[FESTIVAL] Adding festival proximity features...")

FESTIVALS = {
    "diwali":   [pd.Timestamp("2022-10-24"), pd.Timestamp("2023-11-12"), pd.Timestamp("2024-11-01")],
    "holi":     [pd.Timestamp("2022-03-18"), pd.Timestamp("2023-03-08"), pd.Timestamp("2024-03-25")],
    "eid":      [pd.Timestamp("2022-05-03"), pd.Timestamp("2023-04-21"), pd.Timestamp("2024-04-10")],
    "navratri": [pd.Timestamp("2022-09-26"), pd.Timestamp("2023-10-15"), pd.Timestamp("2024-10-03")],
    "pongal":   [pd.Timestamp("2023-01-14"), pd.Timestamp("2024-01-15")],
}

WINDOW = 7

for festival, fdates in FESTIVALS.items():
    df[f"days_to_{festival}"] = df["date"].apply(
        lambda d: min(abs((d - fd).days) for fd in fdates)
    )
    df[f"near_{festival}"] = (df[f"days_to_{festival}"] <= WINDOW).astype(int)

df["near_any_festival"] = df[[c for c in df.columns if c.startswith("near_")]].max(axis=1)
log(f"  Festival proximity coverage: {df['near_any_festival'].mean()*100:.1f}% of rows")


# ─────────────────────────────────────────────
# STEP 8 — COMMODITY CATEGORY (ONE-HOT)
# ─────────────────────────────────────────────
log("\n[CATEGORY] Adding commodity category one-hot features...")

COMMODITY_CATEGORIES = {
    "Cereal":    ["rice", "wheat", "atta"],
    "Pulse":     ["tur_dal", "gram_dal", "urad_dal", "moong_dal", "masoor_dal"],
    "Vegetable": ["tomato", "onion", "potato"],
    "Oil":       ["mustard_oil", "groundnut_oil", "soya_oil", "palm_oil", "sunflower_oil", "vanaspati"],
    "Sugar":     ["sugar", "gur"],
    "Dairy":     ["milk"],
    "Other":     ["tea_loose", "salt"],
}

def get_category(commodity):
    c = str(commodity).lower()
    for cat, keywords in COMMODITY_CATEGORIES.items():
        if c in keywords:
            return cat
    return "Other"

df["commodity_category"] = df["commodity"].apply(get_category)
cat_dummies = pd.get_dummies(df["commodity_category"], prefix="cat").astype(int)
df = pd.concat([df, cat_dummies], axis=1)
log(f"  Categories: {df['commodity_category'].value_counts().to_dict()}")


# ─────────────────────────────────────────────
# STEP 9 — STATE REGION (ONE-HOT)
# ─────────────────────────────────────────────
log("\n[REGION] Adding state region features...")

REGION_MAP = {
    "North":   ["Delhi", "Uttar Pradesh", "Haryana", "Punjab", "Himachal Pradesh",
                "Uttarakhand", "Jammu and Kashmir", "Rajasthan"],
    "South":   ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana"],
    "East":    ["West Bengal", "Bihar", "Odisha", "Jharkhand", "Chhattisgarh",
                "Tripura", "Meghalaya", "Nagaland", "Manipur", "Mizoram",
                "Arunachal Pradesh", "Sikkim"],
    "West":    ["Maharashtra", "Gujarat", "Goa"],
    "Central": ["Madhya Pradesh"],
    "Northeast":["Assam"],
}

def get_region(state):
    for region, states in REGION_MAP.items():
        if state in states:
            return region
    return "Other"

df["region"] = df["state"].apply(get_region)
region_dummies = pd.get_dummies(df["region"], prefix="region").astype(int)
df = pd.concat([df, region_dummies], axis=1)
log(f"  Regions: {df['region'].value_counts().to_dict()}")


# ─────────────────────────────────────────────
# STEP 10 — SAVE
# ─────────────────────────────────────────────
log("\n[FINALIZE] Cleaning up and saving...")

rows_before = len(df)
df = df.dropna(subset=["retail_price", "wholesale_price"])
log(f"  Dropped {rows_before - len(df)} rows with NaN targets. Remaining: {len(df):,}")

# Feature group summary
feature_groups = {
    "Identity" : ["date", "state", "commodity"],
    "Target"   : ["retail_price", "wholesale_price"],
    "Time"     : [c for c in df.columns if any(x in c for x in ["year","month","day","week","quarter","is_weekend","is_month"])],
    "Cyclical" : [c for c in df.columns if "sin" in c or "cos" in c],
    "Lags"     : [c for c in df.columns if "lag" in c],
    "Momentum" : [c for c in df.columns if "chg" in c],
    "Rolling"  : [c for c in df.columns if "roll" in c or "zscore" in c or "range" in c],
    "Spread"   : [c for c in df.columns if "spread" in c or "margin" in c or "ratio" in c],
    "Seasonal" : [c for c in df.columns if any(x in c for x in ["harvest","kharif","rabi","monsoon","summer","winter"])],
    "Festival" : [c for c in df.columns if any(x in c for x in ["festival","diwali","holi","eid","navratri","pongal"])],
    "Category" : [c for c in df.columns if c.startswith("cat_")],
    "Region"   : [c for c in df.columns if c.startswith("region_")],
}

log(f"\n[SUMMARY]")
log(f"  Total rows    : {len(df):,}")
log(f"  Total columns : {len(df.columns)}")
log(f"  Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
log(f"  States        : {df['state'].nunique()}")
log(f"  Commodities   : {df['commodity'].nunique()}")
log(f"\n  Feature groups:")
for group, cols in feature_groups.items():
    if cols:
        log(f"    {group:<12}: {len(cols):>3} cols")

df.to_csv(OUTPUT_PATH, index=False)
log(f"\nSaved → {OUTPUT_PATH}  |  Shape: {df.shape}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f" Report → {REPORT_PATH}")

log("\n" + "="*60)
log("D2 COMPLETE — Ready for D3: Model Training")
log("="*60)