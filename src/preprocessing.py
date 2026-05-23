"""
AgriSense | src/preprocessing.py
=================================
Cleans and preprocesses raw retail and wholesale price data.

What this module does:
1. Loads raw CSVs from data/raw/
2. Fixes known data quality issues (empty columns, Soya Oil dtype, date parsing)
3. Separates aggregate rows (Avg/Max/Min/Modal) from state-level rows
4. Converts wholesale prices from per-quintal to per-kg for apples-to-apples comparison
5. Merges retail + wholesale into one unified dataframe
6. Saves cleaned outputs to data/processed/

Run directly:  python src/preprocessing.py
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# These 4 rows appear every date — they summarise across states, NOT a state themselves
AGGREGATE_LABELS = ['Average Price', 'Maximum Price', 'Minimum Price', 'Modal Price']

# The 22 price commodities (columns we actually care about)
COMMODITY_COLS = [
    'Rice', 'Wheat', 'Atta (Wheat)', 'Gram Dal', 'Tur/Arhar Dal',
    'Urad Dal', 'Moong Dal', 'Masoor Dal', 'Sugar', 'Milk @',
    'Groundnut Oil (Packed)', 'Mustard Oil (Packed)', 'Vanaspati (Packed)',
    'Soya Oil (Packed)', 'Sunflower Oil (Packed)', 'Palm Oil (Packed)',
    'Gur', 'Tea Loose', 'Salt Pack (Iodised)', 'Potato', 'Onion', 'Tomato'
]

# Cleaner display names for plots and dashboards
COMMODITY_RENAME = {
    'Atta (Wheat)': 'Atta',
    'Tur/Arhar Dal': 'Tur Dal',
    'Groundnut Oil (Packed)': 'Groundnut Oil',
    'Mustard Oil (Packed)': 'Mustard Oil',
    'Vanaspati (Packed)': 'Vanaspati',
    'Soya Oil (Packed)': 'Soya Oil',
    'Sunflower Oil (Packed)': 'Sunflower Oil',
    'Palm Oil (Packed)': 'Palm Oil',
    'Salt Pack (Iodised)': 'Salt',
    'Milk @': 'Milk',
}

# Commodity categories — useful for dashboard filters
COMMODITY_CATEGORIES = {
    'Cereals':    ['Rice', 'Wheat', 'Atta'],
    'Pulses':     ['Gram Dal', 'Tur Dal', 'Urad Dal', 'Moong Dal', 'Masoor Dal'],
    'Oils':       ['Groundnut Oil', 'Mustard Oil', 'Vanaspati', 'Soya Oil',
                   'Sunflower Oil', 'Palm Oil'],
    'Sweeteners': ['Sugar', 'Gur'],
    'Dairy':      ['Milk'],
    'Beverages':  ['Tea Loose'],
    'Condiments': ['Salt'],
    'Vegetables': ['Potato', 'Onion', 'Tomato'],
}


# ─────────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────────

def load_raw(price_type: str) -> pd.DataFrame:
    """
    Load a raw CSV file.
    price_type: 'retail' or 'wholesale'
    """
    path = os.path.join(RAW_DIR, f'{price_type}_price_data.csv')
    df = pd.read_csv(path)
    print(f"  Loaded {price_type}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame, price_type: str) -> pd.DataFrame:
    """
    Applies all cleaning steps to a raw dataframe.

    WHY each step exists:
    - Drop unnamed cols:  Retail CSV has 23 trailing empty columns — Excel artefact.
    - Fix Soya Oil:       Read as object dtype because some cells contain " " (whitespace)
                          instead of NaN. We strip whitespace → empty string → NaN → float.
    - Parse dates:        Stored as "DD-MM-YYYY" strings. Pandas needs dayfirst=True.
    - Rename cols:        Shorter names for display purposes.
    """

    # 2a. Drop trailing empty columns (retail only)
    unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed:
        df = df.drop(columns=unnamed)
        print(f"  Dropped {len(unnamed)} empty unnamed columns")

    # 2b. Fix numeric columns — wholesale data stores ALL commodity columns as strings
    #     because some cells contain " " (whitespace) instead of NaN.
    #     pd.to_numeric with errors='coerce' handles this cleanly:
    #       - valid number strings → float
    #       - whitespace / text → NaN
    for col in COMMODITY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2c. Parse dates
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # 2d. Rename commodities to cleaner names
    df = df.rename(columns=COMMODITY_RENAME)

    # 2e. Tag the price type
    df['price_type'] = price_type

    return df


# ─────────────────────────────────────────────
# STEP 3: SPLIT AGGREGATE vs STATE ROWS
# ─────────────────────────────────────────────

def split_aggregates(df: pd.DataFrame):
    """
    The raw data mixes two kinds of rows in 'States/UTs':
      - Actual states (e.g. 'Maharashtra', 'Punjab')
      - Summary rows (Average Price, Maximum Price, Minimum Price, Modal Price)

    We split them because:
      - State rows → time-series modelling, state-wise dashboards
      - Aggregate rows → quick national-level summary, validation checks

    Returns: (state_df, agg_df)
    """
    agg_mask = df['States/UTs'].isin(AGGREGATE_LABELS)
    return df[~agg_mask].copy(), df[agg_mask].copy()


# ─────────────────────────────────────────────
# STEP 4: WHOLESALE UNIT CONVERSION
# ─────────────────────────────────────────────

def convert_wholesale_to_per_kg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wholesale prices are in ₹ per QUINTAL (100 kg).
    Retail prices are in ₹ per KG.

    To compare margin = Retail − Wholesale on the same scale,
    we divide wholesale prices by 100.

    This is logged so you know the transformation happened.
    """
    commodity_cols_present = [c for c in get_renamed_commodity_cols() if c in df.columns]
    df = df.copy()
    df[commodity_cols_present] = df[commodity_cols_present] / 100
    print("  Wholesale prices converted: ₹/quintal → ₹/kg (÷100)")
    return df


def get_renamed_commodity_cols():
    """Returns commodity column names after renaming."""
    renamed = [COMMODITY_RENAME.get(c, c) for c in COMMODITY_COLS]
    return renamed


# ─────────────────────────────────────────────
# STEP 5: MERGE RETAIL + WHOLESALE
# ─────────────────────────────────────────────

def merge_retail_wholesale(retail_states: pd.DataFrame,
                           wholesale_states: pd.DataFrame) -> pd.DataFrame:
    """
    Merges retail and wholesale state-level data on [Date, States/UTs].

    After merge, for each (date, state, commodity) we have:
      - retail_<commodity>  : ₹/kg retail price
      - wholesale_<commodity>: ₹/kg wholesale price (already converted)
      - margin_<commodity>  : retail − wholesale = trader margin in ₹/kg

    Margin tells us how much traders/retailers add between farm gate and consumer.
    High margin = opportunity for procurement / direct sourcing.
    """
    commodity_cols = get_renamed_commodity_cols()
    commodity_cols_present_r = [c for c in commodity_cols if c in retail_states.columns]
    commodity_cols_present_w = [c for c in commodity_cols if c in wholesale_states.columns]

    # Rename commodity cols to indicate source before merge
    retail_renamed = retail_states[['Date', 'States/UTs'] + commodity_cols_present_r].copy()
    retail_renamed.columns = (
        ['Date', 'State'] +
        [f'retail_{c}' for c in commodity_cols_present_r]
    )

    wholesale_renamed = wholesale_states[['Date', 'States/UTs'] + commodity_cols_present_w].copy()
    wholesale_renamed.columns = (
        ['Date', 'State'] +
        [f'wholesale_{c}' for c in commodity_cols_present_w]
    )

    merged = pd.merge(retail_renamed, wholesale_renamed, on=['Date', 'State'], how='inner')

    # Compute margin for each commodity
    common_commodities = [c for c in commodity_cols_present_r if c in commodity_cols_present_w]
    for c in common_commodities:
        if f'retail_{c}' in merged.columns and f'wholesale_{c}' in merged.columns:
            merged[f'margin_{c}'] = merged[f'retail_{c}'] - merged[f'wholesale_{c}']

    print(f"  Merged dataframe: {merged.shape[0]:,} rows × {merged.shape[1]} cols")
    return merged


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n LOADING RAW DATA")
    retail_raw = load_raw('retail')
    wholesale_raw = load_raw('wholesale')

    print("\n CLEANING")
    retail_clean = clean(retail_raw, 'retail')
    wholesale_clean = clean(wholesale_raw, 'wholesale')

    print("\n  SPLITTING AGGREGATES vs STATES")
    retail_states, retail_agg = split_aggregates(retail_clean)
    wholesale_states, wholesale_agg = split_aggregates(wholesale_clean)
    print(f"  Retail  → {len(retail_states):,} state rows | {len(retail_agg):,} agg rows")
    print(f"  Wholesale → {len(wholesale_states):,} state rows | {len(wholesale_agg):,} agg rows")

    print("\n UNIT CONVERSION (Wholesale: ₹/quintal → ₹/kg)")
    wholesale_states = convert_wholesale_to_per_kg(wholesale_states)
    wholesale_agg    = convert_wholesale_to_per_kg(wholesale_agg)

    print("\n MERGING RETAIL + WHOLESALE")
    merged = merge_retail_wholesale(retail_states, wholesale_states)

    print("\n SAVING TO data/processed/")
    retail_states.to_csv(os.path.join(PROCESSED_DIR, 'retail_states.csv'), index=False)
    wholesale_states.to_csv(os.path.join(PROCESSED_DIR, 'wholesale_states.csv'), index=False)
    retail_agg.to_csv(os.path.join(PROCESSED_DIR, 'retail_aggregates.csv'), index=False)
    wholesale_agg.to_csv(os.path.join(PROCESSED_DIR, 'wholesale_aggregates.csv'), index=False)
    merged.to_csv(os.path.join(PROCESSED_DIR, 'merged_retail_wholesale.csv'), index=False)

    print("\n Pipeline complete. Files in data/processed/:")
    for f in os.listdir(PROCESSED_DIR):
        size = os.path.getsize(os.path.join(PROCESSED_DIR, f))
        print(f"   {f}  ({size/1024:.0f} KB)")

    return retail_states, wholesale_states, retail_agg, wholesale_agg, merged


if __name__ == '__main__':
    run_pipeline()