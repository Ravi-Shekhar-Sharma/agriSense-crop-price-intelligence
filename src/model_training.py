"""
AgriSense — Model Training Pipeline
Day 3 | C&A Club Winter Project
Input  : data/processed/features.csv
Outputs: models/ + outputs/model_results.csv + outputs/plots/
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEATURES_PATH  = "data/processed/features.csv"
MODELS_DIR     = "models"
PLOTS_DIR      = "outputs/plots"
RESULTS_PATH   = "outputs/model_results.csv"
REPORT_PATH    = "outputs/model_report.txt"

TRAIN_END = "2024-01-31"
TEST_START = "2024-02-01"

# Set to True for a quick test run (fewer estimators)
QUICK_MODE = False
N_EST = 200 if QUICK_MODE else 500

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ─────────────────────────────────────────────
# STEP 0 — LOAD DATA
# ─────────────────────────────────────────────
log("\n" + "="*62)
log("AGRISENSE — MODEL TRAINING PIPELINE")
log("="*62)

df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
log(f"\n[LOAD] Shape: {df.shape}")
log(f"[LOAD] Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
log(f"[LOAD] Commodities: {sorted(df['commodity'].unique())}")

# ─────────────────────────────────────────────
# STEP 1 — LABEL ENCODING
# State and commodity are raw strings — tree models need integers
# ─────────────────────────────────────────────
log("\n[ENCODE] Label encoding state and commodity...")

le_state = LabelEncoder()
le_commodity = LabelEncoder()

df["state_enc"]     = le_state.fit_transform(df["state"])
df["commodity_enc"] = le_commodity.fit_transform(df["commodity"])

# Save encoders — needed at inference time to encode new input states
encoders = {"state": le_state, "commodity": le_commodity}
with open(f"{MODELS_DIR}/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
log(f"  States encoded: {len(le_state.classes_)} classes")
log(f"  Commodities encoded: {len(le_commodity.classes_)} classes")
log(f"  Saved -> {MODELS_DIR}/label_encoders.pkl")


# ─────────────────────────────────────────────
# STEP 2 — DEFINE FEATURE COLUMNS
# ─────────────────────────────────────────────

# Columns to DROP from feature matrix
# (identifiers, raw strings, or the target itself)
DROP_COLS = [
    "date", "state", "commodity",           # identifiers / raw strings
    "commodity_category", "region",          # string versions (one-hot already encoded)
    "retail_price",                          # TARGET — never leak this into X
]

TARGET = "retail_price"

# All remaining numeric columns are features
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
log(f"\n[FEATURES] Total feature columns: {len(FEATURE_COLS)}")

# Sanity check — make sure target not in features
assert TARGET not in FEATURE_COLS, "TARGET leaked into features!"


# ─────────────────────────────────────────────
# STEP 3 — GLOBAL TRAIN/TEST SPLIT INFO
# ─────────────────────────────────────────────
df_train_all = df[df["date"] <= TRAIN_END]
df_test_all  = df[df["date"] >= TEST_START]

log(f"\n[SPLIT] Train: {df_train_all['date'].min().date()} -> {df_train_all['date'].max().date()} ({len(df_train_all):,} rows)")
log(f"[SPLIT] Test : {df_test_all['date'].min().date()}  -> {df_test_all['date'].max().date()} ({len(df_test_all):,} rows)")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — scale-independent, comparable across commodities."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mpe  = mape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"model": label, "rmse": round(rmse,3), "mae": round(mae,3),
            "mape": round(mpe,2), "r2": round(r2,4)}

def plot_predictions(commodity, dates, y_true, preds_dict, save_path):
    """Line plot: actual vs model predictions over the test period."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true, label="Actual", color="#2c2c54", linewidth=2)
    colors = {"Baseline": "#aaa69d", "XGBoost": "#e55039", "LightGBM": "#27ae60"}
    for model_name, y_pred in preds_dict.items():
        ax.plot(dates, y_pred, label=model_name,
                color=colors.get(model_name, "blue"),
                linewidth=1.5, linestyle="--" if model_name == "Baseline" else "-")
    ax.set_title(f"{commodity.replace('_',' ').title()} — Test Set Predictions (Feb–Aug 2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (Rs/kg)")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)

def plot_feature_importance(commodity, model_name, feature_names, importances, save_path, top_n=15):
    """Horizontal bar chart of top N feature importances."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e55039" if "lag" in feature_names[i] or "roll" in feature_names[i]
              else "#27ae60" if "festival" in feature_names[i] or "harvest" in feature_names[i]
              else "#2980b9"
              for i in indices]
    ax.barh(range(top_n), importances[indices], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i].replace("_", " ") for i in indices], fontsize=9)
    ax.set_title(f"{commodity.replace('_',' ').title()} — {model_name} Top {top_n} Features",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────
# STEP 4 — PER-COMMODITY TRAINING LOOP
# ─────────────────────────────────────────────
log("\n" + "="*62)
log("TRAINING MODELS (1 per commodity × 3 model types = 66 total)")
log("="*62)

commodities = sorted(df["commodity"].unique())
all_results = []

# Commodities to generate full plots for (volatile + stable representatives)
PLOT_COMMODITIES = {"tomato", "onion", "potato", "rice", "tur_dal", "mustard_oil"}

total_start = time.time()

for i, commodity in enumerate(commodities, 1):
    c_start = time.time()
    log(f"\n[{i:02d}/{len(commodities)}] {commodity.upper()}")
    log("-" * 40)

    # Filter to this commodity
    df_c = df[df["commodity"] == commodity].copy()

    # Per-commodity train/test split
    train = df_c[df_c["date"] <= TRAIN_END].copy()
    test  = df_c[df_c["date"] >= TEST_START].copy()

    log(f"  Train rows: {len(train):,} | Test rows: {len(test):,}")

    if len(test) < 30:
        log(f"  SKIP — insufficient test data ({len(test)} rows)")
        continue

    # Feature matrix and target
    # Drop any columns with all NaN in this commodity's data
    valid_features = [c for c in FEATURE_COLS if c in df_c.columns]

    X_train = train[valid_features].copy()
    y_train = train[TARGET].copy()
    X_test  = test[valid_features].copy()
    y_test  = test[TARGET].copy()

    # Drop rows where target is NaN
    train_mask = y_train.notna()
    test_mask  = y_test.notna()
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test,  y_test  = X_test[test_mask],   y_test[test_mask]

    # Fill remaining NaN in features with column median (safety net)
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test  = X_test.fillna(X_train.median(numeric_only=True))  # use train median for test

    # ── BASELINE: Rolling 30-day mean ──────────────────────
    # Predict using the last known 30-day average (no ML needed)
    # This is our benchmark — if XGBoost can't beat this, something's wrong
    if "retail_roll_mean_30d" in X_test.columns:
        baseline_preds = X_test["retail_roll_mean_30d"].values
    else:
        baseline_preds = np.full(len(y_test), y_train.mean())

    baseline_metrics = compute_metrics(y_test, baseline_preds, "Baseline")
    baseline_metrics.update({"commodity": commodity, "n_train": len(y_train), "n_test": len(y_test)})
    all_results.append(baseline_metrics)
    log(f"  Baseline RMSE={baseline_metrics['rmse']:.2f} MAE={baseline_metrics['mae']:.2f} MAPE={baseline_metrics['mape']:.1f}% R2={baseline_metrics['r2']:.3f}")

    # Validation set for early stopping — last 10% of train
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

    # ── XGBOOST ────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(
        n_estimators       = N_EST,
        learning_rate      = 0.05,
        max_depth          = 6,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 5,
        reg_lambda         = 1.0,
        random_state       = 42,
        tree_method        = "hist",
        early_stopping_rounds = 50,
        eval_metric        = "rmse",
        verbosity          = 0,
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = compute_metrics(y_test, xgb_preds, "XGBoost")
    xgb_metrics.update({"commodity": commodity, "n_train": len(y_train), "n_test": len(y_test)})
    all_results.append(xgb_metrics)
    log(f"  XGBoost  RMSE={xgb_metrics['rmse']:.2f} MAE={xgb_metrics['mae']:.2f} MAPE={xgb_metrics['mape']:.1f}% R2={xgb_metrics['r2']:.3f}  [stopped at {xgb_model.best_iteration} rounds]")

    # ── LIGHTGBM ───────────────────────────────────────────
    lgb_model = lgb.LGBMRegressor(
        n_estimators       = N_EST,
        learning_rate      = 0.05,
        max_depth          = 6,
        num_leaves         = 31,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_samples  = 20,
        reg_lambda         = 1.0,
        random_state       = 42,
        verbose            = -1,
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    )
    lgb_preds = lgb_model.predict(X_test)
    lgb_metrics = compute_metrics(y_test, lgb_preds, "LightGBM")
    lgb_metrics.update({"commodity": commodity, "n_train": len(y_train), "n_test": len(y_test)})
    all_results.append(lgb_metrics)
    log(f"  LightGBM RMSE={lgb_metrics['rmse']:.2f} MAE={lgb_metrics['mae']:.2f} MAPE={lgb_metrics['mape']:.1f}% R2={lgb_metrics['r2']:.3f}  [stopped at {lgb_model.best_iteration_} rounds]")

    # ── SAVE MODELS ────────────────────────────────────────
    with open(f"{MODELS_DIR}/xgb_{commodity}.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(f"{MODELS_DIR}/lgbm_{commodity}.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    # ── PLOTS (only for selected commodities) ──────────────
    if commodity in PLOT_COMMODITIES:
        test_dates = test[test_mask]["date"].values

        # Predictions vs Actual
        plot_predictions(
            commodity, test_dates, y_test.values,
            {"Baseline": baseline_preds, "XGBoost": xgb_preds, "LightGBM": lgb_preds},
            f"{PLOTS_DIR}/predictions_{commodity}.png"
        )

        # Feature importance — XGBoost
        xgb_imp = xgb_model.feature_importances_
        plot_feature_importance(
            commodity, "XGBoost",
            valid_features, xgb_imp,
            f"{PLOTS_DIR}/feat_importance_xgb_{commodity}.png"
        )

        # Feature importance — LightGBM
        lgb_imp = lgb_model.feature_importances_
        plot_feature_importance(
            commodity, "LightGBM",
            valid_features, lgb_imp,
            f"{PLOTS_DIR}/feat_importance_lgbm_{commodity}.png"
        )

    elapsed = time.time() - c_start
    log(f"  Done in {elapsed:.1f}s")


# ─────────────────────────────────────────────
# STEP 5 — SAVE RESULTS & SUMMARY
# ─────────────────────────────────────────────
log("\n" + "="*62)
log("RESULTS SUMMARY")
log("="*62)

results_df = pd.DataFrame(all_results)
results_df = results_df[["commodity", "model", "rmse", "mae", "mape", "r2", "n_train", "n_test"]]
results_df.to_csv(RESULTS_PATH, index=False)
log(f"\nSaved -> {RESULTS_PATH}")

# ── Per-commodity best model ──────────────────
log("\n[BEST MODEL PER COMMODITY]")
log(f"{'Commodity':<20} {'Best':<12} {'MAPE%':>7}  {'R2':>7}  vs Baseline MAPE%")
log("-" * 65)

pivot = results_df.pivot_table(index="commodity", columns="model",
                                values=["mape","r2"], aggfunc="first")
for commodity in sorted(results_df["commodity"].unique()):
    rows = results_df[results_df["commodity"] == commodity]
    ml_rows = rows[rows["model"].isin(["XGBoost", "LightGBM"])]
    if ml_rows.empty:
        continue
    best = ml_rows.loc[ml_rows["mape"].idxmin()]
    base = rows[rows["model"] == "Baseline"].iloc[0] if len(rows[rows["model"] == "Baseline"]) > 0 else None
    base_mape = f"{base['mape']:.1f}%" if base is not None else "N/A"
    improvement = f"(was {base_mape})" if base is not None else ""
    log(f"  {commodity:<20} {best['model']:<12} {best['mape']:>6.1f}%  {best['r2']:>7.3f}  {improvement}")

# ── Overall model comparison ──────────────────
log("\n[OVERALL AVERAGE METRICS BY MODEL]")
log(f"{'Model':<12} {'Avg MAPE%':>10} {'Avg RMSE':>10} {'Avg MAE':>10} {'Avg R2':>8}")
log("-"*55)
for model_name in ["Baseline", "XGBoost", "LightGBM"]:
    m = results_df[results_df["model"] == model_name]
    if not m.empty:
        log(f"  {model_name:<12} {m['mape'].mean():>9.2f}%  {m['rmse'].mean():>9.3f}  {m['mae'].mean():>9.3f}  {m['r2'].mean():>7.3f}")

# ── Hardest vs easiest commodities ───────────
log("\n[DIFFICULTY RANKING — by LightGBM MAPE (ascending = easiest)]")
lgbm_results = results_df[results_df["model"] == "LightGBM"].sort_values("mape")
for _, row in lgbm_results.iterrows():
    difficulty = "EASY" if row["mape"] < 5 else "MEDIUM" if row["mape"] < 15 else "HARD"
    log(f"  {row['commodity']:<22} MAPE={row['mape']:>5.1f}%  R2={row['r2']:.3f}  [{difficulty}]")

total_time = time.time() - total_start
log(f"\nTotal training time: {total_time/60:.1f} minutes")
log(f"Models saved: {len(os.listdir(MODELS_DIR))} files in {MODELS_DIR}/")
log(f"Plots saved:  {len(os.listdir(PLOTS_DIR))} files in {PLOTS_DIR}/")

# ── Save report ───────────────────────────────
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"Report saved -> {REPORT_PATH}")

log("\n" + "="*62)
log("D3 COMPLETE -- Ready for D4: Streamlit Dashboard")
log("="*62)