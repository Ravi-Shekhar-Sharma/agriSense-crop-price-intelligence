"""
AgriSense | notebooks/01_eda.py
================================
Day 1 — Exploratory Data Analysis

What we explore here (in order):
  1. Dataset overview — shape, date range, states, commodities
  2. Missing data audit — which states/commodities have gaps
  3. National price trends — how key crops moved over 2 years
  4. Volatility ranking — which crops are most price-unstable
  5. Retail–Wholesale margin analysis — where is money being made
  6. State-wise price ranking — which states pay most/least
  7. Seasonal patterns — month-level averages for vegetables

WHY each analysis matters for AgriSense:
  - Trends     → what to forecast
  - Volatility → risk zones (high volatility = procurement risk)
  - Margin     → intervention opportunity (can we reduce trader margin?)
  - State rank → where to focus the dashboard
  - Seasonal   → harvest cycle signal we'll feed into the model

Run: python notebooks/01_eda.py
Plots saved to: outputs/plots/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves files instead of showing window)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ── path setup so we can import src/ ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import COMMODITY_CATEGORIES, run_pipeline

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi': 130,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

SAVE = lambda name: plt.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_processed():
    proc = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    retail    = pd.read_csv(os.path.join(proc, 'retail_states.csv'),    parse_dates=['Date'])
    wholesale = pd.read_csv(os.path.join(proc, 'wholesale_states.csv'), parse_dates=['Date'])
    merged    = pd.read_csv(os.path.join(proc, 'merged_retail_wholesale.csv'), parse_dates=['Date'])
    return retail, wholesale, merged


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def overview(retail, wholesale, merged):
    print("\n" + "="*60)
    print("1. DATASET OVERVIEW")
    print("="*60)
    print(f"Date range       : {retail['Date'].min().date()} → {retail['Date'].max().date()}")
    print(f"Calendar days    : {(retail['Date'].max() - retail['Date'].min()).days + 1}")
    print(f"Unique dates     : {retail['Date'].nunique()} (retail) | {wholesale['Date'].nunique()} (wholesale)")
    print(f"States           : {retail['States/UTs'].nunique()}")
    print(f"Commodities      : 22")
    print(f"Retail rows      : {len(retail):,}")
    print(f"Wholesale rows   : {len(wholesale):,}")
    print(f"Merged rows      : {len(merged):,}")

    all_states = sorted(retail['States/UTs'].unique())
    print(f"\nAll states ({len(all_states)}):")
    for i, s in enumerate(all_states):
        print(f"  {s}", end='\n' if (i+1) % 4 == 0 else '')
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING DATA AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def missing_data_heatmap(retail, wholesale):
    """
    WHY: Before modelling we need to know where the gaps are.
    A state with 40% missing data needs different treatment than one with 2%.
    We visualise missing % as a heatmap — rows=states, cols=commodities.
    """
    print("\n" + "="*60)
    print("2. MISSING DATA AUDIT")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    for ax, df, label in zip(axes, [retail, wholesale], ['Retail', 'Wholesale']):
        # Renamed commodity columns present in df
        comm_cols = [c for c in df.columns if c not in ['States/UTs', 'Date', 'price_type']]
        miss = (
            df.groupby('States/UTs')[comm_cols]
            .apply(lambda x: x.isna().mean() * 100)
        )
        sns.heatmap(
            miss, ax=ax, cmap='YlOrRd', vmin=0, vmax=100,
            linewidths=0.3, cbar_kws={'label': '% Missing'}
        )
        ax.set_title(f'{label} — Missing Data % by State × Commodity', fontweight='bold')
        ax.set_xlabel('Commodity')
        ax.set_ylabel('State')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=7)

    plt.tight_layout()
    SAVE('01_missing_data_heatmap.png')
    plt.close()
    print("  → Saved: 01_missing_data_heatmap.png")

    # Print top missing combos
    comm_cols = [c for c in retail.columns if c not in ['States/UTs', 'Date', 'price_type']]
    miss_overall = retail[comm_cols].isna().mean() * 100
    high_miss = miss_overall[miss_overall > 5].sort_values(ascending=False)
    if len(high_miss):
        print("\n  Retail commodities with >5% missing data:")
        print(high_miss.to_string())
    else:
        print("\n  ✓ No retail commodity exceeds 5% missing data nationally")


# ─────────────────────────────────────────────────────────────────────────────
# 3. NATIONAL PRICE TRENDS (AVERAGE ACROSS STATES)
# ─────────────────────────────────────────────────────────────────────────────

def national_price_trends(retail):
    """
    WHY: The first question any stakeholder asks is "how have prices moved?"
    We plot the national average (mean across all states) over time.
    We pick 8 representative commodities — 2 per category — for readability.
    Rolling 30-day average smooths daily noise so the trend is visible.
    """
    print("\n" + "="*60)
    print("3. NATIONAL PRICE TRENDS")
    print("="*60)

    highlight_crops = ['Rice', 'Wheat', 'Tur Dal', 'Onion', 'Tomato', 'Potato',
                       'Mustard Oil', 'Sugar']

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    axes = axes.flatten()

    for ax, crop in zip(axes, highlight_crops):
        if crop not in retail.columns:
            ax.set_visible(False)
            continue

        # National daily average (mean of all state prices on that date)
        national = retail.groupby('Date')[crop].mean()
        rolling  = national.rolling(30, min_periods=7).mean()

        ax.fill_between(national.index, national, alpha=0.15, color='steelblue')
        ax.plot(national.index, national, alpha=0.3, color='steelblue', lw=0.8)
        ax.plot(rolling.index,  rolling,  color='steelblue', lw=2, label='30-day avg')

        ax.set_title(crop, fontweight='bold')
        ax.set_ylabel('₹/kg')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.legend(fontsize=8)

        # Print summary stats
        pct_chg = ((national.iloc[-1] - national.iloc[0]) / national.iloc[0]) * 100
        print(f"  {crop:<18} | Start: ₹{national.iloc[0]:.1f} | End: ₹{national.iloc[-1]:.1f} | Change: {pct_chg:+.1f}%")

    fig.suptitle('National Retail Price Trends — 30-Day Rolling Average (Aug 2022 – Aug 2024)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    SAVE('02_national_price_trends.png')
    plt.close()
    print("  → Saved: 02_national_price_trends.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VOLATILITY RANKING
# ─────────────────────────────────────────────────────────────────────────────

def volatility_ranking(retail):
    """
    WHY: Volatility = risk. High CV (Coefficient of Variation = std/mean)
    means the price is unpredictable — bad for procurement planning.
    CV is better than raw std because it's scale-independent:
    ₹10 std on a ₹20 item is catastrophic; ₹10 std on a ₹200 item is fine.

    This chart directly answers: "Which crops need better price stabilisation?"
    — a core question in the AgriSense brief.
    """
    print("\n" + "="*60)
    print("4. VOLATILITY RANKING (Coefficient of Variation)")
    print("="*60)

    national_avg = retail.groupby('Date').mean(numeric_only=True)

    # Coefficient of Variation (CV) = std / mean — expressed as %
    cv = (national_avg.std() / national_avg.mean() * 100).sort_values(ascending=False)
    cv = cv.dropna()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#e74c3c' if v > 20 else '#f39c12' if v > 10 else '#27ae60' for v in cv.values]
    bars = ax.barh(cv.index, cv.values, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, cv.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    ax.set_xlabel('Coefficient of Variation (%) — Higher = More Volatile')
    ax.set_title('Price Volatility Ranking — Retail (Aug 2022 – Aug 2024)', fontweight='bold')
    ax.axvline(20, color='red',    lw=1.5, ls='--', alpha=0.7, label='High risk (>20%)')
    ax.axvline(10, color='orange', lw=1.5, ls='--', alpha=0.7, label='Medium risk (>10%)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    SAVE('03_volatility_ranking.png')
    plt.close()

    print("  Top 5 most volatile commodities:")
    for crop, v in cv.head(5).items():
        print(f"    {crop:<25} CV = {v:.1f}%")
    print("  → Saved: 03_volatility_ranking.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. RETAIL – WHOLESALE MARGIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def margin_analysis(merged):
    """
    WHY: Margin = Retail price − Wholesale price (both in ₹/kg after conversion).
    This is the money traders/retailers add. High margin = consumers pay more
    than needed, farmers get less than possible.

    From a procurement standpoint: states with high margins are exactly where
    direct procurement chains could add the most value.

    Margin % = (margin / wholesale) × 100 — tells us what % markup is applied.
    """
    print("\n" + "="*60)
    print("5. RETAIL–WHOLESALE MARGIN ANALYSIS")
    print("="*60)

    # Identify margin columns
    margin_cols = [c for c in merged.columns if c.startswith('margin_')]
    commodity_names = [c.replace('margin_', '') for c in margin_cols]

    # National average margin per commodity
    avg_margin = merged[margin_cols].mean().values
    avg_retail = merged[[f'retail_{c}' for c in commodity_names if f'retail_{c}' in merged.columns]].mean().values
    avg_wholesale = merged[[f'wholesale_{c}' for c in commodity_names if f'wholesale_{c}' in merged.columns]].mean().values

    # Sort by absolute margin descending
    order = np.argsort(avg_margin)[::-1]
    names_sorted   = [commodity_names[i] for i in order]
    margin_sorted  = avg_margin[order]
    pct_sorted     = [(m/w)*100 if w > 0 else 0
                      for m, w in zip(margin_sorted, avg_wholesale[order])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: absolute margin in ₹/kg
    colors = ['#e74c3c' if m > 5 else '#f39c12' if m > 2 else '#27ae60' for m in margin_sorted]
    ax1.barh(names_sorted, margin_sorted, color=colors, edgecolor='white')
    for i, v in enumerate(margin_sorted):
        ax1.text(v + 0.1, i, f'₹{v:.1f}', va='center', fontsize=9)
    ax1.set_xlabel('Average Margin (₹/kg)')
    ax1.set_title('Retail – Wholesale Margin\n(absolute, ₹/kg)', fontweight='bold')
    ax1.invert_yaxis()

    # Right: margin as % of wholesale
    ax2.barh(names_sorted, pct_sorted, color=colors, edgecolor='white')
    for i, v in enumerate(pct_sorted):
        ax2.text(v + 0.5, i, f'{v:.0f}%', va='center', fontsize=9)
    ax2.set_xlabel('Margin as % of Wholesale Price')
    ax2.set_title('Retail – Wholesale Margin\n(% of wholesale)', fontweight='bold')
    ax2.invert_yaxis()

    plt.suptitle('Trader/Retailer Markup Analysis — National Average', fontsize=13, fontweight='bold')
    plt.tight_layout()
    SAVE('04_margin_analysis.png')
    plt.close()

    print("  Top 5 highest-margin commodities (₹/kg):")
    for n, m, p in zip(names_sorted[:5], margin_sorted[:5], pct_sorted[:5]):
        print(f"    {n:<22} ₹{m:.2f}/kg  ({p:.0f}% markup)")
    print("  → Saved: 04_margin_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. STATE-WISE PRICE RANKING (Focus: 4 key crops)
# ─────────────────────────────────────────────────────────────────────────────

def statewise_ranking(retail):
    """
    WHY: Same commodity, different states → often 2–3x price difference.
    This signals logistics costs, local supply-demand imbalances, or market failures.
    We show state-level averages for 4 crops on a horizontal bar chart.
    """
    print("\n" + "="*60)
    print("6. STATE-WISE PRICE RANKING")
    print("="*60)

    focus_crops = ['Rice', 'Onion', 'Tomato', 'Mustard Oil']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, crop in zip(axes, focus_crops):
        if crop not in retail.columns:
            continue
        state_avg = retail.groupby('States/UTs')[crop].mean().sort_values()
        state_avg = state_avg.dropna()

        cmap_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(state_avg)))
        ax.barh(state_avg.index, state_avg.values, color=cmap_colors, edgecolor='white', linewidth=0.4)
        ax.set_title(f'{crop} — State Average (₹/kg)', fontweight='bold')
        ax.set_xlabel('₹/kg')
        ax.tick_params(axis='y', labelsize=8)

        nat_avg = state_avg.mean()
        ax.axvline(nat_avg, color='navy', lw=1.5, ls='--', alpha=0.7, label=f'National avg ₹{nat_avg:.1f}')
        ax.legend(fontsize=8)

        cheapest = state_avg.idxmin()
        priciest = state_avg.idxmax()
        print(f"  {crop:<15} | Cheapest: {cheapest} (₹{state_avg.min():.1f}) | "
              f"Priciest: {priciest} (₹{state_avg.max():.1f})")

    plt.suptitle('State-Wise Average Retail Prices — Aug 2022 to Aug 2024',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    SAVE('05_statewise_ranking.png')
    plt.close()
    print("  → Saved: 05_statewise_ranking.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SEASONAL PATTERNS (Vegetables — month-level)
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_patterns(retail):
    """
    WHY: Vegetables like Tomato and Onion are notoriously seasonal.
    By plotting month-level averages we can see harvest vs lean season clearly.
    This seasonal signal will become a FEATURE in the time-series model (Day 3).

    The line chart uses month-of-year on x-axis and overlays both years
    to see if the pattern repeats — confirming seasonality is real, not random.
    """
    print("\n" + "="*60)
    print("7. SEASONAL PATTERNS")
    print("="*60)

    crops = ['Tomato', 'Onion', 'Potato']
    retail_copy = retail.copy()
    retail_copy['Year']  = retail_copy['Date'].dt.year
    retail_copy['Month'] = retail_copy['Date'].dt.month

    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors_year = {2022: '#3498db', 2023: '#e74c3c', 2024: '#2ecc71'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, crop in zip(axes, crops):
        if crop not in retail_copy.columns:
            continue

        for year, grp in retail_copy.groupby('Year'):
            monthly = grp.groupby('Month')[crop].mean()
            ax.plot(monthly.index, monthly.values,
                    marker='o', lw=2, color=colors_year.get(year, 'gray'),
                    label=str(year), markersize=5)

        ax.set_title(f'{crop} — Monthly Average (₹/kg)', fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('₹/kg')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names, fontsize=8)
        ax.legend(title='Year', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Identify peak and trough months
        monthly_all = retail_copy.groupby('Month')[crop].mean()
        peak  = int(monthly_all.idxmax())
        trough = int(monthly_all.idxmin())
        print(f"  {crop:<8} | Peak month: {month_names[peak-1]} (₹{monthly_all.max():.1f}) | "
              f"Trough: {month_names[trough-1]} (₹{monthly_all.min():.1f})")

    plt.suptitle('Seasonal Price Patterns — Vegetables (Month-Year Overlay)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    SAVE('06_seasonal_patterns.png')
    plt.close()
    print("  → Saved: 06_seasonal_patterns.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🌾 AgriSense — Day 1 EDA")
    print("Loading processed data...")

    proc_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    if not os.path.exists(os.path.join(proc_dir, 'retail_states.csv')):
        print("Processed data not found — running preprocessing pipeline first...")
        run_pipeline()

    retail, wholesale, merged = load_processed()

    overview(retail, wholesale, merged)
    missing_data_heatmap(retail, wholesale)
    national_price_trends(retail)
    volatility_ranking(retail)
    margin_analysis(merged)
    statewise_ranking(retail)
    seasonal_patterns(retail)

    print("\n" + "="*60)
    print(" EDA complete. All plots saved to outputs/plots/")
    print("="*60)