"""
02_features.py
Turns clean campaign rows into KPIs + calendar features + the ML target.
Depends on: 01_clean_data.py  (needs data/processed/clean_campaigns.csv)
Output    : data/processed/features.csv  and  data/processed/target_info.json
Run       : python src/02_features.py
"""
import json                                        # to save the target threshold for later scripts
from pathlib import Path                           # cross-platform file paths
import numpy as np                                 # inf handling
import pandas as pd                                # tables

ROOT = Path(__file__).resolve().parents[1]         # project root
IN_FILE = ROOT / "data" / "processed" / "clean_campaigns.csv"
OUT_FILE = ROOT / "data" / "processed" / "features.csv"
INFO_FILE = ROOT / "data" / "processed" / "target_info.json"

ROI_TARGET = 1.0        # business rule from the brief: success = ROI >= 1.0 (revenue at least 2x spend)
MIN_CLASS_SHARE = 0.05  # if a class is rarer than 5%, the model has nothing to learn from (see step 7)

# Month -> season. Edit this if your business is in another climate/market.
# Oct-Dec is grouped as "Festive" because Indian retail peaks around Diwali/year-end.
SEASON_MAP = {12: "Festive", 10: "Festive", 11: "Festive",
              1: "Winter", 2: "Winter",
              3: "Summer", 4: "Summer", 5: "Summer",
              6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon"}

# ---------- 1. Load ------------------------------------------------------------
if not IN_FILE.exists():                           # helpful error if Step 3 was skipped
    raise SystemExit(f"Missing {IN_FILE}. Run 01_clean_data.py first.")
df = pd.read_csv(IN_FILE, parse_dates=["start_date", "end_date"])  # parse_dates restores real datetimes
print(f"Loaded {len(df):,} clean campaigns")

# ---------- 2. Marketing KPIs --------------------------------------------------
# Cleaning guaranteed spend, impressions and clicks are > 0, so these divisions are safe.
df["roi"] = (df["revenue"] - df["spend"]) / df["spend"]        # profit per unit spent: 1.0 = earned 2x back
df["roas"] = df["revenue"] / df["spend"]                       # revenue per unit spent (always ROI + 1)
df["ctr"] = df["clicks"] / df["impressions"]                   # click-through rate: how many views became clicks
df["conversion_rate"] = df["conversions"] / df["clicks"]       # how many clicks became customers
# Conversions CAN be 0 (a flop campaign). Dividing by zero gives inf, which breaks averages,
# so we turn inf into NaN ("cost per customer is undefined - nobody was acquired").
df["cpa"] = (df["spend"] / df["conversions"]).replace([np.inf, -np.inf], np.nan)  # cost per acquisition

# ---------- 3. Calendar features ----------------------------------------------
df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days   # how long the campaign ran
df["month"] = df["start_date"].dt.month                        # 1-12: seasonality signal for the model
df["quarter"] = df["start_date"].dt.quarter                    # 1-4: for quarterly business reviews
df["season"] = df["month"].map(SEASON_MAP)                     # Winter/Summer/Monsoon/Festive

# ---------- 4. ML target -------------------------------------------------------
target_used = ROI_TARGET                           # start with the business rule
auto_fallback = False                              # remembers whether we had to change it
share = (df["roi"] >= target_used).mean()          # fraction of campaigns that would be "successful"
print(f"With ROI >= {target_used}: {share:.1%} of campaigns are successful")
if share < MIN_CLASS_SHARE or share > 1 - MIN_CLASS_SHARE:
    # WHY: a target that is ~100% one class makes every model look brilliant by always
    # guessing that class. That is what happens with the Kaggle file (ROI is always 2-8).
    target_used = float(df["roi"].median())        # the median splits the data 50/50
    auto_fallback = True
    print(f"WARNING: target too lopsided to learn from -> switching to the MEDIAN ROI ({target_used:.2f}).")
    print("         Say so in your report: this is a demonstration target, not a business rule.")
df["is_successful"] = (df["roi"] >= target_used).astype(int)   # 1 = hit the target, 0 = missed

# ---------- 5. Save ------------------------------------------------------------
df.to_csv(OUT_FILE, index=False)                   # features for EDA, model, dashboard
INFO_FILE.write_text(json.dumps({"roi_target": target_used, "auto_fallback": auto_fallback}))  # later scripts read this
print(f"\nSuccess rate used for modelling: {df['is_successful'].mean():.1%}")
print(f"Median ROI {df['roi'].median():.2f} | median CTR {df['ctr'].median():.2%} | "
      f"median conversion rate {df['conversion_rate'].median():.2%}")
print(f"Campaigns with zero conversions (CPA undefined): {df['cpa'].isna().sum()}")
print(f"Saved -> {OUT_FILE}")