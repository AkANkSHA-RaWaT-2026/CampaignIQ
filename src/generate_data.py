"""
generate_data.py
Creates a SYNTHETIC but realistic marketing dataset (2,000 rows) with
deliberate messiness, so you can practise cleaning without Kaggle.

Output: data/raw/campaigns_raw.csv
Run   : python src/generate_data.py
"""
from pathlib import Path          # Path handles Windows/Mac/Linux slashes for us
import numpy as np                # numeric random generation
import pandas as pd               # tables

# ---------- 0. Settings ------------------------------------------------------
SEED = 42                          # fixed seed -> you and I get identical data
N_ROWS = 2000                      # number of campaigns requested in the brief
rng = np.random.default_rng(SEED)  # modern NumPy random generator

# Project root = the folder that contains /src, whatever your OS or cwd is
ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "data" / "raw" / "campaigns_raw.csv"

# ---------- 1. Business "rules of the world" ---------------------------------
# WHY: a model can only learn patterns that exist. We plant real, believable
# patterns (channel quality, seasonality, segment fit) so Step 6 has signal.
CHANNELS = {
    #                 median spend (INR), CPM (cost per 1000 views), CTR,  conv. rate, order value (INR)
    "Email":        dict(spend=40_000,  cpm=450, ctr=0.030, cvr=0.060, aov=750),
    "Social Media": dict(spend=120_000, cpm=200, ctr=0.015, cvr=0.030, aov=700),
    "Search Ads":   dict(spend=150_000, cpm=550, ctr=0.040, cvr=0.045, aov=900),
    "Influencer":   dict(spend=200_000, cpm=280, ctr=0.020, cvr=0.028, aov=800),
    "TV/Print":     dict(spend=400_000, cpm=65,  ctr=0.004, cvr=0.020, aov=950),
}
SEGMENTS = ["Young Professionals", "Students", "Families", "Seniors", "Small Business Owners"]
REGIONS = ["North", "South", "East", "West"]

# Month multipliers: festive season (Oct-Dec) sells more, monsoon (Jul-Aug) is soft
SEASON = {1: .95, 2: .95, 3: 1.0, 4: 1.0, 5: .95, 6: .9,
          7: .85, 8: .85, 9: 1.0, 10: 1.25, 11: 1.35, 12: 1.2}

# Segment x channel affinity: >1 means "this segment responds well to this channel"
AFFINITY = {(s, c): 1.0 for s in SEGMENTS for c in CHANNELS}
AFFINITY.update({
    ("Students", "Social Media"): 1.35, ("Students", "Influencer"): 1.30,
    ("Students", "TV/Print"): 0.70, ("Young Professionals", "Email"): 1.25,
    ("Young Professionals", "Search Ads"): 1.20, ("Families", "TV/Print"): 1.30,
    ("Families", "Email"): 1.10, ("Seniors", "TV/Print"): 1.35,
    ("Seniors", "Social Media"): 0.65, ("Seniors", "Influencer"): 0.60,
    ("Small Business Owners", "Search Ads"): 1.30, ("Small Business Owners", "Email"): 1.15,
})

# ---------- 2. Build CLEAN data first ----------------------------------------
channel = rng.choice(list(CHANNELS), size=N_ROWS, p=[.22, .25, .22, .16, .15])
segment = rng.choice(SEGMENTS, size=N_ROWS)
region = rng.choice(REGIONS, size=N_ROWS)

# Random start dates across 2023-2024, duration between 7 and 60 days
start = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, N_ROWS), unit="D")
end = start + pd.to_timedelta(rng.integers(7, 61, N_ROWS), unit="D")

rows = []
for i in range(N_ROWS):
    p = CHANNELS[channel[i]]                                    # this channel's parameters
    spend = np.clip(rng.lognormal(np.log(p["spend"]), 0.5), 5_000, 1_000_000)  # skewed like real budgets
    impressions = int(spend / p["cpm"] * 1000 * rng.uniform(0.8, 1.2))          # spend buys views
    clicks = rng.binomial(impressions, np.clip(p["ctr"] * rng.uniform(0.7, 1.3), 0, 1))
    fit = AFFINITY[(segment[i], channel[i])] * SEASON[start[i].month]           # segment fit x season
    saturation = (spend / p["spend"]) ** -0.12      # diminishing returns: huge budgets convert worse
    cvr = np.clip(p["cvr"] * fit * saturation * rng.uniform(0.75, 1.25), 0, 1)
    conversions = rng.binomial(clicks, cvr)
    revenue = conversions * p["aov"] * rng.uniform(0.8, 1.2)                    # orders x order value
    rows.append((spend, impressions, clicks, conversions, revenue))

df = pd.DataFrame(rows, columns=["spend", "impressions", "clicks", "conversions", "revenue"])
df.insert(0, "campaign_id", [f"CMP-{i:05d}" for i in range(1, N_ROWS + 1)])
df.insert(1, "channel", channel)
df.insert(2, "start_date", start)
df.insert(3, "end_date", end)
df["customer_segment"] = segment
df["region"] = region
df[["spend", "revenue"]] = df[["spend", "revenue"]].round(2)

# ---------- 3. INJECT MESSINESS ----------------------------------------------
def pick(frac):
    """Return the row positions of a random `frac` share of the table."""
    return rng.choice(len(df), size=int(len(df) * frac), replace=False)

# 3a. Dates -> mixed text formats (the #1 real-world headache)
formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%B %d, %Y"]
for col in ["start_date", "end_date"]:
    chosen = rng.choice(formats, size=len(df), p=[.55, .20, .15, .10])
    df[col] = [d.strftime(f) for d, f in zip(pd.to_datetime(df[col]), chosen)]

# 3b. Channel & segment names -> inconsistent casing, spaces, spellings
channel_variants = {
    "Email": ["email", "EMAIL", " Email ", "E-mail"],
    "Social Media": ["social media", "Social media", "SocialMedia", "Social  Media"],
    "Search Ads": ["search ads", "Search ads", "SearchAds", "Search Ad"],
    "Influencer": ["influencer", "INFLUENCER", "Influencers"],
    "TV/Print": ["tv/print", "TV / Print", "TV-Print", "Tv/Print"],
}
df["channel"] = df["channel"].astype(object)      # allow mixed text values
for idx in pick(0.25):
    df.at[idx, "channel"] = rng.choice(channel_variants[df.at[idx, "channel"]])
df["customer_segment"] = df["customer_segment"].astype(object)
for idx in pick(0.10):
    df.at[idx, "customer_segment"] = rng.choice(
        [df.at[idx, "customer_segment"].lower(), df.at[idx, "customer_segment"].upper(),
         " " + df.at[idx, "customer_segment"] + " "])

# 3c. Spend/revenue as text with currency symbols & commas (common in Excel exports)
df["spend"] = df["spend"].astype(object)
df["revenue"] = df["revenue"].astype(object)
for idx in pick(0.05):
    df.at[idx, "spend"] = f"₹{float(df.at[idx, 'spend']):,.2f}"
for idx in pick(0.05):
    df.at[idx, "revenue"] = f"₹{float(df.at[idx, 'revenue']):,.2f}"

# 3d. Missing values
df.loc[pick(0.03), "spend"] = np.nan
df.loc[pick(0.03), "revenue"] = np.nan
df.loc[pick(0.02), "region"] = np.nan
df.loc[pick(0.02), "customer_segment"] = np.nan

# 3e. Negative values (sign errors from bad data entry)
for idx in pick(0.005):
    if isinstance(df.at[idx, "spend"], float) and not np.isnan(df.at[idx, "spend"]):
        df.at[idx, "spend"] = -abs(df.at[idx, "spend"])
for idx in pick(0.005):
    if isinstance(df.at[idx, "revenue"], float) and not np.isnan(df.at[idx, "revenue"]):
        df.at[idx, "revenue"] = -abs(df.at[idx, "revenue"])

# 3f. Impossible rows: more clicks than impressions
bad = pick(0.004)
df.loc[bad, "clicks"] = df.loc[bad, "impressions"] + 50

# 3g. Duplicates: exact copies of ~2% of rows appended at the bottom
df = pd.concat([df, df.iloc[pick(0.02)]], ignore_index=True)

# Shuffle so duplicates are not conveniently at the bottom
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------- 4. Save ----------------------------------------------------------
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)   # create folder if missing
df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows ₹ correctly
print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {OUT_FILE}")
print("Missing values per column:\n", df.isna().sum()[lambda s: s > 0].to_string())