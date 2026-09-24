"""
01_clean_data.py
Cleans data/raw/campaigns_raw.csv -> data/processed/clean_campaigns.csv
Depends on: generate_data.py OR prepare_kaggle.py (run one of them first).
Run: python src/01_clean_data.py
"""
import re                                          # regular expressions, used to normalise text labels
from pathlib import Path                           # cross-platform file paths
import numpy as np                                 # NaT/NaN helpers
import pandas as pd                                # the main data tool

ROOT = Path(__file__).resolve().parents[1]         # project root, independent of where you launch from
IN_FILE = ROOT / "data" / "raw" / "campaigns_raw.csv"          # input written by Step 2
OUT_FILE = ROOT / "data" / "processed" / "clean_campaigns.csv" # output for Step 4

# ---------- Lookup tables (edit these if your data has new spellings) ----------
# Keys are LOWERCASE LETTERS ONLY: "TV / Print", "tv-print", "Tv/Print" all become "tvprint".
CHANNEL_MAP = {                                    # normalised spelling -> official name
    "email": "Email",                              # "E-mail", "EMAIL", " email " ...
    "socialmedia": "Social Media",                 # "social media", "SocialMedia" ...
    "social": "Social Media",                      # short form
    "searchads": "Search Ads",                     # "search ads", "SearchAds" ...
    "searchad": "Search Ads",                      # singular typo
    "search": "Search Ads",                        # Kaggle file calls this just "Search"
    "influencer": "Influencer",                    # any casing
    "influencers": "Influencer",                   # plural typo
    "tvprint": "TV/Print",                         # every slash/dash/space variant
}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%B %d, %Y"]  # formats we expect, tried in order
NUMERIC_COLS = ["spend", "impressions", "clicks", "conversions", "revenue"]  # columns that must be numbers

log = []                                           # we record rows lost at every step for the summary


def record(step, before, after):
    """Store how many rows a step removed - a cleaning script must be auditable."""
    log.append((step, before - after))             # (name, rows removed)


def parse_dates(series):
    """Parse a column holding several date formats, one explicit format at a time."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")  # start with 'unknown'
    text = series.astype(str).str.strip()          # trim stray spaces
    for fmt in DATE_FORMATS:                       # try each known format
        todo = parsed.isna()                       # only rows still unparsed
        parsed[todo] = pd.to_datetime(text[todo], format=fmt, errors="coerce")  # bad ones become NaT
    return parsed                                  # explicit formats avoid the day/month guessing trap


# ---------- 1. Load ------------------------------------------------------------
if not IN_FILE.exists():                           # fail early with a helpful message
    raise SystemExit(f"Missing {IN_FILE}. Run generate_data.py or prepare_kaggle.py first.")
df = pd.read_csv(IN_FILE, encoding="utf-8-sig")    # utf-8-sig removes the invisible BOM Excel adds
df.columns = df.columns.str.strip().str.lower()    # tidy column names: " Spend" -> "spend"
raw_rows = len(df)                                 # remember the starting size
print(f"Loaded {raw_rows:,} rows, {df.shape[1]} columns")

# ---------- 2. Duplicates ------------------------------------------------------
n = len(df)                                        # rows before this step
df = df.drop_duplicates()                          # identical rows are the same campaign copied twice
record("exact duplicate rows", n, len(df))         # log the loss
n = len(df)
df = df.drop_duplicates(subset="campaign_id", keep="first")  # same ID twice = one campaign, keep first
record("repeated campaign_id", n, len(df))

# ---------- 3. Standardise text labels ----------------------------------------
key = df["channel"].astype(str).map(lambda s: re.sub(r"[^a-z]", "", s.lower()))  # letters only, lowercase
official = key.map(CHANNEL_MAP)                    # look up the official name (NaN if unknown)
df["channel"] = official.fillna(df["channel"].astype(str).str.strip().str.title())  # unknown labels: tidy, don't lose
for col in ["customer_segment", "region"]:         # same treatment for both category columns
    df[col] = (df[col].astype("string")            # nullable text type keeps real missing values as <NA>
               .str.strip()                        # remove leading/trailing spaces
               .str.replace(r"\s+", " ", regex=True)  # collapse double spaces
               .str.title()                        # "STUDENTS"/"students" -> "Students"
               .fillna("Unknown"))                 # keep the row: a missing label is not a reason to lose money data
df["campaign_id"] = df["campaign_id"].astype(str).str.strip().str.upper()  # consistent IDs

# ---------- 4. Dates -----------------------------------------------------------
df["start_date"] = parse_dates(df["start_date"])   # mixed text -> real datetimes
df["end_date"] = parse_dates(df["end_date"])
n = len(df)
df = df.dropna(subset=["start_date", "end_date"])  # a campaign with no dates cannot be analysed over time
record("unparseable dates", n, len(df))
n = len(df)
df = df[df["end_date"] >= df["start_date"]]        # a campaign cannot end before it starts
record("end_date before start_date", n, len(df))

# ---------- 5. Numbers stored as text -----------------------------------------
for col in NUMERIC_COLS:                           # "₹92,705.60" -> 92705.60
    as_text = df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)  # keep digits, dot, minus only
    df[col] = pd.to_numeric(as_text, errors="coerce")  # anything unreadable becomes NaN

# ---------- 6. Missing spend / revenue ----------------------------------------
# WHY DROP, NOT FILL? ROI = (revenue - spend) / spend. If we filled a missing
# revenue with an average, we would INVENT the very thing we're trying to measure
# and hide it inside our ML target. Honest analysts drop these rows and report how many.
n = len(df)
df = df.dropna(subset=NUMERIC_COLS)                # any missing core number -> row unusable
record("missing spend/revenue/funnel numbers", n, len(df))

# ---------- 7. Impossible values ----------------------------------------------
n = len(df)
df = df[(df[NUMERIC_COLS] >= 0).all(axis=1)]       # negative money/clicks = data-entry errors -> remove
record("negative values", n, len(df))
n = len(df)
df = df[(df["spend"] > 0) & (df["impressions"] > 0) & (df["clicks"] > 0)]  # zero would cause divide-by-zero in KPIs
record("zero spend/impressions/clicks", n, len(df))
n = len(df)
df = df[(df["clicks"] <= df["impressions"]) & (df["conversions"] <= df["clicks"])]  # funnel can only narrow
record("funnel violations (clicks>impressions etc.)", n, len(df))

# ---------- 8. Final tidy + save ----------------------------------------------
df = df.sort_values("start_date").reset_index(drop=True)   # chronological order helps time charts
for col in ["impressions", "clicks", "conversions"]:       # counts should be whole numbers
    df[col] = df[col].astype(int)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True) # create data/processed if it doesn't exist
df.to_csv(OUT_FILE, index=False)                   # write the cleaned file

# ---------- 9. Summary ---------------------------------------------------------
print("\nRows removed per step:")
for step, lost in log:                             # show the audit trail
    print(f"  {step:<48}{lost:>8,}")
print(f"\nRows kept: {len(df):,} of {raw_rows:,} ({len(df) / raw_rows:.1%})")
print(f"Channels : {sorted(df['channel'].unique())}")
print(f"Segments : {sorted(df['customer_segment'].unique())}")
print(f"Dates    : {df['start_date'].min():%Y-%m-%d} -> {df['start_date'].max():%Y-%m-%d}")
print(f"Saved -> {OUT_FILE}")