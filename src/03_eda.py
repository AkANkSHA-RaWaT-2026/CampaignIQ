"""
03_eda.py
Exploratory analysis: saves 5 charts to reports/figures/ and prints one insight per chart.
Depends on: 02_features.py  (needs data/processed/features.csv and target_info.json)
Run       : python src/03_eda.py
"""
import calendar                                    # month names: 1 -> "Jan"
import json                                        # read the ROI target chosen in Step 4
from pathlib import Path                           # cross-platform file paths

import matplotlib                                  # plotting engine
matplotlib.use("Agg")                              # WHY: draw to files only - no pop-up windows, no GUI crashes on Mac/servers
import matplotlib.pyplot as plt                    # the plotting interface
import numpy as np                                 # sqrt for bubble sizes
import pandas as pd                                # tables
import seaborn as sns                              # nicer statistical charts + heatmaps
from matplotlib.ticker import FuncFormatter        # custom axis labels (1,200,000 -> 1.2M)

ROOT = Path(__file__).resolve().parents[1]         # project root
DATA_FILE = ROOT / "data" / "processed" / "features.csv"
INFO_FILE = ROOT / "data" / "processed" / "target_info.json"
FIG_DIR = ROOT / "reports" / "figures"             # where charts are written
FIG_DIR.mkdir(parents=True, exist_ok=True)         # create it if missing

# ---------- Load ---------------------------------------------------------------
if not DATA_FILE.exists():                         # helpful error if Step 4 was skipped
    raise SystemExit(f"Missing {DATA_FILE}. Run 02_features.py first.")
df = pd.read_csv(DATA_FILE, parse_dates=["start_date", "end_date"])
ROI_TARGET = json.loads(INFO_FILE.read_text())["roi_target"] if INFO_FILE.exists() else 1.0
print(f"Loaded {len(df):,} campaigns | ROI target used: {ROI_TARGET:.2f}\n")

# ---------- Shared styling & helpers -------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")   # clean grid look for every chart
CHANNELS = sorted(df["channel"].unique())              # alphabetical, so colours stay stable across charts
COLORS = dict(zip(CHANNELS, sns.color_palette("Set2", len(CHANNELS))))  # one fixed colour per channel
GOOD, BAD = "#1b9e77", "#d95f02"                       # green = above target, orange = below


def human(v, pos=None):
    """Format big numbers for axes: 1,200,000 -> '1.2M'. `pos` is required by matplotlib's formatter API."""
    for div, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(v) >= div:
            return f"{v / div:.1f}{suffix}".replace(".0", "")
    return f"{v:.0f}"


def agg_roi(spend, revenue):
    """ROI of a GROUP = (total revenue - total spend) / total spend.
    WHY not the average of per-campaign ROIs? A 10-rupee flop and a 10-lakh winner would count equally.
    Summing first weights every rupee equally, which is how a CFO thinks about it."""
    return (revenue - spend) / spend


def save(fig, name):
    """Save a figure at print quality and free its memory."""
    fig.tight_layout()                             # stop labels being clipped
    fig.savefig(FIG_DIR / name, dpi=150)           # 150 dpi = sharp in slides and README
    plt.close(fig)                                 # free memory (matters with many charts)
    print(f"   saved reports/figures/{name}")


# =============================================================================
# a) Revenue & spend over time
# =============================================================================
print("[a] Revenue & spend over time")
monthly = df.groupby(df["start_date"].dt.to_period("M"))[["spend", "revenue"]].sum()  # one row per calendar month
monthly.index = monthly.index.to_timestamp()       # Period -> timestamp so matplotlib can plot it
monthly["roi"] = agg_roi(monthly["spend"], monthly["revenue"])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly.index, monthly["revenue"], marker="o", lw=2.2, color=GOOD, label="Revenue")
ax.plot(monthly.index, monthly["spend"], marker="o", lw=2.2, color=BAD, label="Spend")
ax.fill_between(monthly.index, monthly["spend"], monthly["revenue"],
                where=monthly["revenue"] >= monthly["spend"], interpolate=True, color=GOOD, alpha=0.15)  # profit zone
ax.fill_between(monthly.index, monthly["spend"], monthly["revenue"],
                where=monthly["revenue"] < monthly["spend"], interpolate=True, color=BAD, alpha=0.15)    # loss zone
ax.yaxis.set_major_formatter(FuncFormatter(human))
ax.set(title="Monthly revenue vs spend (by campaign start month)", xlabel="", ylabel="Amount")
ax.legend()
save(fig, "a_revenue_spend_trend.png")
peak = monthly["revenue"].idxmax()                 # month with the highest revenue
best = monthly["roi"].idxmax()                     # month with the best ROI
print(f"   -> Insight: revenue peaks in {peak:%b %Y}; the best-ROI month is {best:%b %Y} "
      f"(ROI {monthly.loc[best, 'roi']:.2f}).\n")

# =============================================================================
# b) ROI by channel
# =============================================================================
print("[b] ROI by channel")
by_ch = df.groupby("channel")[["spend", "revenue"]].sum()
by_ch["roi"] = agg_roi(by_ch["spend"], by_ch["revenue"])
by_ch = by_ch.sort_values("roi", ascending=False)  # best channel on the left

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(by_ch.index, by_ch["roi"],
              color=[GOOD if r >= ROI_TARGET else BAD for r in by_ch["roi"]])  # colour = met target or not
ax.axhline(ROI_TARGET, ls="--", color="black", lw=1, label=f"ROI target ({ROI_TARGET:.2f})")
ax.bar_label(bars, fmt="%.2f", padding=3)          # print the value on top of each bar
ax.set(title="ROI by channel (total revenue vs total spend)", xlabel="", ylabel="ROI")
ax.legend()
save(fig, "b_roi_by_channel.png")
top, bottom = by_ch.index[0], by_ch.index[-1]
print(f"   -> Insight: {top} returns the most (ROI {by_ch.loc[top, 'roi']:.2f}) and {bottom} the least "
      f"(ROI {by_ch.loc[bottom, 'roi']:.2f}); {(by_ch['roi'] >= ROI_TARGET).sum()} of {len(by_ch)} channels meet the target.\n")

# =============================================================================
# c) Spend vs revenue scatter (bubble = conversions)
# =============================================================================
print("[c] Spend vs revenue")
plot_df = df.sample(min(len(df), 3000), random_state=42)   # WHY: 200,000 dots would be an unreadable blob
sizes = 10 + 290 * np.sqrt(plot_df["conversions"] / plot_df["conversions"].max())  # sqrt so area ~ conversions

fig, ax = plt.subplots(figsize=(10, 6.5))
for ch in CHANNELS:                                # one scatter call per channel -> one legend entry each
    m = plot_df["channel"] == ch
    ax.scatter(plot_df.loc[m, "spend"], plot_df.loc[m, "revenue"], s=sizes[m],
               color=COLORS[ch], alpha=0.55, edgecolor="white", linewidth=0.4, label=ch)
lo, hi = plot_df["spend"].min(), plot_df["spend"].max()
xs = np.array([lo, hi])                            # two points are enough to draw a straight reference line
ax.plot(xs, xs, color="gray", ls=":", label="Break-even (revenue = spend)")
ax.plot(xs, xs * (1 + ROI_TARGET), color="black", ls="--", label=f"ROI target line")
ax.set_xscale("log"); ax.set_yscale("log")         # spend is very skewed: log axes show all scales at once
ax.xaxis.set_major_formatter(FuncFormatter(human)); ax.yaxis.set_major_formatter(FuncFormatter(human))
ax.set(title="Spend vs revenue per campaign (bubble size = conversions)", xlabel="Spend (log scale)", ylabel="Revenue (log scale)")
ax.legend(loc="upper left", fontsize=8, markerscale=0.5)
save(fig, "c_spend_vs_revenue.png")
# WHY within each channel? Cheap channels (Email) are also the best ones, so ranking ALL campaigns by
# budget would mix up "small budget" with "good channel". Ranking inside each channel isolates budget size.
q = df.groupby("channel")["spend"].transform(lambda s: pd.qcut(s, 4, labels=False))   # 0 = smallest budgets in its channel
roas_q = df.groupby(q)[["spend", "revenue"]].sum()
roas_q = roas_q["revenue"] / roas_q["spend"]       # ROAS for each budget quartile
verdict = "diminishing returns on big budgets" if roas_q.iloc[-1] < 0.95 * roas_q.iloc[0] else "no clear diminishing returns"
print(f"   -> Insight: ROAS is {roas_q.iloc[0]:.2f} for the smallest-budget quartile vs {roas_q.iloc[-1]:.2f} "
      f"for the largest: {verdict}. {(df['roi'] >= ROI_TARGET).mean():.0%} of campaigns sit above the target line.\n")

# =============================================================================
# d) Heatmap: channel x month
# =============================================================================
print("[d] Channel x month heatmap")
cm = df.groupby(["channel", "month"])[["spend", "revenue"]].sum()
cm["roi"] = agg_roi(cm["spend"], cm["revenue"])
pivot = cm["roi"].unstack("month")                 # rows = channel, columns = month number
pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]   # 1 -> "Jan"

fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=ROI_TARGET,   # green above target, red below
            linewidths=0.5, cbar_kws={"label": "ROI"}, ax=ax)
ax.set(title="ROI by channel and calendar month (all years pooled)", xlabel="", ylabel="")
save(fig, "d_channel_month_heatmap.png")
stacked = pivot.stack()                            # (channel, month) -> ROI, easy to rank
b, w = stacked.idxmax(), stacked.idxmin()
print(f"   -> Insight: best cell is {b[0]} in {b[1]} (ROI {stacked.max():.2f}); "
      f"weakest is {w[0]} in {w[1]} (ROI {stacked.min():.2f}).\n")

# =============================================================================
# e) Segment x channel response rate
# =============================================================================
print("[e] Segment x channel response rate")
known = df[df["customer_segment"] != "Unknown"]    # WHY: 'Unknown' is a data gap, not a customer type
sc = known.groupby(["customer_segment", "channel"])[["conversions", "clicks"]].sum()
sc["rate"] = sc["conversions"] / sc["clicks"]      # response rate = share of clicks that became customers
grid = sc["rate"].unstack("channel")               # rows = segment, columns = channel
labels = grid.apply(lambda col: col.map(lambda v: f"{v:.1%}"))   # text like '4.2%' for the cell labels

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(grid, annot=labels, fmt="", cmap="YlGnBu", linewidths=0.5,
            cbar_kws={"label": "Response rate (conversions / clicks)"}, ax=ax)
ax.set(title="Customer response rate by segment and channel", xlabel="", ylabel="")
save(fig, "e_segment_channel_response.png")
flat = grid.stack()
b, w = flat.idxmax(), flat.idxmin()
print(f"   -> Insight: {b[0]} respond best to {b[1]} ({flat.max():.1%}); "
      f"{w[0]} respond worst to {w[1]} ({flat.min():.1%}).\n")

print(f"Done. Open the folder: {FIG_DIR}")