"""
06_report.py
Writes reports/executive_summary.md: KPIs, top channels, top segment, model quality and three
rule-based recommendations backed by numbers.
Depends on: 02_features.py (features.csv, target_info.json). Uses 04_model.py output if it exists.
Run       : python src/06_report.py
"""
import json                                        # read target + model metrics
from datetime import date                          # stamp the report
from pathlib import Path                           # cross-platform file paths

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "features.csv"
INFO_FILE = ROOT / "data" / "processed" / "target_info.json"
METRICS_FILE = ROOT / "models" / "model_metrics.json"
RAW_FILE = ROOT / "data" / "raw" / "campaigns_raw.csv"
OUT_FILE = ROOT / "reports" / "executive_summary.md"

CURRENCY = "₹"          # change to "$" for the Kaggle file
SHIFT_SHARE = 0.15      # recommendation 1 moves 15% of the weakest channel's budget
MIN_PAIR_N = 15         # a segment x channel cell needs >= 15 campaigns before we trust it
ABS_GAP = 0.25          # a difference must be at least this many ROI points...
REL_GAP = 0.10          # ...AND at least 10% of the leader's ROI to count as "real" (guards against noise)

if not DATA_FILE.exists():
    raise SystemExit(f"Missing {DATA_FILE}. Run steps 1-3 first.")
df = pd.read_csv(DATA_FILE, parse_dates=["start_date", "end_date"])
target = json.loads(INFO_FILE.read_text())["roi_target"] if INFO_FILE.exists() else 1.0
auto_fallback = json.loads(INFO_FILE.read_text()).get("auto_fallback", False) if INFO_FILE.exists() else False
meta = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else None


# ---------- Helpers ------------------------------------------------------------------
def money(v):
    """1,250,000 -> '₹1.25M'."""
    for div, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(v) >= div:
            return f"{CURRENCY}{v / div:.2f}{suffix}"
    return f"{CURRENCY}{v:,.0f}"


def roi_table(frame, by):
    """Group ROI from summed spend/revenue, plus ROAS and the share of campaigns meeting the target."""
    g = frame.groupby(by).agg(spend=("spend", "sum"), revenue=("revenue", "sum"),
                              campaigns=("campaign_id", "count"), hit=("is_successful", "mean"))
    g["roi"] = (g["revenue"] - g["spend"]) / g["spend"]
    g["roas"] = g["revenue"] / g["spend"]
    return g


def is_material(best, worst):
    """True only if the gap is big in absolute AND relative terms - otherwise it is probably noise."""
    gap = best - worst
    return gap >= max(ABS_GAP, REL_GAP * abs(best))


def md_table(headers, rows):
    """Tiny markdown table builder (avoids an extra dependency)."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def top_budget_roas(channel):
    """ROAS of a channel's biggest-budget campaigns: a cautious proxy for what EXTRA money would earn there."""
    c = df[df["channel"] == channel]
    top = c[c["spend"] >= c["spend"].quantile(0.75)]
    return top["revenue"].sum() / top["spend"].sum()


# ---------- Numbers ------------------------------------------------------------------
total_spend, total_rev = df["spend"].sum(), df["revenue"].sum()
overall_roi = (total_rev - total_spend) / total_spend
ch = roi_table(df, "channel").sort_values("roi", ascending=False)
known = df[df["customer_segment"] != "Unknown"]
seg = roi_table(known, "customer_segment").sort_values("roi", ascending=False)
top_seg = seg.index[0]
top_seg_channel = roi_table(known[known["customer_segment"] == top_seg], "channel")["roi"].idxmax()
season = roi_table(df, "season").sort_values("roi", ascending=False)
pairs = roi_table(known, ["customer_segment", "channel"])
pairs = pairs[pairs["campaigns"] >= MIN_PAIR_N].sort_values("roi", ascending=False)

# ---------- Recommendation 1: budget shift -------------------------------------------
best_ch, worst_ch = ch.index[0], ch.index[-1]
if len(ch) > 1 and is_material(ch.loc[best_ch, "roi"], ch.loc[worst_ch, "roi"]):
    moved = SHIFT_SHARE * ch.loc[worst_ch, "spend"]
    marginal = top_budget_roas(best_ch)                   # what extra money in the best channel plausibly earns
    change = moved * (marginal - ch.loc[worst_ch, "roas"])   # gained revenue minus revenue given up
    rec1 = (f"**Shift {SHIFT_SHARE:.0%} of the {worst_ch} budget ({money(moved)}) to {best_ch}.** "
            f"{best_ch} returns {CURRENCY}{ch.loc[best_ch, 'roas']:.2f} per {CURRENCY}1 spent (ROI {ch.loc[best_ch, 'roi']:.2f}) "
            f"versus {CURRENCY}{ch.loc[worst_ch, 'roas']:.2f} for {worst_ch} (ROI {ch.loc[worst_ch, 'roi']:.2f}). "
            f"Even {best_ch}'s largest campaigns earn {CURRENCY}{marginal:.2f} per {CURRENCY}1, so the extra money should not "
            f"collapse in value; the estimated revenue effect is {'+' if change >= 0 else '-'}{money(abs(change))}. "
            f"Note the scale: this is a {moved / ch.loc[best_ch, 'spend']:.0%} increase on {best_ch}'s current budget"
            f"{', which is large, so stage it in tranches and re-measure between them' if moved / ch.loc[best_ch, 'spend'] > 0.5 else ''}: "
            f"channels can saturate.")
else:
    rec1 = (f"**No budget shift recommended.** The best channel ({best_ch}, ROI {ch.loc[best_ch, 'roi']:.2f}) and "
            f"the weakest ({worst_ch}, ROI {ch.loc[worst_ch, 'roi']:.2f}) are too close to justify moving money; "
            f"the gap is within what noise could produce.")

# ---------- Recommendation 2: timing --------------------------------------------------
best_s, worst_s = season.index[0], season.index[-1]
if len(season) > 1 and is_material(season.loc[best_s, "roi"], season.loc[worst_s, "roi"]):
    rec2 = (f"**Concentrate launches in the {best_s} season.** Campaigns started then earned ROI "
            f"{season.loc[best_s, 'roi']:.2f} versus {season.loc[worst_s, 'roi']:.2f} in {worst_s}. "
            f"Move flexible, non-seasonal spend out of {worst_s} and into {best_s}.")
else:
    rec2 = "**No seasonal re-timing recommended.** ROI is similar across seasons, so timing is not a lever in this data."

# ---------- Recommendation 3: segment x channel targeting -----------------------------
if len(pairs) > 1 and is_material(pairs["roi"].iloc[0], pairs["roi"].iloc[-1]):
    (bs, bc), (ws, wc) = pairs.index[0], pairs.index[-1]
    rec3 = (f"**Target {bs} with {bc}; stop pairing {ws} with {wc}.** "
            f"{bs} x {bc} earned ROI {pairs.iloc[0]['roi']:.2f} across {int(pairs.iloc[0]['campaigns'])} campaigns, "
            f"while {ws} x {wc} earned {pairs.iloc[-1]['roi']:.2f} across {int(pairs.iloc[-1]['campaigns'])}.")
else:
    rec3 = "**No segment-channel targeting change recommended.** No pairing stands out beyond normal variation."

# ---------- Model paragraph -----------------------------------------------------------
if meta:
    r = meta["results"][meta["best_model"]]
    verdict = ("useful for screening plans before spending" if r["roc_auc"] >= 0.75 else
               "only weakly informative; use with caution" if r["roc_auc"] >= 0.60 else
               "no better than chance; do not use for decisions")
    model_md = (f"**{meta['best_model']}**, trained on {meta['n_train']:,} campaigns and tested on {meta['n_test']:,} unseen ones.\n\n"
                + md_table(["Metric", "Value"],
                           [["Accuracy", f"{r['accuracy']:.1%}"], ["Baseline (always guess majority)", f"{meta['baseline_accuracy']:.1%}"],
                            ["Precision", f"{r['precision']:.1%}"], ["Recall", f"{r['recall']:.1%}"],
                            ["ROC-AUC", f"{r['roc_auc']:.3f}"]])
                + f"\n\nVerdict: the model is **{verdict}**.")
else:
    model_md = "_No trained model found. Run `python src/04_model.py` and regenerate this report._"

# ---------- Limitations (auto-filled from the actual data) ----------------------------
limits = []
if RAW_FILE.exists():
    raw_rows = len(pd.read_csv(RAW_FILE, usecols=[0]))
    limits.append(f"Cleaning removed {raw_rows - len(df):,} of {raw_rows:,} raw rows ({(raw_rows - len(df)) / raw_rows:.1%}) "
                  f"for duplicates, missing money values or impossible entries.")
if auto_fallback:
    limits.append(f"The ROI target was auto-switched to the median ({target:.2f}) because the data had no real "
                  f"success/failure split. Treat model output as a demonstration.")
limits += [
    "Results are **associations in past data**, not proof that moving budget will cause the same returns.",
    "'Success' means ROI >= %.2f; a different threshold gives different answers." % target,
    "The predictor only sees channel, budget, month and segment. It cannot see creative quality, competitors or pricing.",
    "Predictions outside the historical budget range are extrapolations and unreliable.",
    "If this report was built from the bundled synthetic dataset, every pattern in it is simulated: it demonstrates the method, not the market.",
]

# ---------- Assemble the markdown -----------------------------------------------------
md = f"""# CampaignIQ Executive Summary
_Generated {date.today():%d %b %Y} | {len(df):,} campaigns | {df['start_date'].min():%b %Y} to {df['start_date'].max():%b %Y}_

## 1. Headline numbers
{md_table(["Metric", "Value"], [
    ["Total spend", money(total_spend)], ["Total revenue", money(total_rev)],
    ["Overall ROI", f"{overall_roi:.2f}"], ["Overall ROAS", f"{total_rev / total_spend:.2f}"],
    [f"Campaigns meeting ROI target ({target:.2f})", f"{df['is_successful'].mean():.1%}"]])}

## 2. Top 3 channels by ROI
{md_table(["Rank", "Channel", "ROI", "ROAS", "Spend", "% campaigns on target"],
          [[i + 1, c, f"{r.roi:.2f}", f"{r.roas:.2f}", money(r.spend), f"{r.hit:.0%}"] for i, (c, r) in enumerate(ch.head(3).iterrows())])}

## 3. Top customer segment
**{top_seg}** has the highest ROI ({seg.loc[top_seg, 'roi']:.2f}); it responds best to **{top_seg_channel}**.

## 4. Predictive model
{model_md}

## 5. Recommendations
1. {rec1}
2. {rec2}
3. {rec3}

## 6. Risks and limitations
""" + "\n".join(f"- {x}" for x in limits) + "\n"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(md, encoding="utf-8")            # utf-8 so the currency symbol survives on Windows
print(md)
print(f"\nSaved -> {OUT_FILE}")

# ---------- OPTIONAL: let an LLM rewrite the summary as an executive narrative ----------
# 1) pip install anthropic     2) set the environment variable ANTHROPIC_API_KEY (never paste keys into code)
# Only the aggregated summary is sent, not your raw campaign rows. Remove the leading '# ' to enable.
#
# import os, anthropic
# client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
# reply = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1200,
#     messages=[{"role": "user", "content":
#         "Rewrite this marketing report as a one-page executive narrative for a CMO. "
#         "Keep every number exactly as given and do not invent facts.\n\n" + md}],
# )
# OUT_FILE.with_name("executive_narrative.md").write_text(reply.content[0].text, encoding="utf-8")
# (An OpenAI version is the same idea: swap in openai.OpenAI().chat.completions.create(...).)