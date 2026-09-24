"""
05_predict.py
Score a PLANNED campaign before spending the money.
Depends on: 04_model.py  (needs models/campaign_model.pkl and models/model_metrics.json)
Run as a demo : python src/05_predict.py
Import it     : predict_campaign("Email", 50000, 11, "Students")   (the dashboard does this)
"""
import json                                        # read the saved metrics/metadata
from pathlib import Path                           # cross-platform file paths

import joblib                                      # load the saved model
import pandas as pd                                # the model expects a DataFrame with named columns

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "campaign_model.pkl"
METRICS_FILE = ROOT / "models" / "model_metrics.json"

# Decision bands. WHY these numbers? They are business choices, not maths: "Fund" means we are
# fairly sure of hitting the ROI target; below 40% we'd rather keep the money. Change to taste.
FUND_THRESHOLD = 0.65
AVOID_THRESHOLD = 0.40

_cache = {}                                        # load the model once, not on every call (fast dashboard sliders)


def _load():
    """Load model + metadata the first time, then reuse them."""
    if "model" not in _cache:
        if not MODEL_FILE.exists():                # friendly message instead of a raw traceback
            raise FileNotFoundError(f"{MODEL_FILE} not found. Run: python src/04_model.py")
        _cache["model"] = joblib.load(MODEL_FILE)  # preprocessing + classifier in one object
        _cache["meta"] = json.loads(METRICS_FILE.read_text())
    return _cache["model"], _cache["meta"]


def _match(value, options, label):
    """Case-insensitive lookup so 'email' and ' Email ' both work; clear error if unknown."""
    lookup = {o.lower(): o for o in options}
    key = str(value).strip().lower()
    if key not in lookup:
        raise ValueError(f"Unknown {label} '{value}'. Choose one of: {', '.join(options)}")
    return lookup[key]


def predict_campaign(channel, budget, month, segment):
    """Return the probability a campaign hits the ROI target, plus a Fund / Risky / Avoid call."""
    model, meta = _load()
    channel = _match(channel, meta["channels"], "channel")     # validate BEFORE the model sees the input
    segment = _match(segment, meta["segments"], "segment")
    month = int(month)
    if not 1 <= month <= 12:
        raise ValueError("month must be 1-12")
    if budget <= 0:
        raise ValueError("budget must be positive")

    row = pd.DataFrame([{"channel": channel, "customer_segment": segment,
                         "month": month, "spend": float(budget)}])   # same column names used in training
    prob = float(model.predict_proba(row)[0, 1])   # column 1 = probability of class 'success'

    if prob >= FUND_THRESHOLD:
        recommendation = "Fund"
    elif prob >= AVOID_THRESHOLD:
        recommendation = "Risky"
    else:
        recommendation = "Avoid"

    warning = None                                 # models are unreliable outside the data they saw
    if not meta["spend_min"] <= budget <= meta["spend_max"]:
        warning = (f"Budget is outside the training range ({meta['spend_min']:,.0f} to "
                   f"{meta['spend_max']:,.0f}); treat this result with extra caution.")
    return {"probability": prob, "recommendation": recommendation, "warning": warning,
            "model": meta["best_model"], "roi_target": meta["roi_target"]}


if __name__ == "__main__":
    _, meta = _load()
    segment = meta["segments"][0]                  # demo with the first segment in the data
    budget = meta["spend_median"]                  # and a typical budget
    print(f"Model: {meta['best_model']} | success = ROI >= {meta['roi_target']:.2f}")
    print(f"Segment: {segment} | budget: {budget:,.0f}\n")
    print(f"{'Channel':<14}{'Month':>6}{'P(success)':>12}   Call")
    for month in (2, 11):                          # a quiet month vs the festive peak
        for ch in meta["channels"]:
            r = predict_campaign(ch, budget, month, segment)
            print(f"{ch:<14}{month:>6}{r['probability']:>11.0%}   {r['recommendation']}")
    print("\nError handling demo:")
    try:
        predict_campaign("Billboards", 50000, 5, segment)
    except ValueError as e:
        print("  ->", e)