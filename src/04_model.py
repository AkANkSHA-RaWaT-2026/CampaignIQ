"""
04_model.py
Trains Logistic Regression and Random Forest to predict is_successful, compares them,
saves the best one (with its preprocessing) to models/campaign_model.pkl.
Depends on: 02_features.py  (features.csv + target_info.json)
Run       : python src/04_model.py
"""
import json                                        # save metrics for the dashboard and report
from pathlib import Path                           # cross-platform file paths

import joblib                                      # saves/loads fitted scikit-learn objects
import matplotlib                                  # plotting engine
matplotlib.use("Agg")                              # draw to files only (no pop-up windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer      # apply different preprocessing to different columns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline              # chains preprocessing + model into ONE object
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "features.csv"
INFO_FILE = ROOT / "data" / "processed" / "target_info.json"
MODEL_FILE = ROOT / "models" / "campaign_model.pkl"
METRICS_FILE = ROOT / "models" / "model_metrics.json"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. Choose features: ONLY what is known BEFORE a campaign runs -------
# WHY: roi, roas, ctr, conversion_rate, cpa, revenue, clicks... are RESULTS of the campaign.
# Feeding them in is "data leakage": the model would look brilliant in testing and be useless
# in real life, because at planning time you don't have them. The predictor (Step 7) only
# receives channel, budget, month and segment, so those are the only inputs we train on.
CAT_COLS = ["channel", "customer_segment", "month"]   # month is treated as a category: Nov is not "11x Jan"
NUM_COLS = ["spend"]                                  # the planned budget
FEATURES = CAT_COLS + NUM_COLS
TARGET = "is_successful"
RANDOM_STATE = 42                                     # reproducible results
MAX_ROWS = 30_000                                     # speed cap: see the note where it is used

if not DATA_FILE.exists():
    raise SystemExit(f"Missing {DATA_FILE}. Run 02_features.py first.")
df = pd.read_csv(DATA_FILE)
if len(df) > MAX_ROWS:
    # WHY: with only 4 input features, 30,000 rows teach the model just as much as 200,000, but
    # cross-validating a forest on 200,000 rows takes many minutes. A random sample keeps it fast.
    df = df.sample(MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Large dataset: training on a random sample of {MAX_ROWS:,} rows for speed.")
info = json.loads(INFO_FILE.read_text())              # which ROI threshold defined 'success'
X, y = df[FEATURES], df[TARGET]
print(f"Rows: {len(df):,} | success rate: {y.mean():.1%} | target: ROI >= {info['roi_target']:.2f}")

# ---------- 2. Train / test split ------------------------------------------------
# stratify=y keeps the success/fail ratio identical in both parts, so the test is fair.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows (never touched while training)\n")


# ---------- 3. Model pipelines ---------------------------------------------------
def make_preprocessor():
    """One-hot encode categories; log + standardise the budget.
    log: budgets are heavily skewed (5K to 1M), log makes 'double the budget' the same step everywhere.
    scaler: Logistic Regression needs comparable scales. Trees don't care, but it does no harm.
    Building it INSIDE a Pipeline means the scaler is saved with the model, so training and
    prediction can never drift apart - this is why we don't save a separate scaler file."""
    numeric = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler())])
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),  # unseen label -> all zeros, no crash
                              ("num", numeric, NUM_COLS)])


models = {
    "Logistic Regression": Pipeline([("prep", make_preprocessor()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "Random Forest": Pipeline([("prep", make_preprocessor()),
                               ("clf", RandomForestClassifier(
                                   n_estimators=300,        # 300 trees vote
                                   min_samples_leaf=10,     # leaves need >=10 campaigns: stops memorising noise
                                   n_jobs=-1,               # use all CPU cores
                                   random_state=RANDOM_STATE))]),
}

# ---------- 4. Fit, cross-validate, evaluate -------------------------------------
baseline = max(y_test.mean(), 1 - y_test.mean())      # accuracy of always guessing the majority class
results, fitted = {}, {}
for name, pipe in models.items():
    # CV on the TRAINING data picks the winner. WHY not pick using the test set? Then the test
    # score is no longer a fair, unseen exam - we'd have peeked while choosing.
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)                        # final fit on all training data
    pred = pipe.predict(X_test)                       # hard 0/1 predictions
    proba = pipe.predict_proba(X_test)[:, 1]          # probability of success
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),     # share correct overall
        "precision": precision_score(y_test, pred),   # of campaigns we called winners, how many were
        "recall": recall_score(y_test, pred),         # of real winners, how many we caught
        "f1": f1_score(y_test, pred),                 # balance of precision and recall
        "roc_auc": roc_auc_score(y_test, proba),      # ranking quality: 0.5 = coin flip, 1.0 = perfect
        "cv_roc_auc": cv_auc,
        "confusion": confusion_matrix(y_test, pred).tolist(),
    }
    fitted[name] = (pipe, proba)

table = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc"]].astype(float)
print(table.round(3).to_string())
print(f"\nBaseline (always guess the majority class): accuracy {baseline:.3f}")
for name, r in results.items():
    print(f"\nConfusion matrix - {name}  (rows = actual, cols = predicted; order: fail, success)")
    print(pd.DataFrame(r["confusion"], index=["actual fail", "actual success"],
                       columns=["pred fail", "pred success"]).to_string())

# ---------- 5. Pick and save the best --------------------------------------------
best_name = table["cv_roc_auc"].idxmax()              # winner by cross-validated AUC
best_pipe = fitted[best_name][0]
joblib.dump(best_pipe, MODEL_FILE)                    # the ONE file that contains preprocessing + model
print(f"\nBest model: {best_name} (test ROC-AUC {results[best_name]['roc_auc']:.3f}) -> saved {MODEL_FILE.name}")
if results[best_name]["roc_auc"] < 0.60:
    print("WARNING: ROC-AUC below 0.60 - these features barely predict success. Do not trust the predictor.")

metrics = {                                           # everything the dashboard/report need, in one small file
    "best_model": best_name, "roi_target": info["roi_target"], "auto_fallback": info["auto_fallback"],
    "baseline_accuracy": baseline, "n_train": len(X_train), "n_test": len(X_test),
    "results": results,
    "channels": sorted(df["channel"].unique()),
    "segments": sorted(s for s in df["customer_segment"].unique() if s != "Unknown"),
    "spend_min": float(df["spend"].min()), "spend_median": float(df["spend"].median()),
    "spend_max": float(df["spend"].max()),
}
METRICS_FILE.write_text(json.dumps(metrics, indent=2))

# ---------- 6. Charts ------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")

# 6a. Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(np.array(r["confusion"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["fail", "success"], yticklabels=["fail", "success"])
    ax.set(title=name, xlabel="Predicted", ylabel="Actual")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_confusion_matrices.png", dpi=150); plt.close(fig)

# 6b. ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, (pipe, proba) in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, proba)            # trade-off between catching winners and false alarms
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {results[name]['roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], "k--", label="Coin flip")
ax.set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(FIG_DIR / "g_roc_curves.png", dpi=150); plt.close(fig)

# 6c. Random Forest feature importance, grouped back to the ORIGINAL columns
rf = models["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()          # e.g. 'cat__channel_Email', 'num__spend'
importance = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)


def original_column(feature_name):
    """'cat__channel_Email' -> 'channel'. One-hot splits one column into many; we add them back up."""
    stripped = feature_name.split("__", 1)[1]
    for col in FEATURES:
        if stripped == col or stripped.startswith(col + "_"):
            return col
    return stripped


grouped = importance.groupby(original_column).sum().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(grouped.index, grouped.values, color="#1b9e77")
ax.set(title="What drives campaign success? (Random Forest importance)", xlabel="Share of importance")
fig.tight_layout(); fig.savefig(FIG_DIR / "h_feature_importance.png", dpi=150); plt.close(fig)
print("\nFeature importance (Random Forest):")
print((grouped[::-1] * 100).round(1).astype(str).add("%").to_string())
print("Charts saved: f_confusion_matrices.png, g_roc_curves.png, h_feature_importance.png")