"""
app.py - CampaignIQ interactive dashboard
Depends on: Steps 4 (features.csv) and 6 (model). Missing model -> the predictor tab explains what to run.
Run from the PROJECT ROOT:  streamlit run dashboard/app.py
"""
import calendar                                    # month names for the slider
import importlib.util                              # load 05_predict.py (its name starts with a digit, so `import` can't)
import json                                        # read target + model metadata
from pathlib import Path                           # cross-platform file paths

import pandas as pd
import plotly.express as px                        # quick interactive charts
import plotly.graph_objects as go                  # full control (gauge)
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "features.csv"
INFO_FILE = ROOT / "data" / "processed" / "target_info.json"
METRICS_FILE = ROOT / "models" / "model_metrics.json"
MODEL_FILE = ROOT / "models" / "campaign_model.pkl"
PREDICT_FILE = ROOT / "src" / "05_predict.py"

CURRENCY = "₹"                                     # change to "$" if you are using the Kaggle file
GOOD, BAD = "#1b9e77", "#d95f02"                   # green = meets target, orange = misses

st.set_page_config(page_title="CampaignIQ", page_icon="📈", layout="wide")


# ---------- Cached loaders (Streamlit reruns this whole script on every click; caching keeps it fast) ----------
@st.cache_data
def load_data():
    """Read the engineered dataset once and keep it in memory."""
    df = pd.read_csv(DATA_FILE, parse_dates=["start_date", "end_date"])
    target = json.loads(INFO_FILE.read_text())["roi_target"] if INFO_FILE.exists() else 1.0
    return df, target


@st.cache_resource
def load_predictor():
    """Import src/05_predict.py by file path so the dashboard and the CLI share ONE prediction function."""
    spec = importlib.util.spec_from_file_location("campaign_predict", PREDICT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def money(v):
    """1,250,000 -> '₹1.25M' for compact KPI cards."""
    for div, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(v) >= div:
            return f"{CURRENCY}{v / div:.2f}{suffix}"
    return f"{CURRENCY}{v:,.0f}"


def roi_table(frame, by):
    """Group ROI = (total revenue - total spend) / total spend (weights every unit of money equally)."""
    g = frame.groupby(by)[["spend", "revenue"]].sum()
    g["roi"] = (g["revenue"] - g["spend"]) / g["spend"]
    return g


# ---------- Load data ---------------------------------------------------------------
if not DATA_FILE.exists():
    st.error(f"Missing {DATA_FILE.name}. Run steps 1-3 first (see README), then refresh.")
    st.stop()
df, ROI_TARGET = load_data()

# ---------- Sidebar filters ---------------------------------------------------------
st.sidebar.title("📈 CampaignIQ")
st.sidebar.header("Filters")
d_min, d_max = df["start_date"].min().date(), df["start_date"].max().date()
picked = st.sidebar.date_input("Campaign start date", value=(d_min, d_max), min_value=d_min, max_value=d_max)
if len(picked) != 2:                               # while the user is mid-selection only one date exists
    st.info("Select an end date in the sidebar to continue.")
    st.stop()
all_channels = sorted(df["channel"].unique())
all_segments = sorted(df["customer_segment"].unique())
channels = st.sidebar.multiselect("Channel", all_channels, default=all_channels)
segments = st.sidebar.multiselect("Customer segment", all_segments, default=all_segments)

mask = ((df["start_date"] >= pd.Timestamp(picked[0])) &
        (df["start_date"] < pd.Timestamp(picked[1]) + pd.Timedelta(days=1)) &   # include the whole end day
        df["channel"].isin(channels) & df["customer_segment"].isin(segments))
f = df[mask]
st.sidebar.caption(f"{len(f):,} of {len(df):,} campaigns selected")
if f.empty:
    st.warning("No campaigns match these filters. Widen the date range or add channels/segments.")
    st.stop()

# ---------- Header + KPI cards ------------------------------------------------------
st.title("CampaignIQ: Marketing ROI Dashboard")
by_channel = roi_table(f, "channel").sort_values("roi", ascending=False)
total_spend, total_revenue = f["spend"].sum(), f["revenue"].sum()
overall_roi = (total_revenue - total_spend) / total_spend
best_channel = by_channel.index[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total spend", money(total_spend))
k2.metric("Total revenue", money(total_revenue))
k3.metric("Overall ROI", f"{overall_roi:.2f}", f"{overall_roi - ROI_TARGET:+.2f} vs target {ROI_TARGET:.2f}")
k4.metric("Best channel", best_channel, f"ROI {by_channel.loc[best_channel, 'roi']:.2f}", delta_color="off")

tab_perf, tab_pred, tab_data = st.tabs(["📊 Performance", "🤖 Success predictor", "🗂️ Data"])

# =============================== TAB 1: PERFORMANCE ===============================
with tab_perf:
    c1, c2 = st.columns(2)
    with c1:                                       # channel comparison
        plot = by_channel.reset_index()
        plot["Meets target"] = plot["roi"].ge(ROI_TARGET).map({True: "Yes", False: "No"})
        fig = px.bar(plot, x="channel", y="roi", color="Meets target", text_auto=".2f",
                     color_discrete_map={"Yes": GOOD, "No": BAD}, hover_data={"spend": ":,.0f", "revenue": ":,.0f"},
                     title="ROI by channel")
        fig.add_hline(y=ROI_TARGET, line_dash="dash", annotation_text=f"target {ROI_TARGET:.2f}")
        fig.update_layout(xaxis_title="", yaxis_title="ROI")
        st.plotly_chart(fig, use_container_width=True)
    with c2:                                       # time-series trend
        monthly = f.groupby(f["start_date"].dt.to_period("M"))[["spend", "revenue"]].sum()
        monthly.index = monthly.index.to_timestamp()
        monthly = monthly.rename_axis("month").reset_index().melt("month", var_name="Metric", value_name="Amount")
        monthly["Metric"] = monthly["Metric"].str.title()
        fig = px.line(monthly, x="month", y="Amount", color="Metric", markers=True,
                      color_discrete_map={"Revenue": GOOD, "Spend": BAD}, title="Monthly revenue vs spend")
        fig.update_layout(xaxis_title="", yaxis_title=f"Amount ({CURRENCY})")
        st.plotly_chart(fig, use_container_width=True)

    known = f[f["customer_segment"] != "Unknown"]  # 'Unknown' is a data gap, not a customer type
    if known.empty:
        st.info("Segment breakdown needs at least one known segment in the current selection.")
    else:
        c3, c4 = st.columns(2)
        with c3:                                   # segment breakdown
            seg = roi_table(known, "customer_segment").sort_values("roi").reset_index()
            fig = px.bar(seg, x="roi", y="customer_segment", orientation="h", text_auto=".2f",
                         title="ROI by customer segment", color_discrete_sequence=[GOOD])
            fig.add_vline(x=ROI_TARGET, line_dash="dash")
            fig.update_layout(xaxis_title="ROI", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with c4:                                   # segment x channel
            grid = roi_table(known, ["customer_segment", "channel"])["roi"].unstack("channel")
            fig = px.imshow(grid, text_auto=".2f", aspect="auto", color_continuous_scale="RdYlGn",
                            color_continuous_midpoint=ROI_TARGET, title="ROI: segment x channel")
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

# =============================== TAB 2: PREDICTOR ================================
with tab_pred:
    if not (MODEL_FILE.exists() and METRICS_FILE.exists()):
        st.info("No trained model found. Run `python src/04_model.py`, then refresh this page.")
    else:
        pred = load_predictor()
        meta = json.loads(METRICS_FILE.read_text())
        auc = meta["results"][meta["best_model"]]["roc_auc"]
        st.caption(f"Model: **{meta['best_model']}** | test ROC-AUC **{auc:.2f}** | "
                   f"'success' = ROI ≥ {meta['roi_target']:.2f}")
        if meta.get("auto_fallback"):
            st.warning("The ROI target was auto-switched to the median because the data had no real "
                       "success/failure split. This is a demonstration target, not a business rule.")
        if auc < 0.60:
            st.warning("This model barely beats a coin flip. Do not use these predictions for decisions.")

        left, right = st.columns([1, 1.2])
        with left:
            st.subheader("Plan a campaign")
            p_channel = st.selectbox("Channel", meta["channels"])
            p_segment = st.selectbox("Customer segment", meta["segments"])
            p_month = st.select_slider("Launch month", options=list(range(1, 13)), value=11,
                                       format_func=lambda m: calendar.month_abbr[m])
            lo, hi = int(meta["spend_min"]), int(meta["spend_max"])
            step = max(500, int(round((hi - lo) / 200, -2)))   # ~200 slider positions, rounded to a tidy number
            p_budget = st.slider(f"Budget ({CURRENCY})", min_value=lo, max_value=hi,
                                 value=min(max(int(meta["spend_median"]), lo), hi), step=step)
        result = pred.predict_campaign(p_channel, p_budget, p_month, p_segment)
        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=result["probability"] * 100, number={"suffix": "%"},
                title={"text": "Probability of hitting the ROI target"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                       "steps": [{"range": [0, pred.AVOID_THRESHOLD * 100], "color": "#f4a582"},
                                 {"range": [pred.AVOID_THRESHOLD * 100, pred.FUND_THRESHOLD * 100], "color": "#fee08b"},
                                 {"range": [pred.FUND_THRESHOLD * 100, 100], "color": "#a6d96a"}]}))
            gauge.update_layout(height=280, margin=dict(t=60, b=10))
            st.plotly_chart(gauge, use_container_width=True)
            verdict = {"Fund": st.success, "Risky": st.warning, "Avoid": st.error}[result["recommendation"]]
            verdict(f"**{result['recommendation']}**: {p_channel} for {p_segment} in "
                    f"{calendar.month_abbr[p_month]} at {money(p_budget)}")
            if result["warning"]:
                st.caption("⚠️ " + result["warning"])

        st.subheader("What-if: same plan, every channel")   # the decision the business actually faces
        rows = [{"channel": ch, "probability": pred.predict_campaign(ch, p_budget, p_month, p_segment)["probability"]}
                for ch in meta["channels"]]
        wf = px.bar(pd.DataFrame(rows).sort_values("probability", ascending=False), x="channel", y="probability",
                    text_auto=".0%", color_discrete_sequence=["#4c72b0"])
        wf.add_hline(y=pred.FUND_THRESHOLD, line_dash="dash", annotation_text="Fund")
        wf.add_hline(y=pred.AVOID_THRESHOLD, line_dash="dot", annotation_text="Avoid")
        wf.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="", yaxis_title="P(success)")
        st.plotly_chart(wf, use_container_width=True)

# =============================== TAB 3: DATA =====================================
with tab_data:
    st.caption("First 500 rows of the current selection")
    st.dataframe(f.head(500), use_container_width=True)
    if st.checkbox("Prepare CSV download of all selected rows"):   # only built on demand: big files are slow
        st.download_button("Download CSV", f.to_csv(index=False).encode("utf-8"), "campaigniq_filtered.csv", "text/csv")
