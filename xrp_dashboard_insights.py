import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="XRP Dashboard – Insights (Dev)",
    page_icon="🞎",
    layout="wide",
)

# Hide Streamlit default menu and footer
HIDE_ST_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: auto;}
</style>
"""
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

# Caching helpers to load CSVs
@st.cache_data(show_spinner=False, ttl=600)
def load_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=600)
def load_data(csv_dir: str):
    """Load accounts and percent history plus *_DAILY_LATEST.csv files (excluding __Series1.csv)."""
    accounts_hist = load_csv_safely(os.path.join(csv_dir, "current_stats_accounts_history.csv"))
    percent_hist = load_csv_safely(os.path.join(csv_dir, "current_stats_percent_history.csv"))
    daily_latest = {}
    for p in glob.glob(os.path.join(csv_dir, "*_DAILY_LATEST.csv")):
        if "__Series1.csv" in p:
            continue
        daily_latest[os.path.basename(p)] = load_csv_safely(p)
    return accounts_hist, percent_hist, daily_latest

# Date parsing and filtering

def _parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a date column to a 'date' column."""
    if df is None or df.empty:
        return df
    for c in ["date", "Date", "DATE", "as_of_date", "AsOfDate", "timestamp", "Timestamp"]:
        if c in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df[c], errors="coerce")
            return df
    return df

def apply_accounts_date_floor(df: pd.DataFrame, floor_date: str = "2025-07-04") -> pd.DataFrame:
    """Remove rows before floor_date for accounts history since early rows lack account counts."""
    df = _parse_date_col(df)
    if "date" in df.columns:
        try:
            floor = pd.to_datetime(floor_date)
            df = df[df["date"] >= floor]
        except Exception:
            pass
    return df

def infer_account_columns(df: pd.DataFrame) -> list:
    """Heuristic: return numeric columns that aren't date-related or percent columns."""
    if df is None or df.empty:
        return []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    blacklist = {"date", "Date", "as_of_date", "timestamp"}
    return [c for c in numeric_cols if c not in blacklist and not c.lower().endswith("%")]

# Load data from ./csv relative to this file
CSV_DIR = os.path.join(os.path.dirname(__file__), "csv")
accounts_hist, percent_hist, daily_latest = load_data(CSV_DIR)

# Apply date floor and parse dates
accounts_hist = apply_accounts_date_floor(accounts_hist, "2025-07-04")
accounts_hist = _parse_date_col(accounts_hist)
percent_hist = _parse_date_col(percent_hist)
daily_latest = {k: _parse_date_col(v) for k, v in daily_latest.items()}

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    # Global date range filter based on accounts history dates
    date_range = None
    if accounts_hist is not None and not accounts_hist.empty and "date" in accounts_hist.columns:
        min_date = accounts_hist["date"].min().date()
        max_date = accounts_hist["date"].max().date()
        start, end = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        date_range = (start, end)
    show_qa = st.checkbox("Show Data QA panel", True)

# Filter data by date range

def _clip_df_by_range(df: pd.DataFrame, date_range):
    if df is None or df.empty or date_range is None:
        return df
    if "date" in df.columns:
        start, end = date_range
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()
    return df

accounts_f = _clip_df_by_range(accounts_hist, date_range)
percent_f = _clip_df_by_range(percent_hist, date_range)

# Header section with metrics
c1, c2, c3 = st.columns([1.4, 1, 1])
with c1:
    st.title("XRP Dashboard – Insights (Dev)")
    st.caption("Non-invasive enhancements")
with c2:
    # Last data update across *_DAILY_LATEST
    last_dates = []
    for df in daily_latest.values():
        if df is not None and not df.empty and "date" in df.columns:
            last_dates.append(pd.to_datetime(df["date"].max()))
    if last_dates:
        st.metric("Last Data Update", max(last_dates).strftime("%Y-%m-%d"))
with c3:
    total_rows = len(accounts_hist) + len(percent_hist) + sum(len(df) for df in daily_latest.values())
    st.metric("Rows Loaded", f"{total_rows:,}")

# Narrator Cards (KPIs)
st.markdown("### 🗟️ Narrator Cards")
acct_cols = infer_account_columns(accounts_f)
selected_cols = st.multiselect(
    "Choose KPI columns", options=acct_cols, default=acct_cols[: min(4, len(acct_cols))] if acct_cols else [], help="These KPIs compute 1d/7d/30d deltas over the selected date range."
)

def last_valid_value(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return s.iloc[-1] if not s.empty else np.nan

def delta_over(df: pd.DataFrame, col: str, days: int):
    if df is None or df.empty or col not in df.columns or "date" not in df.columns:
        return np.nan
    df = df.sort_values("date")
    latest = df.iloc[-1]
    cutoff = latest["date"] - pd.Timedelta(days=days)
    past = df[df["date"] <= cutoff]
    if past.empty:
        return np.nan
    return latest[col] - past.iloc[-1][col]

kareas = st.columns(4)
for idx, col in enumerate(selected_cols):
    area = kareas[idx % 4]
    with area:
        curr = last_valid_value(accounts_f, col)
        d1 = delta_over(accounts_f, col, 1)
        d7 = delta_over(accounts_f, col, 7)
        d30 = delta_over(accounts_f, col, 30)
        def humanize(val):
            if pd.isna(val):
                return "—"
            sign = "+" if val > 0 else ""
            return f"{sign}{val:,.0f}"
        st.metric(
            label=col,
            value=f"{curr:,.0f}" if pd.notna(curr) else "—",
            delta=f"1d {humanize(d1)} / 7d {humanize(d7)} / 30d {humanize(d30)}",
        )

# Trends section
st.markdown("### 📈 Trends")
trend_cols = st.multiselect(
    "Select columns to chart", options=acct_cols, default=acct_cols[: min(3, len(acct_cols))] if acct_cols else [], help="Line charts over selected date range."
)
for col in trend_cols:
    dfp = accounts_f[["date", col]].dropna().sort_values("date")
    if not dfp.empty:
        st.line_chart(dfp.set_index("date"))

# Anomaly detection
st.markdown("### 🚨 Anomalies (experimental)")
anom_col = st.selectbox("Column for anomaly scan", options=acct_cols)
if anom_col:
    dfp = accounts_f[["date", anom_col]].dropna().sort_values("date").copy()
    if not dfp.empty:
        dfp["diff"] = dfp[anom_col].diff()
        mu = dfp["diff"].mean()
        sigma = dfp["diff"].std(ddof=0)
        if sigma and sigma > 0:
            dfp["z"] = (dfp["diff"] - mu) / sigma
            thresh = st.slider("Z-score threshold", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
            flagged = dfp.loc[dfp["z"].abs() >= thresh, ["date", anom_col, "diff", "z"]]
            st.dataframe(flagged, use_container_width=True)
        else:
            st.info("Not enough variance to compute anomalies for this column.")

# Data QA panel
if show_qa:
    st.markdown("### 🧪 Data QA")
    qa_tabs = st.tabs(["History Files", "Daily Latest Files", "Schema & Nulls"])
    # Tab 1: history files summary
    with qa_tabs[0]:
        if accounts_hist is not None and not accounts_hist.empty:
            st.write("**current_stats_accounts_history.csv**")
            st.write(
                {
                    "rows": len(accounts_hist),
                    "start": str(accounts_hist["date"].min()) if "date" in accounts_hist.columns else "n/a",
                    "end": str(accounts_hist["date"].max()) if "date" in accounts_hist.columns else "n/a",
                    "columns": list(accounts_hist.columns),
                }
            )
        else:
            st.write("accounts history missing or empty")
        if percent_hist is not None and not percent_hist.empty:
            st.write("**current_stats_percent_history.csv**")
            st.write(
                {
                    "rows": len(percent_hist),
                    "start": str(percent_hist["date"].min()) if "date" in percent_hist.columns else "n/a",
                    "end": str(percent_hist["date"].max()) if "date" in percent_hist.columns else "n/a",
                    "columns": list(percent_hist.columns),
                }
            )
        else:
            st.write("percent history missing or empty")
    # Tab 2: daily latest summary
    with qa_tabs[1]:
        if daily_latest:
            summary = []
            for name, df in daily_latest.items():
                last_dt = df["date"].max() if (df is not None and not df.empty and "date" in df.columns) else None
                summary.append(
                    {
                        "file": name,
                        "rows": len(df),
                        "last_date": str(last_dt) if pd.notna(last_dt) else "n/a",
                        "columns": list(df.columns) if df is not None else [],
                    }
                )
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
        else:
            st.write("No *_DAILY_LATEST.csv files found.")
    # Tab 3: schema & nulls
    with qa_tabs[2]:
        for label, df in [("accounts_history", accounts_hist), ("percent_history", percent_hist)]:
            st.write(f"**{label}**")
            if df is None or df.empty:
                st.write("(empty)")
                continue
            nuls = df.isna().sum().sort_values(ascending=False)
            schema = pd.DataFrame(
                {
                    "column": df.columns,
                    "dtype": [str(df[c].dtype) for c in df.columns],
                    "nulls": [int(nuls.get(c, 0)) for c in df.columns],
                }
            )
            st.dataframe(schema, use_container_width=True)

# Footer
st.divider()
st.caption("This dev add-on app is read-only and designed to coexist with your current layout.")
