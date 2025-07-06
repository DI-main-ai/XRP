import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime

# ----- DARK THEME INJECT -----
st.set_page_config(page_title="XRP Rich List Dashboard", initial_sidebar_state="collapsed", layout="wide")
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #1e222d !important;
        color: #F1F1F1 !important;
    }
    .stApp { background-color: #1e222d !important; }
    .stDataFrame, .stTable, .stMarkdown, .stCaption, .css-1n76uvr {
        background-color: #232334 !important;
        color: #f3f4f6 !important;
    }
    thead tr th, .stDataFrame thead {
        background: #232334 !important;
        color: #fafbfc !important;
        border-bottom: 2px solid #333 !important;
    }
    .stDataFrame tbody, .stTable tbody, .stDataFrame tr, .stTable tr {
        background-color: #232334 !important;
        color: #f3f4f6 !important;
    }
    .stDataFrame td, .stTable td {
        background-color: #232334 !important;
        color: #f3f4f6 !important;
        border-color: #333 !important;
    }
    .css-1544g2n, .css-1v0mbdj, .css-1hynsf2, .st-cq, .st-bb, .stSidebar, .stDropdown, .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    .stSelectbox [data-testid="stSelectbox"] {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    .css-1g6gooi, .css-1d391kg {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    .stDownloadButton > button { background: #3d4053 !important; color: #fff; }
    .sidebar-content, .stSidebar {
        color: #F1F1F1 !important;
        background-color: #232334 !important;
    }
    .stMarkdown svg { color: #E7E7E7 !important; }
    ::-webkit-scrollbar { width: 9px; background: #232334; }
    ::-webkit-scrollbar-thumb { background: #37374f; border-radius: 4px;}
    </style>
    """, unsafe_allow_html=True)

st.title("📊 XRP Rich List Interactive Dashboard")

def format_int(val):
    try:
        v = float(val)
        if v.is_integer():
            return f"{int(v):,}"
        else:
            return f"{v:,.4f}".rstrip('0').rstrip('.')
    except Exception:
        return val


def format_full_number(val):
    try:
        return f"{int(val):,}"
    except Exception:
        return val


def format_millions(val):
    try:
        v = float(val)
    except Exception:
        return val
    if abs(v) >= 1e9:
        return f"{v/1e9:,.2f}B"
    elif abs(v) >= 1e6:
        return f"{v/1e6:,.2f}M"
    elif abs(v) >= 1e3:
        return f"{v/1e3:,.2f}K"
    else:
        return f"{v:,}"

def parse_range_start(range_str):
    # Get the starting number of a range "X - Y"
    if isinstance(range_str, str):
        match = re.match(r"^\s*([\d,]+)", range_str)
        if match:
            return int(match.group(1).replace(",", ""))
    return -1

def pretty_name(name):
    name = name.replace('_', ' ').replace('.csv', '').replace('-', '–')
    name = name.replace('Infinity', '∞')
    return name.title()

def extract_leading_number(name):
    match = re.match(r"^([0-9,_.]+)", name)
    if match:
        try:
            num = float(match.group(1).replace(",", "").replace("_", ""))
            return num
        except:
            return float('inf')
    return float('inf')

def is_not_number_start(name):
    return not re.match(r"^\d", name)
def format_xrp_thresh(x):
    try:
        # If it's a string with a space, get the numeric part and format it with commas
        if isinstance(x, str) and " " in x:
            val = x.split()[0].replace(",", "")
            return f"{float(val):,.4f} XRP" if "." in val else f"{int(float(val)):,} XRP"
        elif pd.isna(x):
            return ""
        else:
            v = float(x)
            return f"{v:,.4f} XRP" if not v.is_integer() else f"{int(v):,} XRP"
    except Exception:
        return str(x)
# ---- MAIN TABS ----
tab2, tab1 = st.tabs(["📋 Current Statistics", "📈 Rich List Charts"])

with tab2:
    st.header("Current XRP Ledger Statistics")

    ACCOUNTS_CSV = "current_stats_accounts_history.csv"
    PERCENT_CSV  = "current_stats_percent_history.csv"

    def normalize_threshold(x):
        if pd.isna(x): return None
        if isinstance(x, str):
            x = x.replace('%','').strip()
        try:
            return float(x)
        except:
            return x

    def normalize_balance_range(x):
        if pd.isna(x): return None
        if isinstance(x, str):
            x = x.replace(',','').split('-')[0].strip()
        try:
            return float(x)
        except:
            return x

    def normalize_threshold(val):
        """Convert thresholds like '0.01 %' to float 0.01 for matching/merging."""
        try:
            if isinstance(val, str):
                return float(val.replace('%','').replace(',','').strip())
            return float(val)
        except Exception:
            return None

    
    def calc_and_display_delta_table(
        df, id_col, delta_cols, table_name, date_col="date", normalize_key_func=None
    ):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        if df[date_col].isnull().any():
            st.warning(f"Some rows in {table_name} have invalid date format.")

        dates_available = sorted(df[date_col].dt.date.unique(), reverse=True)
        sel_date = st.selectbox(
            f"Select Date for {table_name}:", dates_available, 0, key=f"date_{table_name}"
        )
        show_delta = st.checkbox(
            f"Show change vs previous day", value=True, key=f"delta_{table_name}"
        )

        today_df = df[df[date_col].dt.date == sel_date].copy()
        yest_df = df[df[date_col].dt.date == (sel_date - pd.Timedelta(days=1))].copy()

        today_df = today_df.drop_duplicates(subset=[id_col])
        yest_df = yest_df.drop_duplicates(subset=[id_col])

        keep_cols = [date_col] + [id_col] + delta_cols
        today_df = today_df[keep_cols].reset_index(drop=True)
        yest_df = yest_df[keep_cols].reset_index(drop=True)
        # Rename columns to prettify
        pretty_map = {
            "Balance Range (XRP)": "Balance Range (XRP)",
            "Sum in Range (XRP)": "Sum in Range (XRP)",
            "Accounts": "Accounts",
            "Threshold (%)": "Threshold (%)",
            "Accounts ≥ Threshold": "Accounts ≥ Threshold",
            "XRP ≥ Threshold": "XRP ≥ Threshold"
        }
        today_df.columns = [c if c not in pretty_map else pretty_map[c] for c in today_df.columns]
        yest_df.columns = [c if c not in pretty_map else pretty_map[c] for c in yest_df.columns]
        id_col_pretty = id_col if id_col not in pretty_map else pretty_map[id_col]

        # --- Use normalized key column for merge if needed ---
        if normalize_key_func is not None:
            today_df["MergeKey"] = today_df[id_col_pretty].apply(normalize_key_func)
            yest_df["MergeKey"] = yest_df[id_col_pretty].apply(normalize_key_func)
            merge_id = "MergeKey"
        else:
            merge_id = id_col_pretty

        # --- Merge for delta ---
        if show_delta and not yest_df.empty:
            merged = today_df.merge(
                yest_df,
                on=merge_id,
                how="left",
                suffixes=('', '_prev')
            )
            # Add delta columns for each requested delta_col
            for col in delta_cols:
                col_pretty = pretty_map.get(col, col)
                col_prev = f"{col_pretty}_prev"
                if col_prev in merged.columns:
                    merged[f"{col_pretty} Δ"] = merged[col_pretty] - merged[col_prev]
                else:
                    merged[f"{col_pretty} Δ"] = ""
            # Drop prev and merge key columns, keep order
            keep = [c for c in merged.columns if not c.endswith("_prev") and c != "MergeKey"]
            today_df = merged[keep]
        # Formatting
        for c in today_df.columns:
            if "Δ" in c:
                today_df[c] = today_df[c].apply(lambda v: f"{v:+,}" if pd.notnull(v) and str(v).replace('.','',1).replace('-','').isdigit() else "")
            elif "Accounts" in c or "Sum" in c or "XRP" in c:
                today_df[c] = today_df[c].apply(format_int)
        st.subheader(table_name)
        st.markdown(f"<span style='color:#aaa;'>Date: {sel_date}</span>", unsafe_allow_html=True)
        st.dataframe(today_df.drop(columns=[date_col]), use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Download {table_name}",
            data=today_df.to_csv(index=False).encode(),
            file_name=f"{table_name.replace(' ', '_')}_{sel_date}.csv",
            mime='text/csv',
        )

    # Table 1: Number Of Accounts And Sum Of Balance Range
    if os.path.exists(ACCOUNTS_CSV):
        df = pd.read_csv(ACCOUNTS_CSV)
        calc_and_display_delta_table(
            df,
            id_col="Balance Range (XRP)",
            delta_cols=["Accounts", "Sum in Range (XRP)"],
            table_name="Number Of Accounts And Sum Of Balance Range",
            normalize_key_func=normalize_balance_range
        )
    else:
        st.info("current_stats_accounts_history.csv not found.")

    # Table 2: Percentage Of Accounts With Balances Greater Than Or Equal To
    if os.path.exists(PERCENT_CSV):
        df = pd.read_csv(PERCENT_CSV)
        calc_and_display_delta_table(
            df,
            id_col="Threshold (%)",
            delta_cols=["Accounts ≥ Threshold", "XRP ≥ Threshold"],
            table_name="Percentage Of Accounts With Balances Greater Than Or Equal To",
            normalize_key_func=normalize_threshold
        )
    else:
        st.info("current_stats_percent_history.csv not found.")

with tab1:
    # Find all _Series1_DAILY_LATEST.csv files and map them to "base name" for dropdown
    csv_files = [
        f for f in os.listdir('.')
        if f.endswith('_Series1_DAILY_LATEST.csv')
    ]
    file_to_title = {
        f: f.replace('_Series1_DAILY_LATEST.csv', '').replace('_', ' ').replace('-', '–').replace('Infinity', '∞').strip()
        for f in csv_files
    }

    non_num = sorted(
        [f for f, title in file_to_title.items() if is_not_number_start(title)],
        key=lambda x: file_to_title[x]
    )
    num_start = sorted(
        [f for f, title in file_to_title.items() if not is_not_number_start(title)],
        key=lambda x: extract_leading_number(file_to_title[x])
    )
    ordered_csvs = non_num + num_start

    if not ordered_csvs:
        st.error("No CSV files found in this folder!")
        st.stop()

    csv_choice = st.sidebar.selectbox(
        "Choose a data range/table:",
        ordered_csvs,
        format_func=lambda f: file_to_title[f]
    )

    df = pd.read_csv(csv_choice)
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break

    y_label = "XRP"
    if "wallet" in file_to_title[csv_choice].lower():
        y_label = "Wallet Count"

    st.subheader(f"Chart: {file_to_title[csv_choice]}")
    if date_col is not None and 'value' in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True).dt.date
        if df["date"].isnull().any():
            st.warning(f"Some rows in {fname} have invalid date format. Check your CSVs.")
        # Plot chart (abbreviated y-axis)
        fig = px.line(
            df,
            x=date_col if date_col else df.columns[0],
            y='value',
            markers=True,
        )
        fig.update_traces(line=dict(width=3))
        fig.update_yaxes(
            tickformat="~s",  # Abbreviated y-axis: 1.2M, 53K, etc.
            title="Total XRP" if "xrp" in file_to_title[csv_choice].lower() else "Wallet Count"
        )
        fig.update_layout(
            xaxis_title="Date",
            hovermode="x unified",
            hoverlabel=dict(namelength=-1),
            plot_bgcolor='#1e222d',
            paper_bgcolor='#1e222d',
            font=dict(color='#F1F1F1'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Data Table with full value ----
        df_display = df.copy()
        if "wallet count" in file_to_title[csv_choice].lower():
            df_display = df_display.rename(columns={"value": "Wallet Count"})
            value_col = "Wallet Count"
        else:
            df_display = df_display.rename(columns={"value": "Total XRP"})
            value_col = "Total XRP"
        # Format the values as full numbers (with commas)
        if value_col in df_display.columns:
            df_display[value_col] = df_display[value_col].apply(format_full_number)

        # Sort descending by date for table
        df_display = df_display.sort_values(by=date_col, ascending=False).reset_index(drop=True)

        with st.expander("Show Data Table"):
            st.dataframe(
                df_display[[date_col, value_col]],
                use_container_width=True
            )

        st.download_button(
            label="Download this table as CSV",
            data=df_display.to_csv(index=False).encode(),
            file_name=csv_choice,
            mime='text/csv',
        )
    else:
        st.warning("No 'date' or 'value' column found! Chart x-axis may not be time-based.")

    st.caption("Touch, zoom, and pan the chart. Made for XRP data nerds! 🚀")


# ---- OPTIONAL TIP JAR ----
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Like this project?**")
st.sidebar.markdown("Send XRP tips to: `YOUR_XRP_WALLET_ADDRESS`")
