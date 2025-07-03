import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# ----- DARK THEME INJECT -----
st.set_page_config(page_title="XRP Rich List Dashboard", layout="wide")

# For Streamlit Cloud, enforce dark mode
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #1e222d !important;
        color: #F1F1F1 !important;
    }
    .stApp { background-color: #1e222d !important; }
    table, th, td {
        background-color: #25293c !important;
        color: #fff !important;
        border-color: #333 !important;
    }
    thead tr th { 
        white-space: pre-line !important; 
        word-break: break-word !important;
        font-size: 13px !important;
        background: #222235 !important;
        color: #fafbfc !important;
        border-bottom: 2px solid #333 !important;
    }
    .css-1v0mbdj, .st-bb, .st-cq { background-color: #222235 !important; }
    .css-1n76uvr { color: #fff !important; }
    /* Download button tweaks */
    .stDownloadButton > button { background: #3d4053 !important; color: #fff; }
    /* Scrollbar for dataframes */
    ::-webkit-scrollbar { width: 9px; background: #232334; }
    ::-webkit-scrollbar-thumb { background: #37374f; border-radius: 4px;}
    </style>
    """, unsafe_allow_html=True)

st.title("📊 XRP Rich List Interactive Dashboard")

def pretty_name(name):
    name = name.replace('_', ' ').replace('.csv', '').replace('-', '–')
    name = name.replace('Infinity', '∞')
    return name.title()

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

def shorten_col(col):
    if len(col) > 30:
        words = col.split()
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    return col

def format_stats_table(df, table_name):
    df.columns = [shorten_col(str(col)) for col in df.columns]
    for col in df.columns:
        if (
            "sum" in col.lower()
            or "accounts" in col.lower()
            or "balance" in col.lower()
            or "equals" in col.lower()
        ):
            new_vals = []
            for val in df[col]:
                valstr = str(val)
                if "xrp" in valstr.lower():
                    val2 = valstr.replace("XRP", "").replace(",", "").strip()
                    try:
                        val_num = float(val2)
                        formatted = f"{val_num:,.0f} XRP"
                        new_vals.append(formatted)
                    except Exception:
                        new_vals.append(val)
                else:
                    try:
                        val_num = float(valstr.replace(",", ""))
                        formatted = f"{val_num:,.0f}"
                        new_vals.append(formatted)
                    except Exception:
                        new_vals.append(val)
            df[col] = new_vals
    return df

def clean_stat_df(df):
    # Remove duplicate headers and blank rows
    col_headers = [str(c).strip().lower() for c in df.columns]
    def is_duplicate_header(row):
        values = [str(x).strip().lower() for x in row]
        return values == col_headers or all(val in col_headers for val in values)
    df_clean = df[~df.apply(is_duplicate_header, axis=1)].reset_index(drop=True)
    # Remove empty rows
    df_clean = df_clean[~df_clean.isnull().all(axis=1)]
    return df_clean

COLUMN_NAME_MAP = {
    "-- Number of accounts and sum of balance range": "Accounts\nFrom–To",
    "-- Number of accounts and sum of balance range.1": "Balance Range\n(XRP)",
    "-- Number of accounts and sum of balance range.2": "Total in\nRange (XRP)",
}

def rename_columns(df):
    new_cols = []
    for col in df.columns:
        if col in COLUMN_NAME_MAP:
            new_cols.append(COLUMN_NAME_MAP[col])
        elif len(col) > 20:
            words = col.split()
            mid = len(words)//2
            new_cols.append(" ".join(words[:mid]) + "\n" + " ".join(words[mid:]))
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df

# ---- MAIN TABS ----
# Show Current Statistics first, then Rich List Charts second
tab2, tab1 = st.tabs(["📋 Current Statistics", "📈 Rich List Charts"])

with tab2:
    st.header("Current XRP Ledger Statistics")
    stats_csvs = [
        "Number of Accounts and Sum of Balance Range.csv",
        "Percentage of Accounts with Balances Greater Than or Equal to.csv"
    ]
    found_any = False
    for csv_name in stats_csvs:
        if os.path.exists(csv_name):
            found_any = True
            st.subheader(pretty_name(csv_name.replace('.csv', '')))
            stat_df = pd.read_csv(csv_name)
            stat_df = clean_stat_df(stat_df)
            stat_df = format_stats_table(stat_df, csv_name)
            stat_df = rename_columns(stat_df)
            st.dataframe(stat_df, use_container_width=True, hide_index=True)
            st.download_button(
                label=f"Download {csv_name}",
                data=stat_df.to_csv(index=False).encode(),
                file_name=csv_name,
                mime='text/csv',
            )
