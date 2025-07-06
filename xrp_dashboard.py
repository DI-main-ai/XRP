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

# ---- MAIN TABS ----
tab2, tab1 = st.tabs(["📋 Current Statistics", "📈 Rich List Charts"])

with tab2:
    st.header("Current XRP Ledger Statistics")

    # --- Number of Accounts and Sum of Balance Range ---
    ACCOUNTS_CSV = "current_stats_accounts_history.csv"
    PERCENT_CSV  = "current_stats_percent_history.csv"

    if os.path.exists(ACCOUNTS_CSV):
        stat_df = pd.read_csv(ACCOUNTS_CSV)
        if "date" in stat_df.columns:
            # Show latest available date
            stat_df['date'] = pd.to_datetime(stat_df['date'])
            latest_date = stat_df['date'].max()
            st.markdown(f"<span style='color:#aaa;'>Last updated: {latest_date.date()}</span>", unsafe_allow_html=True)
            latest_df = stat_df[stat_df['date'] == latest_date].copy()
        else:
            latest_df = stat_df.copy()

        # Fix columns and sort by descending range start
        colnames = [c.lower() for c in latest_df.columns]
        # Guess columns: accounts, range, sum
        acc_col = next((c for c in latest_df.columns if "account" in c.lower() and "sum" not in c.lower()), latest_df.columns[0])
        range_col = next((c for c in latest_df.columns if "range" in c.lower()), latest_df.columns[1])
        sum_col = next((c for c in latest_df.columns if "sum" in c.lower() or "total" in c.lower()), latest_df.columns[2])

        latest_df = latest_df[[acc_col, range_col, sum_col]].copy()
        latest_df.columns = ["Accounts", "Balance Range (XRP)", "Sum in Range (XRP)"]

        # Sort by descending start of range
        latest_df['__range_start'] = latest_df["Balance Range (XRP)"].apply(parse_range_start)
        latest_df = latest_df.sort_values("__range_start", ascending=False).drop(columns="__range_start")

        # Format numbers
        latest_df["Accounts"] = latest_df["Accounts"].apply(format_int)
        latest_df["Sum in Range (XRP)"] = latest_df["Sum in Range (XRP)"].apply(format_int)

        st.subheader("Number Of Accounts And Sum Of Balance Range")
        st.dataframe(latest_df, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Download Number of Accounts and Sum of Balance Range",
            data=latest_df.to_csv(index=False).encode(),
            file_name="Number_of_Accounts_and_Sum_of_Balance_Range.csv",
            mime='text/csv',
        )
    else:
        st.info("current_stats_accounts_history.csv not found.")

    # --- Percentage of Accounts with Balances Greater Than or Equal to ---
    if os.path.exists(PERCENT_CSV):
        stat_df = pd.read_csv(PERCENT_CSV)
        if "date" in stat_df.columns:
            stat_df['date'] = pd.to_datetime(stat_df['date'])
            latest_date = stat_df['date'].max()
            st.markdown(f"<span style='color:#aaa;'>Last updated: {latest_date.date()}</span>", unsafe_allow_html=True)
            latest_df = stat_df[stat_df['date'] == latest_date].copy()
        else:
            latest_df = stat_df.copy()

        # Guess columns: threshold, accounts, xrp
        thresh_col = next((c for c in latest_df.columns if "%" in c or "thresh" in c.lower()), latest_df.columns[0])
        accounts_col = next((c for c in latest_df.columns if "account" in c.lower()), latest_df.columns[1])
        xrp_col = next((c for c in latest_df.columns if "xrp" in c.lower()), latest_df.columns[2])

        latest_df = latest_df[[thresh_col, accounts_col, xrp_col]].copy()
        latest_df.columns = ["Threshold (%)", "Accounts ≥ Thresh", "XRP ≥ Thresh"]

        latest_df["Accounts ≥ Thresh"] = latest_df["Accounts ≥ Thresh"].apply(format_int)
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
        
        latest_df["XRP ≥ Thresh"] = latest_df["XRP ≥ Thresh"].apply(format_xrp_thresh)


        st.subheader("Percentage Of Accounts With Balances Greater Than Or Equal To")
        st.dataframe(latest_df, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Download Percentage of Accounts with Balances ≥ Threshold",
            data=latest_df.to_csv(index=False).encode(),
            file_name="Percentage_of_Accounts_with_Balances_Greater_Than_or_Equal_to.csv",
            mime='text/csv',
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
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        fig = px.line(
            df,
            x=date_col,
            y='value',
            markers=True,
        )
        fig.update_traces(line=dict(width=3))
        fig.update_yaxes(tickformat=",", title=y_label)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_tickformat=",",
            hovermode="x unified",
            hoverlabel=dict(namelength=-1),
            plot_bgcolor='#1e222d',
            paper_bgcolor='#1e222d',
            font=dict(color='#F1F1F1'),
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show Data Table"):
            st.dataframe(
                df.style.format({"value": format_int}),
                use_container_width=True
            )
        st.download_button(
            label="Download this table as CSV",
            data=df.to_csv(index=False).encode(),
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
