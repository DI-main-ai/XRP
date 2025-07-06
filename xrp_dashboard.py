import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

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

def parse_range_low(x):
    # Extract lowest number from range string like '10,000 - 25,000'
    if pd.isnull(x):
        return float('inf')
    x = str(x)
    if "Infinity" in x:
        return float('inf')
    x = x.split("-")[0].replace(",", "").strip()
    try:
        return float(x)
    except:
        return float('inf')

def format_commas(val):
    try:
        # Handles values like 1,234.56 or 1234 or "5,000 XRP"
        s = str(val).replace(",", "").replace("XRP", "").strip()
        # Don't format as int if it has a decimal
        if '.' in s:
            n = float(s)
            return f"{n:,.4f}".rstrip('0').rstrip('.')
        else:
            return f"{int(float(s)):,}"
    except Exception:
        return val

# ---- MAIN TABS ----
tab2, tab1 = st.tabs(["📋 Current Statistics", "📈 Rich List Charts"])

with tab2:
    st.header("Current XRP Ledger Statistics")

    # ------- ACCOUNTS TABLE (current_stats_accounts_history.csv) -------
    if os.path.exists("current_stats_accounts_history.csv"):
        accounts_df = pd.read_csv("current_stats_accounts_history.csv")
        latest_date = accounts_df['date'].max()
        latest_accounts = accounts_df[accounts_df['date'] == latest_date].copy()
        # Format columns
        latest_accounts["Accounts"] = latest_accounts["Accounts"].apply(format_commas)
        latest_accounts["Sum in Range (XRP)"] = latest_accounts["Sum in Range (XRP)"].apply(format_commas)
        # Order by low end of range
        latest_accounts = latest_accounts.sort_values(by="Balance Range (XRP)", key=lambda col: col.map(parse_range_low))
        latest_accounts = latest_accounts[["Accounts", "Balance Range (XRP)", "Sum in Range (XRP)"]]
        st.subheader("Number Of Accounts And Sum Of Balance Range")
        st.caption(f"Last updated: {latest_date}")
        st.dataframe(latest_accounts, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Download Number of Accounts and Sum of Balance Range.csv",
            data=latest_accounts.to_csv(index=False).encode(),
            file_name="Number of Accounts and Sum of Balance Range.csv",
            mime='text/csv',
        )
    else:
        st.warning("current_stats_accounts_history.csv not found.")

    # ------- PERCENT TABLE (current_stats_percent_history.csv) -------
    if os.path.exists("current_stats_percent_history.csv"):
        percent_df = pd.read_csv("current_stats_percent_history.csv")
        latest_date = percent_df['date'].max()
        latest_percent = percent_df[percent_df['date'] == latest_date].copy()
        latest_percent["Accounts ≥ Threshold"] = latest_percent["Accounts ≥ Threshold"].apply(format_commas)
        # "XRP ≥ Threshold" already has units, just display as is.
        st.subheader("Percentage Of Accounts With Balances Greater Than Or Equal To")
        st.caption(f"Last updated: {latest_date}")
        st.dataframe(latest_percent[["Threshold (%)", "Accounts ≥ Threshold", "XRP ≥ Threshold"]], use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Download Percentage of Accounts with Balances Greater Than or Equal to.csv",
            data=latest_percent.to_csv(index=False).encode(),
            file_name="Percentage of Accounts with Balances Greater Than or Equal to.csv",
            mime='text/csv',
        )
    else:
        st.warning("current_stats_percent_history.csv not found.")

with tab1:
    # Find all _Series1_DAILY_LATEST.csv files and map them to "base name" for dropdown
    csv_files = [
        f for f in os.listdir('.')
        if f.endswith('_Series1_DAILY_LATEST.csv')
    ]
    # Map base names for dropdown
    file_to_title = {
        f: f.replace('_Series1_DAILY_LATEST.csv', '').replace('_', ' ').replace('-', '–').replace('Infinity', '∞').strip()
        for f in csv_files
    }

    # Sort: non-numeric first (alphabetical), then numeric (by leading number)
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

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
    else:
        st.warning("No 'date' column found! Chart x-axis may not be time-based.")

    st.subheader(f"Chart: {file_to_title[csv_choice]}")
    fig = px.line(
        df,
        x=date_col if date_col else df.columns[0],
        y='value',
        markers=True,
    )
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(tickformat=",", title="XRP")
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
            df.style.format({"value": format_millions}),
            use_container_width=True
        )

    st.download_button(
        label="Download this table as CSV",
        data=df.to_csv(index=False).encode(),
        file_name=csv_choice,
        mime='text/csv',
    )

    st.caption("Touch, zoom, and pan the chart. Made for XRP data nerds! 🚀")

# ---- OPTIONAL TIP JAR ----
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Like this project?**")
st.sidebar.markdown("Send XRP tips to: `YOUR_XRP_WALLET_ADDRESS`")
