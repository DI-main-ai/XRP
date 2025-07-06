import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

st.set_page_config(page_title="XRP Rich List Dashboard", initial_sidebar_state="collapsed", layout="wide")

# ---- DARK THEME ----
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

def format_commas(val):
    try:
        v = float(str(val).replace(",", ""))
        return f"{int(v):,}"
    except:
        return val

# Shorten column headers for stats tables
def short_stat_headers(table_name, colnames):
    # Use short/clear headers, still informative
    if "Number of Accounts and Sum of Balance Range" in table_name:
        return ["Accounts", "Range (XRP)", "Total XRP"]
    elif "Percentage of Accounts with Balances Greater Than or Equal to" in table_name:
        return ["Threshold (%)", "Accounts ≥ Thresh", "XRP ≥ Thresh"]
    return [str(c)[:25] for c in colnames]

# ---- MAIN TABS ----
tab2, tab1 = st.tabs(["📋 Current Statistics", "📈 Rich List Charts"])

def parse_range(val):
    # Extracts the starting number from a range string like "100,000 - 500,000"
    if isinstance(val, str):
        match = re.match(r"(\d[\d,]*)", val)
        if match:
            return int(match.group(1).replace(',', ''))
    return float('inf')

def detect_accounts_table(df):
    # Try to find the 'Range' column, 'Sum' column, and 'Accounts' column
    cols = list(df.columns)
    # Try different common variants (case-insensitive, ignore whitespace)
    def norm(x): return x.strip().lower().replace(' ', '')
    candidates = {
        'range': [c for c in cols if 'range' in norm(c) or 'from' in norm(c)],
        'sum': [c for c in cols if 'sum' in norm(c) or 'total' in norm(c) or 'xrp' in norm(c)],
        'accounts': [c for c in cols if 'accounts' in norm(c)],
        'date': [c for c in cols if 'date' in norm(c)]
    }
    # Heuristics to select columns
    range_col = candidates['range'][0] if candidates['range'] else cols[1]
    sum_col = candidates['sum'][0] if candidates['sum'] else cols[-1]
    accounts_col = candidates['accounts'][0] if candidates['accounts'] else cols[0]
    date_col = candidates['date'][0] if candidates['date'] else None

    # If there's a date col at the end (sometimes happens), drop for table display
    keep_cols = [accounts_col, range_col, sum_col]
    return df[keep_cols].rename(columns={
        accounts_col: "Accounts",
        range_col: "Range (XRP)",
        sum_col: "Total XRP"
    })

def detect_percent_table(df):
    cols = list(df.columns)
    def norm(x): return x.strip().lower().replace(' ', '')
    candidates = {
        'thresh': [c for c in cols if 'thresh' in norm(c) or '%' in norm(c) or 'starting' in norm(c)],
        'accounts': [c for c in cols if 'accounts' in norm(c)],
        'xrp': [c for c in cols if 'xrp' in norm(c)],
        'date': [c for c in cols if 'date' in norm(c)]
    }
    thresh_col = candidates['thresh'][0] if candidates['thresh'] else cols[0]
    accounts_col = candidates['accounts'][0] if candidates['accounts'] else cols[1]
    xrp_col = candidates['xrp'][0] if candidates['xrp'] else cols[2]
    date_col = candidates['date'][0] if candidates['date'] else None
    keep_cols = [thresh_col, accounts_col, xrp_col]
    return df[keep_cols].rename(columns={
        thresh_col: "Threshold (%)",
        accounts_col: "Accounts ≥ Thresh",
        xrp_col: "XRP ≥ Thresh"
    })

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

            # Remove rows that are just repeated headers or blank
            stat_df = stat_df[[all(str(cell).strip() for cell in row) and not all(str(cell).lower().startswith(tuple('abcdefghijklmnopqrstuvwxyz')) for cell in row) for _, row in stat_df.iterrows()]]

            if "Number of Accounts" in csv_name:
                stat_df = detect_accounts_table(stat_df)
                # Sort by minimum of range
                stat_df = stat_df.sort_values(by="Range (XRP)", key=lambda col: col.map(parse_range)).reset_index(drop=True)
                # Format numbers
                stat_df["Accounts"] = stat_df["Accounts"].apply(lambda x: f"{int(float(str(x).replace(',',''))):,}" if str(x).replace(",","").isdigit() else x)
                stat_df["Total XRP"] = stat_df["Total XRP"].apply(lambda x: f"{float(str(x).replace(',','')):,.0f}" if str(x).replace(",","").replace(".","").isdigit() else x)
            else:
                stat_df = detect_percent_table(stat_df)
                stat_df["Accounts ≥ Thresh"] = stat_df["Accounts ≥ Thresh"].apply(lambda x: f"{int(float(str(x).replace(',',''))):,}" if str(x).replace(",","").isdigit() else x)
                stat_df["XRP ≥ Thresh"] = stat_df["XRP ≥ Thresh"].apply(lambda x: f"{float(str(x).replace(',','').replace('XRP','')):,.2f} XRP" if isinstance(x,str) and "XRP" in x else x)
            st.dataframe(stat_df, use_container_width=True, hide_index=True)
            st.download_button(
                label=f"Download {csv_name}",
                data=stat_df.to_csv(index=False).encode(),
                file_name=csv_name,
                mime='text/csv',
            )
    if not found_any:
        st.info("No Current Statistics CSVs found. Please add them to the folder.")

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

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
    else:
        st.warning("No 'date' column found! Chart x-axis may not be time-based.")

    st.subheader(f"Chart: {file_to_title[csv_choice]}")
    # Show last update date for this chart
    if date_col is not None:
        last_chart_update = df[date_col].max()
        st.caption(f"Last updated: {last_chart_update}")
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
