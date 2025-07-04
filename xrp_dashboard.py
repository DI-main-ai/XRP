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

# ----------- CLEAN/FORMAT STATS TABLES -------------
def clean_and_rename_stat_df(df, rename_map=None):
    orig_cols = [str(col).strip().lower() for col in df.columns]
    def is_dup_header(row):
        vals = [str(x).strip().lower() for x in row]
        # Allow partial match, some CSVs have extra columns
        return all(v in orig_cols for v in vals if v)
    df = df[~df.apply(is_dup_header, axis=1)].reset_index(drop=True)
    if rename_map:
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    return df

def format_stats_table(df, table_name):
    df = df.copy()
    # Table 1: format 1st and 3rd columns (index 0 and 2)
    if "Number of Accounts and Sum of Balance Range" in table_name:
        for idx in [0, 2]:
            if idx < len(df.columns):
                col = df.columns[idx]
                df[col] = [
                    f"{int(float(str(x).replace(',',''))):,}" if pd.notnull(x) and str(x).replace(",", "").replace(".", "").isdigit() else x
                    for x in df[col]
                ]
    # Table 2: format 2nd column (index 1)
    elif "Percentage of Accounts with Balances Greater Than or Equal to" in table_name:
        if len(df.columns) > 1:
            col = df.columns[1]
            df[col] = [
                f"{int(float(str(x).replace(',',''))):,}" if pd.notnull(x) and str(x).replace(",", "").replace(".", "").isdigit() else x
                for x in df[col]
            ]
    return df
# ----------------------------------------------------

# ---- MAIN TABS ----
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
            # --- Only keep rows where first column starts with a number
            stat_df = stat_df[[col for col in stat_df.columns[:3]]]  # keep at most first 3 cols
            stat_df = stat_df[stat_df[stat_df.columns[0]].astype(str).str.strip().str[0].str.isdigit()]
            # Rename columns to short/clear names
            if "Number of Accounts and Sum of Balance Range" in csv_name:
                stat_df.columns = ["Accounts", "Balance Range (XRP)", "Sum in Range (XRP)"]
            elif "Percentage of Accounts with Balances Greater Than or Equal to" in csv_name:
                stat_df.columns = ["Threshold (%)", "Accounts ≥ Threshold", "XRP ≥ Threshold"]
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
        # No grouping/summing, as file is already 1 row per date!
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
