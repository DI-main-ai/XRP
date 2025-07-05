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

with tab2:
    st.header("Current XRP Ledger Statistics")
    st.markdown("Data source: [rich-list.info](https://rich-list.info/)")

    # ---- USE HISTORY CSVs ----
    stats_files = {
        "Number of Accounts and Sum of Balance Range.csv": "current_stats_accounts_history.csv",
        "Percentage of Accounts with Balances Greater Than or Equal to.csv": "current_stats_percent_history.csv"
    }
    for display_name, history_csv in stats_files.items():
        if not os.path.exists(history_csv):
            st.info(f"No data found for {display_name} (expected {history_csv}).")
            continue

        # Load history, show latest and let user pick previous for comparison
        hist = pd.read_csv(history_csv)
        # Convert date to datetime and sort, keep only columns 0:3
        hist['date'] = pd.to_datetime(hist['date'])
        hist = hist.sort_values('date')
        hist = hist.iloc[:, :4]  # Only use up to 4 columns (date + 3)

        # Option to select any day (default: latest)
        all_dates = hist['date'].dt.date.unique()
        latest_date = all_dates[-1]
        prev_date = all_dates[-2] if len(all_dates) > 1 else None
        sel_date = st.selectbox(
            f"Select date for '{pretty_name(display_name)}':",
            reversed(all_dates),
            format_func=lambda d: d.strftime("%Y-%m-%d")
        )
        latest_tbl = hist[hist['date'].dt.date == sel_date].copy().reset_index(drop=True)
        latest_tbl = latest_tbl.iloc[:, 1:]  # Drop 'date'
        latest_tbl.columns = short_stat_headers(display_name, latest_tbl.columns)

        # Format numbers: add commas to "Accounts" and "Total XRP" (1st & 3rd col for Table1, 2nd col for Table2)
        if "Number of Accounts and Sum of Balance Range" in display_name:
            latest_tbl.iloc[:, 0] = latest_tbl.iloc[:, 0].apply(format_commas)
            if latest_tbl.shape[1] > 2:
                latest_tbl.iloc[:, 2] = latest_tbl.iloc[:, 2].apply(format_commas)
        elif "Percentage of Accounts with Balances Greater Than or Equal to" in display_name:
            if latest_tbl.shape[1] > 1:
                latest_tbl.iloc[:, 1] = latest_tbl.iloc[:, 1].apply(format_commas)

        st.subheader(pretty_name(display_name))
        st.caption(f"Last updated: {latest_date}")
        st.dataframe(latest_tbl, use_container_width=True, hide_index=True)

        # Show previous day's diff if available
        if prev_date is not None and sel_date == latest_date:
            prev_tbl = hist[hist['date'].dt.date == prev_date].copy().reset_index(drop=True)
            prev_tbl = prev_tbl.iloc[:, 1:]
            prev_tbl.columns = latest_tbl.columns

            # Compute absolute and percent change, only for numeric columns
            changes = latest_tbl.copy()
            for col in latest_tbl.columns:
                try:
                    prev_vals = prev_tbl[col].astype(float)
                    latest_vals = latest_tbl[col].astype(float)
                    abs_change = latest_vals - prev_vals
                    pct_change = (abs_change / prev_vals.replace(0, float('nan'))) * 100
                    changes[col] = [f"{a:+,.0f} ({b:+.2f}%)" if pd.notnull(a) else "" for a, b in zip(abs_change, pct_change)]
                except Exception:
                    changes[col] = [""] * len(latest_tbl)
            with st.expander("Show daily change vs previous day"):
                st.dataframe(changes, use_container_width=True, hide_index=True)

        st.download_button(
            label=f"Download history as CSV",
            data=hist.to_csv(index=False).encode(),
            file_name=history_csv,
            mime='text/csv',
        )

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
