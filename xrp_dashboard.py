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

    for table_name, fname, label_map in [
        ("Number Of Accounts And Sum Of Balance Range", ACCOUNTS_CSV,
         {"Accounts": "Accounts", "Balance Range (XRP)": "Balance Range (XRP)", "Sum in Range (XRP)": "Sum in Range (XRP)"}),
        ("Percentage Of Accounts With Balances Greater Than Or Equal To", PERCENT_CSV,
         {"Threshold (%)": "Threshold (%)", "Accounts ≥ Thresh": "Accounts ≥ Thresh", "XRP ≥ Thresh": "XRP ≥ Thresh"})
    ]:
        if not os.path.exists(fname):
            st.info(f"{fname} not found.")
            continue

        df = pd.read_csv(fname)
        if "date" not in df.columns:
            st.warning(f"No date column in {fname}.")
            continue

        df["date"] = pd.to_datetime(df["date"])
        date_options = sorted(df["date"].dt.date.unique(), reverse=True)
        sel_date = st.selectbox(
            f"Select Date for {table_name}:", date_options, 0, key=table_name)
        show_delta = st.checkbox(f"Show change vs previous day", value=True, key=f"delta_{table_name}")

        today_df = df[df["date"].dt.date == sel_date].copy()
        yesterday_df = df[df["date"].dt.date == (sel_date - pd.Timedelta(days=1))].copy()

        # Column detection (robust to column order)
        columns = today_df.columns.tolist()
        # Try to get main 3 columns after 'date'
        data_cols = [c for c in columns if c != "date"][:3]
        today_df = today_df[["date"] + data_cols].reset_index(drop=True)
        today_df.columns = ["Date"] + list(label_map.values())

        if show_delta and not yesterday_df.empty:
            yest_df = yesterday_df.reset_index(drop=True)
            # Match rows by "Balance Range (XRP)" or "Threshold (%)"
            id_col = list(label_map.values())[1]
            yest_df = yest_df[[id_col] + data_cols[1:]]
            today_df = today_df.merge(yest_df, left_on=id_col, right_on=id_col, suffixes=('', '_prev'))
            for col in data_cols[1:]:
                col_now = label_map[col]
                col_prev = f"{col}_prev"
                delta = today_df[col_now] - today_df[col_prev]
                # Try to get percent change if numbers
                try:
                    percent = 100 * delta / today_df[col_prev]
                    percent = percent.replace([float("inf"), -float("inf")], float("nan"))
                except Exception:
                    percent = float("nan")
                today_df[f"{col_now} Δ"] = delta
                today_df[f"{col_now} Δ %"] = percent
            # Remove prev columns
            today_df = today_df[[c for c in today_df.columns if not c.endswith("_prev")]]

        # Format numbers for display
        for c in today_df.columns:
            if "Δ %" in c:
                today_df[c] = today_df[c].apply(lambda v: f"{v:+.2f}%" if pd.notnull(v) else "")
            elif "Δ" in c:
                today_df[c] = today_df[c].apply(lambda v: f"{v:+,.0f}" if pd.notnull(v) and str(v).replace('.','',1).replace('-','').isdigit() else "")
            elif "Accounts" in c or "Sum" in c or "XRP" in c:
                today_df[c] = today_df[c].apply(format_int)

        st.subheader(table_name)
        st.markdown(f"<span style='color:#aaa;'>Date: {sel_date}</span>", unsafe_allow_html=True)
        st.dataframe(today_df.drop(columns="Date"), use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Download {table_name}",
            data=today_df.to_csv(index=False).encode(),
            file_name=f"{table_name.replace(' ', '_')}_{sel_date}.csv",
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
