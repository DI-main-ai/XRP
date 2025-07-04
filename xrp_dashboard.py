import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import numpy as np

# ----- DARK THEME INJECT -----
st.set_page_config(page_title="XRP Rich List Dashboard", initial_sidebar_state="collapsed", layout="wide")



# For Streamlit Cloud, enforce dark mode
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #1e222d !important;
        color: #F1F1F1 !important;
    }
    .stApp { background-color: #1e222d !important; }
    /* Dataframes, tables, and headers */
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
    /* Sidebar and widgets */
    .css-1544g2n, .css-1v0mbdj, .css-1hynsf2, .st-cq, .st-bb, .stSidebar, .stDropdown, .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    /* Dropdown tweaks */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    .stSelectbox [data-testid="stSelectbox"] {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    /* Dropdown menu options */
    .css-1g6gooi, .css-1d391kg {
        background-color: #232334 !important;
        color: #F1F1F1 !important;
    }
    /* Download button tweaks */
    .stDownloadButton > button { background: #3d4053 !important; color: #fff; }
    /* Sidebar font */
    .sidebar-content, .stSidebar {
        color: #F1F1F1 !important;
        background-color: #232334 !important;
    }
    /* Force icon colors */
    .stMarkdown svg { color: #E7E7E7 !important; }
    /* Scrollbar for dataframes */
    ::-webkit-scrollbar { width: 9px; background: #232334; }
    ::-webkit-scrollbar-thumb { background: #37374f; border-radius: 4px;}
    </style>
    """, unsafe_allow_html=True)


st.title("📊 XRP Rich List Interactive Dashboard")
def extract_leading_number(name):
    # Get the first group of digits (possibly with commas/underscores) before a separator
    match = re.match(r"^([0-9,_.]+)", name)
    if match:
        # Remove commas and underscores, convert to float for sorting
        try:
            num = float(match.group(1).replace(",", "").replace("_", ""))
            return num
        except:
            return float('inf')
    return float('inf')  # Non-numeric names sort separately

def is_not_number_start(name):
    # True if not starting with a digit
    return not re.match(r"^\d", name)
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
    if not found_any:
        st.info("No Current Statistics CSVs found. Please add them to the folder.")

with tab1:
    csv_files = [
        f for f in os.listdir('.')
        if f.endswith('.csv')
           and not f.lower().startswith('number of accounts')
           and not f.lower().startswith('percentage of accounts')
           and not f.lower().startswith('current_stats')
    ]

    if not csv_files:
        st.error("No CSV files found in this folder!")
        st.stop()

    non_num = sorted([f for f in csv_files if is_not_number_start(f)], key=pretty_name)
    num_start = sorted(
        [f for f in csv_files if not is_not_number_start(f)],
        key=lambda x: extract_leading_number(os.path.splitext(x)[0])
    )

    # Final ordered list
    ordered_csvs = non_num + num_start

    csv_choice = st.sidebar.selectbox(
        "Choose a data range/table:",
        ordered_csvs,
        format_func=pretty_name
    )

    df = pd.read_csv(csv_choice)
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        df = df.groupby(date_col, as_index=False)['value'].sum()
    else:
        st.warning("No 'date' column found! Chart x-axis may not be time-based.")

    st.subheader(f"Chart: {pretty_name(csv_choice)}")
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
