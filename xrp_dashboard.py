import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

st.set_page_config(page_title="XRP Rich List Dashboard", layout="wide")

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
    # Shorten or wrap long column names for better display
    if len(col) > 30:
        words = col.split()
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    return col

def format_stats_table(df, table_name):
    # Clean up column names for display
    df.columns = [shorten_col(str(col)) for col in df.columns]
    # Format numeric columns with commas
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
                    # Remove XRP, format, add back XRP
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
    # Remove any row that matches the headers (extra header rows in middle of CSV)
    header_set = set(str(x).strip().lower() for x in df.columns)
    mask = []
    for i, row in df.iterrows():
        row_set = set(str(x).strip().lower() for x in row)
        # If row matches at least 2 header names, mark as header row
        if len(header_set & row_set) >= min(2, len(header_set)):
            mask.append(False)
        else:
            mask.append(True)
    df = df[mask].reset_index(drop=True)
    return df

COLUMN_NAME_MAP = {
    "-- Number of accounts and sum of balance range": "Accounts\nFrom–To",
    "-- Number of accounts and sum of balance range.1": "Balance Range\n(XRP)",
    "-- Number of accounts and sum of balance range.2": "Total in\nRange (XRP)",
    # For the second table, add more as needed
}

def rename_columns(df):
    # Rename columns using the map; fallback to auto-wrap if needed
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
tab1, tab2 = st.tabs(["📈 Rich List Charts", "📋 Current Statistics"])

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

    csv_choice = st.sidebar.selectbox(
        "Choose a data range/table:",
        sorted(csv_files, key=pretty_name),
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

# ---- CURRENT STATISTICS ----
with tab2:
    st.header("Current XRP Ledger Statistics")
    stats_csvs = [
        "Number of Accounts and Sum of Balance Range.csv",
        "Percentage of Accounts with Balances Greater Than or Equal to.csv"
    ]
    found_any = False

    # Optional CSS for extra header wrapping (for st.dataframe, not needed for st.table)
    st.markdown("""
        <style>
        thead tr th { white-space: pre-line !important; word-break: break-word !important; font-size: 13px !important;}
        </style>
        """, unsafe_allow_html=True)

    for csv_name in stats_csvs:
        if os.path.exists(csv_name):
            found_any = True
            st.subheader(pretty_name(csv_name.replace('.csv', '')))
            stat_df = pd.read_csv(csv_name)
            stat_df = clean_stat_df(stat_df)
            stat_df = format_stats_table(stat_df, csv_name)
            stat_df = rename_columns(stat_df)
            # Show table with NO row index and full width, no scrollbars
            st.dataframe(stat_df, use_container_width=True, hide_index=True)
            # Use st.table for no scroll and full visibility!
            st.download_button(
                label=f"Download {csv_name}",
                data=stat_df.to_csv(index=False).encode(),
                file_name=csv_name,
                mime='text/csv',
            )
    if not found_any:
        st.info("No Current Statistics CSVs found. Please add them to the folder.")


# ---- OPTIONAL TIP JAR ----
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Like this project?**")
st.sidebar.markdown("Send XRP tips to: `YOUR_XRP_WALLET_ADDRESS`")
