import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="XRP Rich List Dashboard", layout="wide")

st.title("📊 XRP Rich List Interactive Dashboard")

# Helper to prettify file names
def pretty_name(name):
    name = name.replace('_', ' ').replace('.csv', '').replace('-', '–')
    name = name.replace('Infinity', '∞')
    return name.title()

# Format large numbers nicely
def format_millions(val):
    if abs(val) >= 1e9:
        return f"{val/1e9:,.2f}B"
    elif abs(val) >= 1e6:
        return f"{val/1e6:,.2f}M"
    elif abs(val) >= 1e3:
        return f"{val/1e3:,.2f}K"
    else:
        return f"{val:,}"

# Function to get Current Statistics tables from rich-list.info
@st.cache_data(ttl=600)
def get_current_statistics_tables():
    url = 'https://rich-list.info/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    # Find all tables after "Current Statistics"
    tables = []
    for tag in soup.find_all(['h2', 'h3']):
        if "Current Statistics" in tag.text:
            t1 = tag.find_next('table')
            t2 = t1.find_next('table') if t1 else None
            if t1: tables.append(pd.read_html(str(t1))[0])
            if t2: tables.append(pd.read_html(str(t2))[0])
            break
    return tables

# Main Tabs
tab1, tab2 = st.tabs(["📈 Rich List Charts", "📋 Current Statistics"])

with tab1:
    # 1. Find all CSVs in this directory
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

    # 2. Sidebar: Pick a CSV
    csv_choice = st.sidebar.selectbox(
        "Choose a data range/table:",
        sorted(csv_files, key=pretty_name),
        format_func=pretty_name
    )

    # 3. Load the selected CSV
    df = pd.read_csv(csv_choice)

    # 4. Handle date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date  # Date only (hide time)
        # Group by date to avoid duplicates (sums values)
        df = df.groupby(date_col, as_index=False)['value'].sum()
    else:
        st.warning("No 'date' column found! Chart x-axis may not be time-based.")

    # 5. Show interactive Plotly chart with formatted Y axis
    st.subheader(f"Chart: {pretty_name(csv_choice)}")
    fig = px.line(
        df,
        x=date_col if date_col else df.columns[0],
        y='value',
        markers=True,
    )
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(tickformat=",", title="XRP")  # Comma format (e.g., 1,000,000)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_tickformat=",",
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6. Show the data as a table (toggle)
    with st.expander("Show Data Table"):
        st.dataframe(
            df.style.format({"value": format_millions}),
            use_container_width=True
        )

    # 7. Download current data as CSV
    st.download_button(
        label="Download this table as CSV",
        data=df.to_csv(index=False).encode(),
        file_name=csv_choice,
        mime='text/csv',
    )

    st.caption("Touch, zoom, and pan the chart. Made for XRP data nerds! 🚀")

with tab2:
    st.header("Current XRP Ledger Statistics")

    # List of your exact stats table filenames
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
            st.dataframe(
                stat_df.style.format(lambda v: format_millions(v) if pd.api.types.is_number(v) else v),
                use_container_width=True
            )
            st.download_button(
                label=f"Download {csv_name}",
                data=stat_df.to_csv(index=False).encode(),
                file_name=csv_name,
                mime='text/csv',
            )
    if not found_any:
        st.info("No Current Statistics CSVs found. Please add them to the folder.")


# Optional: Tip button or link
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Like this project?**")
st.sidebar.markdown("Send XRP tips to: `YOUR_XRP_WALLET_ADDRESS`")
