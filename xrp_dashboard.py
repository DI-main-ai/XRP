import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="XRP Rich List Dashboard", layout="wide")

st.title("📊 XRP Rich List Interactive Dashboard")

# 1. Find all CSVs in this directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if not csv_files:
    st.error("No CSV files found in this folder!")
    st.stop()

# 2. Sidebar: Pick a CSV
csv_choice = st.sidebar.selectbox("Choose a data range/table:", sorted(csv_files))

# 3. Load the selected CSV
df = pd.read_csv(csv_choice)
# Try to handle date formats (auto-detect column)
date_col = None
for col in df.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col])
else:
    st.warning("No 'date' column found! Chart x-axis may not be time-based.")

# 4. Show interactive Plotly chart
st.subheader(f"Chart: {csv_choice.replace('_', ' ').replace('.csv','')}")
fig = px.line(df, x=date_col if date_col else df.columns[0], y='value', markers=True)
st.plotly_chart(fig, use_container_width=True)

# 5. Show the data as a table
st.subheader("Raw Data Table")
st.dataframe(df, use_container_width=True)

# 6. Download current data as CSV
st.download_button(
    label="Download this table as CSV",
    data=df.to_csv(index=False).encode(),
    file_name=csv_choice,
    mime='text/csv',
)

st.caption("Touch, zoom, and pan the chart. Made for XRP data nerds! 🚀")

