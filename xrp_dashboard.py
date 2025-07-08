import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime

# ----- DARK THEME INJECT -----
st.set_page_config(page_title="XRP Rich List Dashboard", initial_sidebar_state="collapsed", layout="centered")
st.markdown("""
    <style>
    /* Remove Streamlit's default purple background on markdown divs (like your chart titles) */
    div[data-testid="stMarkdownContainer"] > div {
        background: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-bottom: 0.5em !important;
    }    
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
def normalize_balance_range(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.replace(',', '').split('-')[0].strip()
    try:
        return float(x)
    except Exception:
        return x

def clean_numeric(val):
    """Convert to float after removing commas, % signs, 'XRP', spaces."""
    if pd.isnull(val):
        return float('nan')
    if isinstance(val, str):
        val = (
            val.replace(',', '')
               .replace('XRP', '')
               .replace('%', '')
               .strip()
        )
    try:
        return float(val)
    except Exception:
        return float('nan')


def normalize_threshold(val):
    try:
        if isinstance(val, str):
            return float(val.replace('%','').replace(',','').strip())
        return float(val)
    except Exception:
        return None

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

import streamlit as st
import pandas as pd
import os

def format_int(val):
    try:
        v = float(val)
        return f"{int(v):,}" if v.is_integer() else f"{v:,.4f}".rstrip('0').rstrip('.')
    except Exception:
        return val

def clean_numeric(val):
    if pd.isnull(val):
        return float('nan')
    if isinstance(val, str):
        val = val.replace(',', '').replace('XRP','').replace('%','').strip()
    try:
        return float(val)
    except Exception:
        return float('nan')

def normalize_balance_range(x):
    if pd.isna(x): return None
    if isinstance(x, str):
        x = x.replace(',','').split('-')[0].strip()
    try:
        return float(x)
    except:
        return x

def normalize_threshold(val):
    try:
        if isinstance(val, str):
            return float(val.replace('%','').replace(',','').strip())
        return float(val)
    except Exception:
        return None

def calc_and_display_delta_table(
    df, id_col, delta_cols, table_name, date_col="date", normalize_key_func=None, int_delta_cols=[], return_dataframe=False
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isnull().any():
        st.warning(f"Some rows in {table_name} have invalid date format.")

    # Get available dates in *ascending* order (oldest first)
    dates_available = sorted(df[date_col].dt.date.unique())
    sel_date = st.selectbox(
        f"Select Date for {table_name}:", dates_available[::-1], 0, key=f"date_{table_name}"
    )
    show_delta = st.checkbox(
        f"Show change vs previous day", value=True, key=f"delta_{table_name}"
    )

    # This day's data
    today_df = df[df[date_col].dt.date == sel_date].copy()
    today_df = today_df.drop_duplicates(subset=[id_col])
    keep_cols = [date_col, id_col] + delta_cols
    today_df = today_df[keep_cols].reset_index(drop=True)

    # Closest previous day, not necessarily “yesterday”
    prev_dates = [d for d in dates_available if d < sel_date]
    prior_date = max(prev_dates) if prev_dates else None
    yest_df = df[df[date_col].dt.date == prior_date].copy() if prior_date else pd.DataFrame()
    yest_df = yest_df.drop_duplicates(subset=[id_col])
    yest_df = yest_df[keep_cols].reset_index(drop=True) if not yest_df.empty else pd.DataFrame()

    pretty_map = {
        "Balance Range (XRP)": "Balance Range (XRP)",
        "Sum in Range (XRP)": "Sum in Range (XRP)",
        "Accounts": "Accounts",
        "Threshold (%)": "Threshold (%)",
        "Accounts ≥ Threshold": "Accounts ≥ Threshold",
        "XRP ≥ Threshold": "XRP ≥ Threshold"
    }
    today_df.columns = [pretty_map.get(c, c) for c in today_df.columns]
    yest_df.columns = [pretty_map.get(c, c) for c in yest_df.columns]
    id_col_pretty = pretty_map.get(id_col, id_col)

    if normalize_key_func:
        today_df["MergeKey"] = today_df[id_col_pretty].apply(normalize_key_func)
        yest_df["MergeKey"] = yest_df[id_col_pretty].apply(normalize_key_func) if not yest_df.empty else None
        merge_id = "MergeKey"
    else:
        merge_id = id_col_pretty

    # Compute Deltas
    if show_delta and not yest_df.empty:
        merged = today_df.merge(
            yest_df,
            on=merge_id,
            how="left",
            suffixes=('', '_prev')
        )
        # In delta formatting for Δ columns:
        for col in delta_cols:
            col_pretty = pretty_map.get(col, col)
            col_prev = f"{col_pretty}_prev"
            if col_pretty in merged.columns and col_prev in merged.columns:
                delta = merged[col_pretty].apply(clean_numeric) - merged[col_prev].apply(clean_numeric)
                # Better: Custom formatting for each type
                if col_pretty in int_delta_cols:
                    merged[f"{col_pretty} Δ"] = delta.apply(lambda v: f"{int(round(v)):+,d}" if pd.notnull(v) else "")
                elif "XRP" in col_pretty:
                    merged[f"{col_pretty} Δ"] = delta.apply(lambda v: f"{v:+,.4f}".rstrip('0').rstrip('.') + " XRP" if pd.notnull(v) else "")
                else:
                    merged[f"{col_pretty} Δ"] = delta.apply(lambda v: f"{v:+,}" if pd.notnull(v) else "")
            else:
                merged[f"{col_pretty} Δ"] = ""


        keep = [c for c in merged.columns if not c.endswith("_prev") and c != "MergeKey"]
        today_df = merged[keep]

    # Format values (with commas, no exponential)
    for c in today_df.columns:
        if "Δ" in c or "Delta" in c:
            continue  # already formatted
        elif "Accounts" in c or "Sum in Range" in c or "XRP" in c:
            today_df[c] = today_df[c].apply(format_int)
        if c == id_col_pretty:
            today_df[c] = today_df[c]  # Keep as is (do not overwrite with nan!)

    # Show the table
    st.subheader(table_name)
    subtitle = f"<span style='color:#aaa;'>Date: {sel_date}"
    if show_delta and prior_date:
        subtitle += f" (compared to {prior_date})"
    subtitle += "</span>"
    st.markdown(subtitle, unsafe_allow_html=True)
    st.dataframe(today_df.drop(columns=[date_col]), use_container_width=True, hide_index=True)
    st.download_button(
        label=f"Download {table_name}",
        data=today_df.to_csv(index=False).encode(),
        file_name=f"{table_name.replace(' ', '_')}_{sel_date}.csv",
        mime='text/csv',
    )
    if return_dataframe:
        return today_df
    return None


def format_number(x):
    # Format number with commas, no exponentials, keep 2 decimals if needed
    try:
        x = float(x)
        if abs(x) < 1e3:
            return f"{x:,.0f}"
        if abs(x) < 1:
            return f"{x:.6f}"
        if abs(x) % 1 == 0:
            return f"{int(x):,}"
        return f"{x:,.2f}"
    except Exception:
        return x

with tab2:
    st.header("Current XRP Ledger Statistics")

    ACCOUNTS_CSV = "csv/current_stats_accounts_history.csv"
    PERCENT_CSV  = "csv/current_stats_percent_history.csv"

    # ---- Whale Wallet Summary Table at Top ----
    if os.path.exists(ACCOUNTS_CSV):
        df = pd.read_csv(ACCOUNTS_CSV)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        def parse_lower(x):
            try:
                return float(x.split('-')[0].replace(',', '').strip())
            except:
                return 0

        df['min_balance'] = df['Balance Range (XRP)'].apply(parse_lower)
        df['Accounts'] = pd.to_numeric(df['Accounts'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

        summary_rows = []
        for date, g in df.groupby("date"):
            g = g.copy()
            whale_1m = g[g['min_balance'] >= 1_000_000]['Accounts'].sum()
            whale_100k = g[g['min_balance'] >= 100_000]['Accounts'].sum()
            summary_rows.append({
                'Date': date.date(),
                'Wallets ≥ 1M XRP': whale_1m,
                'Wallets ≥ 100K XRP': whale_100k,
            })

        summary = pd.DataFrame(summary_rows).sort_values('Date').reset_index(drop=True)
        summary['Δ vs Prior Day (1M+)'] = summary['Wallets ≥ 1M XRP'].diff().fillna(0).astype(int)
        summary['Δ vs Prior Day (100K+)'] = summary['Wallets ≥ 100K XRP'].diff().fillna(0).astype(int)

        # Format for display
        display_summary = summary.copy()
        for col in ['Wallets ≥ 1M XRP', 'Wallets ≥ 100K XRP', 'Δ vs Prior Day (1M+)', 'Δ vs Prior Day (100K+)']:
            display_summary[col] = display_summary[col].map(format_number)
        display_summary = display_summary.sort_values('Date', ascending=False).reset_index(drop=True)

        st.markdown("### 🦈 Whale Wallet Count Summary (by Day)")
        st.caption("Sum of all XRP wallets holding at least 1,000,000 XRP or 100,000 XRP. Shows daily totals and change from the previous day.")
        st.dataframe(display_summary, use_container_width=True)

       # ---- Table 1: Number Of Accounts And Sum Of Balance Range ----
    if os.path.exists(ACCOUNTS_CSV):
        df = pd.read_csv(ACCOUNTS_CSV)
        try:
            df["Sum in Range (XRP)"] = pd.to_numeric(df["Sum in Range (XRP)"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df["Accounts"] = pd.to_numeric(df["Accounts"].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

            calc_and_display_delta_table(
                df,
                id_col="Balance Range (XRP)",
                delta_cols=["Accounts", "Sum in Range (XRP)"],
                table_name="Number Of Accounts And Sum Of Balance Range",
                normalize_key_func=normalize_balance_range,
                int_delta_cols=["Accounts"]
            )
        except Exception as e:
            st.error(f"Table 1 error: {e}")
    else:
        st.info("current_stats_accounts_history.csv not found.")

    # ---- Table 2: Percentage Of Accounts With Balances Greater Than Or Equal To ----
    if os.path.exists(PERCENT_CSV):
        df = pd.read_csv(PERCENT_CSV)
        try:
            df["Accounts ≥ Threshold"] = pd.to_numeric(df["Accounts ≥ Threshold"].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
            df["XRP ≥ Threshold"] = (
                df["XRP ≥ Threshold"]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('XRP', '', regex=False)
                .str.replace(' ', '', regex=False)
                .astype(float)
            )

            calc_and_display_delta_table(
                df,
                id_col="Threshold (%)",
                delta_cols=["Accounts ≥ Threshold", "XRP ≥ Threshold"],
                table_name="Percentage Of Accounts With Balances Greater Than Or Equal To",
                normalize_key_func=normalize_threshold,
                int_delta_cols=["Accounts ≥ Threshold"]
            )
        except Exception as e:
            st.error(f"Table 2 error: {e}")
    else:
        st.info("current_stats_percent_history.csv not found.")





with tab1:
    st.header("📈 Rich List Charts")

    # List and sort all Series1_DAILY_LATEST.csv files as before
    csv_files = [
        f for f in os.listdir('csv')
        if f.endswith('_Series1_DAILY_LATEST.csv')
    ]
    file_to_title = {
        f: f.replace('_Series1_DAILY_LATEST.csv', '').replace('_', ' ').replace('-', '–').replace('Infinity', '∞').strip()
        for f in csv_files
    }

    # Organize for display: non-numeric first, then numeric by order
    non_num = sorted(
        [f for f, title in file_to_title.items() if is_not_number_start(title)],
        key=lambda x: file_to_title[x]
    )
    num_start = sorted(
        [f for f, title in file_to_title.items() if not is_not_number_start(title)],
        key=lambda x: extract_leading_number(file_to_title[x])
    )
    ordered_csvs = non_num + num_start
    def is_wallet_count_file(filename):
        # Normalize for robust detection
        s = filename.lower().replace('_', ' ').replace('-', ' ').replace('__', ' ')
        return "historic wallet count" in s
        if not ordered_csvs:
            st.error("No CSV files found in this folder!")
            st.stop()
    
    # Loop through and display all charts!
    for csv_file in ordered_csvs:
        df = pd.read_csv(os.path.join('csv', csv_file))
        title = file_to_title[csv_file]
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
    
        # Decide the label based on the title (or file name, whatever is unique)
        value_col = "Wallet Count" if "wallet count" in title.lower() or "historic wallet count" in title.lower() else "Total XRP"
    
        # Rename column for plotting
        df_plot = df.rename(columns={"value": value_col})
        if "–" in title or "-" in title:
            chart_title = f"{title} XRP Balance Range"
        else:
            chart_title = f"{title}"
        # Plot with correct y axis
        fig = px.line(
            df_plot,
            x=date_col,
            y=value_col,  # MUST match the renamed col
            markers=True,
            labels={value_col: value_col}
        )
        fig.update_yaxes(
            tickformat="~s",
            title_text=value_col   # <--- This will match exactly!
        )
        fig.update_traces(
            line=dict(width=1.5),
            marker=dict(size=4, color='#aad8ff', line=dict(width=0)),
            mode="lines+markers",
            hovertemplate=f"<b>%{{x|%b %d, %Y}}</b><br>{value_col}=%{{y:,}}<extra></extra>",
        )
        fig.update_layout(
            xaxis_title="Date",
            hovermode="x",
            xaxis=dict(showspikes=True, spikemode='across', spikethickness=2),
            hoverlabel=dict(namelength=-1),
            plot_bgcolor='#1e222d',
            paper_bgcolor='#1e222d',
            font=dict(color='#F1F1F1'),
            dragmode=False
        )
        st.markdown(
            f"<div style='font-size:1.8em; font-weight:700; margin-bottom:0.2em; margin-top:2.2em'>{chart_title}</div>",
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': False,
            'staticPlot': False,
            'scrollZoom': False,
            'editable': False,
            'doubleClick': 'reset',
        })
    
        # Table (ensure value_col in columns)
        df_display = df_plot.copy()
        df_display[value_col] = df_display[value_col].apply(format_full_number)
        df_display = df_display.sort_values(by=date_col, ascending=False).reset_index(drop=True)
    
        with st.expander("Show Data Table", expanded=False):
            st.dataframe(
                df_display[[date_col, value_col]],
                use_container_width=True
            )
        st.download_button(
            label="Download this table as CSV",
            data=df_display.to_csv(index=False).encode(),
            file_name=csv_file,
            mime='text/csv',
        )
        st.markdown("&nbsp;"*3, unsafe_allow_html=True)  # 3 line breaks
    
        st.caption("Click or Touch any Point on the Chart to see its Hoverbox Value. Tap either Axis to reset Hoverbox")
    
    st.info("Scroll down to see all charts!")





# ---- OPTIONAL TIP JAR ----
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Like this project?**")
st.sidebar.markdown("Send XRP tips to: `YOUR_XRP_WALLET_ADDRESS`")
