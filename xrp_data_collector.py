import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
from io import StringIO
import os
from datetime import datetime

CSV_FOLDER = "csv"
os.makedirs(CSV_FOLDER, exist_ok=True)  # Ensure folder exists

url = 'https://rich-list.info/'
headers = {"User-Agent": "Mozilla/5.0"}
today = datetime.utcnow().date().isoformat()

# ---- 1. Scrape Current Statistics tables and append to history ----

r = requests.get(url, headers=headers, timeout=15)
r.raise_for_status()
soup = BeautifulSoup(r.text, 'html.parser')

# --- 1. Find "Last updated" datetime from the page ---
last_updated_text = None
for tag in soup.find_all(string=re.compile("Last updated:")):
    last_updated_text = tag
    break

if not last_updated_text:
    print("Could not find 'Last updated' on the page. Exiting!")
    exit()

# Extract the timestamp (format: 2025-07-06 03:34:10 UTC)
match = re.search(r"Last updated:\s*([0-9\- :]+) UTC", last_updated_text)
if not match:
    print("Couldn't extract the last updated datetime. Exiting!")
    exit()

last_updated_dt_str = match.group(1)
last_updated_dt = datetime.strptime(last_updated_dt_str, '%Y-%m-%d %H:%M:%S')
print(f"Site last updated at: {last_updated_dt} UTC")

# --- 2. Compare with your local last_updated.txt (or create it if missing) ---
last_updated_file = os.path.join('csv', 'last_updated.txt')

if os.path.exists(last_updated_file):
    with open(last_updated_file, 'r') as f:
        prev_dt_str = f.read().strip()
        if prev_dt_str:
            prev_dt = datetime.strptime(prev_dt_str, '%Y-%m-%d %H:%M:%S')
            if last_updated_dt <= prev_dt:
                print(f"Site last updated ({last_updated_dt}) is NOT newer than our last update ({prev_dt}). Skipping.")
                exit()
else:
    # If file doesn't exist, create the csv folder if needed
    os.makedirs('csv', exist_ok=True)

# --- If we reach here, data is NEW and we continue ---
print("New data detected. Proceeding with update.")

# After a successful update, save the new last_updated_dt
with open(last_updated_file, 'w') as f:
    f.write(last_updated_dt.strftime('%Y-%m-%d %H:%M:%S'))


currentstats_div = soup.find('div', id=lambda x: x and x.lower() == 'currentstats')
if not currentstats_div:
    print("Couldn't find the 'Currentstats' div!")
    exit()

tables = currentstats_div.find_all('table')
if not tables or len(tables) < 2:
    print("No tables found inside 'Currentstats' div.")
    exit()

# Table 1: Number of Accounts and Sum of Balance Range
df1 = pd.read_html(StringIO(str(tables[0])))[0]

df1 = df1.iloc[:, :3]
df1.columns = ["Accounts", "Balance Range (XRP)", "Sum in Range (XRP)"]
df1 = df1[df1["Accounts"].astype(str).str.strip().str.match(r"^\d")].reset_index(drop=True)
df1["date"] = today

hist1 = os.path.join(CSV_FOLDER, "current_stats_accounts_history.csv")
if os.path.exists(hist1):
    prev = pd.read_csv(hist1)
    prev = prev[prev["date"] != today]
    df1 = pd.concat([prev, df1], ignore_index=True)
df1.to_csv(hist1, index=False)
print(f"Updated {hist1} ({len(df1)} rows)")

# Table 2: Percentage of Accounts with Balances Greater Than or Equal to
df2 = pd.read_html(StringIO(str(tables[1])))[0]
df2 = df2.iloc[:, :3]
df2.columns = ["Threshold (%)", "Accounts ≥ Threshold", "XRP ≥ Threshold"]
df2 = df2[df2["Threshold (%)"].astype(str).str.strip().str.match(r"^\d")].reset_index(drop=True)
df2["date"] = today

hist2 = os.path.join(CSV_FOLDER, "current_stats_percent_history.csv")
if os.path.exists(hist2):
    prev = pd.read_csv(hist2)
    prev = prev[prev["date"] != today]
    df2 = pd.concat([prev, df2], ignore_index=True)
df2.to_csv(hist2, index=False)
print(f"Updated {hist2} ({len(df2)} rows)")

# ---- 2. Scrape all CanvasJS chart data and save latest-per-day for Series1 ----

# Download HTML again for fresh script tag scan
html = r.text
pattern = r'title:\s*\{\s*text:\s*"([^"]+)"\s*\}.*?data\s*:\s*(\[[\s\S]*?\])\s*\}\s*\);'
matches = re.findall(pattern, html, re.DOTALL)
print(f"\nFound {len(matches)} chart objects on the page.")

def extract_js_object_array(js_array_str):
    # Robust parse of [{...}, {...}]
    items = []
    bracket_level = 0
    curr = ''
    inside = False
    for c in js_array_str.strip():
        if c == '{':
            if bracket_level == 0:
                curr = ''
            bracket_level += 1
            inside = True
        if inside:
            curr += c
        if c == '}':
            bracket_level -= 1
            if bracket_level == 0:
                items.append(curr)
                curr = ''
                inside = False
    return items

all_charts = []

for i, (title, data_raw) in enumerate(matches):
    print(f"\n{i+1}. Chart: {title}")
    series_objects = extract_js_object_array(data_raw)
    for j, obj in enumerate(series_objects):
        label_match = re.search(r'label\s*:\s*"([^"]+)"', obj)
        label = label_match.group(1) if label_match else f"Series{j+1}"

        # **This is the FIXED regex line!**
        dp_match = re.search(r'dataPoints\s*:\s*(\[[^\]]+\])', obj, re.DOTALL)
        if not dp_match:
            print(f"  Skipping series {j+1} ({label}): no dataPoints found.")
            continue
        datapoints_raw = dp_match.group(1)
        datapoints_json = re.sub(r'([{\s,])([xy]):', r'\1"\2":', datapoints_raw)
        try:
            datapoints = json.loads(datapoints_json)
        except Exception as e:
            print(f"  Skipping {label} due to JSON error: {e}")
            continue

        df = pd.DataFrame(datapoints)
        if not len(df):
            continue
        # Convert timestamps if 'x' is epoch ms
        if df['x'].max() > 1e10:
            df['date'] = pd.to_datetime(df['x'], unit='ms')
        else:
            df['date'] = df['x']
        df = df.rename(columns={'y': 'value'})

        safe_title = title.replace('/', '-').replace(' ', '_')
        safe_label = label.replace('/', '-').replace(' ', '_')
        csv_name = os.path.join(CSV_FOLDER, f"{safe_title}__{safe_label}.csv")
        df.to_csv(csv_name, index=False)
        print(f"    Saved data series '{label}' as: {csv_name}")

        # Also create a DAILY_LATEST version for Series1
        if "Series1" in csv_name:
            df['date_full'] = pd.to_datetime(df['date'])
            df['date_only'] = df['date_full'].dt.date
            df_sorted = df.sort_values('date_full')
            latest_per_day = df_sorted.groupby('date_only').tail(1).sort_values('date_only')
            latest_per_day = latest_per_day[['date_only', 'value']]
            latest_per_day = latest_per_day.rename(columns={'date_only': 'date'})
            daily_name = csv_name.replace('.csv', '_DAILY_LATEST.csv')
            latest_per_day.to_csv(daily_name, index=False)
            print(f"      Saved: {daily_name} ({len(latest_per_day)} rows)")

        all_charts.append({'chart_title': title, 'series_label': label, 'csv': csv_name, 'df': df})
print("\nDone! All data processed.")
