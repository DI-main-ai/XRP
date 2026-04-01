import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

RPC_URL = "https://xrplcluster.com"
CSV_DIR = Path("csv")
LEDGER_CACHE_PATH = CSV_DIR / "_ledger_date_cache.json"

RANGE_SPECS = [
    {"label": "0 - 20", "stem": "0_-_20", "lower": 0, "upper": 20},
    {"label": "20 - 500", "stem": "20_-_500", "lower": 20, "upper": 500},
    {"label": "500 - 1,000", "stem": "500_-_1000", "lower": 500, "upper": 1_000},
    {"label": "1,000 - 5,000", "stem": "1,000_-_5,000", "lower": 1_000, "upper": 5_000},
    {"label": "5,000 - 10,000", "stem": "5,000_-_10,000", "lower": 5_000, "upper": 10_000},
    {"label": "10,000 - 25,000", "stem": "10,000_-_25,000", "lower": 10_000, "upper": 25_000},
    {"label": "25,000 - 50,000", "stem": "25,000_-_50,000", "lower": 25_000, "upper": 50_000},
    {"label": "50,000 - 75,000", "stem": "50,000_-_75,000", "lower": 50_000, "upper": 75_000},
    {"label": "75,000 - 100,000", "stem": "75,000_-_100,000", "lower": 75_000, "upper": 100_000},
    {"label": "100,000 - 500,000", "stem": "100,000_-_500,000", "lower": 100_000, "upper": 500_000},
    {"label": "500,000 - 1,000,000", "stem": "500,000_-_1,000,000", "lower": 500_000, "upper": 1_000_000},
    {"label": "1,000,000 - 5,000,000", "stem": "1,000,000_-_5,000,000", "lower": 1_000_000, "upper": 5_000_000},
    {"label": "5,000,000 - 10,000,000", "stem": "5,000,000_-_10,000,000", "lower": 5_000_000, "upper": 10_000_000},
    {"label": "10,000,000 - 20,000,000", "stem": "10,000,000_-_20,000,000", "lower": 10_000_000, "upper": 20_000_000},
    {"label": "20,000,000 - 100,000,000", "stem": "20,000,000_-_100,000,000", "lower": 20_000_000, "upper": 100_000_000},
    {"label": "100,000,000 - 500,000,000", "stem": "100,000,000_-_500,000,000", "lower": 100_000_000, "upper": 500_000_000},
    {"label": "500,000,000 - 1,000,000,000", "stem": "500,000,000_-_1,000,000,000", "lower": 500_000_000, "upper": 1_000_000_000},
    {"label": "1,000,000,000 - Infinity", "stem": "1,000,000,000_-_Infinity", "lower": 1_000_000_000, "upper": None},
]

PERCENT_THRESHOLDS = [0.01, 0.10, 0.20, 0.50, 1, 2, 3, 4, 5, 10]


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def xrpl_request(method, params=None, max_retries=6):
    payload = {
        "method": method,
        "params": [{**(params or {}), "api_version": 2}],
    }

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = requests.post(RPC_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            result = data.get("result", {})
            if result.get("status") == "error":
                raise RuntimeError(result)
            return result
        except Exception as exc:
            last_exc = exc
            sleep_s = min(2 ** attempt, 30)
            print(f"Request failed for {method} (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                print(f"Retrying in {sleep_s}s...")
                time.sleep(sleep_s)

    raise RuntimeError(f"Failed XRPL request for {method}") from last_exc


def parse_iso_utc(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def get_latest_validated_ledger():
    result = xrpl_request("server_info")
    info = result.get("info", result)
    validated = info.get("validated_ledger") or result.get("validated_ledger")
    if not validated:
        raise RuntimeError("validated_ledger missing from server_info response")

    ledger_hash = validated.get("hash")
    ledger_index = validated.get("seq")
    if ledger_hash is None or ledger_index is None:
        raise RuntimeError(f"Could not read validated ledger hash/index from response: {validated}")

    header = get_ledger_header(int(ledger_index))
    return {
        "ledger_hash": ledger_hash,
        "ledger_index": int(ledger_index),
        "close_time_iso": header["close_time_iso"],
        "close_dt": header["close_dt"],
    }


def get_ledger_header(ledger_index: int):
    result = xrpl_request(
        "ledger",
        {
            "ledger_index": int(ledger_index),
            "transactions": False,
            "expand": False,
            "binary": False,
        },
    )

    ledger = result.get("ledger")
    if not ledger:
        raise RuntimeError(f"No ledger object returned for ledger_index={ledger_index}")

    close_time_iso = ledger["close_time_iso"]
    return {
        "ledger_index": int(ledger["ledger_index"]),
        "ledger_hash": ledger["ledger_hash"],
        "close_time_iso": close_time_iso,
        "close_dt": parse_iso_utc(close_time_iso),
    }


def load_ledger_cache():
    if not LEDGER_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(LEDGER_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_ledger_cache(cache):
    LEDGER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def find_ledger_for_day(day: date, latest_header=None, low_hint=1, high_hint=None):
    """
    Find the last validated ledger whose close time is <= day 23:59:59 UTC.
    """
    cache = load_ledger_cache()
    key = day.isoformat()
    if key in cache:
        cached = cache[key]
        print(f"Using cached ledger for {key}: {cached['ledger_index']}")
        return {
            "ledger_index": int(cached["ledger_index"]),
            "ledger_hash": cached["ledger_hash"],
            "close_time_iso": cached["close_time_iso"],
            "close_dt": parse_iso_utc(cached["close_time_iso"]),
        }

    if latest_header is None:
        latest_header = get_latest_validated_ledger()

    target_dt = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)

    if target_dt >= latest_header["close_dt"]:
        result = latest_header
    else:
        lo = max(1, int(low_hint))
        hi = int(high_hint or latest_header["ledger_index"])
        best = None

        while lo <= hi:
            mid = (lo + hi) // 2
            header = get_ledger_header(mid)

            if header["close_dt"] <= target_dt:
                best = header
                lo = mid + 1
            else:
                hi = mid - 1

        if best is None:
            raise RuntimeError(f"Could not find a ledger for {key}")

        result = best

    cache[key] = {
        "ledger_index": int(result["ledger_index"]),
        "ledger_hash": result["ledger_hash"],
        "close_time_iso": result["close_time_iso"],
    }
    save_ledger_cache(cache)
    return result


def fetch_all_balances_drops(ledger_hash: str, ledger_index: int):
    print(f"Fetching AccountRoot state from ledger {ledger_index} ({ledger_hash})...")
    marker = None
    balances = []
    page = 0

    while True:
        params = {
            "ledger_hash": ledger_hash,
            "type": "account",
            "limit": 256,
            "binary": False,
        }
        if marker is not None:
            params["marker"] = marker

        result = xrpl_request("ledger_data", params)
        state = result.get("state", [])

        for obj in state:
            if obj.get("LedgerEntryType") == "AccountRoot" and "Balance" in obj:
                balances.append(int(obj["Balance"]))

        marker = result.get("marker")
        page += 1

        if page % 250 == 0:
            print(f"Pages: {page} | Accounts collected: {len(balances):,}")

        if not marker:
            break

        time.sleep(0.05)

    if not balances:
        raise RuntimeError("No AccountRoot balances were collected from the ledger.")

    series = pd.Series(balances, dtype="int64").sort_values(ascending=False, ignore_index=True)
    print(f"Collected {len(series):,} accounts from ledger {ledger_index}.")
    return series


def format_threshold_label(threshold):
    if threshold < 1:
        return f"{threshold:.2f}%"
    if float(threshold).is_integer():
        return f"{int(threshold)}%"
    return f"{threshold:g}%"


def append_or_replace_rows(existing_df, new_df, key_cols):
    if existing_df is None or existing_df.empty:
        merged = new_df.copy()
    else:
        existing = existing_df.copy()
        new_copy = new_df.copy()

        for col in key_cols:
            if col in existing.columns:
                existing[col] = existing[col].astype(str)
            if col in new_copy.columns:
                new_copy[col] = new_copy[col].astype(str)

        existing["_key"] = existing[key_cols].agg("||".join, axis=1)
        new_copy["_key"] = new_copy[key_cols].agg("||".join, axis=1)

        existing = existing[~existing["_key"].isin(set(new_copy["_key"]))]
        merged = pd.concat(
            [existing.drop(columns=["_key"]), new_copy.drop(columns=["_key"])],
            ignore_index=True,
        )

    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
        sort_cols = ["date"] + [c for c in key_cols if c != "date"]
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return merged


def write_history_csv(path, new_rows, key_cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_df = pd.read_csv(path) if path.exists() else None
    merged = append_or_replace_rows(existing_df, new_rows, key_cols=key_cols)
    merged.to_csv(path, index=False)
    print(f"Wrote {path}")


def build_accounts_history(balance_drops, snapshot_date):
    rows = []

    for spec in RANGE_SPECS:
        lower_drops = int(spec["lower"] * 1_000_000)
        if spec["upper"] is None:
            mask = balance_drops >= lower_drops
        else:
            upper_drops = int(spec["upper"] * 1_000_000)
            mask = (balance_drops >= lower_drops) & (balance_drops < upper_drops)

        rows.append(
            {
                "Accounts": int(mask.sum()),
                "Balance Range (XRP)": spec["label"],
                "Sum in Range (XRP)": float(balance_drops[mask].sum() / 1_000_000),
                "date": snapshot_date,
            }
        )

    return pd.DataFrame(rows)


def build_percent_history(balance_drops, snapshot_date):
    total_accounts = len(balance_drops)
    rows = []

    for pct in PERCENT_THRESHOLDS:
        count = max(1, int(math.floor(total_accounts * (pct / 100.0))))
        cutoff_balance_xrp = float(balance_drops.iloc[count - 1] / 1_000_000)

        rows.append(
            {
                "Threshold (%)": format_threshold_label(pct),
                "Accounts ≥ Threshold": int(count),
                "XRP ≥ Threshold": f"{cutoff_balance_xrp:,.6f} XRP",
                "date": snapshot_date,
            }
        )

    return pd.DataFrame(rows)


def write_range_series_files(balance_drops, snapshot_date):
    for spec in RANGE_SPECS:
        lower_drops = int(spec["lower"] * 1_000_000)
        if spec["upper"] is None:
            mask = balance_drops >= lower_drops
        else:
            upper_drops = int(spec["upper"] * 1_000_000)
            mask = (balance_drops >= lower_drops) & (balance_drops < upper_drops)

        total_xrp = float(balance_drops[mask].sum() / 1_000_000)
        new_row = pd.DataFrame([{"date": snapshot_date, "value": total_xrp}])

        for suffix in ("__Series1.csv", "__Series1_DAILY_LATEST.csv"):
            path = CSV_DIR / f"{spec['stem']}{suffix}"
            write_history_csv(path, new_row, key_cols=["date"])


def write_historic_wallet_count(balance_drops, snapshot_date):
    new_row = pd.DataFrame([{"date": snapshot_date, "value": int(len(balance_drops))}])

    for suffix in ("__Series1.csv", "__Series1_DAILY_LATEST.csv"):
        path = CSV_DIR / f"Historic_Wallet_Count{suffix}"
        write_history_csv(path, new_row, key_cols=["date"])


def write_last_updated(snapshot_ts):
    path = CSV_DIR / "last_updated.txt"
    path.write_text(snapshot_ts.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(f"Wrote {path}")


def write_snapshot_for_date(snapshot_date_str, balance_drops):
    accounts_history = build_accounts_history(balance_drops, snapshot_date_str)
    percent_history = build_percent_history(balance_drops, snapshot_date_str)

    write_history_csv(
        CSV_DIR / "current_stats_accounts_history.csv",
        accounts_history,
        key_cols=["date", "Balance Range (XRP)"],
    )
    write_history_csv(
        CSV_DIR / "current_stats_percent_history.csv",
        percent_history,
        key_cols=["date", "Threshold (%)"],
    )

    write_range_series_files(balance_drops, snapshot_date_str)
    write_historic_wallet_count(balance_drops, snapshot_date_str)


def get_existing_snapshot_dates():
    path = CSV_DIR / "current_stats_accounts_history.csv"
    if not path.exists():
        return set()

    try:
        df = pd.read_csv(path, usecols=["date"])
        return set(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").unique())
    except Exception:
        return set()


def parse_env_date(name):
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start_day: date, end_day: date):
    cur = start_day
    while cur <= end_day:
        yield cur
        cur += timedelta(days=1)


def resolve_run_dates(today_utc: date):
    start = parse_env_date("BACKFILL_START")
    end = parse_env_date("BACKFILL_END")
    auto_gap = truthy(os.getenv("AUTO_BACKFILL_GAP", "false"))
    skip_existing = not truthy(os.getenv("BACKFILL_INCLUDE_EXISTING", "false"))

    existing_dates = get_existing_snapshot_dates()

    if start or end or auto_gap:
        if auto_gap and not (start or end):
            if existing_dates:
                latest_existing = max(datetime.strptime(d, "%Y-%m-%d").date() for d in existing_dates)
                start = latest_existing + timedelta(days=1)
            else:
                raise RuntimeError("AUTO_BACKFILL_GAP was requested, but no existing history file was found.")
            end = today_utc - timedelta(days=1)
        else:
            if start is None or end is None:
                raise RuntimeError("BACKFILL_START and BACKFILL_END must both be provided in YYYY-MM-DD format.")

        if end < start:
            raise RuntimeError("BACKFILL_END cannot be earlier than BACKFILL_START.")

        days = [d for d in daterange(start, end)]
        if skip_existing:
            days = [d for d in days if d.isoformat() not in existing_dates]

        return "backfill", days

    return "live", [today_utc]


def process_live_day(today_utc: date):
    latest = get_latest_validated_ledger()
    balances = fetch_all_balances_drops(latest["ledger_hash"], latest["ledger_index"])
    write_snapshot_for_date(today_utc.isoformat(), balances)
    write_last_updated(datetime.now(timezone.utc))
    print(f"Live update complete for {today_utc.isoformat()} using ledger {latest['ledger_index']}.")


def process_backfill_days(days):
    if not days:
        print("No dates to backfill.")
        return

    latest = get_latest_validated_ledger()
    low_hint = 1

    for idx, day in enumerate(days, start=1):
        print(f"\n=== Backfill {idx}/{len(days)}: {day.isoformat()} ===")
        header = find_ledger_for_day(day, latest_header=latest, low_hint=low_hint, high_hint=latest["ledger_index"])
        print(
            f"Chosen ledger {header['ledger_index']} for {day.isoformat()} "
            f"(closed {header['close_time_iso']})"
        )

        balances = fetch_all_balances_drops(header["ledger_hash"], header["ledger_index"])
        write_snapshot_for_date(day.isoformat(), balances)

        low_hint = header["ledger_index"]

    write_last_updated(datetime.now(timezone.utc))
    print("Backfill complete.")


def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    today_utc = datetime.now(timezone.utc).date()

    mode, days = resolve_run_dates(today_utc)
    print(f"Mode: {mode}")
    print(f"Dates to process: {[d.isoformat() for d in days]}")

    if mode == "live":
        process_live_day(days[0])
    else:
        process_backfill_days(days)


if __name__ == "__main__":
    main()
