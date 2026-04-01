import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import time

RPC_URL = "https://xrplcluster.com"


def xrpl_request(method, params=None):
    payload = {
        "method": method,
        "params": [{**(params or {}), "api_version": 2}]
    }

    r = requests.post(RPC_URL, json=payload, timeout=60)
    r.raise_for_status()
    result = r.json()["result"]

    if result.get("status") == "error":
        raise Exception(result)

    return result


def fetch_all_accounts():
    print("🔄 Fetching ledger data from XRPL...")

    marker = None
    rows = []
    page = 0

    while True:
        params = {
            "ledger_index": "validated",
            "type": "AccountRoot",
            "limit": 256,
        }

        if marker:
            params["marker"] = marker

        result = xrpl_request("ledger_data", params)

        state = result.get("state", [])

        for obj in state:
            if obj.get("LedgerEntryType") == "AccountRoot":
                balance = int(obj["Balance"]) / 1_000_000

                rows.append({
                    "account": obj["Account"],
                    "balance": balance,
                })

        marker = result.get("marker")
        page += 1

        print(f"Page {page} | Accounts collected: {len(rows)}")

        if not marker:
            break

        time.sleep(0.2)  # be nice to public node

    df = pd.DataFrame(rows)
    return df


def build_rich_list(df):
    print("📊 Building rich list...")

    df = df.sort_values("balance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def save_snapshot(df):
    Path("data").mkdir(exist_ok=True)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    filepath = f"data/rich_list_{date}.csv"
    df.to_csv(filepath, index=False)

    print(f"✅ Saved: {filepath}")


if __name__ == "__main__":
    df = fetch_all_accounts()
    df = build_rich_list(df)
    save_snapshot(df)
