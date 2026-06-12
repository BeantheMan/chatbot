import requests
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from inventory_pricing import build_both_granular

import numpy as np
import pandas as pd

"""
usda_extractor.py
---------------
Extraction and processing of USDA MARS API prices.

Markets covered:
  - Los Angeles Terminal Market  (destination)
  - Fresno FOB                   (CA origin: berries, avocados, melons/grapes in season)
  - McAllen FOB                  (MX origin, TX border: pineapples, watermelon, papaya, limes)
  - Nogales FOB                  (MX origin, AZ border: watermelon, honeydew)
  - Phoenix FOB                  (AZ/CA origin: citrus, grapes)
  - Yakima FOB                   (WA origin: apples, pears)

Mapping of markets without their own USDA report:
  - Mexicali  -> approximated with Phoenix     (IX_FV110 / IX_FV120)
  - Tijuana   -> approximated with Nogales     (IX_FV110 / IX_FV120)
  - Hueneme   -> approximated with LA Terminal (HC_FV010 / HC_FV020)

Data sources:
  USDA MARS API    -> market prices (extracted in memory)
  Inventory.csv    -> inventory to enrich with prices

Main output:
  inventory_priced.csv   -> inventory enriched with USDA prices

Confirmed IDs:
  2306 -> Los Angeles Fruit        (Terminal)
  2307 -> Los Angeles Vegetables   (Terminal)
  2390 -> Fresno Fruit FOB         (FR_FV110)
  2391 -> Fresno Vegetables FOB    (FR_FV120)
  2402 -> McAllen/Nogales/Phoenix Fruit FOB       (IX_FV110) — multi-market in a single slug
  2403 -> McAllen/Nogales/Phoenix Vegetables FOB  (IX_FV120) — multi-market in a single slug
  2412 -> Yakima Fruit FOB         (YA_FV110)
  2413 -> Yakima Vegetables FOB    (YA_FV120)

"""

FILTER_FIELDS = [
    "commodity",
    "variety",
    "properties",
    "package",
    "item_size",
    "group",
    "organic",
    "grade",
    "quality",
    "origin",
    "market_location_name",
    "market_tone_comments",
    "report_end_date",
    "low_price",
    "high_price",
    "mostly_low_price",
    "mostly_high_price",
    "reporter_comment",
]

MARKETS = {
    "Los Angeles - Fruit (HC_FV010)": 2306,
    "Los Angeles - Vegetables (HC_FV020)": 2307,
    "Fresno - Fruit FOB (FR_FV110)": 2390,
    "Fresno - Vegetables FOB (FR_FV120)": 2391,
    "McAllen/Nogales/Phoenix - Fruit FOB (IX_FV110)": 2402,
    "McAllen/Nogales/Phoenix - Vegetables FOB (IX_FV120)": 2403,
    "Yakima - Fruit FOB (YA_FV110)": 2412,
    "Yakima - Vegetables FOB (YA_FV120)": 2413,
}

LA_TERMINAL = "Los Angeles Terminal Market"

FOB_MARKETS = [
    "Fresno (FR) FOB SC",
    "Fresno (FE) FOB SC",
    "Mcallen FOB SC",
    "Nogales FOB SC",
    "Phoenix FOB SC",
    "Yakima FOB SC",
]

ALL_MARKET_COLS = [LA_TERMINAL] + FOB_MARKETS


def get_api_key() -> str:
    """Read the USDA MARS API key from the environment; never hardcode it.

    Set it in a local .env file (see .env.example) or export USDA_MARS_API_KEY.
    """
    import streamlit as st
    api_key = st.secrets.get("USDA_API_KEY")
    if not api_key:
        raise SystemExit("USDA_API_KEY is not set!")
    return str(api_key)

def previous_weekday(date):
    date -= timedelta(days=1)

    while date.weekday() >= 5:  # Saturday or Sunday
        date -= timedelta(days=1)

    return date

def find_list_of_dicts(obj):
    """
    Extract the list of rows from the API payload.
    Handles two structures:
      - Flat list of dicts    (LA Terminal)
      - Dict with key results (Yakima, McAllen, Fresno)
    """
    if isinstance(obj, dict) and "results" in obj:
        candidate = obj["results"]
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return candidate

    best = None

    def walk(x):
        nonlocal best
        if isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                if best is None or len(x) > len(best):
                    best = x
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(obj)
    return best


def fetch_all_markets(markets: dict, n_days: int, api_key: str) -> pd.DataFrame:
    """
    Download n_days days back from end_day for each market.
    Returns a consolidated DataFrame with all the API columns.
    """
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    days: list[str] = [now.strftime("%m/%d/%Y")]

    for _ in range(n_days - 1):
        now = previous_weekday(now)
        days.append(now.strftime("%m/%d/%Y"))
    dfs = []
    for market_name, report_id in markets.items():
        url = f"https://marsapi.ams.usda.gov/services/v1.2/reports/{report_id}/report%20details"
        print(f"  Fetching {market_name} ({report_id})...", end=" ")
        count = 0

        for day in days:
            r = requests.get(
                url,
                params={"q": f"report_date={day}"},
                auth=(api_key, ""),
                timeout=60,
            )
            if r.status_code == 404:
                continue
            r.raise_for_status()

            data = r.json()
            payload = data[0] if isinstance(data, list) and len(data) == 1 else data
            rows = find_list_of_dicts(payload)

            if not rows:
                continue

            df_day = pd.json_normalize(rows)
            df_day["_market_name"] = market_name
            df_day["_report_id"] = report_id
            df_day["_report_date_requested"] = day
            dfs.append(df_day)
            count += len(df_day)

        print(f"{count} rows")

    if not dfs:
        raise SystemExit(
            "No rows returned from any market. Check the API key and that the USDA MARS API is reachable."
        )

    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal raw: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    return df_raw

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and compresses data to minimize token size
    """
    swap_pairs = [
        ["reporter_comment", "rep_cmt"],
        ["appearance", "appear"],
        ["condition", "cond"],
        ["quality", "appearance"],
        ["quality", "condition"],
        ["package", "pkg"],
        ["variety", "var"],
        ["group", "grp"]
    ]
    for target, source in swap_pairs:
        is_empty = (
            df[target].isna()
            | (df[target].astype(str).str.strip() == "")
            | (df[target] == "N/A")
        )
        df[target] = df[target].where(
            ~is_empty, df[source]
        )
    
    df["group"] = df["group"].where(
        pd.isna(df["category"]) | (df["category"] == "N/A"), df["category"]
    )

    district_clean = df["district"].astype("string").str.strip()
    origin_clean = df["origin"].astype("string").str.strip()

    district_filled = (
        district_clean.notna()
        & district_clean.ne("")
        & district_clean.str.upper().ne("N/A")
    )

    origin_filled = (
        origin_clean.notna()
        & origin_clean.ne("")
    )

    both_filled = district_filled & origin_filled
    only_district = district_filled & ~origin_filled

    df.loc[both_filled, "origin"] = (
        origin_clean.loc[both_filled]
        + ", "
        + district_clean.loc[both_filled]
    )

    df.loc[only_district, "origin"] = district_clean.loc[only_district]

    key_cols = [
        "commodity", 
        "variety",
        "properties",
        "package",
        "item_size",
        "category",
        "organic",
        "grade",
        "quality",
        "origin",
        "market_location_name"
    ]

    df = (
        df.sort_values("report_end_date", ascending=False)
        .drop_duplicates(subset=key_cols, keep="first")
    )

    df = df[FILTER_FIELDS].sort_values(
        by=["commodity", "variety", "report_end_date", "properties", "item_size"]
    )
    df.rename(columns={"group": "category"}, inplace=True)
    return df.replace(r"^\s*N/A\s*$", "", regex=True)

def get_usda_data(n_days: int = 2) -> tuple[pd.DataFrame, str]:
    """
    Extracts most recent USDA my market news data for n_days.

    Returns DataFrame and string of csv data tuple
    """
    df: pd.DataFrame = fetch_all_markets(markets=MARKETS, n_days=n_days, api_key=get_api_key())
    # df = pq.read_table("data.parquet").to_pandas()
    df["quality"] = ""

    # --- 2.1 Both granular (commodity + grade + package) ---
    both_granular = build_both_granular(df, LA_TERMINAL, FOB_MARKETS)

    # --- 2.2 Re-aggregate to commodity level for the analytical table ---
    # Average the both-granular prices, collapsing grade and package
    both_commodity = (
        both_granular.groupby(["report_date", "commodity"], sort=True)[
            ["price_LA", "price_FOB", "spread", "spread_pct"]
        ]
        .mean()
        .reset_index()
    )

    # fob_origin: most frequent origin per (date, commodity)
    fob_origin_mode = (
        both_granular.groupby(["report_date", "commodity"])["fob_origin"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
    )
    both_commodity = both_commodity.merge(
        fob_origin_mode, on=["report_date", "commodity"], how="left"
    )
    df = clean_data(df)
    return df, df.to_csv(index=False, na_rep="")

if __name__ == "__main__":
    df, csv_string = get_usda_data(2)
    with open("output.csv", "w", encoding="utf-8", newline="") as f:
        f.write(csv_string)