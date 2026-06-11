"""
inventory_pricing.py
--------------------
Matches inventory files against USDA prices already calculated in memory.
This module does not call the API or read parquet files directly.

Exports three functions used by the main scripts:

    build_both_granular(df_raw, la_terminal, fob_markets)
        Builds the (commodity, variety, grade, package) by market pivot from
        df_raw, which comes directly from the API.

    price_inventory(both, inventory_path, output_path)
        Matches Inventory.csv against USDA prices using fuzzy matching.

    price_missing(both, missing_path, output_path)
        Matches missing.csv against USDA prices using fuzzy matching.

Match: TF-IDF cosine similarity on char n-grams (2-4) using the query
       commodity + variety + grade + package.

Output flags:
    FULL_MATCH       score >= MIN_SCORE  -> prices assigned
    NO_PRICE_MATCH   score <  MIN_SCORE  -> prices are NaN and need review
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Minimum score required to accept a match.
# Matching on 4 fields (commodity+variety+grade+package) calls for a stricter cutoff.
MIN_SCORE     = 0.45
FLAG_FULL     = "FULL_MATCH"
FLAG_NO_MATCH = "NO_PRICE_MATCH"


# =========================================================
# build_both_granular
# =========================================================
def build_both_granular(
    df_raw: pd.DataFrame,
    la_terminal: str,
    fob_markets: list,
) -> pd.DataFrame:
    """
    Build a granular pivot (commodity, variety, grade, package) with LA and FOB
    prices selected independently; they do not need to come from the same day.

    Parameters
    ----------
    df_raw      : Raw DataFrame from fetch_all_markets()
    la_terminal : Exact terminal market string
    fob_markets : List of FOB market strings

    Returns
    -------
    DataFrame with columns:
        commodity, variety, grade, package, report_date,
        price_LA, price_FOB, fob_origin, spread, spread_pct
    """
    df = df_raw.copy()

    # Prices
    df["low_price"]  = pd.to_numeric(df["low_price"],  errors="coerce")
    df["high_price"] = pd.to_numeric(df["high_price"], errors="coerce")
    df["mid_price"]  = 0.5 * (df["low_price"].fillna(df["high_price"]) +
                               df["high_price"].fillna(df["low_price"]))

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["commodity"]   = df["commodity"].astype("string").str.strip().str.title()
    df["variety"]     = df["variety"].astype("string").str.strip().fillna("")
    df["grade"]       = df["grade"].astype("string").str.strip().fillna("")
    df["package"]     = df["package"].astype("string").str.strip().fillna("")

    df = df.dropna(subset=["report_date", "commodity", "mid_price",
                            "market_location_name"])

    KEY_COLS = ["commodity", "variety", "grade", "package"]

    # Group by (date, commodity, variety, grade, package, market)
    df_agg = (
        df.groupby(
            ["report_date"] + KEY_COLS + ["market_location_name"],
            observed=True, sort=True,
        )["mid_price"]
        .mean()
        .reset_index()
    )

    # LA price: most recent row per key
    df_la = (
        df_agg[df_agg["market_location_name"] == la_terminal]
        .sort_values("report_date", ascending=False)
        .groupby(KEY_COLS, sort=False)
        .first()
        .reset_index()
        .rename(columns={"mid_price": "price_LA", "report_date": "date_LA"})
        [KEY_COLS + ["date_LA", "price_LA"]]
    )

    # FOB price: most recent row, then lowest price across all FOB markets
    df_fob_all = df_agg[df_agg["market_location_name"].isin(fob_markets)].copy()

    if len(df_fob_all) > 0:
        df_fob_latest = (
            df_fob_all
            .sort_values("report_date", ascending=False)
            .groupby(KEY_COLS + ["market_location_name"], sort=False)
            .first()
            .reset_index()
        )
        idx_min = (
            df_fob_latest
            .groupby(KEY_COLS)["mid_price"]
            .idxmin(skipna=True)
            .dropna()
        )
        df_fob_best = (
            df_fob_latest.loc[idx_min]
            .rename(columns={
                "mid_price":            "price_FOB",
                "report_date":          "date_FOB",
                "market_location_name": "fob_origin",
            })
            [KEY_COLS + ["date_FOB", "price_FOB", "fob_origin"]]
            .reset_index(drop=True)
        )
    else:
        df_fob_best = pd.DataFrame(
            columns=KEY_COLS + ["date_FOB", "price_FOB", "fob_origin"]
        )

    # Outer join: include rows with only LA, only FOB, or both
    both = df_la.merge(df_fob_best, on=KEY_COLS, how="outer")

    both["report_date"] = both[["date_LA", "date_FOB"]].max(axis=1)

    # Force numeric values before calculating spreads
    both["price_LA"]  = pd.to_numeric(both["price_LA"],  errors="coerce")
    both["price_FOB"] = pd.to_numeric(both["price_FOB"], errors="coerce")

    both["spread"]     = both["price_LA"] - both["price_FOB"]
    both["spread_pct"] = (
        both["spread"] / both["price_FOB"].replace(0, np.nan) * 100
    )

    both = both.reset_index(drop=True)
    return both


# =========================================================
# Inventory COMMODITY normalization
# =========================================================
COMMODITY_MAP = {
    "ANISE":        "Anise",         "APPLE":       "Apples",
    "APRICOT":      "Apricots",      "ARTICHOKE":   "Artichokes",
    "ASPARAGUS":    "Asparagus",     "AVOCADO":     "Avocados",
    "BANANA":       "Bananas",       "BEAN":        "Beans",
    "BEET":         "Beets",         "BELLPEPPER":  "Bell Peppers",
    "BLACKBERRY":   "Blackberries",  "BLUEBERRY":   "Blueberries",
    "BOK CHOY":     "Bok Choy",      "BROCCOLI":    "Broccoli",
    "BRU SPROUT":   "Brussels Sprouts", "CABBAGE":  "Cabbage",
    "CANTALOUPE":   "Cantaloupe",    "CARROT":      "Carrots",
    "CAULIFLOWER":  "Cauliflower",   "CELERY":      "Celery",
    "CHERRY":       "Cherries",      "CILANTRO":    "Cilantro",
    "COCONUT":      "Coconuts",      "CORN":        "Corn",
    "CUCUMBER":     "Cucumbers",     "EGGPLANT":    "Eggplant",
    "FIG":          "Figs",          "GARLIC":      "Garlic",
    "GINGER":       "Ginger",        "GRAPE-BLACK": "Grapes",
    "GRAPE-GREEN":  "Grapes",        "GRAPE-RED":   "Grapes",
    "GRAPEFRUIT":   "Grapefruit",    "GREEN ONION": "Green Onions",
    "HARD SQUASH":  "Squash",        "HERB":        "Herbs",
    "HONEYDEW":     "Honeydew",      "KALE":        "Kale",
    "KIWI":         "Kiwi",          "LEAF":        "Leafy Greens",
    "LEMON":        "Lemons",        "LETTUCE":     "Lettuce",
    "LIME":         "Limes",         "MANGO":       "Mangoes",
    "MELON":        "Melons",        "MUSHROOM":    "Mushrooms",
    "NASHI":        "Pears",         "NECTARINE":   "Nectarines",
    "OKRA":         "Okra",          "ONION":       "Onions",
    "ORANGE":       "Oranges",       "PAPAYA":      "Papayas",
    "PARSLEY":      "Parsley",       "PARSNIP":     "Parsnips",
    "PEA":          "Peas",          "PEACH":       "Peaches",
    "PEAR":         "Pears",         "PEPPER":      "Peppers",
    "PEPPER-R":     "Peppers",       "PINEAPPLE":   "Pineapples",
    "PLANTAIN":     "Plantains",     "PLUM":        "Plums",
    "POMEGRANATE":  "Pomegranates",  "POTATO":      "Potatoes",
    "PUMPKIN":      "Pumpkins",      "RADISH":      "Radishes",
    "RASPBERRY":    "Raspberries",   "RED ONION":   "Onions",
    "ROMAINE":      "Romaine Lettuce", "SPINACH":   "Spinach",
    "SPRING MIX":   "Spring Mix",    "SQUASH":      "Squash",
    "STRAWBERRY":   "Strawberries",  "SWEET ONION": "Onions",
    "TANGERINE":    "Tangerines",    "TOMATO":      "Tomatoes",
    "TURNIP":       "Turnips",       "WATERMELON":  "Watermelons",
    "WHITE ONION":  "Onions",        "YAM":         "Yams",
    "YELLOW ONION": "Onions",        "YUCCA":       "Yuca",
}

def _normalize_commodity(raw: str) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw).strip().upper()
    if s.startswith("ORG"):
        base   = s[3:]
        mapped = COMMODITY_MAP.get(base, base.title())
        return f"Organic {mapped}"
    return COMMODITY_MAP.get(s, s.title())


# =========================================================
# Parse variety, grade, and package from PRODUCT DESCRIPTION
# =========================================================
VARIETY_TOKENS = {
    "RED":        "Red Delicious",    "REDDEL":      "Red Delicious",
    "GOLD":       "Golden Delicious", "GOLDEL":      "Golden Delicious",
    "GAL":        "Gala",             "GALA":        "Gala",
    "FUJ":        "Fuji",             "FUJI":        "Fuji",
    "GRA":        "Granny Smith",     "GRANNY":      "Granny Smith",
    "BRAEBURN":   "Braeburn",         "BRAE":        "Braeburn",
    "HONEYCRISP": "Honeycrisp",       "HC":          "Honeycrisp",
    "COSMICCRISP":"Cosmic Crisp",     "COSMIC":      "Cosmic Crisp",
    "ENVY":       "Envy",             "JAZZ":        "Jazz",
    "AMBROSIA":   "Ambrosia",         "KANZI":       "Kanzi",
    "OPAL":       "Opal",             "EMPIRE":      "Empire",
    "MCINTOSH":   "Mcintosh",         "PINKLADY":    "Pink Lady",
    "BARTLETT":   "Bartlett",         "BART":        "Bartlett",
    "BOSC":       "Bosc",             "ANJOU":       "Anjou",
    "COMICE":     "Comice",           "FORELLE":     "Forelle",
    "SECKEL":     "Seckel",
    "REDGLOBE":   "Red Globe",        "CRIMSON":     "Crimson Seedless",
    "FLAME":      "Flame Seedless",   "THOMPSON":    "Thompson Seedless",
    "COTTON":     "Cotton Candy",     "MUSCAT":      "Muscat",
    "NAVEL":      "Navel",            "VALENCIA":    "Valencia",
    "BLOOD":      "Blood Orange",     "CARA":        "Cara Cara",
    "MEYER":      "Meyer",            "EUREKA":      "Eureka",
    "LISBON":     "Lisbon",           "PERSIAN":     "Persian",
    "BING":       "Bing",             "RAINIER":     "Rainier",
    "LAPIN":      "Lapins",           "BROOKS":      "Brooks",
    "FREESTONE":  "Freestone",        "DONUT":       "Donut",
    "HASS":       "Hass",             "REED":        "Reed",
    "ROMA":       "Roma",             "BEEFSTEAK":   "Beefsteak",
    "HEIRLOOM":   "Heirloom",         "CLUSTER":     "Cluster",
    "ICEBERG":    "Iceberg",          "ROMAINE":     "Romaine",
    "BUTTER":     "Butterhead",
    "RUSSET":     "Russet",           "YUKONGOLD":   "Yukon Gold",
    "REDSKIN":    "Red Skin",         "FINGERLING":  "Fingerling",
    "VIDALIA":    "Vidalia",          "WALLA":       "Walla Walla",
    "HAMI":       "Hami",             "CANARY":      "Canary",
    "CRENSHAW":   "Crenshaw",
    "JALAPENO":   "Jalapeno",         "SERRANO":     "Serrano",
    "ANAHEIM":    "Anaheim",          "HABANERO":    "Habanero",
    "POBLANO":    "Poblano",          "SHISHITO":    "Shishito",
}

GRADE_TOKENS = {
    "XFCY":    "Extra Fancy",   "XF":       "Extra Fancy",
    "FCY":     "Fancy",         "FANCY":    "Fancy",
    "PREM":    "Premium",       "PREMIUM":  "Premium",
    "CHOICE":  "Choice",
    "NO1":     "No. 1",         "NO.1":     "No. 1",
    "US1":     "U.S. No. 1",
    "STD":     "Standard",      "STANDARD": "Standard",
    "CLASS1":  "Class I",       "CLASSI":   "Class I",
    "COMBO":   "Combination",   "1ST":      "First Grade",
}

PACKAGE_PATTERNS = [
    (r"(\d+)\s*#",              lambda m: f"{m.group(1)} lb"),
    (r"(\d+)\s*LBS?",           lambda m: f"{m.group(1)} lb"),
    (r"(\d+)-(\d+)\s*CT",       lambda m: f"{m.group(1)}-{m.group(2)} count"),
    (r"(\d+)\s*CT",             lambda m: f"{m.group(1)} count"),
    (r"\bBU\b",                 lambda m: "bushel"),
    (r"\bBSKT\b",               lambda m: "basket"),
    (r"\bFLT\b|\bFLAT\b",       lambda m: "flat"),
    (r"\bCRTN?\b|\bCARTON\b",   lambda m: "carton"),
    (r"\bBAG\b",                lambda m: "bag"),
    (r"\bSACK\b",               lambda m: "sack"),
    (r"\bBIN\b",                lambda m: "bin"),
    (r"\bTRAY\b",               lambda m: "tray"),
    (r"\bLUG\b",                lambda m: "lug"),
    (r"\bMESH\b",               lambda m: "mesh bag"),
    (r"\bFILM\b",               lambda m: "film bag"),
    (r"\bBUNCH\b",              lambda m: "bunch"),
    (r"\bPLT\b|\bPALLET\b",     lambda m: "pallet"),
    (r"\bPKG\b",                lambda m: "package"),
]

def _parse_variety(desc: str) -> str:
    if pd.isna(desc):
        return ""
    tokens = re.split(r"[\s,/\-]+", str(desc).upper())
    found = [VARIETY_TOKENS[t] for t in tokens if t in VARIETY_TOKENS]
    return " ".join(found)

def _parse_grade(desc: str) -> str:
    if pd.isna(desc):
        return ""
    tokens = re.split(r"[\s,/\-]+", str(desc).upper())
    return " ".join(GRADE_TOKENS[t] for t in tokens if t in GRADE_TOKENS)

def _parse_package(desc: str) -> str:
    if pd.isna(desc):
        return ""
    s = str(desc).upper()
    found = []
    for pattern, formatter in PACKAGE_PATTERNS:
        for m in re.finditer(pattern, s):
            found.append(formatter(m))
    return " ".join(found)

def _build_queries(df_inv: pd.DataFrame) -> pd.DataFrame:
    df = df_inv.copy()
    df["_commodity_norm"] = df["COMMODITY"].apply(_normalize_commodity)
    df["_variety_parsed"] = df["PRODUCT DESCRIPTION"].apply(_parse_variety)
    df["_grade_parsed"]   = df["PRODUCT DESCRIPTION"].apply(_parse_grade)
    df["_package_parsed"] = df["PRODUCT DESCRIPTION"].apply(_parse_package)
    df["usda_query"] = (
        df["_commodity_norm"] + " " +
        df["_variety_parsed"] + " " +
        df["_grade_parsed"]   + " " +
        df["_package_parsed"]
    ).str.lower().str.strip()
    return df


# =========================================================
# Price vocabulary: search_key by combination
# =========================================================
def _build_price_vocab(both: pd.DataFrame) -> pd.DataFrame:
    latest = both.rename(columns={"report_date": "price_date"}).copy()

    latest["search_key"] = (
        latest["commodity"].fillna("") + " " +
        latest["variety"].fillna("")   + " " +
        latest["grade"].fillna("")     + " " +
        latest["package"].fillna("")
    ).str.lower().str.strip()

    latest = latest[latest["search_key"].str.len() > 0].reset_index(drop=True)

    keep = ["commodity", "variety", "grade", "package", "search_key",
            "price_date", "price_LA", "price_FOB", "fob_origin",
            "spread", "spread_pct"]
    keep = [c for c in keep if c in latest.columns]

    n_full = latest.dropna(subset=["price_LA", "price_FOB"]).shape[0]
    print(f"  USDA vocabulary: {len(latest):,} combinations "
          f"({n_full} with LA+FOB, {len(latest)-n_full} with only one side)")

    if len(latest) == 0:
        raise ValueError(
            "Price vocabulary is empty. Check that 'la_terminal' and "
            "'fob_markets' match the raw market_location_name values."
        )
    return latest[keep]


# =========================================================
# Fuzzy join: TF-IDF cosine similarity on the full query.
# Only accepts matches with score >= MIN_SCORE.
# There is no commodity-only fallback; this avoids false intersections.
# =========================================================
def _fuzzy_join(queries: pd.Series, df_prices: pd.DataFrame) -> pd.DataFrame:
    parquet_keys = df_prices["search_key"].tolist()
    inventory_qs = queries.tolist()

    print(f"  USDA vocabulary:   {len(parquet_keys):,} keys")
    print(f"  Queries to match:  {len(inventory_qs):,} rows")

    print("  Fitting TF-IDF (char n-grams 2-4)...")
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
    )
    vec.fit(parquet_keys + inventory_qs)
    mat_p = vec.transform(parquet_keys)
    mat_i = vec.transform(inventory_qs)

    print("  Calculating cosine similarity...")
    price_cols = ["price_date", "price_LA", "price_FOB",
                  "fob_origin", "spread", "spread_pct"]
    results = []
    for i in range(0, len(inventory_qs), 200):
        sims = cosine_similarity(mat_i[i:i + 200], mat_p)
        for j, sim_row in enumerate(sims):
            inv_idx    = i + j
            best_idx   = int(np.argmax(sim_row))
            best_score = float(sim_row[best_idx])

            if best_score >= MIN_SCORE:
                flag      = FLAG_FULL
                match_row = df_prices.iloc[best_idx]
            else:
                flag      = FLAG_NO_MATCH
                match_row = None

            row = {
                "inv_idx":     inv_idx,
                "match_score": round(best_score, 4),
                "usda_query":  inventory_qs[inv_idx],
                "usda_match":  df_prices.iloc[best_idx]["search_key"],
                "price_flag":  flag,
            }
            for col in price_cols:
                row[col] = match_row[col] if match_row is not None else np.nan
            results.append(row)

    return pd.DataFrame(results)


# =========================================================
# _assemble_output - shared helper
# =========================================================
def _print_summary(df_out: pd.DataFrame, output_path: str) -> None:
    counts = df_out["price_flag"].value_counts()
    total  = len(df_out)
    print(f"\n  Output saved to: {output_path}")
    for flag, n in counts.items():
        print(f"    {flag:<22} {n:>4}  ({100*n/total:.0f}%)")
    print(f"    {'TOTAL':<22} {total:>4}")

    for flag in [FLAG_FULL, FLAG_NO_MATCH]:
        subset = df_out[df_out["price_flag"] == flag].head(2)
        if len(subset):
            show = [c for c in ["PRODUCT", "agpluscode", "Commodity Name",
                                 "COMMODITY", "usda_query", "usda_match",
                                 "match_score", "price_LA", "price_FOB", "price_flag"]
                    if c in df_out.columns]
            print(f"\n  Example {flag}:")
            print(subset[show].to_string(index=False))


# =========================================================
# price_inventory — Inventory.csv
# =========================================================
def price_inventory(
    both: pd.DataFrame,
    inventory_path: str = "Inventory.csv",
    output_path: str    = "inventory_priced.csv",
) -> pd.DataFrame:
    """
    Match Inventory.csv against USDA prices.
    The query is built from COMMODITY + PRODUCT DESCRIPTION (variety, grade, package).
    """
    print("\n--- USDA price vocabulary ---")
    df_prices = _build_price_vocab(both)

    print("\n--- Preparing inventory queries ---")
    df_inv = pd.read_csv(inventory_path)
    df_inv = _build_queries(df_inv)
    print(f"  {len(df_inv):,} rows loaded")
    print("\n  Query examples:")
    print(
        df_inv[["PRODUCT", "COMMODITY", "_commodity_norm", "_variety_parsed",
                "_grade_parsed", "_package_parsed", "usda_query"]]
        .head(5).to_string(index=False)
    )

    print("\n--- Fuzzy join ---")
    df_matches = _fuzzy_join(df_inv["usda_query"], df_prices)

    print("\n--- Assembling output ---")
    # Drop internal parsing columns before saving
    internal_cols = ["_commodity_norm", "_variety_parsed", "_grade_parsed",
                     "_package_parsed", "usda_query"]
    df_inv_clean = df_inv.drop(columns=internal_cols, errors="ignore")

    df_out = df_inv_clean.reset_index(drop=True).join(df_matches.set_index("inv_idx"))

    # Move match columns next to prices for easier reading
    match_cols  = ["usda_query", "usda_match", "match_score", "price_flag"]
    price_cols_ = ["price_date", "price_LA", "price_FOB", "fob_origin",
                   "spread", "spread_pct"]
    front = [c for c in df_inv_clean.columns]
    back  = [c for c in df_out.columns if c not in front + match_cols + price_cols_]
    df_out = df_out[front + match_cols + price_cols_ + back]

    df_out.to_csv(output_path, index=False)
    _print_summary(df_out, output_path)
    return df_out


# =========================================================
# price_missing — missing.csv
# =========================================================
def price_missing(
    both: pd.DataFrame,
    missing_path: str = "missing.csv",
    output_path: str  = "missing_priced.csv",
) -> pd.DataFrame:
    """
    Match missing.csv against USDA prices.
    Commodity Name already contains commodity+variety+grade, so it goes
    directly into the query.
    Expected columns: Commodity Name, Package, Item Size, agpluscode
    """
    print("\n--- USDA price vocabulary ---")
    df_prices = _build_price_vocab(both)

    print("\n--- Preparing missing-item queries ---")
    df = pd.read_csv(missing_path)

    # Commodity Name already has commodity+variety+grade; add Package and Item Size.
    df["usda_query"] = (
        df["Commodity Name"].fillna("") + " " +
        df["Package"].fillna("")        + " " +
        df["Item Size"].fillna("").astype(str)
    ).str.lower().str.strip()

    print(f"  {len(df):,} rows loaded")
    print("\n  Query examples:")
    print(
        df[["agpluscode", "Commodity Name", "Package", "Item Size", "usda_query"]]
        .head(5).to_string(index=False)
    )

    print("\n--- Fuzzy join ---")
    df_matches = _fuzzy_join(df["usda_query"], df_prices)

    print("\n--- Assembling output ---")
    df_out = df.reset_index(drop=True).join(df_matches.set_index("inv_idx"))

    # agpluscode first, then match columns, then prices
    front       = ["agpluscode", "Commodity Name", "Package", "Item Size"]
    match_cols  = ["usda_query", "usda_match", "match_score", "price_flag"]
    price_cols_ = ["price_date", "price_LA", "price_FOB", "fob_origin",
                   "spread", "spread_pct"]
    rest  = [c for c in df_out.columns if c not in front + match_cols + price_cols_]
    df_out = df_out[front + match_cols + price_cols_ + rest]

    df_out.to_csv(output_path, index=False)
    _print_summary(df_out, output_path)
    return df_out
