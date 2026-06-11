"""
usda_fob_parser.py
==================
Download and parse the USDA FOB Shipping Point report (fvdfob.pdf).

Return a DataFrame with the following columns:
date, category, commodity, origin, region, package, size, price_low,
price_high, price_avg, unit, quality, condition, misc

Fixes included:
- Forced layout to 2 columns (fixed format of the USDA report)
- RE_PRICE_LINE captures packages that begin with a digit (50 lb, 7/10, 64s, etc.)
- _remove_holiday_notices removes holiday notices injected into the text
- is_valid_package with extended filters for pure numeric noise
- Ignores footer lines "National FOB Review … PAGE N"
"""

import re
import requests
import io
from datetime import datetime
import pandas as pd

try:
    import pdfplumber
except ImportError:
    raise ImportError("Install pdfplumber: pip install pdfplumber")


# ─────────────────────────────────────────────
# 1. DOWNLOAD AND EXTRACTION OF THE PDF
# ─────────────────────────────────────────────

FOB_URL = "https://www.ams.usda.gov/mnreports/fvdfob.pdf"


def download_pdf(url: str = FOB_URL) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def _page_text(page) -> str:
    """
    Extracts text from a page using a fixed 2-column layout.
    The USDA FOB report is always 2 columns.
    """
    mid_x = page.width / 2
    left  = page.crop((0,     0, mid_x,      page.height))
    right = page.crop((mid_x, 0, page.width, page.height))
    left_text  = left.extract_text(x_tolerance=2,  y_tolerance=2) or ""
    right_text = right.extract_text(x_tolerance=2, y_tolerance=2) or ""
    return left_text + "\n" + right_text


def extract_text(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extracts all text from the PDF in 2-column order.
    Returns (full_text, report_date_str).
    """
    full_text   = []
    report_date = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i == 0:
                # Page 1: extract date from header
                header = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                full_text.append(header)
                date_match = re.search(
                    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
                    r"(\w+ \d{1,2},\s*\d{4})",
                    header, re.IGNORECASE
                )
                if date_match:
                    report_date = date_match.group(1).strip()
            else:
                full_text.append(_page_text(page))

    return "\n".join(full_text), report_date


# ─────────────────────────────────────────────
# 2. TEXT PRE-PROCESSING
# ─────────────────────────────────────────────

# Section heading patterns
RE_SALES_FOB = re.compile(
    r"Sales\s+F\.O\.B\..*?(?:Dollars?|USD)",
    re.IGNORECASE | re.DOTALL
)

RE_CATEGORY = re.compile(
    r"^(TROPICAL AND MISC(?:ELLANEOUS)?|VEGETABLES|FRUIT|HERBS?|NUTS?|"
    r"MUSHROOMS?|ORGANIC|SPECIALTY)\b",
    re.IGNORECASE | re.MULTILINE
)

RE_COMMODITY_HEADER = re.compile(
    r"^([A-Z][A-Z &,\-/()]+)\s*$",
    re.MULTILINE
)

# Price line: packaging (may start with a digit), optional size, price
RE_PRICE_LINE = re.compile(
    r"^([\w\/][\w\s\/\-\.,()]*?)\s+"             # packaging (now captures initial digits)
    r"(\d+s?(?:\s+count)?(?:\s+size)?)?\s*"      # optional size
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)"  # price range
    r"(?:\s+(\d+(?:\.\d+)?))?",                  # optional average price
    re.IGNORECASE
)

RE_FOOTER = re.compile(
    r"National FOB Review.{0,60}PAGE\s+\d+",
    re.IGNORECASE
)

HOLIDAY_PATTERNS = [
    re.compile(r"(?:Office|Market|USDA).{0,80}(?:closed|holiday|observance).{0,120}",
               re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:Martin Luther King|Presidents|Memorial|Labor|Thanksgiving|"
               r"Christmas|New Year|Independence|Columbus|Veterans).{0,120}",
               re.IGNORECASE),
]


def _remove_holiday_notices(text: str) -> str:
    """Remove holiday notices that USDA injects into the price text."""
    for pat in HOLIDAY_PATTERNS:
        text = pat.sub("", text)
    return text


def _clean_text(text: str) -> str:
    text = RE_FOOTER.sub("", text)
    text = _remove_holiday_notices(text)
    # Normalize multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# ─────────────────────────────────────────────
# 3. PACKAGING VALIDATION
# ─────────────────────────────────────────────

NOISE_STARTS = {
    "page", "national", "report", "usda", "ams", "prices", "sales",
    "shipping", "point", "prepared", "f.o.b", "market", "news",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october",
    "november", "december",
}


def is_valid_package(text: str) -> bool:
    """Filters lines that are not actual packages."""
    t = text.strip().lower()
    if not t:
        return False
    # Pure number without units
    if re.fullmatch(r"[\d\.]+", t):
        return False
    # Very short (1-2 characters)
    if len(t) <= 2:
        return False
    # Starts with a noise word
    first_word = t.split()[0].rstrip(".,:")
    if first_word in NOISE_STARTS:
        return False
    return True


# ─────────────────────────────────────────────
# 4. CORE PARSING CODE
# ─────────────────────────────────────────────

KNOWN_CATEGORIES = {
    "FRUIT", "VEGETABLES", "TROPICAL AND MISCELLANEOUS",
    "TROPICAL AND MISC", "HERBS", "NUTS", "MUSHROOMS",
    "ORGANIC", "SPECIALTY",
}

KNOWN_ORIGINS = {
    "CALIFORNIA", "FLORIDA", "ARIZONA", "TEXAS", "WASHINGTON",
    "MEXICO", "CHILE", "PERU", "COLOMBIA", "ECUADOR", "GUATEMALA",
    "COSTA RICA", "HONDURAS", "MICHIGAN", "GEORGIA", "NEW JERSEY",
    "OREGON", "IDAHO", "COLORADO", "NEW YORK", "NORTH CAROLINA",
    "SOUTH CAROLINA", "VIRGINIA", "CANADA", "IMPORTED",
}

REGION_KEYWORDS = {
    "DISTRICT": "district",
    "REGION":   "region",
    "AREA":     "area",
    "TERMINAL": "terminal",
}


def _detect_origin_region(line: str) -> tuple[str, str]:
    """Attempts to detect origin and region from a header line."""
    line_up = line.upper()
    origin = ""
    region = ""
    for org in KNOWN_ORIGINS:
        if org in line_up:
            origin = org.title()
            break
    for kw in REGION_KEYWORDS:
        if kw in line_up:
            region = line.strip()
            break
    return origin, region


def parse_fob_text(full_text: str, report_date: str) -> list[dict]:
    """Parse the entire text of the report and return a list of records."""
    full_text = _clean_text(full_text)
    lines = full_text.splitlines()

    records    = []
    category   = ""
    commodity  = ""
    origin     = ""
    region     = ""

    # Attempt to parse date
    try:
        date_obj = datetime.strptime(report_date.replace(",", ""), "%B %d %Y")
    except Exception:
        date_obj = datetime.today()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # Detect category
        line_up = line.upper()
        for cat in KNOWN_CATEGORIES:
            if line_up.startswith(cat):
                category = cat.title()
                break

        # Detect origin / region
        org, reg = _detect_origin_region(line)
        if org:
            origin = org
        if reg:
            region = reg

        # Detect commodity: line all in capital letters, no numbers
        if (line.isupper() and len(line) > 3
                and not re.search(r"\d", line)
                and line_up not in KNOWN_CATEGORIES
                and not any(kw in line_up for kw in REGION_KEYWORDS)):
            commodity = line.title()
            continue

        # Detect price line
        m = RE_PRICE_LINE.match(line)
        if m:
            pkg_raw   = m.group(1).strip()
            size_raw  = (m.group(2) or "").strip()
            price_lo  = float(m.group(3))
            price_hi  = float(m.group(4))
            price_avg = float(m.group(5)) if m.group(5) else round((price_lo + price_hi) / 2, 2)

            if not is_valid_package(pkg_raw):
                continue

            records.append({
                "date":       date_obj,
                "category":   category,
                "commodity":  commodity,
                "origin":     origin,
                "region":     region,
                "package":    pkg_raw,
                "size":       size_raw,
                "price_low":  price_lo,
                "price_high": price_hi,
                "price_avg":  price_avg,
                "unit":       "USD",
                "quality":    "",
                "condition":  "",
                "misc":       "",
            })

    return records


# ─────────────────────────────────────────────
# 5. PRIMARY PUBLIC FUNCTION
# ─────────────────────────────────────────────

def get_fob_dataframe(url: str = FOB_URL) -> tuple[pd.DataFrame, str]:
    """
    Download the USDA FOB report and return (DataFrame, context_string).
    context_string is a text summary to pass to an AI agent.
    """
    pdf_bytes          = download_pdf(url)
    full_text, rpt_date = extract_text(pdf_bytes)
    records            = parse_fob_text(full_text, rpt_date)
    df                 = pd.DataFrame(records)

    if df.empty:
        return df, ""

    # Build context for the AI agent
    lines = [f"USDA FOB Report — {rpt_date}", f"Total entries: {len(df)}",
             f"Commodities: {df['commodity'].nunique()}",
             f"Origins: {', '.join(df['origin'].unique()[:10])}", ""]

    for commodity, grp in df.groupby("commodity"):
        lines.append(f"{commodity}:")
        for _, row in grp.iterrows():
            lines.append(
                f"  {row['package']} {row['size']} "
                f"${row['price_low']:.2f}-${row['price_high']:.2f} "
                f"(avg ${row['price_avg']:.2f}) — {row['origin']}"
            )
        lines.append("")

    context = "\n".join(lines)
    return df, context
