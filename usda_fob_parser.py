"""
usda_fob_parser.py
==================
Descarga y parsea el reporte USDA FOB Shipping Point (fvdfob.pdf).
Devuelve un DataFrame con columnas:
    date, category, commodity, origin, region, package, size, price_low,
    price_high, price_avg, unit, quality, condition, misc

Fixes incluidos:
  - Layout forzado a 2 columnas (formato fijo del reporte USDA)
  - RE_PRICE_LINE captura empaques que empiezan con dígito (50 lb, 7/10, 64s, etc.)
  - _remove_holiday_notices elimina avisos de días feriados inyectados en el texto
  - is_valid_package con filtros ampliados para ruido numérico puro
  - Ignora líneas de pie de página "National FOB Review … PAGE N"
"""

import re
import requests
import io
from datetime import datetime
import pandas as pd

try:
    import pdfplumber
except ImportError:
    raise ImportError("Instala pdfplumber: pip install pdfplumber")


# ─────────────────────────────────────────────
# 1. DESCARGA Y EXTRACCIÓN DEL PDF
# ─────────────────────────────────────────────

FOB_URL = "https://www.ams.usda.gov/mnreports/fvdfob.pdf"


def download_pdf(url: str = FOB_URL) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def _page_text(page) -> str:
    """
    Extrae texto de una página usando layout de 2 columnas fijo.
    El reporte USDA FOB siempre es 2 columnas.
    """
    mid_x = page.width / 2
    left  = page.crop((0,     0, mid_x,      page.height))
    right = page.crop((mid_x, 0, page.width, page.height))
    left_text  = left.extract_text(x_tolerance=2,  y_tolerance=2) or ""
    right_text = right.extract_text(x_tolerance=2, y_tolerance=2) or ""
    return left_text + "\n" + right_text


def extract_text(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extrae todo el texto del PDF en orden de 2 columnas.
    Retorna (full_text, report_date_str).
    """
    full_text   = []
    report_date = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i == 0:
                # Página 1: extraer fecha del encabezado
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
# 2. PRE-PROCESAMIENTO DEL TEXTO
# ─────────────────────────────────────────────

# Patrones de encabezados de sección
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

# Línea de precio: empaque (puede empezar con dígito), tamaño opcional, precio
RE_PRICE_LINE = re.compile(
    r"^([\w\/][\w\s\/\-\.,()]*?)\s+"           # empaque (ahora captura dígitos iniciales)
    r"(\d+s?(?:\s+count)?(?:\s+size)?)?\s*"    # tamaño opcional
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)"  # rango de precio
    r"(?:\s+(\d+(?:\.\d+)?))?",                # precio promedio opcional
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
    """Elimina avisos de días feriados que USDA inyecta en el texto de precios."""
    for pat in HOLIDAY_PATTERNS:
        text = pat.sub("", text)
    return text


def _clean_text(text: str) -> str:
    text = RE_FOOTER.sub("", text)
    text = _remove_holiday_notices(text)
    # Normalizar espacios múltiples
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# ─────────────────────────────────────────────
# 3. VALIDACIÓN DE EMPAQUES
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
    """Filtra líneas que no son realmente empaques."""
    t = text.strip().lower()
    if not t:
        return False
    # Número puro sin unidad
    if re.fullmatch(r"[\d\.]+", t):
        return False
    # Muy corto (1-2 chars)
    if len(t) <= 2:
        return False
    # Empieza con palabra de ruido
    first_word = t.split()[0].rstrip(".,:")
    if first_word in NOISE_STARTS:
        return False
    return True


# ─────────────────────────────────────────────
# 4. PARSEO PRINCIPAL
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
    """Intenta detectar origen y región de una línea de encabezado."""
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
    """Parsea el texto completo del reporte y retorna lista de registros."""
    full_text = _clean_text(full_text)
    lines = full_text.splitlines()

    records    = []
    category   = ""
    commodity  = ""
    origin     = ""
    region     = ""

    # Intentar parsear fecha
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

        # Detectar categoría
        line_up = line.upper()
        for cat in KNOWN_CATEGORIES:
            if line_up.startswith(cat):
                category = cat.title()
                break

        # Detectar origen / región
        org, reg = _detect_origin_region(line)
        if org:
            origin = org
        if reg:
            region = reg

        # Detectar commodity: línea toda en mayúsculas, sin números
        if (line.isupper() and len(line) > 3
                and not re.search(r"\d", line)
                and line_up not in KNOWN_CATEGORIES
                and not any(kw in line_up for kw in REGION_KEYWORDS)):
            commodity = line.title()
            continue

        # Detectar línea de precio
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
# 5. FUNCIÓN PÚBLICA PRINCIPAL
# ─────────────────────────────────────────────

def get_fob_dataframe(url: str = FOB_URL) -> tuple[pd.DataFrame, str]:
    """
    Descarga el reporte USDA FOB y retorna (DataFrame, context_string).
    context_string es un resumen de texto para pasar a un agente AI.
    """
    pdf_bytes          = download_pdf(url)
    full_text, rpt_date = extract_text(pdf_bytes)
    records            = parse_fob_text(full_text, rpt_date)
    df                 = pd.DataFrame(records)

    if df.empty:
        return df, ""

    # Construir contexto para el agente AI
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
