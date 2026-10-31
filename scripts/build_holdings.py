"""Day 2: turn the issuer holdings downloads into one clean table.

Reads every file listed in data/reference/holdings_sources.csv (Vanguard,
iShares, Invesco or HSBC layout), keeps the top N holdings per ETF, puts
everything else into one residual row, and writes
data/reference/etf_holdings.csv with standard columns.

Run from the repo root:
    python scripts/build_holdings.py
"""

import csv
import re
from pathlib import Path

import pandas as pd

from fragility.config import (
    ETF_HOLDINGS_CSV,
    ETF_UNIVERSE_CSV,
    HOLDINGS_SOURCES_CSV,
    NAME_MAP_CSV,
    RAW_DIR,
    RESIDUAL_LABEL,
    TOP_N_HOLDINGS,
)
from fragility.universe import missing_tickers, validate_holdings

OUTPUT_COLS = ["etf_ticker", "holding_id", "holding_name", "holding_ticker",
               "weight", "sector", "country"]

# ---------------------------------------------------------------------------
# 1. Column names: each issuer calls the same thing something different.
#    Every variant on the left maps to our standard name on the right.
# ---------------------------------------------------------------------------
COLUMN_SYNONYMS = {
    "name": ["holding name", "name", "security name", "security", "holding",
             "description", "constituent", "issuer name", "constituent name",
             "company", "company name", "instrument name", "security description",
             "holding description", "asset name", "issuer", "securityname",
             "full name", "fullname", "security_name"],
    "ticker": ["ticker", "ticker symbol", "symbol", "issuer ticker", "bloomberg ticker",
               "bloomberg code", "local ticker"],
    "weight": ["% of market value", "weight (%)", "weight", "% of fund", "% weight",
               "weighting", "% of net assets", "percent of fund", "% of nav",
               "weight %", "market value %", "% of assets", "weight(%)", "weights",
               "% of net asset value", "% of total net assets", "% of portfolio",
               "portfolio %", "portfolio weight", "fund weight", "% of fund nav",
               "weight in fund", "% weighting", "allocation", "% allocation"],
    "sector": ["sector", "gics sector", "industry sector", "industry"],
    "country": ["region", "location", "country", "country of risk",
                "location of risk", "domicile"],
    "asset_class": ["asset class", "asset type", "security type"],
}

# ---------------------------------------------------------------------------
# 2. Sectors: Vanguard uses ICB names, iShares uses GICS names.
#    We translate everything into the 11 GICS sectors.
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "technology": "Information Technology",
    "information technology": "Information Technology",
    "telecommunications": "Communication Services",
    "communication": "Communication Services",
    "communication services": "Communication Services",
    "basic materials": "Materials",
    "materials": "Materials",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "financials": "Financials",
    "energy": "Energy",
    "industrials": "Industrials",
    "real estate": "Real Estate",
    "utilities": "Utilities",
}

# ---------------------------------------------------------------------------
# 3. Countries: Vanguard uses 2-letter codes, iShares uses full names.
# ---------------------------------------------------------------------------
COUNTRY_CODES = {
    "US": "United States", "GB": "United Kingdom", "JP": "Japan", "CN": "China",
    "HK": "Hong Kong", "TW": "Taiwan", "KR": "South Korea", "IN": "India",
    "CA": "Canada", "FR": "France", "DE": "Germany", "CH": "Switzerland",
    "NL": "Netherlands", "AU": "Australia", "SE": "Sweden", "DK": "Denmark",
    "IT": "Italy", "ES": "Spain", "BR": "Brazil", "SA": "Saudi Arabia",
    "ZA": "South Africa", "MX": "Mexico", "IE": "Ireland", "BE": "Belgium",
    "FI": "Finland", "NO": "Norway", "SG": "Singapore", "IL": "Israel",
    "AE": "United Arab Emirates", "TH": "Thailand", "MY": "Malaysia",
    "ID": "Indonesia", "PL": "Poland", "AT": "Austria", "NZ": "New Zealand",
    "PT": "Portugal", "CL": "Chile", "TR": "Turkey", "PH": "Philippines",
    "QA": "Qatar", "KW": "Kuwait", "GR": "Greece", "HU": "Hungary",
    "CZ": "Czech Republic", "CO": "Colombia", "PE": "Peru", "EG": "Egypt",
    "LU": "Luxembourg", "JE": "Jersey", "BM": "Bermuda", "KY": "Cayman Islands",
    "RU": "Russia", "AR": "Argentina", "VN": "Vietnam", "PK": "Pakistan",
    "IS": "Iceland", "MO": "Macau", "UY": "Uruguay", "CY": "Cyprus",
    "RO": "Romania", "KZ": "Kazakhstan", "NG": "Nigeria", "MA": "Morocco",
    "BH": "Bahrain", "OM": "Oman", "BG": "Bulgaria", "HR": "Croatia",
    "SI": "Slovenia", "SK": "Slovakia",
}
COUNTRY_NAME_FIXES = {
    "korea (south)": "South Korea", "korea": "South Korea",
    "republic of korea": "South Korea", "uk": "United Kingdom",
    "usa": "United States", "united states of america": "United States",
}

# Words stripped from the END of company names so "NVIDIA Corp" and
# "NVIDIA CORP" and "Taiwan Semiconductor Manufacturing Co Ltd" match.
NAME_SUFFIXES = {
    "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC",
    "SA", "AG", "NV", "SE", "ASA", "AB", "SPA", "LTDA", "ORD", "REG", "ADR",
    "CLASS", "CL", "A", "B", "C", "THE", "GROUP", "HOLDING",
}


# ---------------------------------------------------------------------------
# Reading raw files
# ---------------------------------------------------------------------------
def rows_to_frame(rows: list) -> pd.DataFrame:
    """List of rows (possibly different lengths) -> DataFrame of text."""
    width = max((len(r) for r in rows), default=0)
    return pd.DataFrame([list(r) + [""] * (width - len(r)) for r in rows], dtype=str)


def read_spreadsheet_xml(path: Path) -> list:
    """Read an 'XML Spreadsheet 2003' file (what iShares calls .xls). One frame per sheet."""
    import xml.etree.ElementTree as ET

    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    index_attr = "{urn:schemas-microsoft-com:office:spreadsheet}Index"
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    root = ET.fromstring(text)

    sheets = []
    for sheet in root.findall("ss:Worksheet", ns):
        rows = []
        for row in sheet.findall("ss:Table/ss:Row", ns):
            values = []
            for cell in row.findall("ss:Cell", ns):
                if cell.get(index_attr):  # skipped empty cells before this one
                    target = int(cell.get(index_attr)) - 1
                    values += [""] * (target - len(values))
                data = cell.find("ss:Data", ns)
                values.append("".join(data.itertext()) if data is not None else "")
            rows.append(values)
        sheets.append(rows_to_frame(rows))
    return sheets


def read_any(path: Path) -> list:
    """Read any issuer file with no header assumptions. Returns one DataFrame per sheet.

    The file's first bytes tell us what it really is, because issuers often
    give files an extension that doesn't match their contents.
    """
    start = path.read_bytes()[:512].lstrip(b"\xef\xbb\xbf \r\n\t")

    if start.startswith(b"PK"):                    # real .xlsx (a zip file)
        return list(pd.read_excel(path, header=None, dtype=str, sheet_name=None,
                                  engine="openpyxl").values())
    if start.startswith(b"\xd0\xcf\x11\xe0"):      # old binary .xls
        return list(pd.read_excel(path, header=None, dtype=str, sheet_name=None,
                                  engine="calamine").values())
    if start.startswith(b"<?xml") or b"<Workbook" in start:  # iShares XML "xls"
        return read_spreadsheet_xml(path)
    if start.lower().startswith((b"<html", b"<!doctype", b"<table")):
        raise ValueError("file is a saved web page, not a spreadsheet; "
                         "download the CSV or Excel version instead")

    # Otherwise treat as CSV. Issuer CSVs have short title lines above the
    # table, so read line by line and pad rows to the same length.
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        return [rows_to_frame(list(csv.reader(f)))]


def clean_label(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def find_header_row(raw: pd.DataFrame):
    """The header is the first row containing both a name and a weight column."""
    for i in range(min(60, len(raw))):
        labels = {clean_label(v) for v in raw.iloc[i]}
        if labels & set(COLUMN_SYNONYMS["name"]) and labels & set(COLUMN_SYNONYMS["weight"]):
            return i
    return None


def preview(sheets: list, n_rows: int = 12) -> str:
    """First rows of every sheet, to paste back when a layout isn't recognised."""
    lines = []
    for s, raw in enumerate(sheets):
        lines.append(f"    --- sheet {s + 1} ---")
        for i in range(min(n_rows, len(raw))):
            cells = [str(v) for v in raw.iloc[i] if str(v) not in {"", "nan", "None"}]
            if cells:
                lines.append(f"    row {i + 1}: " + " | ".join(cells)[:160])
    return "\n".join(lines)


def standardise_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Rename issuer columns to our standard names; drop the rest."""
    rename = {}
    for col in table.columns:
        label = clean_label(col)
        for standard, variants in COLUMN_SYNONYMS.items():
            if label in variants and standard not in rename.values():
                rename[col] = standard
                break
    table = table.rename(columns=rename)
    keep = [c for c in COLUMN_SYNONYMS if c in table.columns]
    return table[keep]


def parse_weight(series: pd.Series) -> pd.Series:
    """'5.0056%' or '5.42' or '1,234' -> float. Returned as fraction (0-1)."""
    numbers = pd.to_numeric(
        series.astype(str).str.replace("%", "").str.replace(",", "").str.strip(),
        errors="coerce",
    )
    # Percent if the total is near 100, fraction if near 1.
    if numbers.sum() > 1.5:
        numbers = numbers / 100
    return numbers


def load_issuer_file(path: Path) -> pd.DataFrame:
    sheets = read_any(path)
    for raw in sheets:  # some issuers put a disclaimer sheet first
        header = find_header_row(raw)
        if header is not None:
            break
    else:
        raise ValueError("could not find a header row with a name and a weight column. "
                         "Send these lines to get the layout added:\n" + preview(sheets))
    table = raw.iloc[header + 1:].copy()
    table.columns = [str(c) for c in raw.iloc[header]]
    table = standardise_columns(table)

    table["weight"] = parse_weight(table["weight"])
    table = table.dropna(subset=["weight"])
    table = table[table["name"].notna() & (table["name"].astype(str).str.strip() != "")]

    # iShares lists cash, futures and FX as rows: keep shares only.
    # The dropped weight ends up in the residual row automatically.
    if "asset_class" in table.columns:
        table = table[table["asset_class"].astype(str).str.lower().str.contains("equity")]

    for col in ["ticker", "sector", "country"]:
        if col not in table.columns:
            table[col] = ""
    return table[["name", "ticker", "weight", "sector", "country"]]


# ---------------------------------------------------------------------------
# Standardising values
# ---------------------------------------------------------------------------
def strip_par_value(name: str) -> str:
    """Invesco adds the share's par value: 'NVIDIA CORP USD0.001' -> 'NVIDIA CORP'."""
    return re.sub(r"\s+[A-Z]{3}\s?\d[\d.,]*\s*$", "", str(name).strip())


def name_key(name: str) -> str:
    """'Taiwan Semiconductor Manufacturing Co Ltd' -> 'TAIWAN SEMICONDUCTOR MANUFACTURING'."""
    text = str(name).upper().replace("'", "").replace("’", "")
    text = re.sub(r"[^A-Z0-9]+", " ", text).split()
    while text and text[-1] in NAME_SUFFIXES:
        text.pop()
    return " ".join(text)


def map_sector(value, unknown: set) -> str:
    label = clean_label(value)
    if label in SECTOR_MAP:
        return SECTOR_MAP[label]
    if label not in {"", "nan", "-"}:
        unknown.add(str(value))
    return "Unknown"


def map_country(value, unknown: set) -> str:
    text = str(value).strip()
    if text.upper() in COUNTRY_CODES:
        return COUNTRY_CODES[text.upper()]
    if clean_label(text) in COUNTRY_NAME_FIXES:
        return COUNTRY_NAME_FIXES[clean_label(text)]
    if len(text) > 2 and text.lower() not in {"nan", "-"}:
        return text  # already a full country name
    if text.lower() not in {"", "nan", "-"}:
        unknown.add(text)
    return "Unknown"


def load_name_map() -> dict:
    """Manual fixes: name_key -> (holding_id, display name, ticker). Created if missing.

    The holding_ticker column is optional: fill it to give a company a ticker
    when no issuer file provides one.
    """
    if not NAME_MAP_CSV.exists():
        pd.DataFrame(columns=["name_key", "holding_id", "display_name",
                              "holding_ticker"]).to_csv(NAME_MAP_CSV, index=False)
        return {}
    df = pd.read_csv(NAME_MAP_CSV, dtype=str).fillna("")
    if "holding_ticker" not in df.columns:
        df["holding_ticker"] = ""
    return {r.name_key: (r.holding_id, r.display_name, r.holding_ticker)
            for r in df.itertuples()}


def clean_ticker(value) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "-", "--", "n/a"} else text


# ---------------------------------------------------------------------------
# Building one ETF
# ---------------------------------------------------------------------------
def build_equity_etf(etf: str, path: Path, name_map: dict,
                     unknown_sectors: set, unknown_countries: set) -> pd.DataFrame:
    table = load_issuer_file(path)

    table["holding_id"] = table["name"].map(strip_par_value).map(name_key)
    table["holding_name"] = table["name"].map(strip_par_value)
    table["ticker"] = table["ticker"].map(clean_ticker)
    for key, (hid, display, ticker) in name_map.items():
        mask = table["holding_id"] == key
        table.loc[mask, "holding_id"] = hid or key
        if display:
            table.loc[mask, "holding_name"] = display
        if ticker:
            table.loc[mask, "ticker"] = ticker

    table["sector"] = table["sector"].map(lambda v: map_sector(v, unknown_sectors))
    table["country"] = table["country"].map(lambda v: map_country(v, unknown_countries))

    # Same company listed twice (e.g. Alphabet class A and class C) -> one row.
    merged = (table.groupby("holding_id", as_index=False)
              .agg(holding_name=("holding_name", "first"),
                   holding_ticker=("ticker", lambda s: next((t for t in s if t), "")),
                   weight=("weight", "sum"),
                   sector=("sector", "first"),
                   country=("country", "first")))

    top = merged.nlargest(TOP_N_HOLDINGS, "weight").copy()
    residual_weight = 1.0 - top["weight"].sum()
    if residual_weight < 0:  # file weights add to a bit over 100%
        top["weight"] = top["weight"] / top["weight"].sum()
        residual_weight = 0.0

    residual = pd.DataFrame([{
        "holding_id": "RESIDUAL", "holding_name": RESIDUAL_LABEL, "holding_ticker": "",
        "weight": residual_weight, "sector": "Mixed", "country": "Mixed",
    }])
    out = pd.concat([top, residual], ignore_index=True)
    out.insert(0, "etf_ticker", etf)
    return out[OUTPUT_COLS]


def build_single_block(row) -> pd.DataFrame:
    """Bond and gold ETFs are treated as one holding: the ETF itself."""
    sector = {"bond": "Bonds", "commodity": "Commodities"}.get(row.asset_class, "Other")
    return pd.DataFrame([{
        "etf_ticker": row.ticker, "holding_id": row.ticker, "holding_name": row.name,
        "holding_ticker": row.ticker, "weight": 1.0, "sector": sector,
        "country": row.region,
    }])[OUTPUT_COLS]


def harmonise_labels(holdings: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Give each company ONE sector and country across all ETFs.

    Issuers sometimes disagree (Vanguard files Alphabet under Technology,
    iShares under Communication Services). iShares, Invesco and HSBC use
    the GICS system natively, so their label wins; Vanguard is the fallback.
    """
    issuer = universe.set_index("ticker")["issuer"]
    df = holdings.copy()
    df["_priority"] = (df["etf_ticker"].map(issuer) == "Vanguard").astype(int)

    real = df[(df["holding_id"] != "RESIDUAL") & (df["sector"] != "Unknown")]
    best = (real.sort_values(["_priority", "weight"], ascending=[True, False])
                .drop_duplicates("holding_id")
                .set_index("holding_id"))

    is_company = df["holding_id"].isin(best.index)
    df.loc[is_company, "sector"] = df.loc[is_company, "holding_id"].map(best["sector"])
    df.loc[is_company, "country"] = df.loc[is_company, "holding_id"].map(best["country"])

    # Display names: Vanguard writes "Apple Inc", iShares "APPLE INC".
    # Prefer the nicer mixed-case spelling for the app.
    names = (real.assign(_upper=real["holding_name"].str.isupper())
                 .sort_values(["_upper", "weight"], ascending=[True, False])
                 .drop_duplicates("holding_id")
                 .set_index("holding_id")["holding_name"])
    df.loc[is_company, "holding_name"] = df.loc[is_company, "holding_id"].map(names)

    # Files with no country column (Invesco): for single-country funds,
    # assume the fund's country. EQQQ is ~95% US-listed companies.
    region_country = {"US": "United States", "UK": "United Kingdom", "Japan": "Japan"}
    fund_country = df["etf_ticker"].map(universe.set_index("ticker")["region"]).map(region_country)
    fill = (df["country"] == "Unknown") & fund_country.notna() & (df["holding_id"] != "RESIDUAL")
    df.loc[fill, "country"] = fund_country[fill]

    # Blank tickers (e.g. a file with no ticker column): borrow the ticker
    # another issuer gives the same company.
    known = (df[(df["holding_ticker"] != "") & (df["holding_id"] != "RESIDUAL")]
             .drop_duplicates("holding_id").set_index("holding_id")["holding_ticker"])
    blank = (df["holding_ticker"] == "") & df["holding_id"].isin(known.index)
    df.loc[blank, "holding_ticker"] = df.loc[blank, "holding_id"].map(known)
    return df.drop(columns="_priority")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    universe = pd.read_csv(ETF_UNIVERSE_CSV, dtype=str).fillna("")
    sources = pd.read_csv(HOLDINGS_SOURCES_CSV, dtype=str).fillna("")
    name_map = load_name_map()

    pieces, problems, summary = [], [], []
    unknown_sectors, unknown_countries = set(), set()

    for row in universe.itertuples(index=False):
        if row.asset_class != "equity":
            pieces.append(build_single_block(row))
            summary.append((row.ticker, "single block", 1, 1.0, 0.0))
            continue

        src = sources[sources["ticker"] == row.ticker]
        if src.empty or not src.iloc[0]["file"]:
            problems.append(f"{row.ticker}: no file listed in holdings_sources.csv")
            continue
        path = RAW_DIR / src.iloc[0]["file"]
        if not path.exists():
            problems.append(f"{row.ticker}: file not found: {path.name}")
            continue

        try:
            etf_table = build_equity_etf(row.ticker, path, name_map,
                                         unknown_sectors, unknown_countries)
        except Exception as e:
            problems.append(f"{row.ticker}: could not read {path.name} ({e})")
            continue

        pieces.append(etf_table)
        top_sum = etf_table.loc[etf_table["holding_id"] != "RESIDUAL", "weight"].sum()
        residual = etf_table.loc[etf_table["holding_id"] == "RESIDUAL", "weight"].sum()
        summary.append((row.ticker, path.name, len(etf_table), top_sum, residual))

        # Record the snapshot date in the universe table.
        universe.loc[universe["ticker"] == row.ticker, "holdings_as_of"] = src.iloc[0]["as_of"]

    if not pieces:
        raise SystemExit("Nothing built. Check holdings_sources.csv and data/raw/.")

    holdings = pd.concat(pieces, ignore_index=True)
    holdings = harmonise_labels(holdings, universe)
    holdings["weight"] = holdings["weight"].round(6)

    # --- Checks (the "done when" from the plan), shared with check_holdings.py ---
    check_problems = validate_holdings(holdings, universe)

    holdings.to_csv(ETF_HOLDINGS_CSV, index=False)
    universe.to_csv(ETF_UNIVERSE_CSV, index=False)

    # --- Report ---
    print("\n=== Built ===")
    print(f"{'ETF':<9}{'source':<26}{'rows':>5}{'top-N %':>9}{'residual %':>12}")
    for etf, src, n, top_sum, resid in summary:
        print(f"{etf:<9}{src:<26}{n:>5}{top_sum * 100:>9.1f}{resid * 100:>12.1f}")

    print(f"\nWrote {len(holdings)} rows to {ETF_HOLDINGS_CSV}")
    print(f"Updated holdings_as_of in {ETF_UNIVERSE_CSV}")

    if problems:
        print("\n=== Problems (fix these) ===")
        for p in problems:
            print(" -", p)
    if unknown_sectors:
        print("\nUnmapped sectors (add to SECTOR_MAP):", sorted(unknown_sectors))
    if unknown_countries:
        print("Unmapped countries (add to COUNTRY_CODES):", sorted(unknown_countries))
    no_ticker = missing_tickers(holdings)
    if not no_ticker.empty:
        print(f"\nNote: {len(no_ticker)} companies have no ticker "
              "(details: python scripts/check_holdings.py)")
    if check_problems:
        print("\n=== Checks failed ===")
        for p in check_problems:
            print(" -", p)

    ok = not problems and not check_problems
    print("\nDONE-WHEN CHECK:", "PASS" if ok else "NOT YET")


if __name__ == "__main__":
    main()