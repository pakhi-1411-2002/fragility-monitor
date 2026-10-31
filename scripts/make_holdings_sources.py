"""Create or update data/reference/holdings_sources.csv automatically.

Looks in data/raw/ for files named TICKER_YYYY-MM-DD.xlsx (or .csv/.xls),
e.g. VWRL_2026-09-30.xlsx, and records which file belongs to which equity
ETF in etf_universe.csv. Safe to re-run after every new download: URLs and
notes you typed in by hand are kept.

Run from the repo root:
    python scripts/make_holdings_sources.py
"""

import re

import pandas as pd

from fragility.config import ETF_UNIVERSE_CSV, HOLDINGS_SOURCES_CSV, RAW_DIR

# Funds that reuse another fund's holdings file (same fund, other share class).
SHARED_FILES = {"VUAG.L": "VUSA.L"}

FILE_PATTERN = re.compile(r"^([A-Z0-9]+)_(\d{4}-\d{2}-\d{2})\.(xlsx|xls|csv)$", re.IGNORECASE)
COLUMNS = ["ticker", "issuer", "file", "as_of", "source_url", "notes"]


def find_raw_files() -> dict:
    """Return {'VWRL': ('VWRL_2026-09-30.xlsx', '2026-09-30'), ...}, newest per ticker."""
    found = {}
    for path in sorted(RAW_DIR.glob("*")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        short, as_of = match.group(1).upper(), match.group(2)
        if short not in found or as_of > found[short][1]:
            found[short] = (path.name, as_of)
    return found


def main() -> None:
    universe = pd.read_csv(ETF_UNIVERSE_CSV, dtype=str).fillna("")
    equity = universe[universe["asset_class"] == "equity"]

    # Keep anything typed by hand in an existing file (URLs, notes).
    existing = {}
    if HOLDINGS_SOURCES_CSV.exists():
        old = pd.read_csv(HOLDINGS_SOURCES_CSV, dtype=str).fillna("")
        existing = {r.ticker: r for r in old.itertuples(index=False)}

    raw_files = find_raw_files()
    rows, missing = [], []

    for etf in equity.itertuples(index=False):
        source_ticker = SHARED_FILES.get(etf.ticker, etf.ticker)
        short = source_ticker.split(".")[0].upper()
        file_name, as_of = raw_files.get(short, ("", ""))

        old = existing.get(etf.ticker)
        notes = old.notes if old is not None else ""
        if etf.ticker in SHARED_FILES and not notes:
            notes = f"same fund as {source_ticker}: uses its file"

        rows.append({
            "ticker": etf.ticker,
            "issuer": etf.issuer,
            "file": file_name,
            "as_of": as_of,
            "source_url": old.source_url if old is not None else "",
            "notes": notes,
        })
        if not file_name:
            missing.append(f"{etf.ticker} (save as {short}_YYYY-MM-DD.xlsx or .csv)")

    pd.DataFrame(rows, columns=COLUMNS).to_csv(HOLDINGS_SOURCES_CSV, index=False)

    found_count = len(rows) - len(missing)
    print(f"Wrote {HOLDINGS_SOURCES_CSV}")
    print(f"{found_count} of {len(rows)} equity ETFs have a holdings file.")
    if missing:
        print("\nStill to download:")
        for m in missing:
            print(" -", m)

    # Files in data/raw/ that don't follow the naming rule, so were ignored.
    ignored = [p.name for p in RAW_DIR.glob("*")
               if p.is_file() and not FILE_PATTERN.match(p.name) and p.name != ".gitkeep"]
    if ignored:
        print("\nIgnored (rename to TICKER_YYYY-MM-DD.ext):")
        for name in ignored:
            print(" -", name)


if __name__ == "__main__":
    main()