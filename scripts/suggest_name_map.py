"""Find companies that appear under two different names across issuers.

Step 1 - suggest (default):
    python scripts/suggest_name_map.py
    -> writes data/reference/name_map_suggestions.csv for you to review.
       Type y in the "accept" column for every pair that IS the same company,
       and n for pairs that are NOT (so they are never suggested again).

Step 2 - apply:
    python scripts/suggest_name_map.py --apply
    -> copies the y rows into data/reference/name_map.csv and remembers
       the n rows in data/reference/name_map_rejected.csv.
       Then re-run scripts/build_holdings.py so the fixes take effect.
"""

import sys
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

from fragility.config import ETF_HOLDINGS_CSV, ETF_UNIVERSE_CSV, NAME_MAP_CSV

SUGGESTIONS_CSV = NAME_MAP_CSV.with_name("name_map_suggestions.csv")
REJECTED_CSV = NAME_MAP_CSV.with_name("name_map_rejected.csv")  # pairs you marked n
SIMILARITY_CUTOFF = 0.88     # 0-1; higher = stricter
MIN_PREFIX_LENGTH = 4        # "GE" must not match "GE VERNOVA"


def squash(text: str) -> str:
    """'EXXON MOBIL' -> 'EXXONMOBIL' so spacing differences don't matter."""
    return str(text).replace(" ", "")


def company_table() -> pd.DataFrame:
    """One row per company id, with the issuers, tickers and countries it appears with."""
    holdings = pd.read_csv(ETF_HOLDINGS_CSV, dtype=str).fillna("")
    holdings["weight"] = holdings["weight"].astype(float)
    universe = pd.read_csv(ETF_UNIVERSE_CSV, dtype=str).fillna("")
    holdings["issuer"] = holdings["etf_ticker"].map(universe.set_index("ticker")["issuer"])

    # Skip the residual rows and the bond/gold single-block rows.
    companies = holdings[(holdings["holding_id"] != "RESIDUAL")
                         & (holdings["holding_id"] != holdings["etf_ticker"])]

    return companies.groupby("holding_id").agg(
        name=("holding_name", "first"),
        issuers=("issuer", lambda s: set(s)),
        etfs=("etf_ticker", lambda s: set(s)),
        tickers=("holding_ticker", lambda s: {t for t in s if t}),
        country=("country", "first"),
        total_weight=("weight", "sum"),
    )


def why_similar(id_a: str, id_b: str, a, b) -> tuple:
    """Return (reason, score) if two ids look like the same company, else (None, 0)."""
    if a.tickers & b.tickers:
        return "same ticker " + ", ".join(sorted(a.tickers & b.tickers)), 1.0

    sa, sb = squash(id_a), squash(id_b)
    short, long_ = sorted([sa, sb], key=len)
    if len(short) >= MIN_PREFIX_LENGTH and long_.startswith(short):
        return "one name starts with the other", 0.95

    score = SequenceMatcher(None, sa, sb).ratio()
    if score >= SIMILARITY_CUTOFF:
        return "very similar spelling", round(score, 2)
    return None, 0


def load_rejected() -> set:
    if not REJECTED_CSV.exists():
        return set()
    df = pd.read_csv(REJECTED_CSV, dtype=str).fillna("")
    return {frozenset(p) for p in zip(df["name_key"], df["holding_id"])}


def suggest() -> None:
    table = company_table()
    rejected = load_rejected()
    rows = []

    for id_a, id_b in combinations(table.index, 2):
        a, b = table.loc[id_a], table.loc[id_b]

        # Each issuer spells names consistently: two ids seen at exactly the
        # same issuers are two different companies, not two spellings.
        if a.issuers == b.issuers or frozenset((id_a, id_b)) in rejected:
            continue

        reason, score = why_similar(id_a, id_b, a, b)
        if not reason:
            continue

        # Keep the id that carries more money; rename the other to it.
        keep, rename = (id_a, id_b) if a.total_weight >= b.total_weight else (id_b, id_a)
        kept, renamed = table.loc[keep], table.loc[rename]

        warnings = []
        if kept.country != renamed.country:
            warnings.append(f"different countries ({kept.country} vs {renamed.country})")
        if kept.etfs & renamed.etfs:
            warnings.append("both appear in the same ETF")

        rows.append({
            "accept": "",
            "name_key": rename,
            "holding_id": keep,
            "display_name": kept["name"],
            "renamed_from": renamed["name"],
            "reason": reason,
            "score": score,
            "check": "; ".join(warnings) or "looks safe",
        })

    out = pd.DataFrame(rows, columns=["accept", "name_key", "holding_id", "display_name",
                                      "renamed_from", "reason", "score", "check"])
    out = out.sort_values(["check", "score"], ascending=[False, False])
    out.to_csv(SUGGESTIONS_CSV, index=False)

    safe = (out["check"] == "looks safe").sum()
    print(f"Checked {len(table)} companies across issuers.")
    print(f"Wrote {len(out)} suggestions to {SUGGESTIONS_CSV}")
    print(f"  {safe} look safe, {len(out) - safe} need a closer look (see 'check' column).")
    print("\nNext: open it, type y in 'accept' for real matches, then run with --apply.")


def apply() -> None:
    if not SUGGESTIONS_CSV.exists():
        raise SystemExit("No suggestions file yet. Run without --apply first.")

    suggestions = pd.read_csv(SUGGESTIONS_CSV, dtype=str).fillna("")
    answer = suggestions["accept"].str.strip().str.lower()
    accepted = suggestions[answer.isin({"y", "yes"})]
    declined = suggestions[answer.isin({"n", "no"})]

    if not declined.empty:
        pairs = declined[["name_key", "holding_id"]]
        if REJECTED_CSV.exists():
            pairs = pd.concat([pd.read_csv(REJECTED_CSV, dtype=str), pairs])
        pairs.drop_duplicates().to_csv(REJECTED_CSV, index=False)
        print(f"Remembered {len(declined)} 'not the same company' answers.")

    if accepted.empty:
        raise SystemExit("No rows marked y in the 'accept' column. Nothing added to name_map.")

    new = accepted[["name_key", "holding_id", "display_name"]]
    if NAME_MAP_CSV.exists():
        existing = pd.read_csv(NAME_MAP_CSV, dtype=str).fillna("")
        new = pd.concat([existing, new], ignore_index=True)
    new = new.drop_duplicates("name_key", keep="last")
    new = new.reindex(columns=["name_key", "holding_id", "display_name", "holding_ticker"],
                      fill_value="").fillna("")
    new.to_csv(NAME_MAP_CSV, index=False)

    print(f"Added {len(accepted)} rows. {NAME_MAP_CSV.name} now has {len(new)} rows.")
    print("Next: python scripts/build_holdings.py  (then re-run this script to check)")


if __name__ == "__main__":
    apply() if "--apply" in sys.argv else suggest()