"""Regional orders report.

Reads data/orders.csv (relative to this file) and answers a single,
unambiguous question about the export, mirroring the attachment-grounded
questions in the GAIA benchmark: the answer can only be produced by actually
opening and parsing the messy spreadsheet.

The export is a real-world-messy CSV:
- comment lines start with "#" and must be skipped
- there is a blank spacer row
- there is a trailing "TOTAL" footer row that is NOT an order
- region casing/whitespace is inconsistent ("EMEA", "emea", "EMEA ")
- status casing is inconsistent ("Shipped", "shipped", "SHIPPED")
- amount is a currency string with a symbol and thousands separators
- one row is billed in EUR and must be excluded (USD-only report)

A row QUALIFIES for this report when, after normalization:
- region == "emea"            (case-insensitive, trimmed)
- status == "shipped"         (case-insensitive, trimmed)
- currency == "USD"
- amount parses to a number >= 0

The report computes, over the qualifying rows:
- qualifying_orders: count of qualifying rows
- total_usd:         sum of amounts, rounded to 2 decimals
- max_order_id:      the order_id of the single largest-amount qualifying row
- REPORT_TOKEN:      sha256 over the sorted qualifying order ids, so two
                     reports agree only if they selected exactly the same rows
"""

import csv
import hashlib
from pathlib import Path

DATA = Path(__file__).parent / "data" / "orders.csv"

TARGET_REGION = "emea"
TARGET_STATUS = "shipped"
TARGET_CURRENCY = "USD"


def load_rows(path=DATA):
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    return list(reader)


def parse_amount(raw):
    if raw is None:
        return None
    cleaned = raw.strip().lstrip("$").replace(",", "")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


def qualifies(row):
    if not (row.get("order_id") or "").strip().isdigit():
        return False
    region = (row.get("region") or "").strip().lower()
    status = (row.get("status") or "").strip().lower()
    currency = (row.get("currency") or "").strip()
    if region != TARGET_REGION or status != TARGET_STATUS:
        return False
    if currency != TARGET_CURRENCY:
        return False
    return parse_amount(row.get("amount")) is not None


def select(rows):
    return [row for row in rows if qualifies(row)]


def report_token(selected):
    ids = sorted(row["order_id"].strip() for row in selected)
    payload = TARGET_REGION + ":" + ",".join(ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def max_order_id(selected):
    if not selected:
        return None
    best = max(selected, key=lambda row: parse_amount(row["amount"]))
    return best["order_id"].strip()


def build_report(path=DATA):
    selected = select(load_rows(path))
    total = sum(parse_amount(row["amount"]) for row in selected)
    return {
        "token": report_token(selected),
        "qualifying_orders": len(selected),
        "total_usd": round(total, 2),
        "max_order_id": max_order_id(selected),
    }


def main():
    report = build_report()
    print(f"REPORT_TOKEN:      {report['token']}")
    print(f"qualifying orders: {report['qualifying_orders']}")
    print(f"total USD:         {report['total_usd']}")
    print(f"largest order id:  {report['max_order_id']}")


if __name__ == "__main__":
    main()
