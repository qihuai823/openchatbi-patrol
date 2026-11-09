"""Build the e-commerce analytics dataset used by the OpenChatBI e-commerce scenario.

The upstream `example/tracking_orders.sqlite` (Spider dataset) has no price or
quantity columns: the item quantity is embedded in the free-text
`order_item_details` column (e.g. ``"retail item qty 8"``), and products only
carry a category name. That makes GMV / AOV / margin impossible to compute.

This script derives a new database that keeps every original column and adds the
analytics columns needed for business analysis:

* ``Products.category``   -- normalized category (``music-2810`` -> ``music``)
* ``Products.unit_price`` -- simulated list price, stable per product
* ``Products.unit_cost``  -- simulated cost of goods, ~65% of list price
* ``Order_Items.quantity``-- parsed from ``order_item_details``
* ``Order_Items.unit_price`` / ``line_amount`` -- denormalized for fast SQL
* ``Orders.order_channel``-- parsed from the first token of ``order_details``
  (``online`` / ``retail`` / ``wholesale`` / ``bulk`` / ``gift`` / ``seasonal`` /
  ``express`` / ``standard`` / ``priority`` / ``verified`` / ``repeat`` /
  ``promotion``), covering all 5893 orders

Note on the monetary scale: the Spider sample genuinely has ~3.2 lines per order
and ~4.9 units per line, i.e. wholesale/distribution volumes rather than 2C
retail. Prices are simulated around that structure, so AOV lands in the
distribution range. The business is therefore framed as omni-channel
distribution (retail storefront + online + wholesale + bulk) rather than pure
consumer e-commerce.

IMPORTANT: prices and costs are SIMULATED with a fixed random seed so the
dataset is reproducible. Only the order/customer/product/shipment structure is
real (it comes from the Spider ``tracking_orders`` sample). Do not present the
monetary figures as real business data.

Usage:
    python scripts/build_ecommerce_dataset.py
    python scripts/build_ecommerce_dataset.py --out example_ecommerce/ecommerce.sqlite
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
from pathlib import Path

DEFAULT_SOURCE = Path("example/tracking_orders.sqlite")
DEFAULT_OUT = Path("example_ecommerce/ecommerce.sqlite")

# Fixed seed: rebuilds must produce byte-identical prices so catalog examples
# and documented SQL results stay stable across machines.
SEED = 20260531

# (min_price, max_price) in USD per normalized category.
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "phone": (89.0, 899.0),
    "electronics": (49.0, 749.0),
    "kitchen": (12.0, 149.0),
    "outdoor": (18.0, 299.0),
    "sports": (12.0, 199.0),
    "clothes": (9.0, 119.0),
    "beauty": (7.0, 89.0),
    "health": (6.0, 59.0),
    "office": (4.0, 79.0),
    "pet": (5.0, 69.0),
    "toy": (6.0, 59.0),
    "garden": (8.0, 119.0),
    "book": (5.0, 39.0),
    "music": (6.0, 34.0),
    "food": (2.0, 29.0),
}
DEFAULT_RANGE = (5.0, 99.0)

# Cost of goods as a share of list price, per category (margin varies by vertical).
MARGIN_BY_CATEGORY: dict[str, float] = {
    "phone": 0.78,
    "electronics": 0.74,
    "book": 0.62,
    "food": 0.70,
    "music": 0.60,
}
DEFAULT_COST_RATIO = 0.65

QTY_RE = re.compile(r"qty\s+(\d+)", re.IGNORECASE)

# First token of ``Orders.order_details`` is a channel label; every one of the
# 5893 orders matches exactly one of these.
ORDER_CHANNELS = (
    "online",
    "retail",
    "wholesale",
    "bulk",
    "gift",
    "seasonal",
    "express",
    "standard",
    "priority",
    "verified",
    "repeat",
    "promotion",
)


def parse_order_channel(details: str | None) -> str:
    """Extract the order channel from the free-text ``order_details``."""
    if not details:
        return "unknown"
    tokens = details.strip().lower().split()
    if tokens and tokens[0] in ORDER_CHANNELS:
        return tokens[0]
    for channel in ORDER_CHANNELS:
        if channel in details.lower():
            return channel
    return "unknown"


def normalize_category(product_name: str | None) -> str:
    """Collapse ``category-code`` names onto their category prefix.

    The Spider sample mixes ``music`` and ``music-2810`` style values; the
    prefix before ``-`` is the category in both cases.
    """
    if not product_name:
        return "unknown"
    return product_name.split("-", 1)[0].strip().lower() or "unknown"


def parse_quantity(details: str | None) -> int:
    """Extract the item quantity from the free-text ``order_item_details``."""
    if not details:
        return 1
    match = QTY_RE.search(details)
    if not match:
        return 1
    return max(int(match.group(1)), 1)


def build(source: Path, out: Path) -> None:
    if not source.exists():
        raise SystemExit(f"source database not found: {source}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    rng = random.Random(SEED)
    src = sqlite3.connect(source)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(out)

    # ---- Products: add category / price / cost -------------------------------
    products = [dict(row) for row in src.execute("SELECT * FROM Products")]
    price_by_product: dict[int, float] = {}
    for product in products:
        category = normalize_category(product["product_name"])
        low, high = PRICE_RANGES.get(category, DEFAULT_RANGE)
        price = round(rng.uniform(low, high), 2)
        cost_ratio = MARGIN_BY_CATEGORY.get(category, DEFAULT_COST_RATIO)
        product["category"] = category
        product["unit_price"] = price
        product["unit_cost"] = round(price * cost_ratio, 2)
        price_by_product[product["product_id"]] = price

    dst.execute(
        """
        CREATE TABLE Products (
            product_id    INTEGER PRIMARY KEY,
            product_name  TEXT,
            category      TEXT NOT NULL,
            unit_price    REAL NOT NULL,
            unit_cost     REAL NOT NULL,
            product_details TEXT
        )
        """
    )
    dst.executemany(
        "INSERT INTO Products VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                p["product_id"],
                p["product_name"],
                p["category"],
                p["unit_price"],
                p["unit_cost"],
                p["product_details"],
            )
            for p in products
        ],
    )

    # ---- Order_Items: add quantity / unit_price / line_amount ---------------
    items = [dict(row) for row in src.execute("SELECT * FROM Order_Items")]
    for item in items:
        item["quantity"] = parse_quantity(item["order_item_details"])
        item["unit_price"] = price_by_product.get(item["product_id"], 0.0)
        item["line_amount"] = round(item["quantity"] * item["unit_price"], 2)

    dst.execute(
        """
        CREATE TABLE Order_Items (
            order_item_id     INTEGER PRIMARY KEY,
            order_id          INTEGER NOT NULL,
            product_id        INTEGER NOT NULL,
            order_item_status TEXT,
            quantity          INTEGER NOT NULL,
            unit_price        REAL NOT NULL,
            line_amount       REAL NOT NULL,
            order_item_details TEXT
        )
        """
    )
    dst.executemany(
        "INSERT INTO Order_Items VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                i["order_item_id"],
                i["order_id"],
                i["product_id"],
                i["order_item_status"],
                i["quantity"],
                i["unit_price"],
                i["line_amount"],
                i["order_item_details"],
            )
            for i in items
        ],
    )

    # ---- Pass-through tables (structure unchanged) --------------------------
    dst.execute(
        """
        CREATE TABLE Customers (
            customer_id      INTEGER PRIMARY KEY,
            customer_name    TEXT,
            customer_details TEXT,
            state            TEXT
        )
        """
    )
    dst.executemany(
        "INSERT INTO Customers VALUES (?, ?, ?, ?)",
        [
            (r["customer_id"], r["customer_name"], r["customer_details"], r["state"])
            for r in src.execute("SELECT * FROM Customers")
        ],
    )

    dst.execute(
        """
        CREATE TABLE Orders (
            order_id          INTEGER PRIMARY KEY,
            customer_id       INTEGER NOT NULL,
            order_status      TEXT,
            order_channel     TEXT NOT NULL,
            date_order_placed TEXT NOT NULL,
            order_details     TEXT
        )
        """
    )
    dst.executemany(
        "INSERT INTO Orders VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                r["order_id"],
                r["customer_id"],
                r["order_status"],
                parse_order_channel(r["order_details"]),
                r["date_order_placed"],
                r["order_details"],
            )
            for r in src.execute("SELECT * FROM Orders")
        ],
    )

    dst.execute(
        """
        CREATE TABLE Invoices (
            invoice_number INTEGER PRIMARY KEY,
            invoice_date   TEXT,
            invoice_details TEXT
        )
        """
    )
    dst.executemany(
        "INSERT INTO Invoices VALUES (?, ?, ?)",
        [(r["invoice_number"], r["invoice_date"], r["invoice_details"]) for r in src.execute("SELECT * FROM Invoices")],
    )

    dst.execute(
        """
        CREATE TABLE Shipments (
            shipment_id              INTEGER PRIMARY KEY,
            order_id                 INTEGER NOT NULL,
            invoice_number           INTEGER,
            shipment_tracking_number TEXT,
            shipment_date            TEXT,
            other_shipment_details   TEXT
        )
        """
    )
    dst.executemany(
        "INSERT INTO Shipments VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                r["shipment_id"],
                r["order_id"],
                r["invoice_number"],
                r["shipment_tracking_number"],
                r["shipment_date"],
                r["other_shipment_details"],
            )
            for r in src.execute("SELECT * FROM Shipments")
        ],
    )

    dst.execute(
        """
        CREATE TABLE Shipment_Items (
            shipment_id   INTEGER NOT NULL,
            order_item_id INTEGER NOT NULL,
            PRIMARY KEY (shipment_id, order_item_id)
        )
        """
    )
    dst.executemany(
        "INSERT INTO Shipment_Items VALUES (?, ?)",
        [(r["shipment_id"], r["order_item_id"]) for r in src.execute("SELECT * FROM Shipment_Items")],
    )

    # ---- Indexes: the analytics queries all filter on dates and join on ids --
    for ddl in (
        "CREATE INDEX idx_orders_date ON Orders(date_order_placed)",
        "CREATE INDEX idx_orders_customer ON Orders(customer_id)",
        "CREATE INDEX idx_orders_channel ON Orders(order_channel)",
        "CREATE INDEX idx_items_order ON Order_Items(order_id)",
        "CREATE INDEX idx_items_product ON Order_Items(product_id)",
        "CREATE INDEX idx_shipments_order ON Shipments(order_id)",
    ):
        dst.execute(ddl)

    dst.commit()

    # ---- Report -------------------------------------------------------------
    def scalar(sql: str) -> float:
        value = dst.execute(sql).fetchone()[0]
        return float(value) if value is not None else 0.0

    gmv = scalar(
        "SELECT SUM(oi.line_amount) FROM Order_Items oi "
        "JOIN Orders o ON oi.order_id = o.order_id "
        "WHERE o.order_status != 'Canceled'"
    )
    orders = scalar("SELECT COUNT(*) FROM Orders WHERE order_status != 'Canceled'")
    categories = scalar("SELECT COUNT(DISTINCT category) FROM Products")
    channels = scalar("SELECT COUNT(DISTINCT order_channel) FROM Orders")
    print(f"wrote {out}")
    print(f"  products          : {len(products)}")
    print(f"  categories        : {int(categories)}")
    print(f"  order items       : {len(items)}")
    print(f"  valid orders      : {int(orders)}")
    print(f"  channels          : {int(channels)}")
    print(f"  GMV (excl. cancel): ${gmv:,.2f}")
    print(f"  AOV               : ${gmv / orders if orders else 0:,.2f}")
    print("\n  channel breakdown (valid orders):")
    print(f"    {'channel':<12}{'orders':>8}{'GMV':>16}{'AOV':>12}")
    for channel, count, channel_gmv in dst.execute(
        "SELECT o.order_channel, COUNT(DISTINCT o.order_id), SUM(oi.line_amount) "
        "FROM Orders o JOIN Order_Items oi ON oi.order_id = o.order_id "
        "WHERE o.order_status != 'Canceled' GROUP BY 1 ORDER BY 3 DESC"
    ):
        avg = channel_gmv / count if count else 0
        print(f"    {channel:<12}{count:>8}{channel_gmv:>16,.2f}{avg:>12,.2f}")

    src.close()
    dst.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
