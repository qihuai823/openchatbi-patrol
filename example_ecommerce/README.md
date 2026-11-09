# E-commerce demo catalog

A self-contained OpenChatBI demo on an e-commerce operations dataset, plus a
proactive anomaly patrol that runs alongside the chat agent.

Everything here is derived from the Spider `tracking_orders` schema. The
original dataset has no money or quantity columns, so `scripts/build_ecommerce_dataset.py`
parses them out of the free-text fields (see "Dataset" below) with a fixed seed
for reproducibility.

## Dataset

| | |
|---|---|
| Orders | 5,893 (4,940 non-canceled) |
| Order lines | 18,727 |
| SKUs / customers / states / categories | 1,000 / 998 / 50 / 15 |
| Sales channels | 12 |
| Date range | 2025-11-30 → 2026-05-31 |
| GMV / AOV | $6,620,663.22 / $1,340.22 |

Derived columns (the raw Spider tables have none of these):

- `Order_Items.quantity` - parsed from `order_item_details` (`qty 3 ...`), 100% coverage
- `Products.category` - normalised from the `music-2810` style product names
- `Orders.order_channel` - the first token of `order_details`, 100% coverage
- `Order_Items.line_amount` - `quantity * unit_price`, the authoritative revenue column
- `Products.unit_price` / `Products.unit_cost` - current list price and COGS

**2025-11 is incomplete**: only 2025-11-30 (46 orders) falls in that month.
`bi.yaml` warns the model about this, and the patrol plan's `min_samples`
keeps it out of the baseline.

## Running it

```bash
# 1. Dependencies. The full `pip install -e .` pulls in gradio/streamlit/pyhive;
#    the demo only needs the subset below.
pip install langgraph langchain-core langchain-openai langchain-community \
            langchain-chroma langchain-mcp-adapters langgraph-checkpoint-sqlite \
            deepagents langmem rank-bm25 python-Levenshtein regex RestrictedPython \
            docker pandas sqlalchemy pyyaml fastapi chromadb

# 2. Credentials. `config.yaml` ships a placeholder so the file can be committed;
#    put the real key in a copy that stays out of version control.
cp example_ecommerce/config.yaml example_ecommerce/config.local.yaml
# edit api_key in config.local.yaml

# 3. Chat
CONFIG_FILE=example_ecommerce/config.local.yaml python run_cli.py "上个月各品类的 GMV 是多少？"

# 4. Patrol (out of band - the agent only reads its output)
python -m openchatbi.patrol --plan example_ecommerce/patrol_plan.yaml \
    --state example_ecommerce/patrol_state.sqlite
CONFIG_FILE=example_ecommerce/config.local.yaml python run_cli.py "现在有什么经营异常吗？"
```

`config.local.yaml`, `*.local.yaml`, `.env` and `patrol_state.sqlite` are all
gitignored.

## LLM notes

`config.yaml` points at DeepSeek through its OpenAI-compatible endpoint, so the
`langchain_openai.ChatOpenAI` client works unchanged. Two things worth knowing:

- **No `embedding_model`.** DeepSeek serves no embeddings API, so the retriever
  falls back to BM25. That is fine at this catalog size (7 tables, 35 columns);
  the loader only warns.
- **The API may rename the model.** Requesting `deepseek-chat` returns
  `deepseek-flash` in `response_metadata`. `observability/pricing.py` matches
  prefixes, so a bare `deepseek` entry is registered as a fallback - without it
  every turn reports a `$0.00` cost.

## Catalog files

| File | Purpose |
|---|---|
| `bi.yaml` | Business glossary: GMV / Net GMV / AOV / return rate / margin definitions |
| `table_info.yaml` | Per-table description, `selection_rule`, `sql_rule`, `derived_metric` |
| `table_columns.csv` | Column registry (35 rows) |
| `common_columns.csv` | Shared column metadata with Chinese aliases (26 rows) |
| `table_spec_columns.csv` | Table-specific overrides - **`unit_price` means different things on `Products` and `Order_Items`** |
| `sql_example.yaml` | 12 few-shot pairs, every one executed against the dataset |
| `table_selection_example.csv` | 14 question → table-set pairs |
| `patrol_plan.yaml` | 5 monitored metrics + suppression rules |

`tests/test_example_ecommerce_catalog.py` pins all of it: no BOM on any CSV,
header schemas, every column carrying metadata, `unit_price` staying
table-specific, and every few-shot SQL executing non-empty.

## Verified answers

Checked against hand-written SQL on the same database:

| Question | Answer |
|---|---|
| 上个月各品类的 GMV（2026-05） | electronics 468,457.44 / outdoor 139,737.83 / kitchen 94,853.22 |
| 总共有多少客户 | 998 |
| 现在有什么经营异常 | 1 medium alert (office category GMV, 2026-05-30) over 13 below-threshold observations |

## Known limitations

- The dataset stops at 2026-05-31, so questions about "last month" (relative to
  now) correctly return "no data" and the agent falls back to the last complete
  month. That is the model behaving well, not a bug.
- Patrol is daily-granularity only and has no seasonality model; holiday and
  promo spikes need explicit `suppress` rules.
- The state database is a single local SQLite file - two concurrent patrol runs
  will contend on the write lock.
