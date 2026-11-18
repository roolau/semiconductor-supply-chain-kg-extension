# Equity research extension

An analyst tool built on top of the Phase 1 semiconductor supply-chain knowledge graph in this repository. It adds no new extraction: every module reads the Phase 1 graph (read-only) and adds logic, evaluation and an interactive website.

**Live site:** https://claude.ai/artifact/GtFG9W1vcjjLKvkJPrD6LV

## Modules

| Module | What it does | Code |
|---|---|---|
| A. Product flows | Labels supply links with what is sold: A supplies B, A makes category X, and B uses X or buys X from another supplier ⇒ A sells X to B | `src/kg/products.py` |
| B. Clustering | Value-chain stage from main products; product peers (Jaccard + average linkage, cut at 0.5); ecosystems (Louvain) | `src/kg/cluster.py` |
| C. Relation prediction | Likely new customers, suppliers and competitors: peer, product and relation-count lists fused with reciprocal rank fusion (RRF) | `src/kg/recommend.py` |
| D. KGQA | An LLM turns a question into a small query plan, which runs on the graph; answers come from the graph only | `src/kg/qa.py`, `src/kg/qa_bench.py` |
| Website | Graph explorer, clusters, company dashboard (market data from yfinance), chat, accuracy tab | `src/kg/site.py`, `src/kg/site_parts.html` |

## Results

| Task | Metric | This system | Baseline |
|---|---|---|---|
| Customers | Hits@10 (345 hidden links, 5 seeds) | 0.545 | 0.441 (rank by relation count) |
| Suppliers | Hits@10 (345 hidden links, 5 seeds) | 0.594 | 0.487 (rank by relation count) |
| Competitors | Hit@10 vs ETO (30 companies) | 0.767 | 0.133 (rank by relation count) |
| Value-chain stage | correct vs ETO (20 companies) | 19/20 | — |
| KGQA | accuracy, held-out paraphrased questions (113) | 94.7% (95% CI 88.9–97.5) | 1.8% (fixed patterns) |
| Product flows | supply links labelled | 161/348 (46%) | — |

Hidden links are chosen at random, not by date. Product labels are coverage, not checked precision.

## Layout

```
equity-research-extension/
├── src/kg/            modules (load, products, cluster, recommend, qa, qa_bench, llm, finance, site, viz, eto, names, categories)
├── notebooks/
│   ├── run_modules.ipynb   modules A–C, evaluation, website build (no network, no LLM)
│   ├── run_qa.ipynb        KGQA benchmark (Gemini paraphrases, Groq planner)
│   └── run_finance.ipynb   market-data snapshot from yfinance
├── tests/             unit tests on small synthetic graphs, plus KGQA checks on the real graph
├── data/reference/eto/  ETO supply-chain data used for evaluation (see SOURCE.md)
├── data/results/      saved results the website and the README numbers come from
└── site/              index.html (built page) and phase1_base.html (the Phase 1 page it extends)
```

## Run

```bash
cd equity-research-extension
pip install -r requirements.txt
pytest -q tests
```

The code reads the Phase 1 KG from `data/` at the repository root (`fused_edges.jsonl`, `canonical_entities.jsonl`); set `P1_KG_DIR` to use another copy. In Colab, the first notebook cell clones this repository. `run_qa.ipynb` needs two API keys, as Colab secrets or environment variables: `GeminiAPIKey` and `GroqAPIKey`. All LLM replies are cached, so interrupted runs resume.

The chat on the website uses Claude when the page is opened in claude.ai; elsewhere it falls back to fixed question patterns.

## Data and credits

- Phase 1 knowledge graph: Team 2 group project (this repository).
- ETO supply-chain data: Emerging Technology Observatory, CSET, Georgetown University, CC BY-NC 4.0.
- Market data: Yahoo Finance via `yfinance`, snapshot of the date stored in the file.
