"""Company financials from Yahoo Finance (yfinance) for the Phase 2 dashboard. Run in Colab: Yahoo is not reachable
from every build environment, and a web page cannot call Yahoo itself (browsers block it, and artifact pages may only load
their own data).

    from src.kg import finance
    snap = finance.fetch_all(finance.tickers_from_kg(kg), P2 / "data/raw/yfinance")
    finance.write_snapshot(snap, P2 / "data/processed/finance/yfinance_snapshot.json")

One ticker per request, one after another, with a pause between them (default 3 s), so the free endpoint is never
hit in parallel. Each ticker is cached for the day: a rerun only fetches what is missing or failed.
The snapshot is labelled "Yahoo Finance via yfinance" with its date; nothing is estimated or filled in.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path

INFO_FIELDS = [
    "shortName", "longName", "quoteType", "exchange", "fullExchangeName", "currency", "financialCurrency",
    "sector", "industry", "country", "city", "website", "fullTimeEmployees",
    "marketCap", "enterpriseValue", "currentPrice", "regularMarketPrice", "previousClose",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiftyDayAverage", "twoHundredDayAverage", "beta",
    "trailingPE", "forwardPE", "priceToSalesTrailing12Months", "priceToBook", "enterpriseToEbitda",
    "enterpriseToRevenue", "trailingPegRatio", "trailingEps", "forwardEps", "dividendYield", "trailingAnnualDividendYield", "payoutRatio",
    "grossMargins", "operatingMargins", "ebitdaMargins", "profitMargins",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth", "returnOnEquity", "returnOnAssets",
    "totalRevenue", "grossProfits", "ebitda", "netIncomeToCommon", "freeCashflow", "operatingCashflow",
    "totalCash", "totalDebt", "debtToEquity", "currentRatio",
    "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
]
STATEMENT_ROWS = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Research And Development"]


def tickers_from_kg(kg) -> dict[str, str]:
    """company_id -> Yahoo ticker, for the companies the Phase 1 graph has a ticker for (the 50 source companies)."""
    return {eid: e["ticker"] for eid, e in sorted(kg.entities.items())
            if e.get("entity_type") == "Company" and e.get("ticker")}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def fetch_one(ticker: str) -> dict:
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info or {}
    out = {"ticker": ticker, "info": {k: _clean(info.get(k)) for k in INFO_FIELDS if _clean(info.get(k)) is not None}}
    try:
        st = t.income_stmt  # annual, newest column first
        years = []
        for col in list(st.columns)[:4]:
            row = {"period_end": str(getattr(col, "date", lambda: col)())}
            for r in STATEMENT_ROWS:
                if r in st.index:
                    row[r] = _clean(st.loc[r, col])
            years.append(row)
        out["annual"] = years
    except Exception as e:  # statement missing for some foreign listings
        out["annual_error"] = repr(e)[:200]
    try:
        h = t.history(period="5y", interval="1wk", auto_adjust=True)
        out["prices"] = [[str(i.date()), round(float(c), 4)] for i, c in h["Close"].dropna().items()]
    except Exception as e:
        out["prices_error"] = repr(e)[:200]
    return out


def fetch_all(tickers: dict[str, str], cache_dir, pause: float = 3.0, today: str | None = None) -> dict:
    """Sequential, one ticker at a time. Returns {company_id: data}; failures are recorded, not guessed."""
    today = today or dt.date.today().isoformat()
    cache = Path(cache_dir) / today
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, (cid, tk) in enumerate(tickers.items()):
        f = cache / f"{tk.replace('/', '_')}.json"
        if f.exists():
            d = json.loads(f.read_text())
            if "error" not in d:
                out[cid] = d
                continue
        try:
            d = fetch_one(tk)
        except Exception as e:
            d = {"ticker": tk, "error": repr(e)[:300]}
        d["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        f.write_text(json.dumps(d, ensure_ascii=False))
        out[cid] = d
        print(f"{i + 1}/{len(tickers)} {tk}: {'error' if 'error' in d else 'ok'}")
        time.sleep(pause)
    return out


def write_snapshot(data: dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dates = sorted(d.get("fetched_at", "")[:10] for d in data.values() if d.get("fetched_at"))
    snap = {"source": "Yahoo Finance via yfinance", "as_of": dates[-1] if dates else None, "companies": data}
    p.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    return p
