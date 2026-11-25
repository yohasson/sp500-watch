"""Market-data providers with a common interface and a local cache.

Primary: Yahoo Finance chart API (unofficial, free; split/dividend-adjusted close available).
Fallback: Stooq daily CSV (no dividend adjustment).
The provider is pluggable: replace with a licensed vendor (Norgate, Sharadar, Tiingo, ...)
by implementing `history(symbol, years) -> DataFrame[open, close, adjclose]`.
"""
import os, time, json, io, concurrent.futures as cf
import numpy as np, pandas as pd, requests
from .config import DATA_DIR

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
PRICE_DIR = os.path.join(DATA_DIR, "prices")

class YahooProvider:
    name = "yahoo"
    hosts = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
    def __init__(self):
        self.s = requests.Session(); self.s.headers.update(UA)
    def history(self, sym, years=10, rng=None):
        rng = rng or f"{years}y"
        ysym = sym.replace("^", "%5E")
        last = None
        for attempt in range(3):
            host = self.hosts[attempt % 2]
            url = f"{host}/v8/finance/chart/{ysym}?range={rng}&interval=1d&events=div%7Csplit&includeAdjustedClose=true"
            try:
                r = self.s.get(url, timeout=12)
                if r.status_code == 429:
                    time.sleep(1 + 2 * attempt); continue
                if r.status_code == 404: return None
                r.raise_for_status()
                res = r.json()["chart"]["result"][0]
                ts = res.get("timestamp")
                if not ts: return None
                q = res["indicators"]["quote"][0]
                adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
                idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
                df = pd.DataFrame({"open": q["open"], "close": q["close"], "adjclose": adj}, index=idx, dtype=float)
                return df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
            except Exception as e:
                last = e; time.sleep(1 + attempt)
        raise RuntimeError(f"yahoo {sym}: {last}")

class StooqProvider:
    name = "stooq"
    def history(self, sym, years=10, rng=None):
        s = sym.lower().replace("-", ".")
        s = "^spx" if s == "^gspc" else (s if s.startswith("^") else s + ".us")
        r = requests.get(f"https://stooq.com/q/d/l/?s={s}&i=d", headers=UA, timeout=8)
        if not r.text.startswith("Date"): return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
        df = df[df.index >= pd.Timestamp.today() - pd.Timedelta(days=365 * years)]
        return pd.DataFrame({"open": df["Open"], "close": df["Close"], "adjclose": df["Close"]})

def yf_bulk(symbols, period, log=print, chunk=100):
    """Bulk download through the yfinance library (handles Yahoo cookies/crumb; works from cloud runners)."""
    try:
        import yfinance as yf
    except Exception as e:
        log(f"[data] yfinance unavailable: {e}"); return {}
    out = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        try:
            df = yf.download(part, period=period, interval="1d", auto_adjust=False, actions=False,
                             group_by="ticker", threads=True, progress=False)
        except Exception as e:
            log(f"[data] yfinance chunk {i} failed: {e}"); continue
        if df is None or df.empty: continue
        for sym in part:
            try:
                sub = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
                d = pd.DataFrame({"open": sub["Open"], "close": sub["Close"],
                                  "adjclose": sub["Adj Close"] if "Adj Close" in sub else sub["Close"]}).dropna(subset=["close"])
                if len(d):
                    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                    out[sym] = d[~d.index.duplicated(keep="last")]
            except Exception:
                pass
        log(f"[data] yfinance {min(i + chunk, len(symbols))}/{len(symbols)} -> {len(out)} ok")
    return out

def _load_cache():
    out = {}
    for f in ("open", "close", "adjclose"):
        p = os.path.join(PRICE_DIR, f"{f}.csv.gz")
        if os.path.exists(p):
            out[f] = pd.read_csv(p, index_col=0, parse_dates=True)
    return out

def _save_cache(frames):
    os.makedirs(PRICE_DIR, exist_ok=True)
    for f, df in frames.items():
        df.sort_index().round(4).to_csv(os.path.join(PRICE_DIR, f"{f}.csv.gz"), compression="gzip")

def update_prices(symbols, years=10, providers=None, workers=12, log=print, budget_s=None):
    """Fill/refresh the cache. Full history for new symbols, recent months for cached ones."""
    providers = providers or [YahooProvider(), StooqProvider()]
    cache = _load_cache()
    have = set(cache["close"].columns) if "close" in cache else set()
    last_date = cache["close"].index.max() if "close" in cache else None
    stale_days = (pd.Timestamp.today().normalize() - last_date).days if last_date is not None else 9999
    deadline = time.time() + (budget_s or float(os.environ.get("ORACLE_DATA_BUDGET", 600)))
    def job(sym):
        if time.time() > deadline: return sym, None, None
        full = sym not in have
        rng = None if full else ("3mo" if stale_days < 80 else "1y")
        for p in providers:
            try:
                df = p.history(sym, years, rng=rng)
                if df is not None and len(df): return sym, df, p.name
            except Exception as e:
                pass
        return sym, None, None
    t0 = time.time()
    got, failed, used = {}, [], {}
    full_syms = [x for x in symbols if x not in have]; part_syms = [x for x in symbols if x in have]
    for group, period in ((full_syms, f"{years}y"), (part_syms, "3mo" if stale_days < 80 else "1y")):
        if group:
            res = yf_bulk(group, period, log=log); got.update(res); used["yfinance"] = used.get("yfinance", 0) + len(res)
    symbols = [x for x in symbols if x not in got]
    log(f"[data] after yfinance: {len(got)} ok, {len(symbols)} left for direct download, {time.time()-t0:.0f}s")
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, (sym, df, pn) in enumerate(ex.map(job, symbols)):
            if i % 100 == 0: log(f"[data] {i}/{len(symbols)} symbols, {time.time()-t0:.0f}s")
            if df is None: failed.append(sym)
            else: got[sym] = df; used[pn] = used.get(pn, 0) + 1
    frames = {}
    for f in ("open", "close", "adjclose"):
        new = pd.DataFrame({s: d[f] for s, d in got.items()})
        old = cache.get(f)
        if old is None: frames[f] = new
        else:
            comb = old.combine_first(new) if new.empty else new.combine_first(old)
            # rescale adjusted history if new dividends changed the adjustment factor
            if f == "adjclose" and not new.empty:
                for s in new.columns:
                    if s in old.columns:
                        ov = old[s].dropna(); nv = new[s].dropna(); common = ov.index.intersection(nv.index)
                        if len(common) and ov[common[0]]:
                            k = nv[common[0]] / ov[common[0]]
                            olds = old[s][old.index < nv.index.min()] * k
                            comb[s] = pd.concat([olds, nv]).reindex(comb.index)
            frames[f] = comb
    if frames["close"].shape[1]:
        _save_cache(frames)
    log(f"[data] fetched {len(got)} symbols ({used}); failed {len(failed)}: {failed[:15]}")
    return frames, failed

def add_histories(got, log=print):
    """Merge externally fetched histories ({sym: DataFrame[open, close, adjclose]}) into the price cache."""
    cache = _load_cache(); frames = {}
    end = _trim_sparse_tail(cache)["adjclose"].index.max() if "adjclose" in cache else None
    for f in ("open", "close", "adjclose"):
        new = pd.DataFrame({s: d[f] for s, d in got.items()})
        if end is not None: new = new[new.index <= end]   # never add dates the main source does not have yet
        frames[f] = cache[f].combine_first(new) if f in cache else new
    _save_cache(frames); log(f"[data] merged {len(got)} extra histories")

def load_prices():
    c = _load_cache()
    if "close" not in c: raise RuntimeError("no price cache; run update first")
    return _trim_sparse_tail(c)

def _trim_sparse_tail(c):
    """Drop trailing dates that only a few stocks have (e.g. a date added by a secondary source)."""
    n = c["adjclose"].notna().sum(axis=1)
    if not len(n): return c
    full = n.rolling(20, min_periods=1).median().shift(1).fillna(n.max())
    keep = len(n)
    while keep > 1 and n.iloc[keep - 1] < 0.5 * full.iloc[keep - 1]: keep -= 1
    if keep == len(n): return c
    idx = n.index[:keep]
    return {k: (v.loc[idx] if isinstance(v, pd.DataFrame) and v.index.equals(n.index) else v) for k, v in c.items()}
