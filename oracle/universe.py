"""S&P 500 membership with dates (point-in-time eligibility).

Source: Wikipedia "List of S&P 500 companies" (current members with date added,
plus the "Selected changes" table with additions and removals). Cached to data/universe.json.
Limitation: Wikipedia's change table is incomplete for older years; removed tickers whose
price history is no longer available from the price provider are skipped.
"""
import io, json, os, re, datetime as dt
import pandas as pd, requests
from .config import DATA_DIR

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
URL2 = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
URL3 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies_(changes)"
CACHE = os.path.join(DATA_DIR, "universe.json")
UA = {"User-Agent": "Mozilla/5.0 (sp500-oracle research; contact via GitHub)"}

def _norm(t):
    return str(t).strip().upper().replace(".", "-")

def fetch_universe() -> dict:
    html = requests.get(URL, headers=UA, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    flat = lambda c: "_".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
    cur = next(t for t in tables if {"Symbol", "GICS Sector"} <= {flat(c) for c in t.columns})
    cur.columns = [flat(c) for c in cur.columns]
    members = {}
    for _, r in cur.iterrows():
        t = _norm(r["Symbol"])
        added = pd.to_datetime(str(r.get("Date added", ""))[:10], errors="coerce")
        members[t] = {"name": str(r.get("Security", t)), "sector": str(r["GICS Sector"]),
                      "sub_industry": str(r.get("GICS Sub-Industry", "")),
                      "start": None if pd.isna(added) else added.strftime("%Y-%m-%d"), "end": None}
    removed = {}
    try:
        cands = [t for t in tables if any("removed" in flat(c).lower() for c in t.columns)]
        if not cands:   # the change log moved to a separate article
            for u in (URL2, URL3):
                try:
                    more = pd.read_html(io.StringIO(requests.get(u, headers=UA, timeout=30).text))
                    cands = [t for t in more if any("removed" in flat(c).lower() for c in t.columns)]
                    if cands: break
                except Exception as e:
                    print(f"[universe] {u}: {e!r}")
        ch = max(cands, key=len) if cands else next(iter(()))
        ch.columns = [flat(c) for c in ch.columns]
        dcol = next(c for c in ch.columns if "date" in c.lower())
        rcol = [c for c in ch.columns if "removed" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())]
        ncol = [c for c in ch.columns if "removed" in c.lower() and ("security" in c.lower() or "name" in c.lower())]
        for _, r in ch.iterrows():
            d = pd.to_datetime(str(r[dcol]), errors="coerce")
            if pd.isna(d) or not rcol: continue
            t = r[rcol[0]]
            if not isinstance(t, str) or not t.strip() or t.strip().lower() == "nan": continue
            t = _norm(t)
            if t in members: continue   # removed earlier, re-added later: keep current membership
            if t not in removed or d.strftime("%Y-%m-%d") > removed[t]["end"]:
                removed[t] = {"name": str(r[ncol[0]]) if ncol else t, "sector": "Unknown", "sub_industry": "",
                              "start": None, "end": d.strftime("%Y-%m-%d")}
    except StopIteration:
        print("[universe] change table not found on any page; removed stocks skipped. Tables seen: " + " | ".join(str([flat(c) for c in t.columns][:6]) for t in tables[:4]))
    print(f"[universe] {len(members)} current members, {len(removed)} removed tickers with dates")
    uni = {"fetched_utc": dt.datetime.utcnow().isoformat(timespec="seconds"), "source": URL,
           "members": members, "removed": removed}
    os.makedirs(DATA_DIR, exist_ok=True)
    json.dump(uni, open(CACHE, "w"), indent=0)
    return uni

SEED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "universe_seed.csv")

def _seed():
    df = pd.read_csv(SEED, keep_default_na=False)
    return {"fetched_utc": "seed", "source": "bundled seed (Nov 2026)", "removed": {},
            "members": {r.ticker: {"name": r.ticker, "sector": r.sector, "sub_industry": "", "start": r.start or None, "end": None} for r in df.itertuples()}}

def load_universe(refresh: bool = True) -> dict:
    if refresh:
        try:
            u = fetch_universe()
            if len(u["members"]) >= 480: return u
            print(f"[universe] only {len(u['members'])} members parsed; using cache/seed")
        except Exception as e:  # network/parse failure -> use cache, then the bundled seed
            print(f"[universe] refresh failed ({e!r}); using cache/seed")
    if os.path.exists(CACHE): return json.load(open(CACHE))
    if os.path.exists(SEED): return _seed()
    raise RuntimeError("no S&P 500 member list available (Wikipedia unreachable and no cache)")

def membership_frame(uni: dict, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Boolean frame: True where ticker was an index member on that date."""
    out = {}
    for t in tickers:
        info = uni["members"].get(t) or uni["removed"].get(t)
        if info is None:
            out[t] = pd.Series(False, index=dates); continue
        s = pd.Timestamp(info["start"]) if info.get("start") else dates[0]
        e = pd.Timestamp(info["end"]) if info.get("end") else dates[-1] + pd.Timedelta(days=1)
        out[t] = pd.Series((dates >= s) & (dates < e), index=dates)
    return pd.DataFrame(out)

def all_tickers(uni: dict, since: str):
    cur = list(uni["members"])
    rem = [t for t, v in uni["removed"].items() if v["end"] >= since]
    return cur, rem
