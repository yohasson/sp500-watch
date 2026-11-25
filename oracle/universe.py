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
CACHE = os.path.join(DATA_DIR, "universe.json")
UA = {"User-Agent": "Mozilla/5.0 (sp500-oracle research; contact via GitHub)"}

def _norm(t):
    return str(t).strip().upper().replace(".", "-")

def fetch_universe() -> dict:
    html = requests.get(URL, headers=UA, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    cur = tables[0]
    cur.columns = [str(c) for c in cur.columns]
    members = {}
    for _, r in cur.iterrows():
        t = _norm(r["Symbol"])
        added = pd.to_datetime(str(r.get("Date added", ""))[:10], errors="coerce")
        members[t] = {"name": str(r["Security"]), "sector": str(r["GICS Sector"]),
                      "sub_industry": str(r.get("GICS Sub-Industry", "")),
                      "start": None if pd.isna(added) else added.strftime("%Y-%m-%d"), "end": None}
    removed = {}
    if len(tables) > 1:
        ch = tables[1]
        ch.columns = ["_".join([str(x) for x in c]) if isinstance(c, tuple) else str(c) for c in ch.columns]
        dcol = [c for c in ch.columns if c.lower().startswith("date") or "effective" in c.lower()][0]
        rcol = [c for c in ch.columns if c.lower().startswith("removed") and "ticker" in c.lower()]
        ncol = [c for c in ch.columns if c.lower().startswith("removed") and "security" in c.lower()]
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
    uni = {"fetched_utc": dt.datetime.utcnow().isoformat(timespec="seconds"), "source": URL,
           "members": members, "removed": removed}
    os.makedirs(DATA_DIR, exist_ok=True)
    json.dump(uni, open(CACHE, "w"), indent=0)
    return uni

def load_universe(refresh: bool = True) -> dict:
    if refresh:
        try:
            return fetch_universe()
        except Exception as e:  # network failure -> use cache
            print(f"[universe] refresh failed ({e}); using cache")
    return json.load(open(CACHE))

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
