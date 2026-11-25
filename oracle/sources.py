"""Additional data sources, all free and reachable from GitHub runners.

- SEC EDGAR (official): company financial facts (10-K/10-Q XBRL) and filing events (8-K), used point-in-time
  by FILING date, so the replay never sees numbers before they were public.
- Wikipedia page views (Wikimedia API): investor-attention signal, lagged one day.
- FRED (St. Louis Fed): yield-curve slope and high-yield credit spread, lagged one day.
- CBOE: official VIX history.
- Nasdaq quote API: independent second source used to cross-check the latest closing prices.
- Yahoo Finance RSS: latest headlines per stock (display only; no history, so not used by the model).
- Tiingo (optional, needs a free API key in the TIINGO_API_KEY secret): price history for stocks that left the index.
"""
import os, io, re, json, time, gzip, random, datetime as dt, concurrent.futures as cf
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests
from .config import DATA_DIR

SRC = os.path.join(DATA_DIR, "src")
SEC_UA = os.environ.get("SEC_USER_AGENT") or "SP500Watch Research sp500watch.research@gmail.com"
BUA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
       "Accept": "application/json, text/plain, */*"}
WUA = {"User-Agent": "SP500Watch/1.0 (https://github.com/yohasson/sp500-watch)"}


def _p(*a): return os.path.join(SRC, *a)
def _fresh(path, hours):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < hours * 3600


class _Rate:
    def __init__(self, per_s): self.gap = 1.0 / per_s; self.t = 0.0
    def wait(self):
        now = time.time(); d = self.t + self.gap - now
        if d > 0: time.sleep(d)
        self.t = max(now, self.t + self.gap)


# ----------------------------------------------------------------------------- Wikipedia: CIK + article titles
def wiki_meta(log=print):
    """Ticker -> {cik, article} from the S&P 500 list page (current members)."""
    path = _p("wiki_meta.json")
    if _fresh(path, 20): return json.load(open(path))
    os.makedirs(SRC, exist_ok=True)
    try:
        from lxml import html as LH
        doc = LH.fromstring(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=WUA, timeout=30).content)
        tab = doc.xpath('//table[@id="constituents"]')[0]
        heads = [h.text_content().strip() for h in tab.xpath('.//tr[1]/th')]
        ci = next(i for i, h in enumerate(heads) if h.upper() == "CIK")
        out = {}
        for tr in tab.xpath('.//tr[position()>1]'):
            tds = tr.xpath('./td')
            if len(tds) <= ci: continue
            t = tds[0].text_content().strip().upper().replace(".", "-")
            a = tds[1].xpath('.//a/@href')
            art = a[0].split("/wiki/")[-1] if a and "/wiki/" in a[0] else None
            cik = re.sub(r"\D", "", tds[ci].text_content())
            out[t] = {"cik": int(cik) if cik else None, "article": art}
        json.dump(out, open(path, "w")); log(f"[src] wiki meta for {len(out)} members")
        return out
    except Exception as e:
        log(f"[src] wiki meta failed: {e!r}")
        return json.load(open(path)) if os.path.exists(path) else {}


# ----------------------------------------------------------------------------- SEC EDGAR
CONCEPTS = {
    "ni": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"), ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic")],
    "rev": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"), ("us-gaap", "SalesRevenueNet"),
            ("us-gaap", "RevenuesNetOfInterestExpense")],
    "gp": [("us-gaap", "GrossProfit")],
    "cogs": [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsAndServicesSold"), ("us-gaap", "CostOfGoodsSold")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "assets": [("us-gaap", "Assets")],
    "equity": [("us-gaap", "StockholdersEquity"), ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")],
}
INSTANT = {"assets", "equity", "shares"}


def _sec_get(url, rate, tries=3):
    for k in range(tries):
        rate.wait()
        try:
            r = requests.get(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}, timeout=40)
            if r.status_code == 200: return r.json()
            if r.status_code == 404: return None
            if r.status_code in (403, 429): time.sleep(2 + 3 * k)
        except Exception:
            time.sleep(1 + k)
    return None


def _extract_facts(js, t):
    rows = []
    facts = (js or {}).get("facts", {})
    for key, cands in CONCEPTS.items():
        for tax, tag in cands:
            node = facts.get(tax, {}).get(tag)
            if not node: continue
            units = node.get("units", {})
            vals = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
            for v in vals:
                form = v.get("form", "")
                if not (form.startswith("10-K") or form.startswith("10-Q")) or form.endswith("/A"): continue
                rows.append((t, key, v.get("start"), v["end"], float(v["val"]), v["filed"], tag))
            # keep every candidate tag: companies switch tags over time; dedupe later by period (first filed wins)
    return rows


def _extract_events(js, t, rate):
    out = []
    def take(f):
        for form, d in zip(f.get("form", []), f.get("filingDate", [])):
            if form in ("8-K", "8-K/A"): out.append((t, d))
    if not js: return out
    take(js.get("filings", {}).get("recent", {}))
    for extra in js.get("filings", {}).get("files", []):
        if extra.get("filingTo", "9999") < "2018-01-01": continue
        e = _sec_get("https://data.sec.gov/submissions/" + extra["name"], rate)
        if e: take(e)
    return out


def sec_update(meta, log=print, hours=20, workers=6):
    """Refresh SEC facts and 8-K events for all members with a CIK (at most once per `hours`)."""
    fpath, epath = _p("sec_facts.csv.gz"), _p("sec_events.csv.gz")
    if _fresh(fpath, hours) and _fresh(epath, hours):
        return pd.read_csv(fpath), pd.read_csv(epath)
    os.makedirs(SRC, exist_ok=True)
    rate = _Rate(8)   # SEC fair-access limit is 10 requests/second
    todo = [(t, m["cik"]) for t, m in meta.items() if m.get("cik")]
    facts, events, ok = [], [], 0
    t0 = time.time()
    def job(x):
        t, cik = x; c = f"{int(cik):010d}"
        fj = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c}.json", rate)
        sj = _sec_get(f"https://data.sec.gov/submissions/CIK{c}.json", rate)
        return t, (_extract_facts(fj, t) if fj else None), (_extract_events(sj, t, rate) if sj else None)
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, (t, fr, er) in enumerate(ex.map(job, todo)):
            if fr: facts += fr; ok += 1
            if er: events += er
            if i % 100 == 0: log(f"[sec] {i}/{len(todo)} companies, {time.time()-t0:.0f}s")
    if ok < 0.5 * max(len(todo), 1):
        log(f"[sec] only {ok}/{len(todo)} companies fetched; keeping the previous cache")
        if os.path.exists(fpath): return pd.read_csv(fpath), pd.read_csv(epath)
        if not facts: return None, None
    F = pd.DataFrame(facts, columns=["t", "k", "start", "end", "val", "filed", "tag"])
    Ev = pd.DataFrame(events, columns=["t", "date"]).drop_duplicates()
    F.to_csv(fpath, index=False, compression="gzip"); Ev.to_csv(epath, index=False, compression="gzip")
    log(f"[sec] facts for {ok}/{len(todo)} companies ({len(F):,} values), {len(Ev):,} 8-K filings, {time.time()-t0:.0f}s")
    return F, Ev


def _quarters(g):
    """Quarterly values (end, val, avail) from duration facts: direct ~3-month facts + differences of year-to-date facts."""
    g = g.dropna(subset=["start"]).copy()
    g["start"] = pd.to_datetime(g["start"]); g["end"] = pd.to_datetime(g["end"]); g["filed"] = pd.to_datetime(g["filed"])
    g = g.sort_values("filed").drop_duplicates(["start", "end"], keep="first")      # first publication = point in time
    g["dur"] = (g["end"] - g["start"]).dt.days
    q = {}
    for _, r in g[(g.dur >= 80) & (g.dur <= 100)].iterrows():
        q[r.end] = (r.val, r.filed)
    for s, grp in g[g.dur >= 80].groupby("start"):
        grp = grp.sort_values("end"); prev = None
        for _, r in grp.iterrows():
            if prev is not None and 80 <= (r.end - prev.end).days <= 100 and r.end not in q:
                q[r.end] = (r.val - prev.val, max(r.filed, prev.filed))
            prev = r
    for _, r in g[(g.dur >= 350) & (g.dur <= 380)].iterrows():   # Q4 = full year minus the three reported quarters
        if r.end in q: continue
        inside = [(e, v) for e, v in q.items() if r.start < e < r.end]
        if len(inside) == 3:
            q[r.end] = (r.val - sum(v for _, (v, _f) in inside), max([r.filed] + [f for _, (_v, f) in inside]))
    if not q: return pd.DataFrame(columns=["val", "avail"])
    out = pd.DataFrame([(e, v, f) for e, (v, f) in q.items()], columns=["end", "val", "avail"]).sort_values("end").set_index("end")
    return out


def _ttm(qd):
    """Trailing-12-month sum at each quarter end where 4 consecutive quarters exist."""
    if len(qd) < 4: return pd.DataFrame(columns=["val", "avail"])
    ends = qd.index; rows = []
    for i in range(3, len(ends)):
        span = (ends[i] - ends[i - 3]).days
        if 250 <= span <= 290:
            rows.append((ends[i], qd["val"].iloc[i - 3:i + 1].sum(), qd["avail"].iloc[i - 3:i + 1].max()))
    return pd.DataFrame(rows, columns=["end", "val", "avail"]).set_index("end")


def _instant(g):
    g = g.copy(); g["end"] = pd.to_datetime(g["end"]); g["filed"] = pd.to_datetime(g["filed"])
    g = g.sort_values("filed").drop_duplicates(["end"], keep="first").sort_values("end")
    return pd.DataFrame({"val": g["val"].values, "avail": g["filed"].values}, index=g["end"].values)


def _asof_daily(points, idx):
    """Value known on each trading day: a point becomes usable the trading day AFTER it was filed."""
    if points is None or not len(points): return pd.Series(np.nan, index=idx)
    p = points.sort_values("avail")
    s = pd.Series(p["val"].values, index=pd.to_datetime(p["avail"].values) + pd.Timedelta(days=1))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def fundamental_features(F, Ev, px, stocks, log=print):
    """Daily point-in-time features (rows = trading days, cols = tickers). NaN where unknown."""
    A = px["adjclose"]; C = px["close"]; idx = A.index
    out = {k: pd.DataFrame(np.nan, index=idx, columns=stocks) for k in ("ey", "bm", "gpa", "accr", "revg", "sue")}
    out["evt"] = pd.DataFrame(0.0, index=idx, columns=stocks)
    if F is None or not len(F): return out
    n = 0
    bad = 0
    for t, g in F.groupby("t"):
        if t not in stocks or t not in A.columns: continue
        try:
            _one_company(t, g, out, A, C, idx); n += 1
        except Exception as e:
            bad += 1
            if bad <= 3: log(f"[src] fundamentals skipped for {t}: {e!r}")
    if bad: log(f"[src] fundamentals skipped for {bad} companies")
    _events(Ev, out, stocks, idx)
    for k, v in out.items():
        out[k] = v.replace([np.inf, -np.inf], np.nan)
    log(f"[src] fundamentals computed for {n} companies")
    return out


def _one_company(t, g, out, A, C, idx):
    if True:
        q = {k: _quarters(g[g.k == k]) for k in ("ni", "rev", "gp", "cogs", "cfo")}
        inst = {k: _instant(g[g.k == k]) for k in INSTANT}
        ttm = {k: _ttm(v) for k, v in q.items()}
        ni = _asof_daily(ttm["ni"], idx); eq = _asof_daily(inst["equity"], idx); assets = _asof_daily(inst["assets"], idx)
        sh = _asof_daily(inst["shares"], idx)
        # market cap without split errors: shares x (close/adjclose at the day the share count became usable) x adjclose today
        ratio = (C[t] / A[t])
        if len(inst["shares"]):
            sp = inst["shares"].sort_values("avail"); days = pd.DatetimeIndex(pd.to_datetime(sp["avail"].values) + pd.Timedelta(days=1)).unique()
            rpts = pd.Series(ratio.reindex(idx.union(days)).bfill().reindex(days).values, index=days)
            rpts = rpts[~rpts.index.duplicated(keep="last")]
            r_at = rpts.reindex(idx.union(rpts.index)).ffill().reindex(idx)
        else:
            r_at = pd.Series(np.nan, index=idx)
        mcap = sh * r_at * A[t]
        mcap = mcap.where(mcap > 0)
        out["ey"][t] = ni / mcap
        out["bm"][t] = eq / mcap
        gp_ttm = ttm["gp"]
        if not len(gp_ttm) and len(ttm["rev"]) and len(ttm["cogs"]):
            j = ttm["rev"].join(ttm["cogs"], lsuffix="_r", rsuffix="_c", how="inner")
            gp_ttm = pd.DataFrame({"val": j.val_r - j.val_c, "avail": j[["avail_r", "avail_c"]].max(axis=1)})
        out["gpa"][t] = _asof_daily(gp_ttm, idx) / assets.where(assets > 0)
        if len(ttm["cfo"]) and len(ttm["ni"]):
            j = ttm["ni"].join(ttm["cfo"], lsuffix="_n", rsuffix="_c", how="inner")
            acc = pd.DataFrame({"val": -(j.val_n - j.val_c), "avail": j[["avail_n", "avail_c"]].max(axis=1)})
            out["accr"][t] = _asof_daily(acc, idx) / assets.where(assets > 0)
        rv = ttm["rev"]
        if len(rv) >= 5:
            g4 = rv["val"] / rv["val"].shift(4) - 1
            ok = [(rv.index[i] - rv.index[i - 4]).days in range(350, 380) for i in range(len(rv))] if len(rv) else []
            gr = pd.DataFrame({"val": g4.where(pd.Series(ok, index=rv.index)), "avail": rv["avail"]}).dropna()
            out["revg"][t] = _asof_daily(gr, idx)
        qn = q["ni"]
        if len(qn) >= 9:
            d4 = qn["val"] - qn["val"].shift(4)
            sd = d4.rolling(8, min_periods=6).std().shift(1)
            su = pd.DataFrame({"val": (d4 / sd).clip(-5, 5), "avail": qn["avail"]}).dropna()
            s_d = _asof_daily(su, idx)
            # earnings-surprise drift: only use the surprise for ~60 trading days after the report
            last_av = _asof_daily(pd.DataFrame({"val": pd.to_datetime(su["avail"]).map(pd.Timestamp.toordinal), "avail": su["avail"]}), idx)
            age = pd.Series(idx.map(pd.Timestamp.toordinal), index=idx) - last_av
            out["sue"][t] = s_d.where(age <= 90)


def _events(Ev, out, stocks, idx):
    if Ev is not None and len(Ev):
        ev = Ev[Ev.t.isin(stocks)].copy(); ev["date"] = pd.to_datetime(ev["date"])
        cnt = ev.groupby(["date", "t"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(idx.union(cnt.index), fill_value=0)
        roll = cnt.rolling("7D").sum().reindex(idx).fillna(0)   # 8-K filings in the last 7 calendar days (filed by that day)
        out["evt"] = roll.reindex(columns=stocks).fillna(0).shift(1).fillna(0)  # usable the next trading day


# ----------------------------------------------------------------------------- Wikipedia page views (attention)
def pageviews(meta, idx, stocks, log=print, hours=20, workers=4):
    path = _p("pageviews.csv.gz")
    V = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else None
    n_art = sum(1 for m in meta.values() if m.get("article"))
    if V is None or not _fresh(path, hours) or V.shape[1] < 0.8 * n_art:      # refetch when stale or incomplete
        os.makedirs(SRC, exist_ok=True)
        start = (idx.min() - pd.Timedelta(days=90)).strftime("%Y%m%d"); end = dt.date.today().strftime("%Y%m%d")
        arts = {t: m["article"] for t, m in meta.items() if m.get("article")}
        have = set(V.columns) if V is not None else set()
        arts = dict(sorted(arts.items(), key=lambda x: x[0] in have))   # companies still missing go first
        rate = _Rate(12); deadline = time.time() + float(os.environ.get("ORACLE_PV_BUDGET", 90))
        old = V
        def job(x):
            t, a = x
            if time.time() > deadline: return t, None
            u = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{a}/daily/{start}/{end}"
            for k in range(5):
                rate.wait()
                try:
                    r = requests.get(u, headers=WUA, timeout=30)
                    if r.status_code == 429:
                        if time.time() > deadline: return t, None
                        time.sleep(1 + k); continue
                    if r.status_code == 200:
                        it = r.json().get("items", [])
                        return t, pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in it}, dtype=float)
                    if r.status_code == 404: return t, None
                except Exception: time.sleep(1)
            return t, None
        got = {}
        with cf.ThreadPoolExecutor(workers) as ex:
            for t, s in ex.map(job, arts.items()):
                if s is not None and len(s): got[t] = s
        if not got and old is None:
            log("[src] page views: nothing fetched")
            return pd.DataFrame(np.nan, index=idx, columns=stocks)
        V = pd.DataFrame(got)
        if old is not None: V = V.combine_first(old) if len(got) else old   # keep earlier coverage, add new companies
        V.to_csv(path, compression="gzip")
        log(f"[src] page views for {len(got)} companies this run, {V.shape[1]} in total (time budget {os.environ.get('ORACLE_PV_BUDGET', 90)}s)")
    V = V.sort_index()
    att = np.log((V.rolling(5, min_periods=3).mean() + 10) / (V.rolling(60, min_periods=30).mean() + 10))
    att.index = att.index + pd.Timedelta(days=1)     # daily counts are published after the day ends
    return att.reindex(idx.union(att.index)).ffill(limit=3).reindex(idx).reindex(columns=stocks)


# ----------------------------------------------------------------------------- FRED + CBOE macro
def macro(idx, log=print, hours=20):
    path = _p("macro.csv.gz")
    if _fresh(path, hours):
        M = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        os.makedirs(SRC, exist_ok=True); cols = {}
        for sid in ("T10Y2Y", "BAMLH0A0HYM2", "VIXCLS"):
            try:
                df = pd.read_csv(io.StringIO(requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", headers=WUA, timeout=30).text))
                df.columns = ["d", "v"]; cols[sid] = pd.to_numeric(df["v"], errors="coerce").set_axis(pd.to_datetime(df["d"]))
            except Exception as e: log(f"[src] FRED {sid} failed: {e!r}")
        try:
            df = pd.read_csv(io.StringIO(requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", headers=BUA, timeout=30).text))
            cols["VIX_CBOE"] = df.set_index(pd.to_datetime(df["DATE"]))["CLOSE"].astype(float)
        except Exception as e: log(f"[src] CBOE VIX failed: {e!r}")
        if not cols: return None
        M = pd.DataFrame(cols); M.to_csv(path, compression="gzip")
        log(f"[src] macro series: {list(cols)}")
    M = M.sort_index()
    lag = M.copy(); lag.index = lag.index + pd.Timedelta(days=1)   # published after the day ends
    return lag.reindex(idx.union(lag.index)).ffill(limit=5).reindex(idx)


# ----------------------------------------------------------------------------- Nasdaq price cross-check
def _num(x):
    try: return float(str(x).replace("$", "").replace(",", ""))
    except Exception: return np.nan


def crosscheck(px, tickers, must=(), n=80, log=print):
    """Compare the last 5 daily closes with Nasdaq's own history for a random sample (+ the picks)."""
    C = px["close"]; last = C.index.max()
    pool = [t for t in tickers if t in C.columns]
    sample = list(dict.fromkeys(list(must) + random.Random(int(time.time() // 86400)).sample(pool, min(n, len(pool)))))
    frm = (last - pd.Timedelta(days=12)).strftime("%Y-%m-%d"); to = last.strftime("%Y-%m-%d")
    def job(t):
        u = f"https://api.nasdaq.com/api/quote/{t.replace('-', '.')}/historical?assetclass=stocks&fromdate={frm}&todate={to}&limit=20"
        try:
            j = requests.get(u, headers=BUA, timeout=20).json()
            rows = ((j.get("data") or {}).get("tradesTable") or {}).get("rows") or []
            return t, {pd.to_datetime(r["date"]): _num(r["close"]) for r in rows}
        except Exception:
            return t, None
    res = {}
    with cf.ThreadPoolExecutor(6) as ex:
        for t, d in ex.map(job, sample): res[t] = d
    checked = agree = 0; bad = []
    for t, d in res.items():
        if not d: continue
        for day, v in d.items():
            if day in C.index and not pd.isna(C.at[day, t]) and v and not np.isnan(v):
                checked += 1
                if abs(C.at[day, t] / v - 1) <= 0.005: agree += 1
                else: bad.append(f"{t} {day.date()} {C.at[day, t]:.2f} vs {v:.2f}")
    out = {"stocks": sum(1 for d in res.values() if d), "sampled": len(sample), "points": checked,
           "agree": (agree / checked) if checked else None, "mismatches": bad[:10]}
    log(f"[check] Nasdaq cross-check: {out['stocks']}/{len(sample)} stocks, {checked} closes, agreement {out['agree']}")
    return out


# ----------------------------------------------------------------------------- headlines
def headlines(tickers, log=print, per=5, workers=8):
    def job(t):
        u = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
        try:
            r = requests.get(u, headers=BUA, timeout=15)
            root = ET.fromstring(r.content); items = []
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip(); link = (it.findtext("link") or "").strip()
                d = pd.to_datetime(it.findtext("pubDate"), errors="coerce", utc=True)
                if title and link.startswith("http"):
                    items.append({"title": title[:180], "link": link, "date": None if pd.isna(d) else d.strftime("%Y-%m-%d %H:%M")})
            items.sort(key=lambda x: x["date"] or "", reverse=True)
            return t, items[:per]
        except Exception:
            return t, None
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for t, it in ex.map(job, tickers):
            if it: out[t] = it
    log(f"[src] headlines for {len(out)}/{len(tickers)} stocks")
    return out


def recent_filings(Ev, tickers, days=30):
    if Ev is None or not len(Ev): return {}
    ev = Ev.copy(); ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[ev.date >= pd.Timestamp.today() - pd.Timedelta(days=days)]
    return {t: sorted(g["date"].dt.strftime("%Y-%m-%d").tolist(), reverse=True)[:8] for t, g in ev.groupby("t") if t in tickers}


# ----------------------------------------------------------------------------- optional: Tiingo for stocks that left the index
def tiingo_history(symbols, years, log=print):
    """Free Tiingo plan: about 50 requests/hour, so fetch at most ORACLE_TIINGO_MAX per run; filled histories
    stay in the price cache, and tickers Tiingo does not know are remembered so later runs skip them."""
    key = os.environ.get("TIINGO_API_KEY")
    if not key or not symbols: return {}
    skipf = _p("tiingo_skip.json")
    try: skip = set(json.load(open(skipf)))
    except Exception: skip = set()
    todo = [s for s in symbols if s not in skip][:int(os.environ.get("ORACLE_TIINGO_MAX", "45"))]
    start = (pd.Timestamp.today() - pd.Timedelta(days=365 * years)).strftime("%Y-%m-%d"); got = {}; codes = {}
    for s in todo:
        try:
            r = requests.get(f"https://api.tiingo.com/tiingo/daily/{s.lower().replace('.', '-')}/prices",
                             params={"startDate": start, "token": key}, timeout=30)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
            if r.status_code == 404: skip.add(s); continue
            if r.status_code in (401, 403, 429): log(f"[tiingo] stopped at {s}: HTTP {r.status_code}"); break
            if r.status_code != 200: continue
            df = pd.DataFrame(r.json())
            if not len(df): skip.add(s); continue
            df.index = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            got[s] = pd.DataFrame({"open": df["open"], "close": df["close"], "adjclose": df["adjClose"]})
            time.sleep(0.5)
        except Exception as e:
            log(f"[tiingo] {s}: {e!r}")
    try:
        os.makedirs(SRC, exist_ok=True); json.dump(sorted(skip), open(skipf, "w"))
    except Exception: pass
    log(f"[tiingo] history for {len(got)}/{len(todo)} tried ({len(symbols)} still missing, {len(skip)} unknown to Tiingo); HTTP {codes}")
    return got
