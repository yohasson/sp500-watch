"""One-off connectivity probe for candidate data sources (run on GitHub Actions)."""
import json, time, urllib.request, os
UA = {"User-Agent": "sp500-watch research yohasson@users.noreply.github.com", "Accept": "*/*"}
BUA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "Accept": "application/json, text/plain, */*"}
T = {
 "sec_tickers": ("https://www.sec.gov/files/company_tickers.json", UA),
 "sec_facts": ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json", UA),
 "sec_submissions": ("https://data.sec.gov/submissions/CIK0000320193.json", UA),
 "gdelt_doc": ("https://api.gdeltproject.org/api/v2/doc/doc?query=%22Apple%20Inc%22&mode=artlist&maxrecords=5&format=json&timespan=3d", UA),
 "nasdaq_hist": ("https://api.nasdaq.com/api/quote/AAPL/historical?assetclass=stocks&fromdate=2026-10-01&todate=2026-11-24&limit=100", BUA),
 "nasdaq_hist_delisted": ("https://api.nasdaq.com/api/quote/TWTR/historical?assetclass=stocks&fromdate=2020-01-01&todate=2022-11-01&limit=100", BUA),
 "stooq": ("https://stooq.com/q/d/l/?s=aapl.us&i=d", BUA),
 "stooq_delisted": ("https://stooq.com/q/d/l/?s=twtr.us&i=d", BUA),
 "wiki_pageviews": ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Apple_Inc./daily/20261101/20261124", UA),
 "google_news_rss": ("https://news.google.com/rss/search?q=%22Apple%22+stock&hl=en-US&gl=US&ceid=US:en", BUA),
 "yahoo_rss": ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US", BUA),
 "fred_vix": ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", UA),
 "fred_t10y2y": ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y", UA),
 "cboe_vix": ("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", BUA),
 "tiingo_nokey": ("https://api.tiingo.com/tiingo/daily/aapl/prices", UA),
}
out = {}
for k, (u, h) in T.items():
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=25); b = r.read()
        out[k] = {"ok": True, "status": r.status, "bytes": len(b), "head": b[:300].decode("utf-8", "replace"), "s": round(time.time() - t0, 1)}
    except Exception as e:
        out[k] = {"ok": False, "err": str(e)[:200], "s": round(time.time() - t0, 1)}
    print(k, out[k].get("ok"), out[k].get("bytes"), out[k].get("err", ""), flush=True)
    time.sleep(1)
try:
    import yfinance as yf
    for t in ["TWTR", "ATVI", "SIVB", "XLNX"]:
        try:
            df = yf.Ticker(t).history(period="max"); out["yf_" + t] = {"rows": len(df)}
        except Exception as e: out["yf_" + t] = {"err": str(e)[:100]}
except Exception as e: out["yf"] = str(e)
os.makedirs("data", exist_ok=True); json.dump(out, open("data/probe.json", "w"), indent=1)
