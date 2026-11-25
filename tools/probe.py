"""One-off connectivity probe for candidate data sources (run on GitHub Actions)."""
import json, time, urllib.request, os
UA = {"User-Agent": "sp500-watch research yohasson@users.noreply.github.com", "Accept": "*/*"}
BUA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "Accept": "application/json, text/plain, */*"}
T = {
 "sec_ua1": ("https://data.sec.gov/submissions/CIK0000320193.json", {"User-Agent": "SP500Watch Research sp500watch.research@gmail.com", "Accept-Encoding": "identity"}),
 "sec_ua2": ("https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/Revenues.json", {"User-Agent": "Mozilla/5.0 (compatible; SP500Watch/1.0; +https://github.com/yohasson/sp500-watch) contact sp500watch.research@gmail.com"}),
 "sec_www": ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom", {"User-Agent": "SP500Watch Research sp500watch.research@gmail.com"}),
 "sec_bulk_head": ("https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip", {"User-Agent": "SP500Watch Research sp500watch.research@gmail.com", "Range": "bytes=0-1000"}),
 "wiki_list_html": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", UA),
 "nasdaq_quote_batch": ("https://api.nasdaq.com/api/quote/watchlist?symbol=aapl%7cstocks&symbol=msft%7cstocks", BUA),
}
out = {}
for k, (u, h) in T.items():
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=25); b = r.read(2000)
        out[k] = {"ok": True, "status": r.status, "bytes": len(b), "head": b[:300].decode("utf-8", "replace"), "s": round(time.time() - t0, 1)}
    except Exception as e:
        out[k] = {"ok": False, "err": str(e)[:200], "s": round(time.time() - t0, 1)}
    print(k, out[k].get("ok"), out[k].get("bytes"), out[k].get("err", ""), flush=True)
    time.sleep(1)
if False:
    import yfinance as yf
    for t in ["TWTR", "ATVI", "SIVB", "XLNX"]:
        try:
            df = yf.Ticker(t).history(period="max"); out["yf_" + t] = {"rows": len(df)}
        except Exception as e: out["yf_" + t] = {"err": str(e)[:100]}
except Exception as e: out["yf"] = str(e)
os.makedirs("data", exist_ok=True); json.dump(out, open("data/probe.json", "w"), indent=1)
