"""Builds site/index.html + site/data.json (interactive dashboard) and site/weekly_report.md."""
import os, json, math, datetime as dt
import numpy as np, pandas as pd
from .config import CFG, DOCS_DIR, DATA_DIR
from . import engine as E, ledger as L

SHORT = {"rev5": "Weak last week (bounce-back candidate)", "rev20": "Weak last month (bounce-back candidate)",
         "mom12_1": "Strong 12-month trend", "mom6_1": "Strong 6-month trend", "lowvol": "Calm, steady price",
         "hi52": "Near its 52-week high", "lowbeta": "Less sensitive to market swings", "secrev5": "Lagged its sector last week"}

def _r(x, n=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return None
    return round(float(x), n)

def regime_text(vix):
    if vix is None: return ("Unknown", "")
    if vix < 15: return ("Calm", "Volatility is low; markets are relaxed.")
    if vix < 22: return ("Normal", "Volatility is in its usual range.")
    if vix < 30: return ("Nervous", "Volatility is elevated; expect bigger daily swings.")
    return ("Stressed", "Volatility is high; one-week predictions are least reliable now.")

def build(px, rep, live, evaluation, sims, ic, cal, status, names, sectors, entry, exit_, last, uni, daily=None, daily_all=None):
    os.makedirs(DOCS_DIR, exist_ok=True)
    A = px["adjclose"]
    vix = float(px["close"]["^VIX"].dropna().iloc[-1]) if "^VIX" in px["close"] else None
    vix_prev = float(px["close"]["^VIX"].dropna().iloc[-6]) if "^VIX" in px["close"] else None
    reg, reg_txt = regime_text(vix)
    spy = A[CFG.benchmark]
    # weekly price series for drill-down (52 weeks)
    wk = A.resample("W-FRI").last().iloc[-53:]
    spy_w = (wk[CFG.benchmark] / wk[CFG.benchmark].iloc[0]).round(4).tolist()
    weeks = [d.strftime("%Y-%m-%d") for d in wk.index]
    # rank history (last 26 replay weeks + live)
    rp = rep.copy(); rp["pct"] = rp.groupby("date")["ens"].rank(pct=True, ascending=False)
    hist_dates = sorted(rp.date.unique())[-26:]
    piv = rp[rp.date.isin(hist_dates)].pivot(index="date", columns="ticker", values="pct")
    stocks = []
    for t, r in live.iterrows():
        drivers = sorted([(k, r[f"f_{k}"]) for k in E.FACTORS if not pd.isna(r[f"f_{k}"])], key=lambda x: -x[1])
        why = [SHORT[k] for k, v in drivers[:2] if v >= 0.7]
        ser = wk[t] if t in wk.columns else None
        stocks.append({
            "t": t, "n": r["name"], "s": r["sector"], "rk": int(r["rank"]), "sc": _r(r["ens"], 4), "b": r["bucket"], "l": r["label"],
            "x": _r(r["exp_excess"], 5), "p": _r(r["p_beat"], 3), "port": bool(r["in_portfolio"]), "a": r["action"],
            "c": _r(r["last_close"], 2), "why": why,
            "f": {k: _r(r[f"f_{k}"], 3) for k in E.FACTORS},
            "m": {"gbm": _r(live["gbm"].rank(pct=True)[t], 3), "ridge": _r(live["ridge"].rank(pct=True)[t], 3), "blend": _r(live["blend"].rank(pct=True)[t], 3)},
            "px": None if ser is None or ser.isna().all() else [_r(v, 4) for v in (ser / ser.dropna().iloc[0])],
            "h": [None if (t not in piv.columns or pd.isna(piv.loc[d, t])) else _r(1 - piv.loc[d, t], 3) for d in hist_dates],
            "r1w": _r(A[t].iloc[-1] / A[t].iloc[-6] - 1, 4) if t in A.columns else None,
        })
    # replay cumulative (chosen config) vs benchmark
    ch = sims["chosen"]; cum = (1 + ch.net).cumprod()
    cum_s = [[d.strftime("%Y-%m-%d"), _r(v - 1, 4)] for d, v in cum.items()]
    ic26 = ic.rolling(26).mean().dropna()
    # sealed forecasts & grades
    grades = L.grades(); fcs = []
    for r in L.forecasts():
        g = grades.get(r["id"])
        fcs.append({"id": r["id"], "decision": r["decision_date"], "entry": r.get("entry_date"), "exit": r["exit_date"],
                    "model": r.get("model_version", ""), "conv": r.get("convention"), "sha": r["file_sha256"][:12],
                    "status": "graded" if g else ("open" if pd.Timestamp(r["exit_date"]) >= last else "awaiting data"),
                    "ic": _r(g["rank_ic"], 3) if g else None, "port": _r(g["portfolio_excess"], 4) if g else None,
                    "hit": _r(g["portfolio_hit_rate"], 2) if g else None, "bench": _r(g["benchmark_return"], 4) if g else None,
                    "best": g.get("best") if g else None, "worst": g.get("worst") if g else None})
    # running result of the latest sealed weekly forecast (updated after every close)
    running = None
    sysf = [r for r in L.forecasts() if r.get("convention") == "next_open"]
    if sysf:
        fr = sysf[-1]; en = pd.Timestamp(fr["entry_date"]); ex = pd.Timestamp(fr["exit_date"])
        if en in A.index and en <= A.index.max():
            upto = min(A.index.max(), ex)
            O, C = px["open"], px["close"]
            p0 = O.loc[en] * (A.loc[en] / C.loc[en]); rr = A.loc[upto] / p0 - 1
            picks = [t for t in fr.get("portfolio", []) if t in rr.index and not pd.isna(rr[t])]
            if picks:
                running = {"id": fr["id"], "entry": str(en.date()), "exit": str(ex.date()), "asof": str(upto.date()), "done": bool(upto >= ex),
                           "picks": {t: _r(rr[t], 4) for t in picks}, "avg": _r(float(rr[picks].mean()), 4), "bench": _r(float(rr[CFG.benchmark]), 4)}
    port = live[live.in_portfolio]
    sells = live[live.action == "SELL"].index.tolist()
    plan_entry, plan_exit, plan_port = entry, exit_, port.index.tolist()
    # between weekly seals, the plan shown is the sealed one (not today's re-ranking)
    if sysf and pd.Timestamp(sysf[-1]["exit_date"]) >= A.index.max() and sysf[-1]["decision_date"] != str(last.date()):
        fr = sysf[-1]
        try:
            fc = pd.read_csv(os.path.join(os.path.dirname(L.LEDGER), fr["file"])).set_index("ticker")
            plan_port = [t for t in fr.get("portfolio", []) if t in fc.index]
            acts = fc["action"].fillna("").to_dict()
            for st in stocks:
                st["port"] = st["t"] in plan_port
                st["a"] = acts.get(st["t"], "") if st["port"] or acts.get(st["t"], "") == "SELL" else ""
            sells = [t for t, a in acts.items() if a == "SELL"]
            plan_entry, plan_exit = pd.Timestamp(fr["entry_date"]), pd.Timestamp(fr["exit_date"])
        except Exception as e:
            print(f"[dashboard] could not load sealed plan: {e}")
    ev = evaluation; pc = ev["portfolio"]["chosen"]
    data = {
        "generated": status["run_utc"], "asof": str(last.date()), "entry": str(plan_entry.date()), "exit": str(plan_exit.date()),
        "regime": {"name": reg, "text": reg_txt, "vix": _r(vix, 1), "vix_prev": _r(vix_prev, 1),
                   "spy_1w": _r(spy.iloc[-1] / spy.iloc[-6] - 1, 4), "spy_1m": _r(spy.iloc[-1] / spy.iloc[-22] - 1, 4)},
        "seal": status["steps"].get("seal"), "ks": status["kill_switches"],
        "eval": {"weeks": ev["weeks"], "first": ev["first"], "last": ev["last"], "ic": _r(ev["ic"]["ic_mean"], 4), "t": _r(ev["ic"]["t"], 2),
                 "pctpos": _r(ev["ic"]["pct_pos"], 3), "p_ic_pos": _r(ev["ic"]["p_ic_pos"], 3), "weeks_needed": _r(ev["ic"]["weeks_needed"], 0),
                 "band13": [_r(x, 3) for x in ev["ic"]["band_13"]], "dsr": _r(ev["deflated_sharpe_prob"], 3),
                 "gross": _r(pc["gross_wk"], 5), "net": _r(pc["net_wk"], 5), "turn": _r(pc["turnover"], 3), "ann": _r(pc["net_ann"], 4),
                 "be": _r(pc["breakeven_oneway"], 5), "hitw": _r(pc["hit_weeks"], 3), "dd": _r(ev["max_drawdown_chosen"], 3),
                 "byyear": {k: _r(v, 4) for k, v in ev["ic_by_year"].items()}, "cost": _r(ev["cost_oneway"], 5),
                 "alts": {k: {"net": _r(v["net_ann"], 4), "turn": _r(v["turnover"], 3)} for k, v in ev["portfolio"].items()},
                 "league": {k: {kk: _r(vv, 4) for kk, vv in v.items()} for k, v in ev.get("league", {}).items()}},
        "cal": [{"b": b, "m": _r(v["mean_excess"], 5), "p": _r(v["p_beat"], 3)} for b, v in ev["calibration"].items()],
        "cum": cum_s, "ic26": [[d.strftime("%Y-%m-%d"), _r(v, 4)] for d, v in ic26.items()],
        "weeks": weeks, "spyw": spy_w, "hist_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in hist_dates],
        "stocks": stocks, "portfolio": plan_port, "sells": sells, "forecasts": fcs,
        "cfg": {"n": CFG.portfolio_n, "k": CFG.portfolio_hold_k, "model": CFG.model_version},
        "health": status["steps"]["data"],
        "running": running,
        "daily": [{"d": r["date"].strftime("%Y-%m-%d"), "p": r["picks"], "x": r["exit"].strftime("%Y-%m-%d") if "exit" in r else None,
                   "v": _r(1000 * (1 + r["ret"]), 2) if "ret" in r else None, "b": _r(1000 * (1 + r["bench"]), 2) if "bench" in r else None,
                   "e": r.get("each")} for r in (daily or [])],   # e = top-10 picks in rank order with their returns
    }
    # blind daily hindsight: one small file per stock (loaded when the stock is opened) + daily summary
    if daily_all is not None and len(daily_all):
        sd = os.path.join(DOCS_DIR, "stocks"); os.makedirs(sd, exist_ok=True)
        da = daily_all.copy(); da["d"] = da["date"].dt.strftime("%Y-%m-%d")
        for t, g in da.groupby("ticker"):
            if not str(t).replace("-", "").isalnum(): continue
            json.dump({"t": t, "d": g["d"].tolist(), "r": g["rank"].astype(int).tolist(),
                       "p": [_r(x, 5) for x in g["proj"]], "a": [_r(x, 5) for x in g["actual"]]},
                      open(os.path.join(sd, f"{t}.json"), "w"), separators=(",", ":"))
        dk = da.dropna(subset=["actual"])
        dic = dk.groupby("d").apply(lambda x: x["pct"].corr(x["actual"], method="spearman") * -1)
        hit = dk.assign(ok=np.sign(dk["proj"]) == np.sign(dk["actual"])).groupby("d")["ok"].mean()
        data["hind"] = {"days": int(dk["d"].nunique()), "rows": int(len(dk)), "ic": _r(dic.mean(), 4), "pos": _r((dic > 0).mean(), 3),
                        "dir_hit": _r(hit.mean(), 3), "series": [[k, _r(v, 3)] for k, v in dic.items()]}
    tpl = open(os.path.join(os.path.dirname(__file__), "dashboard_template.html"), encoding="utf-8").read()
    open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8").write(tpl)
    def _clean(o):
        if isinstance(o, float): return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
        return o
    data = _clean(data)
    json.dump(data, open(os.path.join(DOCS_DIR, "data.json"), "w"), separators=(",", ":"), allow_nan=False)
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    os.makedirs(DATA_DIR, exist_ok=True); open(os.path.join(DATA_DIR, "dashboard_offline.html"), "w", encoding="utf-8").write(html)
    _report(data)
    return data

def _pct(x, d=1, sign=True):
    return "n/a" if x is None else (f"{x*100:+.{d}f}%" if sign else f"{x*100:.{d}f}%")

def _report(d):
    e = d["eval"]; st = {s["t"]: s for s in d["stocks"]}
    lines = [f"# S&P 500 Oracle: weekly plan", "",
             f"**Data as of:** {d['asof']} close · **Holding window:** buy at the open on {d['entry']}, sell at the close on {d['exit']}",
             f"**Market mood:** {d['regime']['name']} (VIX {d['regime']['vix']}) · {d['regime']['text']}", "",
             "## This week's 10 picks (equal amounts)", "", "| # | Stock | Sector | Action | Main signals |", "|---|---|---|---|---|"]
    for i, t in enumerate(d["portfolio"], 1):
        s = st[t]; lines.append(f"| {i} | **{t}** {s['n']} | {s['s']} | {s['a']} | {'; '.join(s['why']) or 'Model blend'} |")
    if d["sells"]: lines += ["", f"**Sell (left the top {d['cfg']['k']}):** " + ", ".join(d["sells"])]
    lines += ["", "## How much to trust this", "",
              f"- Over {e['weeks']} replayed weeks ({e['first']} to {e['last']}), this plan beat the S&P 500 by {_pct(e['gross'],2)} a week on average before costs ({_pct(e['net'],2)} after costs at a $0-commission broker).",
              f"- It beat the index in {_pct(e['hitw'],0,False)} of weeks. The worst stretch was {_pct(e['dd'],0)} behind the index.",
              f"- Probability the edge is real (not luck): about {_pct(e['p_ic_pos'],0,False)}. Statistically proving it would take ~{int(e['weeks_needed'] or 0)} weeks of results.",
              "", "_Research output, not financial advice. Paper-trade first._"]
    open(os.path.join(DOCS_DIR, "weekly_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
