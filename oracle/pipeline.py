"""End-to-end run: data -> replay -> live forecast -> seal -> grade -> dashboard."""
import os, json, datetime as dt, math, hashlib
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from .config import CFG, DATA_DIR, DOCS_DIR, LEDGER_DIR
from . import universe as U, data as D, engine as E, ledger as L
from .calendar import next_trading_days, is_last_trading_day_of_week

def _now_utc(): return dt.datetime.now(dt.timezone.utc)

def market_open_utc(day: pd.Timestamp) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, 9, 30, tzinfo=ZoneInfo("America/New_York")).astimezone(dt.timezone.utc)

def run(update=True, seal=True, log=print, offline_universe=False):
    t0 = _now_utc(); status = {"run_utc": t0.isoformat(timespec="seconds"), "steps": {}}
    uni = U.load_universe(refresh=not offline_universe)
    cur, rem = U.all_tickers(uni, (pd.Timestamp.today() - pd.Timedelta(days=365 * CFG.history_years)).strftime("%Y-%m-%d"))
    syms = cur + rem + [CFG.benchmark, "RSP", "^VIX"]
    failed = []
    if update:
        _, failed = D.update_prices(syms, CFG.history_years, log=log)
    px = D.load_prices()
    A = px["adjclose"]; last = A.index.max()
    have_cur = [t for t in cur if t in A.columns and not pd.isna(A[t].iloc[-1])]
    health = len(have_cur) / max(len(cur), 1)
    log(f"[run] data health {health:.0%} ({len(have_cur)}/{len(cur)}), last date {last.date()}, {(_now_utc()-t0).seconds}s")
    if health < 0.6 or CFG.benchmark not in A.columns:
        status["steps"]["data"] = {"health": health, "failed": failed[:30], "note": "too little data; site not rebuilt"}
        os.makedirs(DATA_DIR, exist_ok=True); json.dump(status, open(os.path.join(DATA_DIR, "status.json"), "w"), indent=1, default=str)
        log("[run] stopping early: not enough fresh prices (the next run continues the download)"); return status
    status["steps"]["data"] = {"last_date": str(last.date()), "symbols": int(A.shape[1]), "current_members_with_data": len(have_cur),
                               "current_members": len(cur), "removed_with_data": int(sum(t in A.columns for t in rem)), "health": health,
                               "failed": failed[:30]}
    sectors = {t: v["sector"] for t, v in {**uni["removed"], **uni["members"]}.items()}
    names = {t: v["name"] for t, v in {**uni["removed"], **uni["members"]}.items()}
    tick = [t for t in A.columns if t in sectors]
    member = U.membership_frame(uni, A.index, tick)
    log(f"[run] prices ready {A.shape}, {(_now_utc()-t0).seconds}s")
    R = E.Replay(px, sectors, member)
    rep = R.run(log=log)
    log(f"[run] replay done {(_now_utc()-t0).seconds}s")
    os.makedirs(DATA_DIR, exist_ok=True)
    rep.to_csv(os.path.join(DATA_DIR, "replay.csv.gz"), index=False, compression="gzip", float_format="%.5f")
    # ---- evaluation
    ic = E.ic_series(rep); ic_cc = E.ic_series(rep, target="fwd_cc")
    cost = CFG.oneway_cost()
    configs = {"chosen": (CFG.portfolio_n, CFG.portfolio_hold_k, CFG.max_per_sector), "no_sector_cap": (CFG.portfolio_n, CFG.portfolio_hold_k, None),
               "top20_weekly": (20, 20, None), "top10_weekly": (10, 10, None), "top20_hold60": (20, 60, None)}
    sims = {k: E.simulate(rep, n, kk, cost, sectors=sectors, cap=c) for k, (n, kk, c) in configs.items()}
    srs = [s.net.mean() / s.net.std() for s in sims.values()]
    cal = E.calibration(rep)
    pw = E.power_stats(ic)
    ch = sims["chosen"]
    eq = (1 + ch.net).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    trials = 42 + len(configs)
    evaluation = {"weeks": int(len(ic)), "first": str(ic.index.min().date()), "last": str(ic.index.max().date()),
                  "ic": pw, "ic_close_to_close": float(ic_cc.mean()),
                  "ic_by_year": {str(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()},
                  "portfolio": {k: {"gross_wk": float(s.gross.mean()), "net_wk": float(s.net.mean()), "turnover": float(s.turnover.mean()),
                                    "net_ann": float((1 + s.net.mean()) ** 52 - 1), "breakeven_oneway": float(s.gross.mean() / (2 * s.turnover.mean())),
                                    "hit_weeks": float((s.net > 0).mean())} for k, s in sims.items()},
                  "max_drawdown_chosen": dd, "deflated_sharpe_prob": E.deflated_sharpe(ch.net.values, trials, float(np.var(srs))), "trials": trials,
                  "calibration": {b: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()} for b, row in cal.iterrows()},
                  "cost_oneway": cost}
    league = {}
    for col, label in [("ens", "Ensemble (used)"), ("gbm", "Boosted trees"), ("ridge", "Linear model"), ("blend", "Simple signal blend")]:
        i2 = E.ic_series(rep, col=col); s2 = E.simulate(rep, CFG.portfolio_n, CFG.portfolio_hold_k, cost, col=col, sectors=sectors, cap=CFG.max_per_sector)
        league[label] = {"ic": float(i2.mean()), "t": float(i2.mean() / i2.std() * np.sqrt(len(i2))), "net_ann": float((1 + s2.net.mean()) ** 52 - 1),
                         "last52": float(i2.iloc[-52:].mean())}
    evaluation["league"] = league
    # ---- live scores
    live, ntrain = R.live(last)
    live = live.copy(); live["pct"] = live["ens"].rank(pct=True, ascending=False)
    live["rank"] = live["ens"].rank(ascending=False, method="first").astype(int)
    live["bucket"] = live["pct"].map(E.bucket_of)
    live["label"] = live["bucket"].map(lambda b: E.label_for(b, cal))
    live["exp_excess"] = live["bucket"].map(cal["mean_excess"]); live["p_beat"] = live["bucket"].map(cal["p_beat"])
    prev_port = []
    fcs = L.forecasts()
    sys_fcs = [r for r in fcs if r.get("convention") == "next_open"]
    if sys_fcs:
        pf = pd.read_csv(os.path.join(LEDGER_DIR, sys_fcs[-1]["file"]))
        prev_port = pf[pf.in_portfolio.astype(bool)].ticker.tolist()
    rk = live["rank"]
    port = E.pick_portfolio(rk, prev_port, CFG.portfolio_n, CFG.portfolio_hold_k, sectors, CFG.max_per_sector)
    live["in_portfolio"] = live.index.isin(port)
    live["action"] = ["BUY" if (s in port and s not in prev_port) else "KEEP" if s in port else "SELL" if s in prev_port else "" for s in live.index]
    for k in E.FACTORS: live[f"f_{k}"] = (R.F[k].loc[last].reindex(live.index) + 0.5)
    live["name"] = [names.get(t, t) for t in live.index]; live["sector"] = [sectors.get(t, "") for t in live.index]
    live["last_close"] = px["close"].loc[last].reindex(live.index)
    live = live.sort_values("rank")
    entry, *_, exit_ = next_trading_days(last, CFG.horizon_days, None)
    daily_all, daily = R.daily_backtest(rep, days=252, top=10)
    daily_all.to_csv(os.path.join(DATA_DIR, "hindsight_daily.csv.gz"), index=False, compression="gzip", float_format="%.5f")
    # ---- seal
    sealed = None; seal_note = ""
    open_fc = [r for r in sys_fcs if pd.Timestamp(r["exit_date"]) >= last + pd.Timedelta(days=1)]
    due = is_last_trading_day_of_week(last) or not open_fc
    already = any(r["decision_date"] == str(last.date()) for r in sys_fcs)
    if seal and due and not already:
        if _now_utc() >= market_open_utc(entry):
            seal_note = "late: market already open for the entry day; not sealed"
        elif health < 0.95:
            seal_note = f"blocked: data health {health:.0%} below 95% (KS-5)"
        else:
            cols = ["ticker", "name", "sector", "rank", "ens", "bucket", "label", "exp_excess", "p_beat", "in_portfolio", "action", "last_close"] + [f"f_{k}" for k in E.FACTORS]
            out = live.reset_index().rename(columns={"index": "ticker"})[cols].rename(columns={"ens": "score"})
            meta = {"id": f"{last.date()}_{CFG.model_version}", "decision_date": str(last.date()), "entry_date": str(entry.date()),
                    "exit_date": str(exit_.date()), "convention": "next_open", "model_version": CFG.model_version,
                    "config_sha256": CFG.hash(), "code_sha": os.environ.get("GITHUB_SHA", "local"),
                    "data_sha256": hashlib.sha256(A.loc[last].round(4).to_json().encode()).hexdigest(), "train_weeks": int(ntrain),
                    "portfolio": port}
            sealed = L.seal(out, meta); seal_note = "sealed" if sealed else "already sealed"
    elif already: seal_note = "already sealed for this decision date"
    else: seal_note = "not due (forecast already open and not the last trading day of the week)"
    status["steps"]["seal"] = seal_note
    graded = L.grade_due(px, CFG.benchmark, log=log)
    problems = L.verify()
    # ---- kill switches
    grades = L.grades(); live_ics = [g["rank_ic"] for fid, g in grades.items() if fid.endswith(CFG.model_version)]
    band26 = pw["band_26"]
    lp = [g["portfolio_excess"] for fid, g in grades.items() if fid.endswith(CFG.model_version)]
    leq = np.cumprod([1 + x for x in lp]) if lp else np.array([1.0]); live_dd = float((leq / np.maximum.accumulate(leq) - 1).min())
    ks = [
        {"id": "Data health", "ok": bool(health >= 0.95), "detail": f"{health:.0%} of current members have fresh prices"},
        {"id": "Ledger integrity", "ok": bool(not problems), "detail": "all seals verified" if not problems else "; ".join(problems)},
        {"id": "Live skill band", "ok": bool((len(live_ics) < 26) or (np.mean(live_ics[-26:]) >= band26[0])),
         "detail": f"{len(live_ics)} live weeks graded; pause if 26-week average falls below {band26[0]:+.3f}"},
        {"id": "Drawdown vs S&P 500", "ok": bool(live_dd > -0.25), "detail": f"paper portfolio {live_dd:+.1%} from its best level vs the index (pause at -25%)"},
    ]
    status["kill_switches"] = ks; status["evaluation"] = evaluation
    json.dump(status, open(os.path.join(DATA_DIR, "status.json"), "w"), indent=1, default=str)
    from . import dashboard
    dashboard.build(px=px, rep=rep, live=live, evaluation=evaluation, sims=sims, ic=ic, cal=cal, status=status,
                    names=names, sectors=sectors, entry=entry, exit_=exit_, last=last, uni=uni, daily=daily, daily_all=daily_all)
    log(f"[run] done: last data {last.date()}, seal: {seal_note}, graded {len(graded)}, ledger problems {len(problems)}")
    return status
