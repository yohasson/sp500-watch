"""Pre-registered improvement search (written 25 Nov 2026, before any of these variants were run).

Protocol
- Every decision week from the walk-forward replay is split into DEVELOPMENT (all weeks before the final 52)
  and HOLDOUT (the final 52 weeks). Choices are made on development weeks only.
- Stage 1: each single change below is compared with the baseline on development weeks.
- Stage 2: from each group, the change with the best development result after costs is kept if it beats the
  baseline there; kept changes are combined automatically (no hand-picking).
- The combined model is adopted only if, on the HOLDOUT year, (a) its result after costs is at least the
  baseline's, (b) its rank accuracy is at least the baseline's and above zero, and (c) on development weeks its
  result after costs beats the baseline. The number of variants tried is recorded for the overfitting check.
All results are written, including the ones that lose."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle import universe as U, data as D, engine as E, pipeline as P
from oracle.config import CFG, DATA_DIR

GROUPS = {
    "target":   {"sector_target": {"target": "sector"}},
    "inputs":   {"extra_price": {"extra_features": True}, "fund": {"fundamentals": True}, "macro": {"macro": True},
                 "fund+extra": {"fundamentals": True, "extra_features": True}},
    "ensemble": {"blend_only": {"ensemble": "blend"}, "gbm_only": {"ensemble": "gbm"}, "ridge_only": {"ensemble": "ridge"},
                 "gbm+ridge": {"ensemble": "gbm+ridge"}},
    "size":     {"top20": {"portfolio_n": 20, "max_per_sector": 5}, "top30": {"portfolio_n": 30, "max_per_sector": 7}},
    "holding":  {"hold_top100": {"portfolio_hold_k": 100}, "hold_top25": {"portfolio_hold_k": 25}},
    "gate":     {"gate26": {"_gate": 26}, "gate52": {"_gate": 52}},
}
MODEL_KEYS = ("extra_features", "target", "ensemble", "fundamentals", "macro")
HOLD_WEEKS = 52

def metrics(rep, sim, weeks):
    ic = E.ic_series(rep); ic = ic[ic.index.isin(weeks)]; s = sim[sim.index.isin(weeks)]
    sd = s.net.std()
    return {"weeks": int(len(s)), "ic": float(ic.mean()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 2 else None,
            "net_ann": float((1 + s.net.mean()) ** 52 - 1), "sharpe": float(s.net.mean() / sd * np.sqrt(52)) if sd > 0 else 0.0,
            "turnover": float(s.turnover.mean()), "invested": float(s["in"].mean()) if "in" in s else 1.0,
            "hit": float((s.net > 0).mean())}

def main():
    uni = U.load_universe(refresh=True)
    cur, rem = U.all_tickers(uni, "2018-01-01")
    D.update_prices(cur + rem + [CFG.benchmark, "RSP", "^VIX"], CFG.history_years)
    px = D.load_prices(); A = px["adjclose"]
    sectors = {t: v["sector"] for t, v in {**uni["removed"], **uni["members"]}.items()}
    tick = [t for t in A.columns if t in sectors]; member = U.membership_frame(uni, A.index, tick)
    extra, macro, _ = P.load_signals(px, uni, tick, update=True)
    base_cfg = {k: getattr(CFG, k) for k in MODEL_KEYS + ("portfolio_n", "portfolio_hold_k", "max_per_sector")}
    cache = {}

    def run(ov):
        cfg = {**base_cfg, **{k: v for k, v in ov.items() if not k.startswith("_")}}
        mkey = json.dumps({k: cfg[k] for k in MODEL_KEYS}, sort_keys=True)
        if mkey not in cache:
            for k in MODEL_KEYS: setattr(CFG, k, cfg[k])
            cache[mkey] = E.Replay(px, sectors, member, CFG, extra=extra, macro=macro).run(log=lambda *a: None)
        rep = cache[mkey]
        sim = E.simulate(rep, cfg["portfolio_n"], cfg["portfolio_hold_k"], CFG.oneway_cost(), sectors=sectors,
                         cap=cfg["max_per_sector"], gate=ov.get("_gate", 0))
        return rep, sim

    t0 = time.time(); rep, sim = run({})
    weeks = sorted(sim.index); hold = set(weeks[-HOLD_WEEKS:]); dev = set(weeks[:-HOLD_WEEKS])
    out = {"baseline": {"group": None, "ov": {}, "dev": metrics(rep, sim, dev), "hold": metrics(rep, sim, hold)}}
    print("baseline", json.dumps(out["baseline"]), round(time.time() - t0), flush=True)
    for g, vs in GROUPS.items():
        for name, ov in vs.items():
            t1 = time.time(); r, s = run(ov)
            out[name] = {"group": g, "ov": ov, "dev": metrics(r, s, dev), "hold": metrics(r, s, hold)}
            print(name, json.dumps(out[name]), round(time.time() - t1), flush=True)
    b = out["baseline"]["dev"]; combo = {}; kept = []
    for g, vs in GROUPS.items():
        best = max(vs, key=lambda n: out[n]["dev"]["net_ann"])
        if out[best]["dev"]["net_ann"] > b["net_ann"] and out[best]["dev"]["ic"] >= b["ic"] - 0.002:
            combo.update(vs[best]); kept.append(best)
    if kept:
        r, s = run(combo)
        out["combined"] = {"group": "combined", "ov": combo, "dev": metrics(r, s, dev), "hold": metrics(r, s, hold)}
    else:
        out["combined"] = {"group": "combined", "ov": {}, "dev": out["baseline"]["dev"], "hold": out["baseline"]["hold"]}
    c, bh = out["combined"], out["baseline"]["hold"]
    adopt = bool(kept) and c["hold"]["net_ann"] >= bh["net_ann"] and c["hold"]["ic"] >= bh["ic"] and c["hold"]["ic"] > 0 \
        and c["dev"]["net_ann"] > b["net_ann"]
    res = {"run_utc": pd.Timestamp.utcnow().isoformat(), "holdout_start": str(min(hold).date()), "holdout_weeks": len(hold),
           "dev_weeks": len(dev), "trials": len(out), "kept": kept, "combined": combo, "adopt": adopt, "results": out,
           "rule": __doc__.split("Protocol")[1].strip()}
    print("RESULT kept", kept, "adopt", adopt, json.dumps(out["combined"]), flush=True)
    json.dump(res, open(os.path.join(DATA_DIR, "search.json"), "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
