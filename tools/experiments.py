"""Compare pre-defined model variants on the same point-in-time replay. Writes data/experiments.json.
All variants are reported (no silent cherry-picking); the number of variants feeds the overfitting check."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle import universe as U, data as D, engine as E, pipeline as P
from oracle.config import CFG, DATA_DIR

# Pre-registered adoption rule (written before the results were seen, 25 Nov 2026):
# a new-signal variant replaces the baseline only if (1) its full-period t-stat is at least 0.5 higher,
# (2) its last-2-year IC is not lower, and (3) its net yearly result after costs is not lower.
VARIANTS = {
    "baseline": {},
    "fund": {"fundamentals": True},
    "macro": {"macro": True},
    "fund+macro": {"fundamentals": True, "macro": True},
    "no_leavers (bias check)": {"_no_leavers": True},
}

def main():
    uni = U.load_universe(refresh=True)
    cur, rem = U.all_tickers(uni, "2018-01-01")
    D.update_prices(cur + rem + [CFG.benchmark, "RSP", "^VIX"], CFG.history_years)
    px = D.load_prices(); A = px["adjclose"]
    sectors = {t: v["sector"] for t, v in {**uni["removed"], **uni["members"]}.items()}
    tick = [t for t in A.columns if t in sectors]; member = U.membership_frame(uni, A.index, tick)
    extra, macro, sinfo = P.load_signals(px, uni, tick, update=True)
    print("sources", json.dumps(sinfo["sources"]), flush=True)
    cur_only = [t for t in tick if t in uni["members"]]
    member_cur = U.membership_frame(uni, A.index, cur_only)
    base = {k: getattr(CFG, k) for k in ("extra_features", "target", "ensemble", "fundamentals", "macro")}
    out = {}
    for name, ov in VARIANTS.items():
        t0 = time.time()
        for k, v in {**base, **ov}.items():
            if not k.startswith("_"): setattr(CFG, k, v)
        mem = member_cur if ov.get("_no_leavers") else member
        R = E.Replay(px, sectors, mem, CFG, extra=extra, macro=macro); rep = R.run(log=lambda *a: None)
        ic = E.ic_series(rep); sim = E.simulate(rep, CFG.portfolio_n, CFG.portfolio_hold_k, CFG.oneway_cost(), sectors=sectors, cap=CFG.max_per_sector)
        recent = ic[ic.index >= ic.index.max() - pd.Timedelta(days=730)]
        out[name] = {"overrides": ov, "weeks": int(len(ic)), "ic": float(ic.mean()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
                     "ic_last2y": float(recent.mean()), "t_last2y": float(recent.mean() / recent.std() * np.sqrt(len(recent))),
                     "net_ann": float((1 + sim.net.mean()) ** 52 - 1), "turnover": float(sim.turnover.mean()), "secs": round(time.time() - t0)}
        print(name, json.dumps(out[name]), flush=True)
    for k, v in base.items(): setattr(CFG, k, v)
    b = out["baseline"]; verdict = {}
    for k in ("fund", "macro", "fund+macro"):
        v = out[k]; verdict[k] = bool(v["t"] >= b["t"] + 0.5 and v["ic_last2y"] >= b["ic_last2y"] and v["net_ann"] >= b["net_ann"])
    nl = out["no_leavers (bias check)"]
    bias = {"leavers_with_data": len([t for t in tick if t not in uni["members"]]), "leavers_total": len([t for t, v in uni["removed"].items() if (v.get("end") or "") >= str(A.index.min().date())]),
            "ic_without": nl["ic"], "ic_with": b["ic"], "net_ann_without": nl["net_ann"], "net_ann_with": b["net_ann"]}
    print("verdict", json.dumps(verdict), "bias", json.dumps(bias), flush=True)
    json.dump({"run_utc": pd.Timestamp.utcnow().isoformat(), "variants": out, "adopt": verdict, "survivorship": bias,
               "rule": "adopt only if t >= baseline t + 0.5, last-2y IC >= baseline, net/yr >= baseline"}, open(os.path.join(DATA_DIR, "experiments.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
