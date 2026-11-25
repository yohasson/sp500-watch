"""Compare pre-defined model variants on the same point-in-time replay. Writes data/experiments.json.
All variants are reported (no silent cherry-picking); the number of variants feeds the overfitting check."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle import universe as U, data as D, engine as E
from oracle.config import CFG, DATA_DIR

VARIANTS = {
    "baseline": {},
    "extra_features": {"extra_features": True},
    "sector_target": {"target": "sector"},
    "extra+sector": {"extra_features": True, "target": "sector"},
    "gbm_only": {"ensemble": "gbm"},
    "gbm+ridge": {"ensemble": "gbm+ridge"},
    "extra+gbm+ridge": {"extra_features": True, "ensemble": "gbm+ridge"},
    "ridge_only": {"ensemble": "ridge"},
    "blend_only": {"ensemble": "blend"},
    "gbm+blend": {"ensemble": "gbm+blend"},
}

def main():
    uni = U.load_universe(refresh=True)
    cur, rem = U.all_tickers(uni, "2018-01-01")
    D.update_prices(cur + rem + [CFG.benchmark, "RSP", "^VIX"], CFG.history_years)
    px = D.load_prices(); A = px["adjclose"]
    sectors = {t: v["sector"] for t, v in {**uni["removed"], **uni["members"]}.items()}
    tick = [t for t in A.columns if t in sectors]; member = U.membership_frame(uni, A.index, tick)
    base = {k: getattr(CFG, k) for k in ("extra_features", "target", "ensemble")}
    out = {}
    for name, ov in VARIANTS.items():
        t0 = time.time()
        for k, v in {**base, **ov}.items(): setattr(CFG, k, v)
        R = E.Replay(px, sectors, member, CFG); rep = R.run(log=lambda *a: None)
        ic = E.ic_series(rep); sim = E.simulate(rep, CFG.portfolio_n, CFG.portfolio_hold_k, CFG.oneway_cost(), sectors=sectors, cap=CFG.max_per_sector)
        recent = ic[ic.index >= ic.index.max() - pd.Timedelta(days=730)]
        out[name] = {"overrides": ov, "weeks": int(len(ic)), "ic": float(ic.mean()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
                     "ic_last2y": float(recent.mean()), "t_last2y": float(recent.mean() / recent.std() * np.sqrt(len(recent))),
                     "net_ann": float((1 + sim.net.mean()) ** 52 - 1), "turnover": float(sim.turnover.mean()), "secs": round(time.time() - t0)}
        print(name, json.dumps(out[name]), flush=True)
    for k, v in base.items(): setattr(CFG, k, v)
    json.dump({"run_utc": pd.Timestamp.utcnow().isoformat(), "variants": out}, open(os.path.join(DATA_DIR, "experiments.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
