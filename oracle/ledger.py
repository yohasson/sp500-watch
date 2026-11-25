"""Append-only, hash-chained ledger of sealed forecasts and their grades."""
import os, json, hashlib, datetime as dt
import pandas as pd, numpy as np
from .config import LEDGER_DIR

LEDGER = os.path.join(LEDGER_DIR, "ledger.jsonl")
FC_DIR = os.path.join(LEDGER_DIR, "forecasts")

def _sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

def read():
    if not os.path.exists(LEDGER): return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]

def append(rec: dict):
    recs = read(); prev = recs[-1]["record_hash"] if recs else "GENESIS"
    rec = dict(rec); rec["prev_hash"] = prev
    rec["record_hash"] = _sha(json.dumps(rec, sort_keys=True).encode())
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(LEDGER, "a") as f: f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec

def verify():
    """Recompute the hash chain and every forecast file hash. Returns list of problems."""
    probs, prev = [], "GENESIS"
    for r in read():
        body = {k: v for k, v in r.items() if k != "record_hash"}
        if r["prev_hash"] != prev: probs.append(f"chain break at {r.get('id')}")
        if _sha(json.dumps(body, sort_keys=True).encode()) != r["record_hash"]: probs.append(f"record altered: {r.get('id')}")
        if r["type"] == "forecast":
            p = os.path.join(LEDGER_DIR, r["file"])
            if not os.path.exists(p) or _sha(open(p, "rb").read()) != r["file_sha256"]: probs.append(f"forecast file altered: {r['id']}")
        prev = r["record_hash"]
    return probs

def seal(df: pd.DataFrame, meta: dict) -> dict:
    """Write the canonical forecast CSV and append a ledger record (idempotent per id)."""
    fid = meta["id"]
    if any(r["type"] == "forecast" and r["id"] == fid for r in read()):
        return None
    os.makedirs(FC_DIR, exist_ok=True)
    rel = f"forecasts/{fid}.csv"; path = os.path.join(LEDGER_DIR, rel)
    df.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    rec = {"type": "forecast", "id": fid, "file": rel, "file_sha256": _sha(open(path, "rb").read()),
           "sealed_utc": dt.datetime.utcnow().isoformat(timespec="seconds"), **meta}
    return append(rec)

def forecasts():
    return [r for r in read() if r["type"] == "forecast"]

def grades():
    return {r["forecast_id"]: r for r in read() if r["type"] == "grade"}

def grade_due(px, bench="SPY", log=print):
    """Grade every sealed forecast whose exit date is in the data."""
    O, C, A = px["open"], px["close"], px["adjclose"]; done = grades(); new = []
    last = A.index.max()
    for r in forecasts():
        if r["id"] in done: continue
        ex = pd.Timestamp(r["exit_date"]); en = pd.Timestamp(r["entry_date"]); dd = pd.Timestamp(r["decision_date"])
        if ex > last or ex not in A.index: continue
        f = pd.read_csv(os.path.join(LEDGER_DIR, r["file"]))
        tick = [t for t in f.ticker if t in A.columns]
        if r.get("convention") == "close_to_close":
            p0 = A.loc[dd]
        else:
            if en not in A.index: continue
            p0 = O.loc[en] * (A.loc[en] / C.loc[en])
        ret = A.loc[ex] / p0 - 1; exr = (ret[tick] - ret[bench]).dropna()
        f = f.set_index("ticker").loc[exr.index]; f["excess"] = exr
        port = f[f.get("in_portfolio", pd.Series(False, index=f.index)).astype(bool)] if "in_portfolio" in f else f.nsmallest(10, "rank")
        if port.empty: port = f.nsmallest(10, "rank")
        top20 = f.nsmallest(20, "rank").excess; bot20 = f.nlargest(20, "rank").excess
        g = {"type": "grade", "id": f"grade-{r['id']}", "forecast_id": r["id"], "graded_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
             "rank_ic": float(-f["rank"].corr(f.excess, method="spearman")), "n_stocks": int(len(f)),
             "portfolio_excess": float(port.excess.mean()), "portfolio_hit_rate": float((port.excess > 0).mean()),
             "top20_excess": float(top20.mean()), "bottom20_excess": float(bot20.mean()),
             "benchmark_return": float(ret[bench]), "best": port.excess.idxmax(), "worst": port.excess.idxmin(),
             "portfolio_detail": {k: round(float(v), 5) for k, v in port.excess.items()}}
        new.append(append(g)); log(f"[grade] {r['id']}: IC {g['rank_ic']:+.3f}, portfolio vs {bench} {g['portfolio_excess']:+.2%}")
    return new
