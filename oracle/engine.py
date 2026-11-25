"""Features, models, walk-forward replay and evaluation."""
import numpy as np, pandas as pd, warnings, math
from scipy.stats import norm, skew, kurtosis
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from .config import CFG
warnings.filterwarnings("ignore")

FACTORS = ["rev5", "rev20", "mom12_1", "mom6_1", "lowvol", "hi52", "lowbeta", "secrev5"]
EXTRA = ["rev1", "vrev5", "max21", "idrev5", "gap1", "ma50"]
FACTOR_LABELS = {"rev5": "Bounce-back after a weak week (1-week reversal)", "rev20": "Bounce-back after a weak month",
                 "mom12_1": "12-month trend (momentum)", "mom6_1": "6-month trend", "lowvol": "Calm price moves (low volatility)",
                 "hi52": "Close to its 52-week high", "lowbeta": "Moves less than the market (low beta)",
                 "secrev5": "Lagged its sector last week"}
REGIME = ["vix", "mkt20", "mkt5", "disp5"]

def decision_dates(idx: pd.DatetimeIndex):
    """Last trading day of each ISO week present in the data."""
    s = pd.Series(idx, index=idx)
    wk = s.groupby([idx.isocalendar().year.values, idx.isocalendar().week.values]).max()
    return pd.DatetimeIndex(sorted(wk.values))

def build_features(px, sectors, member, bench="SPY"):
    A = px["adjclose"]; stocks = [c for c in A.columns if c in member.columns]
    S = A[stocks]; R = S.pct_change(fill_method=None); M = A[bench]; RM = M.pct_change(fill_method=None)
    f = {}
    f["rev5"] = -(S / S.shift(5) - 1)
    f["rev20"] = -(S / S.shift(21) - 1)
    f["mom12_1"] = S.shift(21) / S.shift(252) - 1
    f["mom6_1"] = S.shift(21) / S.shift(126) - 1
    f["lowvol"] = -R.rolling(60, min_periods=40).std()
    f["hi52"] = S / S.rolling(252, min_periods=200).max()
    cov = R.rolling(252, min_periods=200).cov(RM); var = RM.rolling(252, min_periods=200).var()
    f["lowbeta"] = -(cov.div(var, axis=0))
    r5 = S / S.shift(5) - 1
    sec = pd.Series({s: sectors.get(s, "Unknown") for s in stocks})
    f["secrev5"] = -(r5 - r5.T.groupby(sec).transform("mean").T)
    vol = R.rolling(60, min_periods=40).std()
    f["rev1"] = -R
    f["vrev5"] = f["rev5"] / (vol * np.sqrt(5))
    f["max21"] = -R.rolling(21, min_periods=15).max()
    beta = cov.div(var, axis=0); m5 = M / M.shift(5) - 1
    f["idrev5"] = -(r5 - beta.mul(m5, axis=0))
    if "open" in px and "close" in px:
        O = px["open"][stocks]; C = px["close"][stocks]
        f["gap1"] = -(O / C.shift(1) - 1)
    else:
        f["gap1"] = f["rev1"] * 0
    f["ma50"] = -(S / S.rolling(50, min_periods=40).mean() - 1)
    vix = A["^VIX"] if "^VIX" in A.columns else px["close"].get("^VIX", pd.Series(20.0, index=A.index))
    reg = pd.DataFrame({"vix": vix, "mkt20": M / M.shift(21) - 1, "mkt5": M / M.shift(5) - 1,
                        "disp5": r5.where(member.reindex(r5.index).fillna(False)).std(axis=1)})
    return f, reg, stocks

def forward_excess(px, stocks, dates, bench="SPY", horizon=5, entry="open"):
    """Excess total return vs benchmark: entry at next open (or decision close), exit at close of day `horizon`."""
    O, C, A = px["open"], px["close"], px["adjclose"]
    idx = A.index; pos = idx.get_indexer(dates)
    rows, exits = {}, {}
    for d, p in zip(dates, pos):
        if p < 0 or p + horizon >= len(idx): continue
        e0, e1 = idx[p + 1], idx[p + horizon]
        if entry == "open":
            fac = (A.loc[e0] / C.loc[e0])
            p0 = O.loc[e0] * fac
        else:
            p0 = A.loc[d]
        ret = A.loc[e1] / p0 - 1
        rows[d] = ret[stocks] - ret[bench]; exits[d] = e1
    return pd.DataFrame(rows).T, pd.Series(exits)

def xrank(df):
    return df.rank(axis=1, pct=True) - 0.5

class Replay:
    def __init__(self, px, sectors, member, cfg=CFG):
        self.cfg = cfg; self.px = px
        self.f, self.reg, self.stocks = build_features(px, sectors, member, cfg.benchmark)
        self.member = member.reindex(px["adjclose"].index).fillna(False)[self.stocks]
        self.factors = FACTORS + (EXTRA if getattr(cfg, "extra_features", False) else [])
        self.F = {k: xrank(v.where(self.member)) for k, v in self.f.items() if k in self.factors}
        self.sectors = sectors
        self.dates = decision_dates(px["adjclose"].index)
        self.dates = self.dates[self.dates >= px["adjclose"].index[260]]
        self.fwd, self.exit = forward_excess(px, self.stocks, self.dates, cfg.benchmark, cfg.horizon_days, "open")
        self.fwd = self.fwd.where(self.member.reindex(self.fwd.index))
        self.fwd_cc, _ = forward_excess(px, self.stocks, self.dates, cfg.benchmark, cfg.horizon_days, "close")
        self.model_hist = []

    def panel(self, d):
        if not hasattr(self, "_pc"): self._pc = {}
        if d in self._pc: return self._pc[d]
        X = self._panel(d); self._pc[d] = X; return X

    def _train_rows(self, t):
        if not hasattr(self, "_tc"): self._tc = {}
        if t not in self._tc:
            X = self.panel(t); y = self.fwd.loc[t].reindex(X.index).rank(pct=True) - 0.5
            if getattr(self.cfg, "target", "market") == "sector":
                fw = self.fwd.loc[t].reindex(X.index); sec = pd.Series({k: self.sectors.get(k, "U") for k in X.index})
                y = fw.groupby(sec).rank(pct=True) - 0.5
            m = X[self.factors].notna().all(1) & y.notna()
            Xd = None
            if m.sum() >= 50:
                Xd = X[m].copy(); Xd["y"] = y[m]
            self._tc[t] = Xd
        return self._tc[t]

    def _panel(self, d):
        X = pd.DataFrame({k: self.F[k].loc[d] for k in self.factors})
        for c in REGIME: X[c] = self.reg.loc[d, c]
        return X[self.member.loc[d]]

    def blend(self, d):
        return sum(self.F[k].loc[d].fillna(0) for k in self.cfg.factor_blend) / len(self.cfg.factor_blend)

    def _train(self, d):
        rows = []
        for t in self.dates:
            if t not in self.exit.index or self.exit[t] > d: continue      # purge: label must be known at d
            Xd = self._train_rows(t)
            if Xd is not None: rows.append(Xd)
        if not rows: return None
        tr = pd.concat(rows); X = tr[self.factors + REGIME].fillna(0); y = tr["y"]
        gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
                                           l2_regularization=1.0, random_state=self.cfg.random_seed).fit(X, y)
        rd = Ridge(alpha=10).fit(X, y)
        return gb, rd, len(rows)

    def score(self, d, models):
        X = self.panel(d); Xf = X[self.factors + REGIME].fillna(0)
        gb, rd = models[0], models[1]
        g = pd.Series(gb.predict(Xf), index=X.index); r = pd.Series(rd.predict(Xf), index=X.index)
        b = self.blend(d)[X.index]
        mode = getattr(self.cfg, "ensemble", "gbm+ridge+blend")
        parts = {"gbm": g.rank(pct=True), "ridge": r.rank(pct=True), "blend": b.rank(pct=True)}
        use = [k for k in mode.split("+") if k in parts] or list(parts)
        ens = sum(parts[k] for k in use) / len(use)
        return pd.DataFrame({"ens": ens, "gbm": g, "ridge": r, "blend": b})

    def run(self, log=print):
        """Walk-forward over all decision dates; models retrained every N weeks on purged history."""
        out = []; models = None; last_train = None
        start = self.cfg.min_train_weeks
        for i, d in enumerate(self.dates):
            if i < start: continue
            if models is None or (i - start) % self.cfg.retrain_every_weeks == 0:
                m = self._train(d)
                if m is None: continue
                models = m; self.model_hist.append((d, m))
            sc = self.score(d, models); sc["date"] = d; sc["fwd"] = self.fwd.loc[d].reindex(sc.index) if d in self.fwd.index else np.nan
            sc["fwd_cc"] = self.fwd_cc.loc[d].reindex(sc.index) if d in self.fwd_cc.index else np.nan
            out.append(sc.reset_index().rename(columns={"index": "ticker"}))
        res = pd.concat(out, ignore_index=True)
        log(f"[replay] {res.date.nunique()} decision weeks {res.date.min().date()}..{res.date.max().date()}")
        return res

    def live(self, d):
        models = self._train(d)
        self.model_hist.append((d, models))
        return self.score(d, models), models[2]

    def daily_backtest(self, rep, days=252, top=5):
        """Blind daily hindsight analysis for every stock on every trading day of the last `days`.

        For each day d, using ONLY information known at d's close:
          - features from prices up to d,
          - the latest model whose training labels had all ended by its training date (<= d),
          - a projection table (rank group -> average next-week return vs SPY) built from replay weeks
            whose outcome was already known at d.
        The real outcome (buy next open, sell at the close 5 trading days later) is attached afterwards
        for grading only. Returns (all_rows DataFrame, top-5 rows list)."""
        O, C, A = self.px["open"], self.px["close"], self.px["adjclose"]
        idx = A.index; h = self.cfg.horizon_days; bench = self.cfg.benchmark
        hist = sorted(self.model_hist, key=lambda x: x[0])
        r = rep.dropna(subset=["fwd"]).copy()
        r["pct"] = r.groupby("date")["ens"].rank(pct=True, ascending=False)
        r["bucket"] = r["pct"].map(bucket_of)
        r["known"] = r["date"].map(self.exit)
        names = [b[2] for b in BUCKETS]
        rows, tops = [], []
        for p in range(max(0, len(idx) - days), len(idx)):
            d = idx[p]
            usable = [m for t, m in hist if t <= d]
            if not usable: continue
            X = self.panel(d)
            if len(X) < 100: continue
            sc = self.score(d, usable[-1])["ens"]
            pct = sc.rank(pct=True, ascending=False)
            kn = r[r["known"] <= d]
            proj = kn.groupby("bucket")["fwd"].mean().reindex(names).fillna(0.0) if len(kn) else pd.Series(0.0, index=names)
            b = pct.map(bucket_of)
            df = pd.DataFrame({"date": d, "ticker": sc.index, "rank": sc.rank(ascending=False, method="first").astype(int).values,
                               "pct": pct.values, "proj": b.map(proj).values})
            picks = list(sc.sort_values(ascending=False).index[:top])
            row = {"date": d, "picks": picks}
            if p + h < len(idx):
                e0, e1 = idx[p + 1], idx[p + h]
                p0 = O.loc[e0] * (A.loc[e0] / C.loc[e0]); ret = A.loc[e1] / p0 - 1
                df["actual"] = (ret.reindex(df.ticker) - ret[bench]).values
                rp = ret[picks].astype(float)
                row.update({"exit": e1, "ret": float(rp.iloc[:5].mean()), "bench": float(ret[bench]),
                            "each": {k: round(float(v), 4) for k, v in rp.items()}})
            else:
                df["actual"] = np.nan
            rows.append(df); tops.append(row)
        return pd.concat(rows, ignore_index=True), tops

# ---------------- evaluation ----------------
def ic_series(rep, col="ens", target="fwd"):
    g = rep.dropna(subset=[target]).groupby("date")
    return g.apply(lambda x: x[col].corr(x[target], method="spearman")).dropna()

def pick_portfolio(rk, held, n, k, sectors=None, cap=None):
    """Top-n by rank, keeping current holdings while they stay in the top k; at most `cap` per sector."""
    keep = [s for s in held if s in rk.index and rk[s] <= k]
    new = list(keep); cnt = {}
    for s in new: cnt[(sectors or {}).get(s)] = cnt.get((sectors or {}).get(s), 0) + 1
    for s in rk.sort_values().index:
        if len(new) >= n: break
        if s in new: continue
        sec = (sectors or {}).get(s)
        if cap and sectors and cnt.get(sec, 0) >= cap: continue
        new.append(s); cnt[sec] = cnt.get(sec, 0) + 1
    return new

def simulate(rep, n, k, cost_oneway, target="fwd", col="ens", sectors=None, cap=None):
    held, rows = [], []
    for d, x in rep.dropna(subset=[target]).groupby("date"):
        rk = x.set_index("ticker")[col].rank(ascending=False)
        new = pick_portfolio(rk, held, n, k, sectors, cap)
        to = len(set(new) - set(held)) / n if held else 1.0
        y = x.set_index("ticker")[target]
        g = y[new].mean(); rows.append({"date": d, "gross": g, "turnover": to, "net": g - 2 * cost_oneway * to, "names": new})
        held = new
    return pd.DataFrame(rows).set_index("date")

BUCKETS = [(0, .02, "Top 2%"), (.02, .10, "Top 2–10%"), (.10, .20, "Top 10–20%"), (.20, .50, "Upper middle"),
           (.50, .80, "Lower middle"), (.80, .90, "Bottom 10–20%"), (.90, .98, "Bottom 2–10%"), (.98, 1.0001, "Bottom 2%")]

def bucket_of(pct_from_top):
    for lo, hi, name in BUCKETS:
        if lo < pct_from_top <= hi or (lo == 0 and pct_from_top <= hi): return name
    return BUCKETS[-1][2]

def calibration(rep, col="ens", target="fwd"):
    r = rep.dropna(subset=[target]).copy()
    r["pct"] = r.groupby("date")[col].rank(pct=True, ascending=False)
    r["bucket"] = r["pct"].map(bucket_of)
    wk = r.groupby(["bucket", "date"])[target].mean()
    cal = r.groupby("bucket")[target].agg(mean_excess="mean", p_beat=lambda v: (v > 0).mean(), n="size")
    w = wk.groupby("bucket").agg(["mean", "std", "count"])
    cal["t"] = w["mean"] / w["std"] * np.sqrt(w["count"])
    order = [b[2] for b in BUCKETS]
    return cal.reindex(order)

def label_for(bucket, cal, cfg=CFG):
    """Label from the bucket's replay edge over the average stock (not over SPY), net of round-trip cost."""
    rt = cfg.roundtrip_cost()
    if bucket not in cal.index: return "Hold"
    m = cal.loc[bucket, "mean_excess"] - cal["mean_excess"].mean()
    p = cal.loc[bucket, "p_beat"]
    if bucket == "Top 2%" and m >= cfg.strong_buy_cost_mult * rt: return "Strong Buy"
    if bucket in ("Top 2%", "Top 2–10%") and m >= cfg.buy_cost_mult * rt: return "Buy"
    if bucket in ("Bottom 2–10%", "Bottom 2%") and m <= -rt and p < 0.5: return "Avoid"
    return "Hold"

def deflated_sharpe(r, trials, sr_var):
    r = np.asarray(r); T = len(r); sr = r.mean() / r.std()
    g3 = skew(r); g4 = kurtosis(r, fisher=False)
    emax = math.sqrt(max(sr_var, 1e-12)) * ((1 - 0.5772) * norm.ppf(1 - 1 / trials) + 0.5772 * norm.ppf(1 - 1 / (trials * math.e)))
    return float(norm.cdf((sr - emax) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))))

def power_stats(ic):
    m, sd, n = ic.mean(), ic.std(), len(ic)
    prior = 0.02; pv = 1 / (1 / prior ** 2 + n / sd ** 2); pm = pv * (n * m / sd ** 2)
    out = {"ic_mean": m, "ic_sd": sd, "weeks": n, "t": m / sd * math.sqrt(n), "pct_pos": float((ic > 0).mean()),
           "weeks_needed": (2 * sd / m) ** 2 if m > 0 else None, "posterior_mean": pm,
           "p_ic_pos": float(1 - norm.cdf(0, pm, math.sqrt(pv)))}
    for k in (13, 26, 52):
        out[f"band_{k}"] = [m - 1.96 * sd / math.sqrt(k), m + 1.96 * sd / math.sqrt(k)]
    return out
