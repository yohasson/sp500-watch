"""Central configuration. Every value here is sealed (hashed) with each forecast."""
from dataclasses import dataclass, asdict, field
import json, hashlib, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ORACLE_DATA_DIR", os.path.join(ROOT, "data"))
LEDGER_DIR = os.path.join(ROOT, "ledger")
DOCS_DIR = os.path.join(ROOT, "site")

@dataclass
class Config:
    model_version: str = "ensemble-v1.1"
    horizon_days: int = 5              # trading days: entry at next open, exit at close of day 5
    history_years: int = 10            # price history requested from the provider
    min_train_weeks: int = int(os.environ.get("ORACLE_MIN_TRAIN", 104))         # walk-forward: minimum training history
    retrain_every_weeks: int = 13
    portfolio_n: int = 10              # buy the top N ...
    portfolio_hold_k: int = 50         # ... and keep holding while still in the top K
    max_per_sector: int = 3            # diversification cap per GICS sector
    # cost profile (one-way, fraction of traded value); default = $0-commission US broker
    broker_fee_min_usd: float = 0.0
    broker_fee_pct: float = 0.0
    half_spread_pct: float = 0.0003
    account_usd: float = 50000.0
    # label rule multipliers on round-trip cost
    strong_buy_cost_mult: float = 3.0
    buy_cost_mult: float = 2.0
    benchmark: str = "SPY"
    random_seed: int = 0
    factor_blend: tuple = ("rev5", "secrev5", "mom12_1", "lowvol", "hi52")

    def oneway_cost(self) -> float:
        pos = self.account_usd / max(self.portfolio_n, 1)
        fee = max(self.broker_fee_min_usd / pos, self.broker_fee_pct) if pos > 0 else 0.0
        return fee + self.half_spread_pct

    def roundtrip_cost(self) -> float:
        return 2 * self.oneway_cost()

    def to_dict(self):
        d = asdict(self); d["factor_blend"] = list(self.factor_blend); return d

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

CFG = Config()
