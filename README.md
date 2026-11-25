# S&P 500 Oracle

Personal dashboard: **https://yohasson.github.io/sp500-watch/**

The Oracle ranks every S&P 500 stock each week for the next 5 trading days, suggests 10 picks, **seals** the full forecast with a SHA-256 fingerprint before the market opens, and **grades itself** automatically afterwards. A history replay applies the same rules to past weeks using only what was known at the time.

Research tool, not financial advice. Paper-trade first.

## How it runs
- `.github/workflows/oracle.yml` runs after every US close (22:30 UTC, Mon–Fri) and on Monday before the open.
  It updates prices, replays history, seals a new forecast on the last trading day of the week, grades finished forecasts, and publishes the site to the `gh-pages` branch.
- To trigger a run by hand: bump the number in `tools/run.txt` (or use *Run workflow* in the Actions tab).
- Personal settings in the page (amount invested, broker fees) are stored on the device only.

## Layout
| Path | What |
|---|---|
| `oracle/` | Python package: data providers, universe (point-in-time S&P 500 membership), engine (features, models, walk-forward replay), ledger (seal and grade), pipeline, dashboard |
| `ledger/ledger.jsonl` | Append-only, hash-chained record of every sealed forecast and grade |
| `ledger/forecasts/` | The sealed forecast files (their SHA-256 is in the ledger) |
| `site/` | The published dashboard (`index.html` + `data.json`, installable as an app) |
| `data/status.json` | Last run status, evaluation and checks |

## Method (short)
- **Universe:** S&P 500 members on each date (Wikipedia list and change history); stocks only count after they joined.
- **Signals (8):** 1-week and 1-month reversal, 12-1 and 6-1 month momentum, low volatility, near 52-week high, low beta, sector-relative 1-week reversal; plus market mood (VIX, S&P 500 trend, dispersion).
- **Models:** gradient-boosted trees + ridge regression (trained walk-forward, labels purged until known) + a simple signal blend; ensemble = average rank.
- **Portfolio:** top 10, keep while in the top 50, max 3 per sector, equal weights.
- **Trading convention:** decide after the close, buy at the next open, sell at the close of the 5th trading day; total returns (dividends included via adjusted prices), measured against SPY.
- **Labels:** from the replay's calibration table and the cost profile (Strong Buy / Buy / Hold / Avoid).

## Data
Free, unofficial Yahoo Finance chart API (fallback: Stooq). Replace `oracle/data.py` providers with a licensed feed for production use.
