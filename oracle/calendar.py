"""NYSE trading calendar helpers (full-day closures; half days count as trading days)."""
import pandas as pd
NYSE_HOLIDAYS = pd.to_datetime([
 "2025-01-01","2025-01-09","2025-01-20","2025-02-17","2025-04-18","2025-05-26","2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
 "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25","2026-06-19","2026-07-03","2026-09-07","2026-11-26","2026-12-25",
 "2027-01-01","2027-01-18","2027-02-15","2027-03-26","2027-05-31","2027-06-18","2027-07-05","2027-09-06","2027-11-25","2027-12-24",
 "2028-01-17","2028-02-21","2028-04-14","2028-05-29","2028-06-19","2028-07-04","2028-09-04","2028-11-23","2028-12-25"])

def next_trading_days(after: pd.Timestamp, n: int, known: pd.DatetimeIndex = None):
    """The n trading days strictly after `after` (uses known data dates first, then the calendar)."""
    out = []
    if known is not None:
        out = [d for d in known if d > after][:n]
    d = out[-1] if out else after
    while len(out) < n:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
            out.append(d)
    return out

def is_last_trading_day_of_week(d: pd.Timestamp) -> bool:
    nxt = next_trading_days(d, 1)[0]
    return nxt.isocalendar()[1] != d.isocalendar()[1] or (nxt - d).days > 3
