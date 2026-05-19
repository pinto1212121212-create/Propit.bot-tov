"""
Realistic NQ-like 15-minute data generator.

Encodes known properties of NQ E-mini futures:
  • ~1.0-1.5% daily realized volatility
  • Intraday volatility pattern: NY open spike, lunch lull, close ramp
  • Mix of trending days (~30%) and ranging days (~70%)
  • Occasional news spikes (3-5 per month)
  • Weekly drift (markets tend up slightly long-term)
  • Realistic volume profile (V-shaped: open-lunch-close)

NOT actual NQ data. But realistic enough that RELATIVE performance
between strategies is meaningful.
"""
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

def make_realistic_nq(days: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bars_per_day = 24 * 4  # 15-min bars
    n_bars = days * bars_per_day

    start = datetime(2025, 11, 1, 0, 0, tzinfo=pytz.UTC)
    times = pd.date_range(start, periods=n_bars, freq="15min", tz="UTC")
    times_eastern = times.tz_convert("America/New_York")

    # ──── 1. Intraday volatility profile (US Eastern Time) ────
    # Peaks at NY open (09:30 ET), low at lunch (12-13 ET), ramps at close (15-16 ET)
    minutes_since_midnight_et = times_eastern.hour * 60 + times_eastern.minute
    vol_profile = np.ones(n_bars) * 0.6  # base vol
    # NY open spike (09:30 = 570)
    open_dist = np.minimum(np.abs(minutes_since_midnight_et - 570), 120) / 120
    vol_profile += 1.4 * np.exp(-open_dist * 3)
    # Lunch lull (12:30 = 750)
    lunch_dist = np.abs(minutes_since_midnight_et - 750) / 90
    vol_profile -= 0.3 * np.exp(-lunch_dist * 2)
    # Close ramp (15:30 = 930)
    close_dist = np.minimum(np.abs(minutes_since_midnight_et - 930), 120) / 120
    vol_profile += 0.8 * np.exp(-close_dist * 3)
    # Pre-market and after-hours: very low vol
    is_off_hours = (minutes_since_midnight_et < 4*60) | (minutes_since_midnight_et > 16*60)
    vol_profile = np.where(is_off_hours, vol_profile * 0.3, vol_profile)
    vol_profile = np.maximum(vol_profile, 0.1)

    # ──── 2. Day-level regime (trending vs ranging) ────
    day_idx = np.arange(n_bars) // bars_per_day
    n_days_total = days
    regime = rng.choice([0, 1, 2], size=n_days_total, p=[0.50, 0.30, 0.20])
    # 0 = ranging, 1 = trending up, 2 = trending down
    day_drift = np.where(regime == 1, +0.0002, np.where(regime == 2, -0.0002, 0.0))
    bar_drift = day_drift[day_idx]

    # ──── 3. Base returns ────
    base_sigma = 0.0008  # per 15-min bar
    sigma_series = base_sigma * vol_profile
    rets = rng.normal(bar_drift, sigma_series, n_bars)

    # ──── 4. Add occasional news spikes ────
    n_news = int(days / 30 * 4)  # ~4 per month
    spike_idx = rng.integers(0, n_bars, n_news)
    spike_size = rng.normal(0, 0.005, n_news) * rng.choice([-1, 1], n_news)
    rets[spike_idx] += spike_size

    # ──── 5. Build close prices ────
    close = 17000 * np.exp(np.cumsum(rets))

    # ──── 6. Derive OHLC ────
    open_ = np.r_[close[0], close[:-1]]
    bar_range_pct = np.abs(rets) + sigma_series * 0.5  # realistic range
    half_range = bar_range_pct * close
    high = np.maximum(open_, close) + half_range * rng.uniform(0.2, 0.8, n_bars)
    low  = np.minimum(open_, close) - half_range * rng.uniform(0.2, 0.8, n_bars)

    # ──── 7. Volume profile ────
    base_vol = 2000
    vol = (base_vol * (1 + vol_profile)).astype(int)
    vol = vol + rng.integers(-300, 300, n_bars)
    vol = np.maximum(vol, 100)

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vol,
    }, index=times)

if __name__ == "__main__":
    df = make_realistic_nq(days=90)
    print(f"Generated {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['close'].min():.0f} to {df['close'].max():.0f}")
    print(f"Daily returns std: {df['close'].pct_change(96).std()*100:.2f}%")
    print(f"Per-bar returns std: {df['close'].pct_change().std()*100:.3f}%")
    # Show 15-min bar volatility by hour-of-day (Asia/Jerusalem)
    df["hour_il"] = df.index.tz_convert("Asia/Jerusalem").hour
    print("\nVolatility by Israel hour:")
    print(df.groupby("hour_il")["close"].apply(lambda s: s.pct_change().std()*100).round(3))
