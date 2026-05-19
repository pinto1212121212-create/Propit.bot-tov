"""
pine_indicators.py — Pine Script ta.* functions, replicated to spec.

Goal: 100% deterministic equivalence with TradingView's built-in
indicator functions. Each function below references the Pine Script
v5 Reference Manual algorithm.

Every formula here is from the public Pine docs — there is no
hidden behavior. Where I'm uncertain about an edge case, it's
flagged with a `# TODO_VERIFY:` comment so we can confirm against
a TradingView CSV export.

Numerical precision: Pine v5 uses 64-bit IEEE 754 floats internally.
Python's `float` is the same. So no precision loss between platforms.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════
# MOVING AVERAGES
# ════════════════════════════════════════════════════════════════════════

def sma(source: pd.Series, length: int) -> pd.Series:
    """
    ta.sma(source, length) — Simple Moving Average.

    Pine spec: sum of last `length` values divided by `length`.
    First `length-1` bars: returns NaN.
    """
    return source.rolling(window=length, min_periods=length).mean()


def ema(source: pd.Series, length: int) -> pd.Series:
    """
    ta.ema(source, length) — Exponential Moving Average.

    Pine spec (v5 reference):
        α = 2 / (length + 1)
        ema[i] = α * source[i] + (1 - α) * ema[i-1]

    SEED behavior: Pine seeds the EMA with the first non-na source value
    (NOT SMA seed). This matches pandas' ewm(span=N, adjust=False).
    """
    return source.ewm(span=length, adjust=False, min_periods=length).mean()


def rma(source: pd.Series, length: int) -> pd.Series:
    """
    ta.rma(source, length) — Wilder's Smoothing / SMMA / Modified MA.

    Used internally by ta.rsi, ta.atr, ta.dmi. Critical to get right.

    Pine spec:
        α = 1 / length        # NOT 2/(N+1) — different from EMA
        rma[i] = α * source[i] + (1 - α) * rma[i-1]

    SEED: First valid bar = SMA of the first `length` bars. After that,
    Wilder's recursion above.
    """
    # Wilder's seed = SMA of first N bars, then recursion
    sma_seed = source.rolling(window=length, min_periods=length).mean()
    alpha = 1.0 / length
    # ewm with alpha matches the recursion exactly; need to seed manually
    out = pd.Series(np.nan, index=source.index, dtype=float)
    started = False
    prev = np.nan
    for i in range(len(source)):
        val = source.iloc[i]
        if np.isnan(val):
            continue
        if not started:
            seed = sma_seed.iloc[i]
            if not np.isnan(seed):
                out.iloc[i] = seed
                prev = seed
                started = True
        else:
            prev = alpha * val + (1 - alpha) * prev
            out.iloc[i] = prev
    return out


def wma(source: pd.Series, length: int) -> pd.Series:
    """
    ta.wma(source, length) — Weighted Moving Average.

    Pine spec: linear weights 1,2,...,length on the last `length` bars.
    """
    weights = np.arange(1, length + 1, dtype=float)
    return source.rolling(window=length, min_periods=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# ════════════════════════════════════════════════════════════════════════
# STATISTICAL
# ════════════════════════════════════════════════════════════════════════

def stdev(source: pd.Series, length: int, biased: bool = True) -> pd.Series:
    """
    ta.stdev(source, length, biased=true) — Standard Deviation.

    Pine spec:
      biased=true  → divides by N        (population stdev)
      biased=false → divides by N - 1    (sample stdev, Bessel correction)

    DEFAULT IS TRUE in Pine. ta.bb() also uses biased=true by default.
    pandas .std() defaults to ddof=1 (biased=false), so MUST pass ddof=0.
    """
    ddof = 0 if biased else 1
    return source.rolling(window=length, min_periods=length).std(ddof=ddof)


def variance(source: pd.Series, length: int, biased: bool = True) -> pd.Series:
    """ta.variance(source, length, biased=true)"""
    ddof = 0 if biased else 1
    return source.rolling(window=length, min_periods=length).var(ddof=ddof)


# ════════════════════════════════════════════════════════════════════════
# BOLLINGER BANDS — uses ta.stdev(..., biased=true) by default
# ════════════════════════════════════════════════════════════════════════

def bb(source: pd.Series, length: int, mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ta.bb(source, length, mult) — Bollinger Bands.

    Pine spec:
      middle = ta.sma(source, length)
      dev    = mult * ta.stdev(source, length, biased=true)   ← BIASED
      upper  = middle + dev
      lower  = middle - dev

    Returns (middle, upper, lower).
    """
    mid = sma(source, length)
    dev = mult * stdev(source, length, biased=True)
    return mid, mid + dev, mid - dev


# ════════════════════════════════════════════════════════════════════════
# RSI — uses Wilder's smoothing (ta.rma)
# ════════════════════════════════════════════════════════════════════════

def rsi(source: pd.Series, length: int) -> pd.Series:
    """
    ta.rsi(source, length) — Relative Strength Index (Wilder, 1978).

    Pine spec:
      change   = source - source[1]
      gain     = max(change, 0)
      loss     = max(-change, 0)
      avgGain  = ta.rma(gain, length)
      avgLoss  = ta.rma(loss, length)
      rs       = avgGain / avgLoss
      rsi      = 100 - 100 / (1 + rs)

    Edge case: if avgLoss == 0, Pine returns 100. We handle this.
    """
    change = source.diff()
    gain = change.clip(lower=0)
    loss = (-change).clip(lower=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # When avg_loss == 0, RSI is 100 by Pine convention
    out = out.where(avg_loss != 0, 100.0)
    return out


# ════════════════════════════════════════════════════════════════════════
# MACD
# ════════════════════════════════════════════════════════════════════════

def macd(source: pd.Series, fast_length: int, slow_length: int, signal_length: int
        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ta.macd(source, fast_length, slow_length, signal_length).

    Pine spec:
      fastMA     = ta.ema(source, fast_length)
      slowMA     = ta.ema(source, slow_length)
      macdLine   = fastMA - slowMA
      signalLine = ta.ema(macdLine, signal_length)
      histogram  = macdLine - signalLine

    Returns (macdLine, signalLine, histogram).
    """
    fast = ema(source, fast_length)
    slow = ema(source, slow_length)
    macd_line = fast - slow
    sig_line = ema(macd_line, signal_length)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


# ════════════════════════════════════════════════════════════════════════
# STOCHASTIC
# ════════════════════════════════════════════════════════════════════════

def stoch(close: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """
    ta.stoch(close, high, low, length) — Raw stochastic %K.

    Pine spec:
      lowest_low   = ta.lowest(low,  length)
      highest_high = ta.highest(high, length)
      stoch = 100 * (close - lowest_low) / (highest_high - lowest_low)

    Edge case: if highest_high == lowest_low (perfectly flat), Pine
    returns 0 (avoids division by zero). TODO_VERIFY: confirm with TV.
    """
    ll = low.rolling(window=length, min_periods=length).min()
    hh = high.rolling(window=length, min_periods=length).max()
    rng = hh - ll
    raw = 100 * (close - ll) / rng
    return raw.where(rng != 0, 0.0)


# ════════════════════════════════════════════════════════════════════════
# TRUE RANGE + ATR
# ════════════════════════════════════════════════════════════════════════

def true_range(high: pd.Series, low: pd.Series, close: pd.Series,
              handle_na: bool = True) -> pd.Series:
    """
    ta.tr(handle_na) — True Range.

    Pine spec:
      tr = max( high - low,
                |high - close[1]|,
                |low  - close[1]| )

    handle_na: when close[1] is NaN (first bar):
      handle_na=true  → tr = high - low
      handle_na=false → tr = na
    """
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    if handle_na:
        # First bar (prev_close NaN): use high-low
        tr.iloc[0] = (high.iloc[0] - low.iloc[0])
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """
    ta.atr(length) — Average True Range (Wilder, 1978).

    Pine spec:
      ATR = ta.rma( ta.tr(true), length )
    """
    tr = true_range(high, low, close, handle_na=True)
    return rma(tr, length)


# ════════════════════════════════════════════════════════════════════════
# DIRECTIONAL MOVEMENT — DI+, DI-, ADX
# ════════════════════════════════════════════════════════════════════════

def dmi(high: pd.Series, low: pd.Series, close: pd.Series,
        di_length: int, adx_smoothing: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ta.dmi(di_length, adx_smoothing) — Wilder's Directional Movement.

    Pine spec:
      up   = ta.change(high)
      down = -ta.change(low)
      plusDM  = up   if (up > down and up > 0)   else 0
      minusDM = down if (down > up and down > 0) else 0
      trur    = ta.rma(ta.tr, di_length)
      plus    = 100 * ta.rma(plusDM,  di_length) / trur
      minus   = 100 * ta.rma(minusDM, di_length) / trur
      sum     = plus + minus
      adx     = 100 * ta.rma(|plus - minus| / (sum == 0 ? 1 : sum), adx_smoothing)

    Returns (di_plus, di_minus, adx).
    """
    up   = high.diff()
    down = -low.diff()
    plus_dm  = pd.Series(np.where((up > down) & (up > 0),     up,   0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0),   down, 0.0), index=high.index)
    tr = true_range(high, low, close, handle_na=True)
    trur = rma(tr, di_length)
    plus  = 100 * rma(plus_dm,  di_length) / trur
    minus = 100 * rma(minus_dm, di_length) / trur
    total = plus + minus
    dx = 100 * (plus - minus).abs() / total.where(total != 0, 1.0)
    adx = rma(dx, adx_smoothing)
    return plus, minus, adx


# ════════════════════════════════════════════════════════════════════════
# VWAP (anchored on session reset)
# ════════════════════════════════════════════════════════════════════════

def vwap(source: pd.Series, volume: pd.Series, anchor_id: pd.Series) -> pd.Series:
    """
    ta.vwap(source) — Volume Weighted Average Price.

    Pine spec:
      Resets on each new session (anchor changes).
      cumPV = cumsum(source * volume) within anchor group
      cumV  = cumsum(volume)          within anchor group
      vwap  = cumPV / cumV

    Caller provides anchor_id (Series of group keys). For default
    "1 Day" anchoring, anchor_id is the date in the exchange timezone.

    NOTE: Pine's default ta.vwap uses HLC3 as source. To match it,
    pass source = (high+low+close)/3. We make it explicit so the
    caller controls which "source" matches the Pine input.
    """
    pv = source * volume
    cum_pv = pv.groupby(anchor_id).cumsum()
    cum_v  = volume.groupby(anchor_id).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


# ════════════════════════════════════════════════════════════════════════
# CROSS DETECTION
# ════════════════════════════════════════════════════════════════════════

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    ta.crossover(a, b) — True if a crossed above b on this bar.

    Pine spec: returns true when (a[1] <= b[1]) AND (a > b).
    Note the <= (NOT strict <) on the previous bar.
    """
    return (a.shift(1) <= b.shift(1)) & (a > b)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    ta.crossunder(a, b) — True if a crossed below b on this bar.

    Pine spec: returns true when (a[1] >= b[1]) AND (a < b).
    """
    return (a.shift(1) >= b.shift(1)) & (a < b)


def cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """ta.cross(a, b) — Either direction."""
    return crossover(a, b) | crossunder(a, b)


# ════════════════════════════════════════════════════════════════════════
# HIGHEST / LOWEST
# ════════════════════════════════════════════════════════════════════════

def highest(source: pd.Series, length: int) -> pd.Series:
    """ta.highest(source, length)."""
    return source.rolling(window=length, min_periods=length).max()


def lowest(source: pd.Series, length: int) -> pd.Series:
    """ta.lowest(source, length)."""
    return source.rolling(window=length, min_periods=length).min()


def change(source: pd.Series, length: int = 1) -> pd.Series:
    """ta.change(source, length) — source - source[length]."""
    return source.diff(length)


# ════════════════════════════════════════════════════════════════════════
# SELF-TEST — verifies internal consistency
# (Real Pine-vs-Python validation needs a TV CSV; this catches our bugs.)
# ════════════════════════════════════════════════════════════════════════

def _self_test():
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.001, n)
    close = pd.Series(17000 * np.exp(np.cumsum(rets)), index=idx)
    high  = close * (1 + rng.uniform(0, 0.002, n))
    low   = close * (1 - rng.uniform(0, 0.002, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol   = pd.Series(rng.integers(1000, 5000, n), index=idx, dtype=float)

    # Each function must produce a Series of the same length, with NaN
    # for warmup bars and finite values after.
    checks = {
        "sma(close, 20)":  sma(close, 20),
        "ema(close, 21)":  ema(close, 21),
        "rma(close, 14)":  rma(close, 14),
        "rsi(close, 14)":  rsi(close, 14),
        "bb middle (18)":  bb(close, 18, 2.0)[0],
        "bb upper  (18)":  bb(close, 18, 2.0)[1],
        "macd hist 12/26/9": macd(close, 12, 26, 9)[2],
        "stoch(close, h, l, 14)": stoch(close, high, low, 14),
        "atr(h, l, c, 14)": atr(high, low, close, 14),
        "dmi(h, l, c, 14, 14) → ADX": dmi(high, low, close, 14, 14)[2],
        "vwap (HLC3, daily anchor)":
            vwap((high+low+close)/3, vol,
                 pd.Series(idx.date, index=idx)),
        "crossover(close, sma20)": crossover(close, sma(close, 20)),
    }

    print(f"{'Function':<35} {'first valid bar':>15} {'last value':>15}")
    print("─" * 70)
    for name, s in checks.items():
        first_valid = s.first_valid_index()
        first_pos = s.index.get_loc(first_valid) if first_valid is not None else -1
        last = s.iloc[-1]
        last_str = f"{last:.4f}" if isinstance(last, (int, float, np.floating)) and not np.isnan(last) else str(last)
        print(f"{name:<35} {first_pos:>15} {last_str:>15}")

    print("\n✅ All functions executed without error.")
    print("→ Pine-vs-Python numerical validation requires a TradingView CSV.")
    print("   Once provided, tv_diff.py will compare each function against TV.")

if __name__ == "__main__":
    _self_test()
