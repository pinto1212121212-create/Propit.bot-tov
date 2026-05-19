"""
ProfitBot Jerusalem v3.2 — Python Backtest Engine
==================================================

Re-implementation of the Pine Script strategy in Python for offline backtesting.

SCOPE: Only indicators that do NOT require parallel assets are implemented.
       (BB, RSI, MACD, STOCH, VWAP, VOL, RSI DIV)
       SMT, IM, PSP are SKIPPED per user request.

Usage:
    python3 profitbot_backtest.py
"""

from __future__ import annotations
import os
import sys
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytz

# ────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors Pine inputs 1:1)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Sessions
    use_s1: bool = True
    sess1: tuple = (time(4, 0), time(8, 0))
    use_s2: bool = False
    sess2: tuple = (time(12, 0), time(16, 0))
    use_close: bool = True
    close_time: time = time(8, 0)
    tz: str = "Asia/Jerusalem"

    # Risk
    contracts: int = 1
    max_loss: float = 500.0
    max_profit: float = 500.0
    max_trades: int = 6
    stop_ticks: int = 50
    rr_ratio: float = 2.0
    cd_opp: int = 2
    use_trail: bool = False
    trail_atr: int = 14
    trail_mult: float = 1.5

    # Days (Sunday=0 ... Saturday=6 in our internal mapping below)
    use_sun: bool = True
    use_mon: bool = True
    use_tue: bool = True
    use_wed: bool = True
    use_thu: bool = True
    use_fri: bool = False
    use_sat: bool = False
    ind_rev: bool = False
    use_dna: bool = False
    dna_body: float = 0.3
    use_regime: bool = False
    regime_adx: int = 20

    # Indicator toggles
    use_stoch: bool = True
    use_bb: bool = True
    use_macd: bool = False
    use_vwap: bool = False
    use_rsi: bool = False
    use_rsidiv: bool = False
    use_vol: bool = False

    # Indicator params
    rsi_len: int = 14
    rsi_os: int = 30
    rsi_ob: int = 70
    bb_len: int = 18
    bb_std: float = 2.0
    ma_fast: int = 12
    ma_slow: int = 26
    ma_sig: int = 9
    stoch_k: int = 6
    stoch_ks: int = 1
    div_len: int = 14
    div_lb: int = 3
    vol_ma_len: int = 20

    # Instrument (NQ futures)
    tick_size: float = 0.25
    point_value: float = 20.0   # $/point for NQ
    contract_commission: float = 2.0  # $ per contract per side

# ────────────────────────────────────────────────────────────────────────────
# INDICATORS  (vectorized, matches Pine ta.* semantics)
# ────────────────────────────────────────────────────────────────────────────

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing — Pine's ta.rma."""
    alpha = 1.0 / n
    return s.ewm(alpha=alpha, adjust=False, min_periods=n).mean()

def rsi(s: pd.Series, n: int) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    avg_up = rma(up, n)
    avg_dn = rma(dn, n)
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def bb(s: pd.Series, n: int, k: float):
    mid = sma(s, n)
    dev = s.rolling(n, min_periods=n).std(ddof=0) * k
    return mid, mid + dev, mid - dev

def macd(s: pd.Series, fast: int, slow: int, sig: int):
    line = ema(s, fast) - ema(s, slow)
    signal = ema(line, sig)
    return line, signal, line - signal

def stoch(close: pd.Series, high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    return 100 * (close - ll) / (hh - ll).replace(0, np.nan)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)

def dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int):
    """Returns (DI+, DI-, ADX). Matches Pine ta.dmi(n, n)."""
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    plus_dm  = pd.Series(plus_dm,  index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_n = rma(tr, n)
    di_plus  = 100 * rma(plus_dm,  n) / atr_n
    di_minus = 100 * rma(minus_dm, n) / atr_n
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = rma(dx, n)
    return di_plus, di_minus, adx

def vwap_session(df: pd.DataFrame, session_id: pd.Series) -> pd.Series:
    """VWAP that resets on session_id change (daily VWAP)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    grp = session_id
    cum_pv = pv.groupby(grp).cumsum()
    cum_v  = df["volume"].groupby(grp).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)

def crossover(a: pd.Series, b) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a: pd.Series, b) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))

# ────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int           # +1 long, -1 short
    entry_px: float
    exit_px: float
    qty: int
    pnl: float
    exit_reason: str

@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = None
    initial_capital: float = 10_000.0

    @property
    def stats(self) -> dict:
        if not self.trades:
            return {"trades": 0}
        pnls = np.array([t.pnl for t in self.trades])
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        ec = self.equity_curve.dropna()
        peak = ec.cummax(); dd = (ec - peak)
        return {
            "trades":         len(self.trades),
            "wins":           int((pnls > 0).sum()),
            "losses":         int((pnls < 0).sum()),
            "win_rate":       float((pnls > 0).mean() * 100),
            "total_pnl":      float(pnls.sum()),
            "avg_win":        float(wins.mean()) if len(wins)  else 0.0,
            "avg_loss":       float(losses.mean()) if len(losses) else 0.0,
            "profit_factor":  float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            "max_drawdown":   float(dd.min()),
            "final_equity":   float(ec.iloc[-1]),
            "return_pct":     float((ec.iloc[-1] / self.initial_capital - 1) * 100),
        }

def in_session(dt_local: pd.Timestamp, start: time, end: time) -> bool:
    t = dt_local.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end

def run_backtest(df: pd.DataFrame, cfg: Config, initial_capital: float = 10_000.0) -> BacktestResult:
    """
    df: DataFrame indexed by UTC timestamps with columns: open, high, low, close, volume
    """
    tz = pytz.timezone(cfg.tz)
    idx_local = df.index.tz_convert(tz)
    df = df.copy()
    df["local_time"] = idx_local
    df["dow"] = idx_local.dayofweek            # Mon=0..Sun=6  (we'll remap below)
    # Pine's dayofweek: Sun=1..Sat=7. We'll use Python convention but map in dayOk.
    df["session_id"] = idx_local.date  # daily session boundary for VWAP

    in_s1 = pd.Series([cfg.use_s1 and in_session(t, *cfg.sess1) for t in idx_local], index=df.index, dtype=bool)
    in_s2 = pd.Series([cfg.use_s2 and in_session(t, *cfg.sess2) for t in idx_local], index=df.index, dtype=bool)
    # Python weekday: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    is_weekday = (df["dow"] != 5).astype(bool)
    in_s1 = (in_s1 & is_weekday).astype(bool)
    in_s2 = (in_s2 & is_weekday).astype(bool)
    in_sess = (in_s1 | in_s2).astype(bool)
    in_sess_prev = in_sess.shift(1, fill_value=False).astype(bool)

    # Day filter (Python weekday → flag)
    day_map = {
        6: cfg.use_sun, 0: cfg.use_mon, 1: cfg.use_tue, 2: cfg.use_wed,
        3: cfg.use_thu, 4: cfg.use_fri, 5: cfg.use_sat,
    }
    day_ok = df["dow"].map(day_map)

    # ── INDICATORS ───────────────────────────────────────────────────────
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    rsi_v = rsi(c, cfg.rsi_len)
    rsi_long  = rsi_v <= cfg.rsi_os
    rsi_short = rsi_v >= cfg.rsi_ob

    bb_mid, bb_up, bb_dn = bb(c, cfg.bb_len, cfg.bb_std)
    bb_long  = crossunder(c, bb_mid) & in_sess & in_sess_prev
    bb_short = crossover (c, bb_mid) & in_sess & in_sess_prev

    vwap_v = vwap_session(df, df["session_id"])
    vwap_long  = crossunder(c, vwap_v) & in_sess & in_sess_prev
    vwap_short = crossover (c, vwap_v) & in_sess & in_sess_prev

    _ml, _sl, macd_hist = macd(c, cfg.ma_fast, cfg.ma_slow, cfg.ma_sig)
    macd_long  = crossover (macd_hist, 0) & in_sess & in_sess_prev
    macd_short = crossunder(macd_hist, 0) & in_sess & in_sess_prev

    stoch_k = sma(stoch(c, h, l, cfg.stoch_k), cfg.stoch_ks)
    stoch_long  = crossover (stoch_k, 50) & in_sess & in_sess_prev
    stoch_short = crossunder(stoch_k, 50) & in_sess & in_sess_prev

    rsi_div = rsi(c, cfg.div_len)
    lowest_l_prev  = l.rolling(cfg.div_lb).min().shift(1)
    lowest_r_prev  = rsi_div.rolling(cfg.div_lb).min().shift(1)
    highest_h_prev = h.rolling(cfg.div_lb).max().shift(1)
    highest_r_prev = rsi_div.rolling(cfg.div_lb).max().shift(1)
    bull_div = (l < lowest_l_prev) & (rsi_div > lowest_r_prev)
    bear_div = (h > highest_h_prev) & (rsi_div < highest_r_prev)

    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr_ma = sma(tr, cfg.vol_ma_len)
    vol_raw   = (tr > tr_ma) & (tr.shift(1) <= tr_ma.shift(1))
    vol_long  = vol_raw & (c > o)
    vol_short = vol_raw & (c < o)

    # DNA candle filter
    body  = (c - o).abs()
    rng   = h - l
    body_ratio = np.where(rng > 0, body / rng, 0.0)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    if cfg.use_dna:
        dna_long  = (body_ratio >= cfg.dna_body) & (c > o) & (lower_wick < body)
        dna_short = (body_ratio >= cfg.dna_body) & (c < o) & (upper_wick < body)
    else:
        dna_long  = pd.Series(True, index=df.index)
        dna_short = pd.Series(True, index=df.index)

    # Regime
    _dp, _dm, adx_v = dmi(h, l, c, 14)
    regime_ok = pd.Series(True, index=df.index) if not cfg.use_regime else (adx_v >= cfg.regime_adx)

    # Combine signals
    or_long = (cfg.use_stoch  & stoch_long.fillna(False)) | \
              (cfg.use_bb     & bb_long.fillna(False))    | \
              (cfg.use_macd   & macd_long.fillna(False))  | \
              (cfg.use_vwap   & vwap_long.fillna(False))  | \
              (cfg.use_rsi    & rsi_long.fillna(False))   | \
              (cfg.use_rsidiv & bull_div.fillna(False))   | \
              (cfg.use_vol    & vol_long.fillna(False))
    or_short= (cfg.use_stoch  & stoch_short.fillna(False)) | \
              (cfg.use_bb     & bb_short.fillna(False))    | \
              (cfg.use_macd   & macd_short.fillna(False))  | \
              (cfg.use_vwap   & vwap_short.fillna(False))  | \
              (cfg.use_rsi    & rsi_short.fillna(False))   | \
              (cfg.use_rsidiv & bear_div.fillna(False))    | \
              (cfg.use_vol    & vol_short.fillna(False))

    if cfg.ind_rev:
        long_sig, short_sig = or_short, or_long
    else:
        long_sig, short_sig = or_long, or_short

    first_session_bar = in_sess & (~in_sess_prev)

    atr_trail = atr(h, l, c, cfg.trail_atr)

    # ── BAR-BY-BAR LOOP (sequential, mirrors Pine execution order) ────────
    trades: list[Trade] = []
    equity = initial_capital
    day_eq = initial_capital
    day_cnt = 0
    last_entry_bar = -10_000
    last_entry_dir = 0
    current_day = None

    # Open position state
    pos_dir = 0     # 0/+1/-1
    pos_qty = 0
    pos_entry_px = 0.0
    pos_entry_time = None
    sl_px = 0.0
    tp_px = 0.0
    trail_level = float("nan")

    equity_at_close = []
    times_at_close  = []

    tick = cfg.tick_size
    pv   = cfg.point_value
    comm = cfg.contract_commission

    # Pre-compute close-time minutes
    close_hm = cfg.close_time.hour * 100 + cfg.close_time.minute

    df_arr = df[["open", "high", "low", "close"]].to_numpy()
    in_sess_arr = in_sess.to_numpy()
    first_session_bar_arr = first_session_bar.to_numpy()
    long_sig_arr  = long_sig.to_numpy()
    short_sig_arr = short_sig.to_numpy()
    dna_long_arr  = (dna_long if isinstance(dna_long, pd.Series) else pd.Series(dna_long, index=df.index)).to_numpy()
    dna_short_arr = (dna_short if isinstance(dna_short, pd.Series) else pd.Series(dna_short, index=df.index)).to_numpy()
    regime_ok_arr = regime_ok.to_numpy()
    day_ok_arr    = day_ok.to_numpy()
    atr_arr       = atr_trail.to_numpy()
    local_times   = idx_local

    n = len(df)
    for i in range(n):
        ts = df.index[i]
        local_ts = local_times[i]
        op, hi, lo, cl = df_arr[i]
        if np.isnan(cl):
            continue

        # New day reset
        day_str = local_ts.date()
        if current_day is None:
            current_day = day_str
        elif day_str != current_day:
            current_day = day_str
            day_eq = equity
            day_cnt = 0
            last_entry_bar = -10_000
            last_entry_dir = 0

        hm = local_ts.hour * 100 + local_ts.minute
        daily_pnl = equity - day_eq
        daily_loss_ok   = daily_pnl > -abs(cfg.max_loss)
        daily_profit_ok = daily_pnl <  abs(cfg.max_profit)
        trades_ok = day_cnt < cfg.max_trades

        # ── Manage open position: check SL/TP/Trail first (intra-bar) ────
        if pos_dir != 0:
            exit_px = None; reason = ""
            if cfg.use_trail and not math.isnan(trail_level):
                if pos_dir > 0:
                    trail_level = max(trail_level, cl - atr_arr[i] * cfg.trail_mult) if not math.isnan(atr_arr[i]) else trail_level
                    if cl < trail_level:
                        exit_px, reason = cl, "🔁 Trail"
                else:
                    trail_level = min(trail_level, cl + atr_arr[i] * cfg.trail_mult) if not math.isnan(atr_arr[i]) else trail_level
                    if cl > trail_level:
                        exit_px, reason = cl, "🔁 Trail"
            else:
                # Standard SL/TP check (bar-touch model)
                if pos_dir > 0:
                    if lo <= sl_px:
                        exit_px, reason = sl_px, "SL"
                    elif hi >= tp_px:
                        exit_px, reason = tp_px, "TP"
                else:
                    if hi >= sl_px:
                        exit_px, reason = sl_px, "SL"
                    elif lo <= tp_px:
                        exit_px, reason = tp_px, "TP"

            # Forced closes (mirror Pine priority order)
            if exit_px is None:
                close_reason = ""
                if cfg.use_close and hm >= close_hm:                close_reason = "⏰ סגירה"
                elif not in_sess_arr[i]:                            close_reason = "🏁 סשן"
                elif not daily_loss_ok:                             close_reason = "❌ הפסד"
                elif not daily_profit_ok:                           close_reason = "✅ רווח"
                if close_reason:
                    exit_px, reason = cl, close_reason

            if exit_px is not None:
                pnl = (exit_px - pos_entry_px) * pos_dir * pos_qty * pv
                pnl -= 2 * comm * pos_qty  # entry + exit commission
                equity += pnl
                trades.append(Trade(pos_entry_time, ts, pos_dir, pos_entry_px, exit_px, pos_qty, pnl, reason))
                pos_dir = 0; pos_qty = 0
                trail_level = float("nan")

        # ── Entries (at close, mirrors process_orders_on_close=true) ──────
        base_ok = (in_sess_arr[i] and daily_loss_ok and daily_profit_ok
                   and trades_ok and bool(day_ok_arr[i]) and pos_dir == 0)
        bars_since_entry = i - last_entry_bar

        can_long  = (base_ok and bool(dna_long_arr[i])  and bool(regime_ok_arr[i])
                     and not first_session_bar_arr[i]
                     and (last_entry_dir == 0 or last_entry_dir == 1
                          or (last_entry_dir == -1 and bars_since_entry >= cfg.cd_opp)))
        can_short = (base_ok and bool(dna_short_arr[i]) and bool(regime_ok_arr[i])
                     and not first_session_bar_arr[i]
                     and (last_entry_dir == 0 or last_entry_dir == -1
                          or (last_entry_dir == 1 and bars_since_entry >= cfg.cd_opp)))

        if long_sig_arr[i] and can_long:
            pos_dir = 1; pos_qty = cfg.contracts
            pos_entry_px = cl; pos_entry_time = ts
            sl_px = cl - cfg.stop_ticks * tick
            tp_px = cl + cfg.stop_ticks * tick * cfg.rr_ratio
            trail_level = cl - atr_arr[i] * cfg.trail_mult if cfg.use_trail and not math.isnan(atr_arr[i]) else float("nan")
            day_cnt += 1
            last_entry_bar = i; last_entry_dir = 1
        elif short_sig_arr[i] and can_short:
            pos_dir = -1; pos_qty = cfg.contracts
            pos_entry_px = cl; pos_entry_time = ts
            sl_px = cl + cfg.stop_ticks * tick
            tp_px = cl - cfg.stop_ticks * tick * cfg.rr_ratio
            trail_level = cl + atr_arr[i] * cfg.trail_mult if cfg.use_trail and not math.isnan(atr_arr[i]) else float("nan")
            day_cnt += 1
            last_entry_bar = i; last_entry_dir = -1

        equity_at_close.append(equity)
        times_at_close.append(ts)

    eq = pd.Series(equity_at_close, index=times_at_close, name="equity")
    return BacktestResult(trades=trades, equity_curve=eq, initial_capital=initial_capital)

# ────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ────────────────────────────────────────────────────────────────────────────

def load_yfinance(ticker: str = "NQ=F", period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]

def make_synthetic(days: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic 15-min data for engine validation only — NOT a real backtest."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, 0, 0, tzinfo=pytz.UTC)
    n_bars = days * 24 * 4
    times = pd.date_range(start, periods=n_bars, freq="15min", tz="UTC")
    # Geometric Brownian motion on the close path, then derive realistic candles
    drift = 0.00002
    vol   = 0.002
    rets  = rng.normal(drift, vol, n_bars)
    close = 17000 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    # Intra-bar range relative to bar magnitude
    half_rng = np.abs(close - open_) * rng.uniform(0.4, 1.5, n_bars) + close * 0.0005
    raw_high = np.maximum(open_, close) + half_rng
    raw_low  = np.minimum(open_, close) - half_rng
    volume = rng.integers(500, 5000, n_bars)
    return pd.DataFrame({"open": open_, "high": raw_high, "low": raw_low,
                         "close": close, "volume": volume}, index=times)

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def print_report(res: BacktestResult, label: str = "Backtest"):
    s = res.stats
    print(f"\n┌─ {label} ─────────────────────────────")
    if s.get("trades", 0) == 0:
        print("│  No trades."); print("└" + "─" * 45); return
    print(f"│  Trades:        {s['trades']:>10}")
    print(f"│  Win rate:      {s['win_rate']:>9.1f}%")
    print(f"│  Wins/Losses:   {s['wins']:>5}/{s['losses']:<5}")
    print(f"│  Total PnL:     ${s['total_pnl']:>10,.2f}")
    print(f"│  Avg win:       ${s['avg_win']:>10,.2f}")
    print(f"│  Avg loss:      ${s['avg_loss']:>10,.2f}")
    print(f"│  Profit factor: {s['profit_factor']:>10.2f}")
    print(f"│  Max drawdown:  ${s['max_drawdown']:>10,.2f}")
    print(f"│  Final equity:  ${s['final_equity']:>10,.2f}")
    print(f"│  Return:        {s['return_pct']:>9.2f}%")
    print("└" + "─" * 45)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["yfinance", "csv", "synthetic"], default="synthetic")
    p.add_argument("--csv", help="Path to CSV with columns: timestamp,open,high,low,close,volume")
    p.add_argument("--ticker", default="NQ=F")
    p.add_argument("--period", default="60d")
    p.add_argument("--interval", default="15m")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    if args.source == "yfinance":
        df = load_yfinance(args.ticker, args.period, args.interval)
        label = f"NQ {args.period} @ {args.interval}"
    elif args.source == "csv":
        df = load_csv(args.csv)
        label = f"CSV: {args.csv}"
    else:
        df = make_synthetic()
        label = "SYNTHETIC (engine validation only)"

    print(f"Data: {df.shape[0]} bars | {df.index[0]} → {df.index[-1]}")

    cfg = Config()  # default = STOCH + BB only (matches Pine default)
    res = run_backtest(df, cfg)
    print_report(res, label)

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        res.equity_curve.plot(ax=ax, color="#00C853")
        ax.set_title(f"ProfitBot v3.2 — Equity Curve ({label})")
        ax.set_ylabel("Equity ($)"); ax.grid(alpha=0.3)
        out = "/tmp/equity.png"
        plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
        print(f"Plot saved: {out}")

    return res

if __name__ == "__main__":
    main()
