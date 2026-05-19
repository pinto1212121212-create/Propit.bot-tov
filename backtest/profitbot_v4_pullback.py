"""
ProfitBot Jerusalem v4.0 — Momentum Pullback (Python backtest)

3-layer entry: direction alignment → context filters → pullback trigger.
No crossover entries. Structure-based SL, RR-based TP.
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytz

# Reuse indicators + helpers from v3.2 engine
sys.path.insert(0, "/home/user/Propit.bot-tov/backtest")
from profitbot_backtest import (
    sma, ema, rma, rsi, bb, macd, atr, dmi,
    crossover, crossunder, in_session, make_synthetic,
    Trade, BacktestResult, print_report,
)

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigV4:
    # Sessions
    use_s1: bool = True
    sess1: tuple = (time(8, 0), time(12, 0))
    use_s2: bool = True
    sess2: tuple = (time(13, 0), time(16, 0))
    use_close: bool = True
    close_time: time = time(16, 0)
    tz: str = "Asia/Jerusalem"

    # Risk
    contracts: int = 1
    max_loss: float = 500.0
    max_profit: float = 500.0
    max_trades: int = 4
    rr_ratio: float = 2.0
    cd_bars: int = 4
    sl_buffer_t: int = 4
    sl_max_t: int = 80

    # Day mask (Python: Mon=0..Sun=6)
    use_sun: bool = True; use_mon: bool = True; use_tue: bool = True
    use_wed: bool = True; use_thu: bool = True
    use_fri: bool = False; use_sat: bool = False

    # Strategy params
    ema_len: int = 21
    macd_fast: int = 12
    macd_slow: int = 26
    macd_sig: int = 9
    adx_min: float = 20.0
    bbw_lb: int = 100
    bbw_pctile: float = 40.0
    vol_mult: float = 1.2
    pullback_lb: int = 3
    ema_touch_p: float = 0.15

    # Instrument
    tick_size: float = 0.25
    point_value: float = 20.0
    contract_commission: float = 2.0

# ────────────────────────────────────────────────────────────────────────────
# BACKTEST
# ────────────────────────────────────────────────────────────────────────────

def percentile_linear(s: pd.Series, n: int, p: float) -> pd.Series:
    """Rolling percentile via linear interpolation, mirrors ta.percentile_linear_interpolation."""
    return s.rolling(n, min_periods=n).quantile(p / 100.0, interpolation="linear")

def run_backtest_v4(df: pd.DataFrame, cfg: ConfigV4, initial_capital: float = 10_000.0) -> BacktestResult:
    tz = pytz.timezone(cfg.tz)
    idx_local = df.index.tz_convert(tz)
    df = df.copy()
    df["local_time"] = idx_local
    df["dow"] = idx_local.dayofweek
    df["session_id"] = idx_local.date

    in_s1 = pd.Series([cfg.use_s1 and in_session(t, *cfg.sess1) for t in idx_local], index=df.index, dtype=bool)
    in_s2 = pd.Series([cfg.use_s2 and in_session(t, *cfg.sess2) for t in idx_local], index=df.index, dtype=bool)
    in_sess = (in_s1 | in_s2).astype(bool)
    in_sess_prev = in_sess.shift(1, fill_value=False).astype(bool)

    day_map = {6: cfg.use_sun, 0: cfg.use_mon, 1: cfg.use_tue, 2: cfg.use_wed,
               3: cfg.use_thu, 4: cfg.use_fri, 5: cfg.use_sat}
    day_ok = pd.Series(df["dow"].map(day_map).values, index=df.index, dtype=bool)

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    ema21 = ema(c, cfg.ema_len)
    ema_slope = ema21 - ema21.shift(3)

    # Session-reset VWAP
    tp = (h + l + c) / 3
    pv = tp * v
    cum_pv = pv.groupby(df["session_id"]).cumsum()
    cum_v  = v.groupby(df["session_id"]).cumsum()
    vwap_v = cum_pv / cum_v.replace(0, np.nan)

    _ml, _sl, macd_hist = macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_sig)
    _dp, _dm, adx_v = dmi(h, l, c, 14)
    bb_mid, bb_up, bb_dn = bb(c, 20, 2.0)
    bbw = (bb_up - bb_dn) / bb_mid * 100
    bbw_thresh = percentile_linear(bbw, cfg.bbw_lb, cfg.bbw_pctile)
    atr_v = atr(h, l, c, 14)
    vol_ma = sma(v, 20)

    # Layer 1 — direction
    dir_long  = (ema_slope > 0) & (c > vwap_v) & (macd_hist > 0)
    dir_short = (ema_slope < 0) & (c < vwap_v) & (macd_hist < 0)

    # Layer 2 — context
    adx_ok = adx_v >= cfg.adx_min
    bbw_ok = bbw >= bbw_thresh
    vol_ok = v >= vol_ma * cfg.vol_mult
    context_ok = adx_ok & bbw_ok & vol_ok

    # Layer 3 — touched EMA in last `pullback_lb` bars
    proximity = cfg.ema_touch_p * atr_v
    touched = pd.Series(False, index=df.index)
    for i in range(cfg.pullback_lb + 1):
        diff_low  = (l.shift(i)  - ema21.shift(i)).abs()
        diff_high = (h.shift(i) - ema21.shift(i)).abs()
        prox_i = proximity.shift(i)
        touched = touched | (diff_low <= prox_i) | (diff_high <= prox_i)

    # Reversal candle triggers
    body = (c - o).abs()
    rng  = h - l
    body_ratio = np.where(rng > 0, body / rng, 0.0)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l

    bullish_rev = (c > o) & (body_ratio >= 0.4) & (lower_wick > body * 0.5)
    bearish_rev = (c < o) & (body_ratio >= 0.4) & (upper_wick > body * 0.5)

    bull_engulf = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    bear_engulf = (c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))

    trigger_long  = (bullish_rev | bull_engulf) & (c > ema21)
    trigger_short = (bearish_rev | bear_engulf) & (c < ema21)

    first_session_bar = in_sess & (~in_sess_prev)

    # ── Simulation loop ──────────────────────────────────────────────────
    trades: list[Trade] = []
    equity = initial_capital
    day_eq = initial_capital
    day_cnt = 0
    last_entry_bar = -10_000
    current_day = None

    pos_dir = 0; pos_qty = 0
    pos_entry_px = 0.0; pos_entry_time = None
    sl_px = 0.0; tp_px = 0.0

    eq_values = []; eq_times = []

    tick = cfg.tick_size
    pv_   = cfg.point_value
    comm = cfg.contract_commission
    close_hm = cfg.close_time.hour * 100 + cfg.close_time.minute

    # Pre-roll lows/highs for SL
    pullback_low  = l.rolling(cfg.pullback_lb + 1, min_periods=1).min()
    pullback_high = h.rolling(cfg.pullback_lb + 1, min_periods=1).max()

    arr = df[["open", "high", "low", "close"]].to_numpy()
    sl_buf = cfg.sl_buffer_t * tick
    sl_max = cfg.sl_max_t * tick

    bool_cols = {
        "in_sess": in_sess.to_numpy(),
        "first_sb": first_session_bar.to_numpy(),
        "day_ok": day_ok.to_numpy(),
        "dir_long": dir_long.fillna(False).to_numpy(),
        "dir_short": dir_short.fillna(False).to_numpy(),
        "context_ok": context_ok.fillna(False).to_numpy(),
        "touched": touched.to_numpy(),
        "trig_long": trigger_long.fillna(False).to_numpy(),
        "trig_short": trigger_short.fillna(False).to_numpy(),
    }
    pb_low  = pullback_low.to_numpy()
    pb_high = pullback_high.to_numpy()
    local_times = idx_local

    for i in range(len(df)):
        ts = df.index[i]
        local_ts = local_times[i]
        op, hi, lo, cl = arr[i]
        if np.isnan(cl):
            continue

        day = local_ts.date()
        if current_day is None:
            current_day = day
        elif day != current_day:
            current_day = day
            day_eq = equity
            day_cnt = 0
            last_entry_bar = -10_000

        hm = local_ts.hour * 100 + local_ts.minute
        daily_pnl = equity - day_eq
        daily_loss_ok   = daily_pnl > -abs(cfg.max_loss)
        daily_profit_ok = daily_pnl <  abs(cfg.max_profit)
        trades_ok = day_cnt < cfg.max_trades

        # ── Manage open position ─────────────────────────────────────────
        if pos_dir != 0:
            exit_px = None; reason = ""
            if pos_dir > 0:
                if lo <= sl_px:      exit_px, reason = sl_px, "SL"
                elif hi >= tp_px:    exit_px, reason = tp_px, "TP"
            else:
                if hi >= sl_px:      exit_px, reason = sl_px, "SL"
                elif lo <= tp_px:    exit_px, reason = tp_px, "TP"

            if exit_px is None:
                if cfg.use_close and hm >= close_hm:      exit_px, reason = cl, "⏰ Close"
                elif not bool_cols["in_sess"][i]:         exit_px, reason = cl, "🏁 Session"
                elif not daily_loss_ok:                   exit_px, reason = cl, "❌ Daily Loss"
                elif not daily_profit_ok:                 exit_px, reason = cl, "✅ Daily Profit"

            if exit_px is not None:
                pnl = (exit_px - pos_entry_px) * pos_dir * pos_qty * pv_
                pnl -= 2 * comm * pos_qty
                equity += pnl
                trades.append(Trade(pos_entry_time, ts, pos_dir, pos_entry_px, exit_px, pos_qty, pnl, reason))
                pos_dir = 0; pos_qty = 0

        # ── Entries ──────────────────────────────────────────────────────
        base_ok = (bool_cols["in_sess"][i] and daily_loss_ok and daily_profit_ok
                   and trades_ok and bool_cols["day_ok"][i] and pos_dir == 0)
        cooldown_ok = (i - last_entry_bar) >= cfg.cd_bars
        no_first = not bool_cols["first_sb"][i]

        can_long = (base_ok and bool_cols["dir_long"][i] and bool_cols["context_ok"][i]
                    and bool_cols["touched"][i] and bool_cols["trig_long"][i]
                    and no_first and cooldown_ok)
        can_short = (base_ok and bool_cols["dir_short"][i] and bool_cols["context_ok"][i]
                     and bool_cols["touched"][i] and bool_cols["trig_short"][i]
                     and no_first and cooldown_ok)

        if can_long:
            sl_raw = pb_low[i] - sl_buf
            sl_px = max(sl_raw, cl - sl_max)
            risk = cl - sl_px
            if risk > 0:
                tp_px = cl + risk * cfg.rr_ratio
                pos_dir = 1; pos_qty = cfg.contracts
                pos_entry_px = cl; pos_entry_time = ts
                day_cnt += 1; last_entry_bar = i
        elif can_short:
            sl_raw = pb_high[i] + sl_buf
            sl_px = min(sl_raw, cl + sl_max)
            risk = sl_px - cl
            if risk > 0:
                tp_px = cl - risk * cfg.rr_ratio
                pos_dir = -1; pos_qty = cfg.contracts
                pos_entry_px = cl; pos_entry_time = ts
                day_cnt += 1; last_entry_bar = i

        eq_values.append(equity); eq_times.append(ts)

    eq = pd.Series(eq_values, index=eq_times, name="equity")
    return BacktestResult(trades=trades, equity_curve=eq, initial_capital=initial_capital)

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["yfinance", "csv", "synthetic"], default="synthetic")
    p.add_argument("--csv")
    p.add_argument("--ticker", default="NQ=F")
    p.add_argument("--period", default="60d")
    p.add_argument("--interval", default="15m")
    args = p.parse_args()

    if args.source == "yfinance":
        from profitbot_backtest import load_yfinance
        df = load_yfinance(args.ticker, args.period, args.interval)
        label = f"NQ {args.period} @ {args.interval}"
    elif args.source == "csv":
        from profitbot_backtest import load_csv
        df = load_csv(args.csv)
        label = f"CSV: {args.csv}"
    else:
        df = make_synthetic()
        label = "SYNTHETIC (engine validation only)"

    print(f"Data: {df.shape[0]} bars | {df.index[0]} → {df.index[-1]}")

    cfg = ConfigV4()
    res = run_backtest_v4(df, cfg)
    print_report(res, f"v4.0 Pullback — {label}")

    if res.trades:
        df_t = pd.DataFrame([{
            "entry":  t.entry_time, "exit": t.exit_time, "dir": "L" if t.direction > 0 else "S",
            "entry_px": round(t.entry_px, 2), "exit_px": round(t.exit_px, 2),
            "pnl": round(t.pnl, 2), "reason": t.exit_reason,
        } for t in res.trades])
        print("\nFirst 10 trades:")
        print(df_t.head(10).to_string(index=False))
        print("\nExit reason breakdown:")
        print(df_t["reason"].value_counts().to_string())
        # Trades per day
        df_t["date"] = pd.to_datetime(df_t["entry"]).dt.date
        per_day = df_t.groupby("date").size()
        print(f"\nTrades per day — mean: {per_day.mean():.2f}, max: {per_day.max()}, min: {per_day.min()}")

    return res

if __name__ == "__main__":
    main()
