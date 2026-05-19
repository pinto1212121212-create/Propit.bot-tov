"""
Strategy search — sweep indicator combos × sessions × stop_ticks on
realistic NQ data, rank by composite prop-firm score.

Each backtest is fully vectorized except for the entry/exit sim loop.
Total runtime for ~150 configs: ~20 seconds.
"""
from __future__ import annotations
import itertools, sys, time, math
from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, "/home/user/Propit.bot-tov/backtest")
from realistic_nq import make_realistic_nq
from profitbot_backtest import sma, ema, rsi, bb, macd, dmi

TICK     = 0.25
POINT_V  = 20.0
COMM     = 2.0  # per side per contract

# ════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE ALL INDICATOR SIGNALS (one-time, expensive)
# ════════════════════════════════════════════════════════════════════════

def precompute_signals(df: pd.DataFrame):
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # RSI (default 14, OS 30, OB 70)
    rsi_v   = rsi(c, 14)

    # BB(18) crossover mid
    bb_mid, _, _ = bb(c, 18, 2.0)
    bb_long  = (c.shift(1) >= bb_mid.shift(1)) & (c < bb_mid)
    bb_short = (c.shift(1) <= bb_mid.shift(1)) & (c > bb_mid)

    # MACD(12,26,9) histogram zero-cross
    _, _, macd_h = macd(c, 12, 26, 9)
    macd_long  = (macd_h.shift(1) <= 0) & (macd_h > 0)
    macd_short = (macd_h.shift(1) >= 0) & (macd_h < 0)

    # VWAP (session reset by date)
    date_id = df.index.tz_convert("Asia/Jerusalem").date
    tp = (h + l + c) / 3
    pv = tp * v
    cum_pv = pv.groupby(date_id).cumsum()
    cum_v  = v.groupby(date_id).cumsum()
    vwap_v = cum_pv / cum_v.replace(0, np.nan)
    vwap_long  = (c.shift(1) >= vwap_v.shift(1)) & (c < vwap_v)
    vwap_short = (c.shift(1) <= vwap_v.shift(1)) & (c > vwap_v)

    # STOCH(6,1) cross 50
    ll = l.rolling(6).min(); hh = h.rolling(6).max()
    stoch_k = ((c - ll) / (hh - ll) * 100).rolling(1).mean()
    stoch_long  = (stoch_k.shift(1) <= 50) & (stoch_k > 50)
    stoch_short = (stoch_k.shift(1) >= 50) & (stoch_k < 50)

    # VOL = TR cross of MA20
    prev_c = c.shift(1)
    tr = np.maximum.reduce([h-l, (h-prev_c).abs(), (l-prev_c).abs()])
    tr = pd.Series(tr, index=df.index)
    tr_ma = sma(tr, 20)
    vol_raw   = (tr > tr_ma) & (tr.shift(1) <= tr_ma.shift(1))
    vol_long  = vol_raw & (c > o)
    vol_short = vol_raw & (c < o)

    # RSI threshold (state, not crossover — fires every bar while extreme)
    rsi_long  = rsi_v <= 30
    rsi_short = rsi_v >= 70

    return {
        "STOCH": (stoch_long.fillna(False), stoch_short.fillna(False)),
        "BB":    (bb_long.fillna(False),    bb_short.fillna(False)),
        "MACD":  (macd_long.fillna(False),  macd_short.fillna(False)),
        "VWAP":  (vwap_long.fillna(False),  vwap_short.fillna(False)),
        "RSI":   (rsi_long.fillna(False),   rsi_short.fillna(False)),
        "VOL":   (vol_long.fillna(False),   vol_short.fillna(False)),
    }

# ════════════════════════════════════════════════════════════════════════
# SESSION MASK
# ════════════════════════════════════════════════════════════════════════

def session_mask(idx, start: dtime, end: dtime) -> np.ndarray:
    """1 if local time within [start, end), else 0."""
    local = idx.tz_convert("Asia/Jerusalem")
    h = local.hour; m = local.minute
    cur = h*60 + m
    s = start.hour*60 + start.minute
    e = end.hour*60 + end.minute
    return ((cur >= s) & (cur < e) & (local.dayofweek < 5))  # Mon-Fri, already numpy

# ════════════════════════════════════════════════════════════════════════
# SIMULATION (vectorized except the sequential entry-exit logic)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    trades: int
    wins:   int
    losses: int
    pnl:    float
    win_rate: float
    avg_win:  float
    avg_loss: float
    max_dd:   float
    max_consec_loss: int
    expectancy: float
    sharpe_d:   float    # daily Sharpe approx

def simulate(df: pd.DataFrame, in_sess: np.ndarray,
             or_long: np.ndarray, or_short: np.ndarray,
             stop_ticks: int = 30, rr: float = 2.0,
             max_loss: float = 700, max_trades: int = 4,
             cd_opp: int = 4, last_bar_guard: int = 3,
             initial_cap: float = 10_000) -> SimResult:

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    in_sess_prev = np.concatenate(([False], in_sess[:-1]))
    valid_signal = in_sess & in_sess_prev

    or_l = or_long & valid_signal
    or_s = or_short & valid_signal

    # Last-bar guard: count consecutive in_sess bars before each, find session ends
    # For prop firm style sessions (continuous block), block the last N bars.
    # Simpler: pre-compute "bars until session ends" via forward fill
    session_end = ~in_sess & in_sess_prev   # the bar right after session ends is one_after
    # Build "bars_to_end" via reverse cumsum within each session
    end_bar = np.zeros(len(in_sess), dtype=int)
    last_end = len(in_sess)
    for i in range(len(in_sess)-1, -1, -1):
        if not in_sess[i]:
            last_end = i
        end_bar[i] = last_end
    bars_to_end = end_bar - np.arange(len(in_sess))
    block_last = in_sess & (bars_to_end <= last_bar_guard)
    or_l = or_l & ~block_last
    or_s = or_s & ~block_last

    sl_pts = stop_ticks * TICK
    tp_pts = sl_pts * rr
    pnl_win  = tp_pts * POINT_V - 2*COMM
    pnl_loss = -sl_pts * POINT_V - 2*COMM

    # State
    equity = initial_cap
    day_eq = initial_cap
    day_cnt = 0
    last_entry = -10_000
    last_dir = 0
    cur_day = None

    trades = []  # (pnl, day)
    eq_curve = []

    # Date for daily reset (use Asia/Jerusalem date as "day")
    local_dates = df.index.tz_convert("Asia/Jerusalem").date

    # Active position
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0

    for i in range(len(df)):
        day = local_dates[i]
        if cur_day is None:
            cur_day = day
        elif day != cur_day:
            cur_day = day
            day_eq = equity
            day_cnt = 0
            last_entry = -10_000
            last_dir = 0

        # Manage open
        if pos_dir != 0:
            exit_px = None
            if pos_dir > 0:
                if l[i] <= pos_sl: exit_px = pos_sl
                elif h[i] >= pos_tp: exit_px = pos_tp
            else:
                if h[i] >= pos_sl: exit_px = pos_sl
                elif l[i] <= pos_tp: exit_px = pos_tp
            if exit_px is None and not in_sess[i]:
                exit_px = c[i]
            if exit_px is not None:
                pnl = (exit_px - pos_entry) * pos_dir * POINT_V - 2*COMM
                equity += pnl
                trades.append((pnl, day))
                pos_dir = 0

        # Daily cap check
        daily_pnl = equity - day_eq
        if daily_pnl <= -max_loss or day_cnt >= max_trades:
            eq_curve.append(equity)
            continue

        # Entry
        bars_since = i - last_entry
        if pos_dir == 0 and in_sess[i]:
            if or_l[i] and (last_dir != -1 or bars_since >= cd_opp):
                pos_dir = 1
                pos_entry = c[i]
                pos_sl = c[i] - sl_pts
                pos_tp = c[i] + tp_pts
                day_cnt += 1
                last_entry = i
                last_dir = 1
            elif or_s[i] and (last_dir != 1 or bars_since >= cd_opp):
                pos_dir = -1
                pos_entry = c[i]
                pos_sl = c[i] + sl_pts
                pos_tp = c[i] - tp_pts
                day_cnt += 1
                last_entry = i
                last_dir = -1

        eq_curve.append(equity)

    # Close any leftover at the end
    if pos_dir != 0:
        pnl = (c[-1] - pos_entry) * pos_dir * POINT_V - 2*COMM
        equity += pnl
        trades.append((pnl, local_dates[-1]))

    # Metrics
    if not trades:
        return SimResult(0,0,0,0.0,0.0,0,0,0,0,0.0,0.0)
    pnls = np.array([t[0] for t in trades])
    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    win_rate = wins / len(trades) if trades else 0
    avg_w = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_l = pnls[pnls < 0].mean() if losses > 0 else 0
    eq_arr = np.array(eq_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak)
    max_dd = dd.min() if len(dd) > 0 else 0
    # Consec losses
    streak = 0; max_streak = 0
    for p in pnls:
        if p < 0:
            streak += 1; max_streak = max(max_streak, streak)
        else:
            streak = 0
    expectancy = pnls.mean()
    # Daily Sharpe approx
    df_t = pd.DataFrame({"pnl": pnls, "day": [t[1] for t in trades]})
    daily = df_t.groupby("day")["pnl"].sum()
    sharpe = daily.mean() / daily.std() if daily.std() > 0 else 0
    return SimResult(
        trades=len(trades), wins=wins, losses=losses, pnl=float(pnls.sum()),
        win_rate=float(win_rate), avg_win=float(avg_w), avg_loss=float(avg_l),
        max_dd=float(max_dd), max_consec_loss=int(max_streak),
        expectancy=float(expectancy), sharpe_d=float(sharpe),
    )

# ════════════════════════════════════════════════════════════════════════
# SEARCH SPACE
# ════════════════════════════════════════════════════════════════════════

INDICATOR_POOL = ["STOCH", "BB", "MACD", "VWAP", "RSI", "VOL"]

SESSIONS = [
    ("04:00-08:00", dtime(4,0),  dtime(8,0)),
    ("06:00-10:00", dtime(6,0),  dtime(10,0)),
    ("08:00-12:00", dtime(8,0),  dtime(12,0)),
    ("10:00-14:00", dtime(10,0), dtime(14,0)),
    ("13:00-17:00", dtime(13,0), dtime(17,0)),
    ("14:00-18:00", dtime(14,0), dtime(18,0)),
    ("16:00-20:00", dtime(16,0), dtime(20,0)),
    ("16:30-20:30", dtime(16,30),dtime(20,30)),
    ("18:00-22:00", dtime(18,0), dtime(22,0)),
]
STOP_TICKS = [20, 30, 40]
RR         = [1.5, 2.0, 2.5]

def combine_signals(signals: dict, combo: tuple) -> tuple:
    long_arr  = np.zeros_like(signals[combo[0]][0].values, dtype=bool)
    short_arr = np.zeros_like(signals[combo[0]][1].values, dtype=bool)
    for ind in combo:
        long_arr  = long_arr  | signals[ind][0].values
        short_arr = short_arr | signals[ind][1].values
    return long_arr, short_arr

def composite_score(r: SimResult, days: int) -> float:
    """Prop-firm friendly score: high PnL, smooth equity, enough trades."""
    if r.trades < days * 0.3:  # need at least 0.3 trades/day
        return -9999
    if r.max_dd < -2500:        # rule out blowups
        return -9999
    # PnL weighted by sharpe; penalize huge drawdowns
    return r.pnl + 200 * r.sharpe_d + 50 * r.win_rate - 0.5 * abs(r.max_dd)

# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("Generating realistic NQ data...")
    df = make_realistic_nq(days=90, seed=42)
    n_days = (df.index[-1] - df.index[0]).days
    print(f"  {len(df)} bars over {n_days} days")

    print("Pre-computing indicator signals...")
    signals = precompute_signals(df)

    # Build combos: pairs, triples, quads
    combos = []
    for k in (2, 3, 4):
        for c in itertools.combinations(INDICATOR_POOL, k):
            combos.append(c)
    print(f"Indicator combos: {len(combos)}")
    print(f"Session windows: {len(SESSIONS)}")
    print(f"Stop_ticks values: {STOP_TICKS}")

    # Search: fix RR=2.0, sweep combos × sessions × stop_ticks (manageable)
    rr_fixed = 2.0
    print(f"\nRunning {len(combos) * len(SESSIONS) * len(STOP_TICKS)} simulations...")

    results = []
    for s_label, s_start, s_end in SESSIONS:
        in_sess = session_mask(df.index, s_start, s_end)
        for combo in combos:
            or_l, or_s = combine_signals(signals, combo)
            for st in STOP_TICKS:
                r = simulate(df, in_sess, or_l, or_s, stop_ticks=st, rr=rr_fixed)
                score = composite_score(r, n_days)
                results.append({
                    "session": s_label,
                    "combo":   "+".join(combo),
                    "n_ind":   len(combo),
                    "stop":    st,
                    "trades":  r.trades,
                    "win_rate":round(r.win_rate*100, 1),
                    "pnl":     round(r.pnl, 0),
                    "avg_win": round(r.avg_win, 0),
                    "avg_loss":round(r.avg_loss, 0),
                    "max_dd":  round(r.max_dd, 0),
                    "consec_L":r.max_consec_loss,
                    "sharpe":  round(r.sharpe_d, 2),
                    "exp":     round(r.expectancy, 1),
                    "score":   round(score, 0),
                })

    res_df = pd.DataFrame(results).sort_values("score", ascending=False)
    print(f"\nDone in {time.time()-t0:.1f}s")

    # Save
    res_df.to_csv("/home/user/Propit.bot-tov/backtest/search_results.csv", index=False)
    print(f"Saved {len(res_df)} rows → backtest/search_results.csv")

    # Top 15
    print("\n" + "="*110)
    print("TOP 15 STRATEGIES (by composite score)")
    print("="*110)
    print(res_df.head(15).to_string(index=False))

    # Best per session
    print("\n" + "="*110)
    print("BEST PER SESSION WINDOW")
    print("="*110)
    best_per_session = res_df.groupby("session").head(1).sort_values("score", ascending=False)
    print(best_per_session.to_string(index=False))

    # Best per indicator combo size
    print("\n" + "="*110)
    print("BEST BY COMBO SIZE")
    print("="*110)
    print(res_df.groupby("n_ind").head(3).sort_values(["n_ind", "score"], ascending=[True, False]).to_string(index=False))

    return res_df

if __name__ == "__main__":
    main()
