"""
SMA crossover strategy signals.

Generates long/short signals based on a set of fast and slow simple moving
averages: long when all fast SMAs are above all slow SMAs; short when all fast
SMAs are below all slow SMAs; otherwise flat.
"""

import pandas as pd
from typing import List
from ..features.indicators import sma

def sma_crossover_signals(prices: pd.Series, fast_windows: List[int], slow_windows: List[int]) -> pd.Series:
    """Compute SMA crossover signals from price series.

    Args:
        prices (pd.Series): Price series indexed by datetime.
        fast_windows (List[int]): Window sizes for fast SMAs.
        slow_windows (List[int]): Window sizes for slow SMAs.

    Returns:
        pd.Series: Signal in {-1, 0, 1} aligned to `prices`.
    """
    df = pd.DataFrame(index=prices.index)
    for w in fast_windows:
        df[f"fast_{w}"] = sma(prices, w)
    for w in slow_windows:
        df[f"slow_{w}"] = sma(prices, w)
    fast_stack = df[[c for c in df.columns if c.startswith("fast_")]]
    slow_stack = df[[c for c in df.columns if c.startswith("slow_")]]
    cond_long = (fast_stack.min(axis=1) > slow_stack.max(axis=1))
    cond_short = (fast_stack.max(axis=1) < slow_stack.min(axis=1))
    sig = pd.Series(0.0, index=prices.index)
    sig[cond_long] = 1.0
    sig[cond_short] = -1.0
    return sig
