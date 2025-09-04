"""
Regime labeling utilities.

Currently includes a simple high-volatility regime indicator based on rolling
standard deviation exceeding its rolling quantile threshold.
"""

import pandas as pd

def volatility_regime(returns: pd.Series, lookback: int = 60, quantile: float = 0.7) -> pd.Series:
    """Binary high-vol regime: 1 when vol > rolling quantile, else 0.

    Args:
        returns (pd.Series): Per-period (log) returns.
        lookback (int): Window for both stdev and its quantile.
        quantile (float): Quantile threshold, e.g., 0.7.

    Returns:
        pd.Series: 0/1 regime series aligned to `returns`.
    """
    rolling = returns.rolling(lookback, min_periods=lookback).std()
    q = rolling.rolling(lookback, min_periods=lookback).quantile(quantile)
    return (rolling > q).astype(int)
