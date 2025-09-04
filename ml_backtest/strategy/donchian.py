"""
Donchian channel breakout signals.

Enters long when close breaks above the prior upper band; enters short when
close breaks below the prior lower band. Exits use shorter exit bands.
Maintains position state across time.
"""

import pandas as pd
from ..features.indicators import donchian

def donchian_signals(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 55, exit_lookback: int = 20) -> pd.Series:
    """Generate Donchian breakout signals with separate exit lookback.

    Args:
        high, low, close (pd.Series): OHLC series aligned on the same index.
        lookback (int): Lookback for entry Donchian channel.
        exit_lookback (int): Lookback for exit Donchian channel.

    Returns:
        pd.Series: Position signal in {-1, 0, 1} with 1-bar lag on entries.
    """
    upper, lower, _ = donchian(high, low, lookback)
    upper_exit, lower_exit, _ = donchian(high, low, exit_lookback)
    long_entry = close > upper.shift(1)
    short_entry = close < lower.shift(1)
    sig = pd.Series(0.0, index=close.index)
    position = 0.0
    for t in range(1, len(close)):
        if position == 0.0:
            if long_entry.iloc[t]:
                position = 1.0
            elif short_entry.iloc[t]:
                position = -1.0
        elif position == 1.0 and close.iloc[t] < lower_exit.shift(1).iloc[t]:
            position = 0.0
        elif position == -1.0 and close.iloc[t] > upper_exit.shift(1).iloc[t]:
            position = 0.0
        sig.iloc[t] = position
    return sig
