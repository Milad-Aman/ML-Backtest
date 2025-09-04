import pandas as pd
from src.strategy.sma import sma_crossover_signals

def test_signal_is_t_only():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    price = pd.Series(range(100), index=idx).astype(float)
    sig = sma_crossover_signals(price, [5], [20])
    assert sig.index.equals(price.index)
