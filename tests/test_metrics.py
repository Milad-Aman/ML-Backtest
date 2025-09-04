import pandas as pd
import numpy as np
from src.eval.metrics import sharpe, max_drawdown

def test_sharpe_basic():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    s = sharpe(r, ann_factor=1)
    assert np.isfinite(s)

def test_mdd_monotonic():
    eq = pd.Series([1,1.1,1.21,1.0,0.9,1.3])
    mdd = max_drawdown(eq)
    assert mdd <= 0.0
