"""
Walkforward backtesting utilities.

This module implements a simple walkforward evaluation framework:
- `rolling_windows` yields sequential train/test index splits with a fixed
  training window size (`min_train`) and a fixed forward test window size
  (`test_size`). The window advances by `test_size` each fold.
- `walkforward_run` orchestrates strategy signal generation per fold, executes
  a simple trading engine, collects per-fold metrics, and concatenates the
  out-of-sample (OOS) returns, positions, and equity.

Strategies supported via `config["strategy"]["name"]`:
- "sma": Simple moving average crossover (see `ml_backtest/strategy/sma.py`).
- "donchian": Donchian channel breakout (see `ml_backtest/strategy/donchian.py`).
- "ml_classifier": Supervised classifier over engineered features
  (see `ml_backtest/ml/pipelines.py` and `ml_backtest/features/featureset.py`).

Execution parameters are read from `config["execution"]` and passed to
`backtest_t1` (see `ml_backtest/execution/engine.py`). The framework avoids lookahead by
fitting models and generating signals using only data within each fold.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..execution.engine import backtest_t1
from ..strategy.sma import sma_crossover_signals
from ..strategy.donchian import donchian_signals
from ..features.featureset import build_feature_matrix
from ..ml.pipelines import build_model, fit_predict_proba
from .metrics import summarize

def rolling_windows(index, min_train: int, test_size: int):
    """Generator for sequential train/test splits without overlap.

    The first fold uses `index[0 : min_train]` for training and
    `index[min_train : min_train + test_size]` for testing. Each subsequent fold
    advances the start by `test_size`, keeping the train window size fixed at
    `min_train`.

    Args:
        index (pd.Index): Data index (typically timezone-aware datetime).
        min_train (int): Number of samples in each training window.
        test_size (int): Number of samples in each test window.

    Yields:
        Tuple[pd.Index, pd.Index]: Train and test indices for each fold.
    """
    start = 0
    n = len(index)
    while True:
        train_end = start + min_train
        test_end = train_end + test_size
        if test_end > n:
            break
        train_idx = index[start:train_end]
        test_idx = index[train_end:test_end]
        yield train_idx, test_idx
        start += test_size

def walkforward_run(config, df: pd.DataFrame):  # could include (rng: np.random.Generator) to support randomness
    """Run walkforward backtesting with the configured strategy and execution.

    Expected `config` structure (keys used):
    - `walkforward.min_train` (int): Train window size per fold.
      `walkforward.test_size` (int): Test window size per fold.
    - `strategy.name` (str): One of {"sma", "donchian", "ml_classifier"}.
      `strategy.params` (dict): Strategy-specific parameters, e.g.:
        - sma: `fast_windows` (int), `slow_windows` (int)
        - donchian: `lookback` (int), `exit_lookback` (int)
        - ml_classifier: `features` (dict), optional `model` (str), optional
          `prob_threshold` (float in (0,1))
    - `execution` (dict): Passed to `backtest_t1`, expects keys
      `fees_bps`, `slippage_bps`, `target_vol_annual`, `vol_lookback`, `max_leverage`.

    Data assumptions:
    - `df` contains `Adj Close` and, depending on strategy, may require `High`, `Low`.
    - Returns are log-differenced on `Adj Close` and the first NaN is dropped.
    - No lookahead: training and signal generation use only fold-local data.

    Args:
        config (dict): Strategy, execution, and walkforward parameters.
        df (pd.DataFrame): OHLCV price data indexed by datetime.

    Returns:
        folds_df (pd.DataFrame): One row per fold with metrics from `summarize`,
            augmented with `start` and `end` date strings for the test slice.
        oos_ret (pd.Series): Concatenated OOS net pnl (log returns) across folds.
        oos_pos (pd.Series): Concatenated OOS positions across folds.
        oos_eq (pd.DataFrame): Concatenated OOS equity curve with column `equity`.
    """
    price = df["Adj Close"]
    ret = np.log(price).diff().dropna()
    df = df.loc[ret.index]

    min_train = int(config["walkforward"]["min_train"])
    test_size = int(config["walkforward"]["test_size"])

    folds = []
    oos_returns = []
    oos_positions = []
    oos_equity_df = []

    for ti, vi in rolling_windows(df.index, min_train=min_train, test_size=test_size):
        train = df.loc[ti]
        test = df.loc[vi]

        if config["strategy"]["name"] == "sma":
            p = config["strategy"]["params"]
            sig_test  = sma_crossover_signals(test["Adj Close"], p["fast_windows"], p["slow_windows"])
        elif config["strategy"]["name"] == "donchian":
            p = config["strategy"]["params"]
            sig_test  = donchian_signals(test["High"], test["Low"], test["Adj Close"], p["lookback"], p["exit_lookback"])
        elif config["strategy"]["name"] == "ml_classifier":
            p = config["strategy"]["params"]
            X = build_feature_matrix(df, **p["features"])
            y = (np.log(df["Adj Close"]).diff().shift(-1) > 0).astype(int)
            Xy = pd.concat([X, y.rename("y")], axis=1).dropna()
            train_idx = Xy.index.intersection(ti)
            test_idx  = Xy.index.intersection(vi)
            X_train, y_train = Xy.loc[train_idx].drop(columns=["y"]), Xy.loc[train_idx, "y"]
            X_test  = Xy.loc[test_idx].drop(columns=["y"])
            model = build_model(p.get("model","random_forest"))
            proba = fit_predict_proba(model, X_train, y_train, X_test)
            thr = float(p.get("prob_threshold", 0.55))
            sig_test = pd.Series(0.0, index=test_idx)
            sig_test[proba > thr] = 1.0
            sig_test[proba < (1-thr)] = -1.0
        else:
            raise ValueError("Unknown strategy")

        exe = config["execution"]
        bt_test = backtest_t1(test["Adj Close"], np.log(test["Adj Close"]).diff(), sig_test,
                              fees_bps=exe["fees_bps"],
                              slippage_bps=exe["slippage_bps"],
                              target_vol_annual=exe["target_vol_annual"],
                              vol_lookback=exe["vol_lookback"],
                              max_leverage=exe["max_leverage"])
        # bt_train not needed for now -- could be used for in-sample metrics or checks

        metrics = summarize(bt_test["pnl_net"], bt_test["pos"])
        metrics["start"] = str(test.index[0].date())
        metrics["end"] = str(test.index[-1].date())
        folds.append(metrics)
        oos_returns.append(bt_test["pnl_net"])
        oos_positions.append(bt_test["pos"])
        oos_equity_df.append(bt_test[["equity"]])

    folds_df = pd.DataFrame(folds)
    oos_ret = pd.concat(oos_returns).sort_index()
    oos_pos = pd.concat(oos_positions).sort_index()
    oos_eq = pd.concat(oos_equity_df).sort_index()
    return folds_df, oos_ret, oos_pos, oos_eq
