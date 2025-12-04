import numpy as np
import pandas as pd

from sp500_backtest import run_buy_and_hold_backtest, run_ma_crossover_backtest


def _generate_prices():
    """Create a deterministic upward price series for tests."""

    dates = pd.bdate_range(start="2020-01-01", periods=260)
    steps = np.arange(len(dates)) * 0.05
    prices = pd.Series(100 + steps, index=dates, name="Adj Close")
    return prices


def test_buy_and_hold_runs():
    prices = _generate_prices()
    result = run_buy_and_hold_backtest(prices)
    assert result.final_value > 0
    assert result.total_return > 0


def test_ma_crossover_runs():
    prices = _generate_prices()
    result = run_ma_crossover_backtest(prices, short_window=10, long_window=30)
    assert result.final_value > 0
    # strategy might not outperform buy & hold, but should produce a finite number
    assert result.max_drawdown <= 0
