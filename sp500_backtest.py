"""
Simple S&P 500 backtesting helper using yfinance data.

This module downloads historical S&P 500 prices from Yahoo Finance
using ``yfinance`` and performs a moving-average crossover backtest.
The example is intentionally lightweight so it can serve as a starting
point for custom strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    initial_capital: float
    final_portfolio_value: float
    total_return: float
    trades: int
    equity_curve: pd.Series


def download_sp500_data(
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    ticker: str = "^GSPC",
) -> pd.DataFrame:
    """Download S&P 500 historical data using yfinance.

    Parameters
    ----------
    start:
        ISO formatted start date (``YYYY-MM-DD``).
    end:
        Optional ISO formatted end date. If omitted, the latest available data
        is returned.
    interval:
        Sampling frequency understood by Yahoo Finance (``"1d"``, ``"1wk"``,
        ``"1mo"``, etc.).
    ticker:
        Yahoo Finance ticker. Defaults to ``"^GSPC"`` (S&P 500 index).

    Returns
    -------
    pd.DataFrame
        Price history including ``Open``, ``High``, ``Low``, ``Close`` and
        ``Volume`` columns.
    """

    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if data.empty:
        raise ValueError("No data downloaded. Check ticker or date range.")
    return data


def moving_average_crossover_backtest(
    price_data: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Run a simple moving-average crossover backtest.

    The strategy goes long when the short moving average crosses above the
    long moving average, and exits (goes to cash) when the short average
    crosses below the long average. Position size is always 100% of the
    available capital, and the strategy does not short the market.

    Parameters
    ----------
    price_data:
        ``pandas.DataFrame`` with a ``Close`` column.
    short_window:
        Number of periods for the short moving average.
    long_window:
        Number of periods for the long moving average.
    initial_capital:
        Starting cash value.

    Returns
    -------
    BacktestResult
        Summary metrics plus the equity curve.
    """

    prices = price_data["Close"].copy()
    if prices.isna().all():
        raise ValueError("Price series contains only NaN values.")

    signals = pd.DataFrame(index=prices.index)
    signals["short_ma"] = prices.rolling(window=short_window, min_periods=1).mean()
    signals["long_ma"] = prices.rolling(window=long_window, min_periods=1).mean()

    # Generate trading signals: 1 for long, 0 for cash
    signals["signal"] = 0
    signals.loc[signals["short_ma"] > signals["long_ma"], "signal"] = 1

    # Calculate positions (changes in signal)
    signals["positions"] = signals["signal"].diff().fillna(0)

    # Calculate returns
    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = daily_returns * signals["signal"].shift().fillna(0)

    # Equity curve
    equity_curve = (1 + strategy_returns).cumprod() * initial_capital
    final_value = float(equity_curve.iloc[-1])
    total_return = (final_value - initial_capital) / initial_capital
    trades = int(signals["positions"].abs().sum())

    return BacktestResult(
        initial_capital=initial_capital,
        final_portfolio_value=final_value,
        total_return=total_return,
        trades=trades,
        equity_curve=equity_curve,
    )


def consecutive_down_day_leverage_backtest(
    price_data: pd.DataFrame,
    negative_days_required: int = 1,
    leverage: float = 3.0,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Backtest a leveraged long entry after consecutive down days.

    The strategy stays in cash until the previous ``negative_days_required``
    sessions all posted negative returns. On the following session it goes
    ``leverage`` times long for that day only. Positions are reset to cash
    at each close and the strategy never shorts.

    Parameters
    ----------
    price_data:
        ``pandas.DataFrame`` with a ``Close`` column.
    negative_days_required:
        Number of consecutive negative return days that must occur before a
        leveraged long position is opened the next day. Must be at least 1.
    leverage:
        Size of the leveraged long exposure applied when the condition is met.
    initial_capital:
        Starting cash value.

    Returns
    -------
    BacktestResult
        Summary metrics plus the equity curve.
    """

    if negative_days_required < 1:
        raise ValueError("negative_days_required must be at least 1")

    prices = price_data["Close"].copy()
    if prices.isna().all():
        raise ValueError("Price series contains only NaN values.")

    daily_returns = prices.pct_change().fillna(0)
    negative_flags = daily_returns < 0

    negative_run_met = (
        negative_flags.rolling(
            window=negative_days_required, min_periods=negative_days_required
        )
        .apply(lambda window: 1.0 if bool(window.all()) else 0.0)
        .fillna(0)
    )

    # Shift so that today's position is based solely on yesterday's information
    signals = negative_run_met.shift(1).fillna(0)

    strategy_returns = daily_returns * leverage * signals
    equity_curve = (1 + strategy_returns).cumprod() * initial_capital

    final_value = float(equity_curve.iloc[-1])
    total_return = (final_value - initial_capital) / initial_capital
    trades = int((signals > 0).sum())

    return BacktestResult(
        initial_capital=initial_capital,
        final_portfolio_value=final_value,
        total_return=total_return,
        trades=trades,
        equity_curve=equity_curve,
    )


def main():
    """Example execution printing a simple summary to stdout."""

    data = download_sp500_data(start="1980-01-01")
    crossover_result = moving_average_crossover_backtest(data)

    print("Simple Moving Average Crossover (50/200) on S&P 500")
    print(f"Initial capital: ${crossover_result.initial_capital:,.2f}")
    print(f"Final portfolio value: ${crossover_result.final_portfolio_value:,.2f}")
    print(f"Total return: {crossover_result.total_return:.2%}")
    print(f"Trades executed: {crossover_result.trades}\n")

    for negative_days in (1, 2, 3):
        leverage_result = consecutive_down_day_leverage_backtest(
            data, negative_days_required=negative_days
        )
        print(
            "Leveraged long after negative days: "
            f"{negative_days} down-day trigger"
        )
        print(f"Initial capital: ${leverage_result.initial_capital:,.2f}")
        print(f"Final portfolio value: ${leverage_result.final_portfolio_value:,.2f}")
        print(f"Total return: {leverage_result.total_return:.2%}")
        print(f"Trades executed: {leverage_result.trades}\n")


if __name__ == "__main__":
    main()
