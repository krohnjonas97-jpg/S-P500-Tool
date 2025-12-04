"""Backtesting utilities for S&P 500 strategies.

This module contains small, easy-to-read helpers for downloading data with
`yfinance` and for running simple buy & hold and moving-average crossover
backtests. The goal is to stay dependency-light and keep the code approachable
for beginners.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class BacktestResult:
    """Container for common performance statistics."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    final_value: float
    total_return: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float

    def to_rows(self) -> list[list[str]]:
        """Return values formatted for simple console tables."""

        return [
            ["Strategie", self.name],
            ["Start", self.start.date().isoformat()],
            ["Ende", self.end.date().isoformat()],
            ["Endwert (USD)", f"{self.final_value:,.2f}"],
            ["Gesamtrendite", f"{self.total_return * 100:.2f}%"],
            ["Jährliche Rendite", f"{self.annual_return * 100:.2f}%"],
            ["Volatilität (p.a.)", f"{self.annual_volatility * 100:.2f}%"],
            ["Max. Drawdown", f"{self.max_drawdown * 100:.2f}%"],
        ]


def download_sp500_data(
    start: str = "2010-01-01", end: Optional[str] = None
) -> pd.Series:
    """Download adjusted closing prices for the S&P 500.

    Parameters
    ----------
    start: str
        Start date in ISO format (YYYY-MM-DD).
    end: Optional[str]
        End date in ISO format; defaults to today.

    Returns
    -------
    pandas.Series
        Adjusted close prices indexed by date.

    Raises
    ------
    RuntimeError
        If the download fails or returns no data.
    """

    end = end or _dt.date.today().isoformat()
    try:
        data = yf.download("^GSPC", start=start, end=end, progress=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Download der S&P-500-Daten fehlgeschlagen: {exc}") from exc

    if data is None or data.empty:
        raise RuntimeError("Keine Daten geladen – bitte Internetverbindung und Datum prüfen.")

    if "Adj Close" not in data.columns:
        raise RuntimeError("Die erwartete Spalte 'Adj Close' fehlt im Download.")

    prices = data["Adj Close"].dropna()
    if prices.empty:
        raise RuntimeError("Zeitreihe ist leer nach dem Bereinigen von fehlenden Werten.")

    return prices


def _ensure_prices(prices: pd.Series) -> pd.Series:
    """Validate input prices and return a cleaned copy."""

    if prices is None:
        raise ValueError("Preisdaten dürfen nicht None sein.")
    if prices.empty:
        raise ValueError("Preisdaten sind leer.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Preisdaten benötigen einen DatetimeIndex.")
    return prices.sort_index().copy()


def _calculate_metrics(equity_curve: pd.Series, initial_capital: float) -> BacktestResult:
    """Calculate performance statistics from an equity curve."""

    if equity_curve.empty:
        raise ValueError("Die Equity-Kurve ist leer und kann nicht ausgewertet werden.")

    returns = equity_curve.pct_change().dropna()
    start, end = equity_curve.index[0], equity_curve.index[-1]

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    if returns.empty:
        annual_return = 0.0
        annual_volatility = 0.0
    else:
        periods_per_year = 252
        annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(periods_per_year)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    return BacktestResult(
        name="",
        start=start,
        end=end,
        final_value=float(equity_curve.iloc[-1]),
        total_return=float(total_return),
        annual_return=float(annual_return),
        annual_volatility=float(annual_volatility),
        max_drawdown=float(max_drawdown),
    )


def run_buy_and_hold_backtest(prices: pd.Series, initial_capital: float = 10_000.0) -> BacktestResult:
    """Simple buy & hold backtest on the provided price series."""

    cleaned_prices = _ensure_prices(prices)
    equity_curve = initial_capital * (cleaned_prices / cleaned_prices.iloc[0])
    result = _calculate_metrics(equity_curve, initial_capital)
    result.name = "Buy & Hold"
    return result


def run_ma_crossover_backtest(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Moving-average crossover backtest with simple long/cash logic."""

    cleaned_prices = _ensure_prices(prices)
    data = pd.DataFrame({"price": cleaned_prices})
    data["ma_short"] = data["price"].rolling(window=short_window, min_periods=short_window).mean()
    data["ma_long"] = data["price"].rolling(window=long_window, min_periods=long_window).mean()

    data.dropna(inplace=True)
    if data.empty:
        raise ValueError("Nicht genügend Datenpunkte für die gleitenden Durchschnitte.")

    signal = (data["ma_short"] > data["ma_long"]).astype(float)
    position = signal.shift(1).fillna(0.0)

    returns = data["price"].pct_change().fillna(0.0)
    strategy_returns = position * returns
    equity_curve = (1 + strategy_returns).cumprod() * initial_capital

    result = _calculate_metrics(equity_curve, initial_capital)
    result.name = f"MA Crossover ({short_window}/{long_window})"
    return result


def run_backtests(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    initial_capital: float = 10_000.0,
) -> tuple[BacktestResult, BacktestResult]:
    """Convenience wrapper to download data and run both strategies."""

    prices = download_sp500_data(start=start, end=end)
    buy_hold = run_buy_and_hold_backtest(prices, initial_capital=initial_capital)
    ma_crossover = run_ma_crossover_backtest(
        prices,
        short_window=50,
        long_window=200,
        initial_capital=initial_capital,
    )
    return buy_hold, ma_crossover


def format_table(rows: list[list[str]]) -> str:
    """Create a simple aligned text table for console output."""

    col_width = max(len(row[0]) for row in rows) + 2
    lines = []
    for left, right in rows:
        lines.append(f"{left.ljust(col_width)}{right}")
    return "\n".join(lines)
