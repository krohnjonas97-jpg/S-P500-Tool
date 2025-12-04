"""Einstiegspunkt für den S&P-500-Backtest."""

from __future__ import annotations

from textwrap import indent

from sp500_backtest import (
    BacktestResult,
    download_sp500_data,
    format_table,
    run_buy_and_hold_backtest,
    run_ma_crossover_backtest,
)


def print_result(result: BacktestResult) -> None:
    """Print a formatted block with the statistics of a single backtest."""

    print("=" * 60)
    print(result.name)
    print("=" * 60)
    print(indent(format_table(result.to_rows()), "  "))
    print()


def main() -> None:
    print("Lade S&P-500-Daten über yfinance ...")
    try:
        prices = download_sp500_data(start="2010-01-01")
    except Exception as exc:  # pragma: no cover - user friendly output
        print("Fehler beim Laden der Daten:")
        print(exc)
        return

    print(f"Datenpunkte geladen: {len(prices)}")

    print("\nStarte Buy & Hold Backtest ...")
    buy_hold = run_buy_and_hold_backtest(prices)
    print_result(buy_hold)

    print("Starte Moving-Average-Crossover Backtest ...")
    ma_result = run_ma_crossover_backtest(prices, short_window=50, long_window=200)
    print_result(ma_result)


if __name__ == "__main__":
    main()
