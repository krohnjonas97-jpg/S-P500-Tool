S&P500 Tool – Backtests für den S&P 500

## Features
- Download historischer S&P 500 Kurse via `yfinance`.
- Einfache Moving-Average-Crossover-Strategie als Beispiel.
- Parametrisierbarer Test: Nach *n* negativen Tagen am Folgetag 3x long gehen.

## Nutzung
```
python sp500_backtest.py
```

Das Script lädt die Daten ab 2015, führt die Strategien aus und gibt eine kurze
Auswertung auf der Konsole aus. Der Parameter `negative_days_required`
kontrolliert, nach wie vielen roten Tagen die 3x-Long-Position eröffnet wird
(`1`, `2` oder `3` sind vorkonfiguriert).
