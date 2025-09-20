"""Command line utility to backtest TradingAgents decisions.

This script evaluates the TradingAgents graph over a range of trading dates
for one or more tickers and computes a simple daily return series by
interpreting BUY/SELL/HOLD signals. Results for each ticker are stored as CSV
logs alongside a JSON summary file.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yfinance as yf

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


SignalRecord = Dict[str, object]
Summary = Dict[str, object]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the TradingAgents strategy over a date range."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols to evaluate (e.g. AAPL MSFT).",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date for the backtest in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date for the backtest in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Starting capital used to compute the equity curve.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory where backtest artifacts will be written.",
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional path to a JSON file with TradingAgents configuration "
            "overrides."
        ),
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Force the agents to use only online data tools (disable offline cache).",
    )
    parser.add_argument(
        "--analysts",
        nargs="+",
        choices=["market", "social", "news", "fundamentals"],
        default=["market", "social", "news", "fundamentals"],
        help="Subset of analysts to include in the workflow.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable LangGraph debug streaming for troubleshooting.",
    )
    return parser.parse_args()


def _load_config(config_path: str | None, online_only: bool) -> Dict[str, object]:
    config = DEFAULT_CONFIG.copy()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as config_file:
            overrides = json.load(config_file)
        config.update(overrides)
    if online_only:
        config["online_tools"] = True
        config["offline_tools"] = False
    return config


def _fetch_price_series(ticker: str, start: date, end: date) -> pd.Series:
    """Download adjusted close prices including a small forward buffer."""
    buffer_days = 7
    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=buffer_days)).isoformat(),
        progress=False,
    )
    if data.empty or "Adj Close" not in data:
        raise ValueError(f"No price data available for {ticker} in requested range.")

    prices = data["Adj Close"].copy()
    if hasattr(prices.index, "tz_localize"):
        try:
            prices = prices.tz_localize(None)
        except (TypeError, AttributeError):
            # Index is already timezone naive
            pass

    return prices.sort_index()


def _normalise_signal(signal: str) -> str:
    cleaned = signal.strip().upper()
    if not cleaned:
        return "HOLD"
    token = cleaned.split()[0]
    if token not in {"BUY", "SELL", "HOLD"}:
        return "HOLD"
    return token


def _run_backtest_for_ticker(
    ticker: str,
    start: date,
    end: date,
    initial_capital: float,
    base_config: Dict[str, object],
    analysts: Iterable[str],
    output_dir: Path,
    debug: bool,
) -> Tuple[pd.DataFrame, Summary]:
    price_series = _fetch_price_series(ticker, start, end)
    if len(price_series) < 2:
        raise ValueError(f"Insufficient price history for {ticker} to run backtest.")

    graph = TradingAgentsGraph(
        selected_analysts=list(analysts),
        debug=debug,
        config=base_config.copy(),
    )

    records: List[SignalRecord] = []
    position = 0
    equity = initial_capital
    index_list = list(price_series.index)

    for idx, current_ts in enumerate(index_list[:-1]):
        current_date = current_ts.date()
        if current_date < start or current_date > end:
            continue

        current_price = float(price_series.iloc[idx])
        next_price = float(price_series.iloc[idx + 1])
        trade_date_str = current_date.strftime("%Y-%m-%d")

        try:
            _, raw_signal = graph.propagate(ticker, trade_date_str)
        except Exception as exc:  # pragma: no cover - safety net for runtime errors
            raise RuntimeError(
                f"Failed to generate decision for {ticker} on {trade_date_str}."
            ) from exc

        signal = _normalise_signal(raw_signal)
        if signal == "BUY":
            position = 1
        elif signal == "SELL":
            position = -1

        daily_return = position * (next_price - current_price) / current_price
        equity *= 1.0 + daily_return

        records.append(
            {
                "date": trade_date_str,
                "signal": signal,
                "position": position,
                "close": current_price,
                "next_close": next_price,
                "daily_return": daily_return,
                "equity": equity,
            }
        )

    results = pd.DataFrame(records)
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    results_path = ticker_dir / "backtest_log.csv"
    results.to_csv(results_path, index=False)

    total_return = (results["daily_return"].add(1.0).prod() - 1.0) if not results.empty else 0.0
    summary: Summary = {
        "ticker": ticker,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "num_observations": int(len(results)),
        "final_equity": equity,
        "total_return": total_return,
        "long_days": int((results["position"] == 1).sum()) if not results.empty else 0,
        "short_days": int((results["position"] == -1).sum()) if not results.empty else 0,
        "flat_days": int((results["position"] == 0).sum()) if not results.empty else 0,
        "log_path": str(results_path),
    }

    summary_path = ticker_dir / "backtest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return results, summary


def main() -> None:
    args = _parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if start >= end:
        raise ValueError("start-date must be earlier than end-date.")

    config = _load_config(args.config, args.online_only)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Summary] = []
    for ticker in args.tickers:
        print(f"Running TradingAgents backtest for {ticker}...")
        _, summary = _run_backtest_for_ticker(
            ticker=ticker,
            start=start,
            end=end,
            initial_capital=args.initial_capital,
            base_config=config,
            analysts=args.analysts,
            output_dir=output_dir,
            debug=args.debug,
        )
        summaries.append(summary)
        print(
            f"  {ticker}: total_return={summary['total_return']:.2%}, "
            f"final_equity={summary['final_equity']:.2f}"
        )

    if summaries:
        print("\nBacktest complete. Summary files saved to:")
        for summary in summaries:
            print(f"  {summary['ticker']}: {summary['log_path']}")
    else:
        print("No tickers were processed.")


if __name__ == "__main__":
    main()
