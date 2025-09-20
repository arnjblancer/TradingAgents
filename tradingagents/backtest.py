"""Command line utility to backtest TradingAgents decisions.

This script evaluates the TradingAgents graph over a range of trading dates
for one or more tickers and computes a simple daily return series by
interpreting BUY/SELL/HOLD signals. Results for each ticker are stored as CSV
logs alongside a JSON summary file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


SignalRecord = Dict[str, object]
Summary = Dict[str, object]


class _MovingAverageFallbackGraph:
    """Simple deterministic strategy used when the full graph cannot start."""

    def __init__(self, price_series: pd.Series) -> None:
        self._prices = price_series

    def propagate(self, ticker: str, trade_date: str):  # pragma: no cover - simple shim
        ts = pd.to_datetime(trade_date)
        history = self._prices.loc[:ts]
        if len(history) < 2:
            return {}, "HOLD"

        short_window = min(3, len(history))
        long_window = min(7, len(history))
        short_ma = history.tail(short_window).mean()
        long_ma = history.tail(long_window).mean()

        if short_ma > long_ma * 1.001:
            signal = "BUY"
        elif short_ma < long_ma * 0.999:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {}, signal


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


def _generate_synthetic_price_series(
    ticker: str, start: date, end: date, buffer_days: int
) -> pd.Series:
    """Generate a deterministic synthetic price series as a last-resort fallback."""

    # Extend the end date with the same buffer used for real downloads to keep
    # behaviour consistent with the online path.
    synthetic_end = end + timedelta(days=buffer_days)
    date_index = pd.date_range(start=start, end=synthetic_end, freq="B")
    if len(date_index) < 2:
        # Ensure at least two observations so the backtest can proceed.
        date_index = pd.date_range(start=start, periods=2, freq="B")

    seed_material = f"{ticker}-{start.isoformat()}-{end.isoformat()}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)

    base_price = 80.0 + (seed % 5000) / 100.0
    drift = 0.0005
    volatility = 0.02
    returns = rng.normal(loc=drift, scale=volatility, size=len(date_index))
    prices = base_price * np.exp(np.cumsum(returns))

    series = pd.Series(prices, index=date_index, name="Adj Close")
    return series


def _fetch_price_series(ticker: str, start: date, end: date) -> pd.Series:
    """Download adjusted close prices, falling back to synthetic data if required."""

    buffer_days = 7
    try:
        data = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=buffer_days)).isoformat(),
            progress=False,
        )
    except Exception:
        data = pd.DataFrame()

    if data.empty or "Adj Close" not in data:
        print(
            f"  Warning: Unable to download price data for {ticker}. "
            "Using synthetic price series instead."
        )
        prices = _generate_synthetic_price_series(ticker, start, end, buffer_days)
    else:
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

    try:
        graph = TradingAgentsGraph(
            selected_analysts=list(analysts),
            debug=debug,
            config=base_config.copy(),
        )
    except Exception as exc:  # pragma: no cover - defensive offline fallback
        print(
            "  Warning: Unable to initialise TradingAgentsGraph. "
            "Using moving-average fallback signals instead."
        )
        print(f"           Reason: {exc}")
        graph = _MovingAverageFallbackGraph(price_series)

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

    if results.empty:
        total_return = 0.0
        max_drawdown = 0.0
        annual_average_return = 0.0
    else:
        total_return = results["daily_return"].add(1.0).prod() - 1.0

        equity_curve = results["equity"]
        running_max = equity_curve.cummax().replace(0.0, np.nan)
        drawdowns = equity_curve.divide(running_max) - 1.0
        drawdowns = drawdowns.fillna(0.0)
        max_drawdown = float(abs(drawdowns.min()))

        final_equity = float(equity_curve.iloc[-1])
        if initial_capital > 0 and final_equity > 0:
            years = max(len(results) / 252.0, 1.0 / 252.0)
            annual_average_return = float(
                np.power(final_equity / initial_capital, 1.0 / years) - 1.0
            )
        elif initial_capital > 0 and final_equity == 0:
            annual_average_return = -1.0
        else:
            annual_average_return = -1.0

    summary: Summary = {
        "ticker": ticker,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "num_observations": int(len(results)),
        "final_equity": equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "annual_average_return": annual_average_return,
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
            "  "
            + (
                f"{ticker}: total_return={summary['total_return']:.2%}, "
                f"annual_avg_return={summary['annual_average_return']:.2%}, "
                f"max_drawdown={summary['max_drawdown']:.2%}, "
                f"final_equity={summary['final_equity']:.2f}"
            )
        )

    if summaries:
        print("\nBacktest complete. Summary files saved to:")
        for summary in summaries:
            print(f"  {summary['ticker']}: {summary['log_path']}")
    else:
        print("No tickers were processed.")


if __name__ == "__main__":
    main()
