#!/usr/bin/env python3
"""Backtest della strategia su candele storiche 5-min.

Uso:
    python3 run_backtest.py                      # scarica ~7 giorni di BTC/USDT 5m
    python3 run_backtest.py --days 30            # più storico
    python3 run_backtest.py --csv dati.csv       # da file CSV (ts_ms,o,h,l,c,v)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.backtester import run_backtest
from engine.config import load_config
from engine.data_feed import DataFeed


def download_history(cfg, days: int) -> np.ndarray:
    """Scarica storico a pagine di 1000 candele via feed pubblico."""
    feed = DataFeed(cfg.exchange.id, cfg.exchange.symbol,
                    cfg.exchange.timeframe, cfg.ensemble.lookback_candles)
    needed = days * 288 + cfg.ensemble.lookback_candles
    rows: list[list[float]] = []

    if feed._ccxt is not None:
        tf_ms = feed.tf_seconds * 1000
        since = int((time.time() - (needed + 10) * feed.tf_seconds) * 1000)
        while len(rows) < needed:
            batch = feed._ccxt.fetch_ohlcv(cfg.exchange.symbol, cfg.exchange.timeframe,
                                           since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + tf_ms
            if len(batch) < 1000:
                break
    else:
        # Fallback REST Binance: singole pagine all'indietro non supportate qui,
        # si limita alle ultime 1000 candele (~3.5 giorni).
        candles = feed.fetch_ohlcv(min(needed, 1000))
        rows = [[c.ts * 1000, c.open, c.high, c.low, c.close, c.volume] for c in candles]

    arr = np.array(rows, dtype=float)
    # Scarta l'ultima candela se non ancora chiusa (stesso criterio di
    # DataFeed._drop_open_candle): una candela parziale nel backtest è
    # look-ahead bias.
    if len(arr) and arr[-1, 0] / 1000.0 + feed.tf_seconds > time.time():
        arr = arr[:-1]
    print(f"scaricate {len(arr)} candele {cfg.exchange.timeframe} di {cfg.exchange.symbol}")
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--csv", help="file CSV ts_ms,open,high,low,close,volume (no header)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    cfg = load_config(here / args.config)

    if args.csv:
        ohlcv = np.loadtxt(args.csv, delimiter=",", ndmin=2)
    else:
        ohlcv = download_history(cfg, args.days)

    t0 = time.time()
    result = run_backtest(cfg, ohlcv, seed=args.seed)
    print(f"\n=== BACKTEST {cfg.exchange.symbol} {cfg.exchange.timeframe} "
          f"({time.time() - t0:.1f}s) ===")
    print(result.summary())
    print("\nNB: un backtest positivo NON garantisce profitti futuri. "
          "Valida in paper trading per settimane prima di considerare il live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
