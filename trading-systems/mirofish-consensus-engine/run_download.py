#!/usr/bin/env python3
"""Scarica e mette in cache lo storico 5-min per la validazione (voce D2).

Il walk-forward serio richiede ANNI di candele, non giorni: il download è
incrementale e riprende da dove si era fermato, così si può accumulare lo
storico in più sessioni senza riscaricare nulla.

Uso:
    python3 run_download.py --days 1460                 # ~4 anni di BTC/USDT 5m
    python3 run_download.py --days 365 --out data/btc.csv

Formato output CSV (senza header): ts_ms,open,high,low,close,volume
Compatibile con: run_backtest.py --csv  e  run_walkforward.py --csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.config import load_config
from engine.data_feed import DataFeed, TIMEFRAME_SECONDS


def load_cache(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    return arr if len(arr) else None


def download(cfg, days: int, out: Path) -> np.ndarray:
    feed = DataFeed(cfg.exchange.id, cfg.exchange.symbol,
                    cfg.exchange.timeframe, cfg.ensemble.lookback_candles)
    if not feed.has_ccxt:
        raise SystemExit(
            "ccxt è richiesto per il download paginato dello storico "
            "(pip install ccxt). Il fallback REST copre solo ~1000 candele."
        )
    tf_s = TIMEFRAME_SECONDS[cfg.exchange.timeframe]
    tf_ms = tf_s * 1000
    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - days * 86_400_000

    cached = load_cache(out)
    rows: list[list[float]] = cached.tolist() if cached is not None else []
    if rows:
        # Riparte dalla candela successiva all'ultima in cache.
        since = int(rows[-1][0]) + tf_ms
        print(f"cache: {len(rows)} candele in {out}, riprendo dal "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(since / 1000))} UTC")
    else:
        since = target_start_ms

    fetched = 0
    while since < now_ms:
        batch = feed.fetch_ohlcv_since(since, limit=1000)
        if not batch:
            break
        rows.extend([[float(v) for v in r[:6]] for r in batch])
        fetched += len(batch)
        since = batch[-1][0] + tf_ms
        if fetched % 10_000 < 1000:
            print(f"  scaricate {fetched} nuove candele "
                  f"(fino al {time.strftime('%Y-%m-%d', time.gmtime(since / 1000))})",
                  flush=True)
        if len(batch) < 1000:
            break

    arr = np.array(rows, dtype=float)
    if len(arr) == 0:
        raise SystemExit("nessuna candela scaricata")

    # Dedup + ordina + scarta l'eventuale candela ancora aperta.
    _, idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[idx]
    if arr[-1, 0] / 1000.0 + tf_s > time.time():
        arr = arr[:-1]

    # Controllo integrità: buchi nella serie (manutenzioni exchange, ecc.).
    gaps = np.diff(arr[:, 0])
    n_gaps = int(np.sum(gaps > tf_ms))
    if n_gaps:
        print(f"ATTENZIONE: {n_gaps} buchi temporali nella serie "
              f"(normale su storici lunghi: manutenzioni/outage exchange)")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, arr, delimiter=",",
               fmt=["%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.6f"])
    days_span = (arr[-1, 0] - arr[0, 0]) / 86_400_000
    print(f"salvate {len(arr)} candele ({days_span:.1f} giorni) in {out}")
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default=None,
                    help="default: data/<SYMBOL>_<TF>.csv")
    args = ap.parse_args()

    cfg = load_config(HERE / args.config)
    default_name = (cfg.exchange.symbol.replace("/", "") + "_"
                    + cfg.exchange.timeframe + ".csv")
    out = Path(args.out) if args.out else HERE / "data" / default_name
    download(cfg, args.days, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
