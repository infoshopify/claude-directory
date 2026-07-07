#!/usr/bin/env python3
"""Walk-forward validation da linea di comando (Fase 1 del piano: D1 + D2).

Questo è il guard-rail di tutto il progetto: qualunque modifica alla strategia
entra solo se migliora i risultati OUT-OF-SAMPLE prodotti da questo script.

Uso tipico (dopo aver scaricato lo storico con run_download.py):
    python3 run_download.py --days 1460
    python3 run_walkforward.py --csv data/BTCUSDT_5m.csv --grid medium
    python3 run_walkforward.py --csv data/BTCUSDT_5m.csv --stress-costs 2.0

Parametri chiave:
    --train-days / --test-days   dimensione finestre (default 30/10 giorni)
    --grid small|medium          griglia parametri (4 o 18 configurazioni)
    --fast                       meno percorsi Monte Carlo (ricerca rapida);
                                 il run finale di conferma va fatto SENZA --fast
    --stress-costs X             moltiplica fees+slippage (robustezza D2)

Output: report a video + runtime/walkforward_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.config import load_config
from engine.data_feed import TIMEFRAME_SECONDS
from engine.walkforward import GRIDS, run_walkforward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--csv", required=True,
                    help="storico CSV ts_ms,o,h,l,c,v (vedi run_download.py)")
    ap.add_argument("--train-days", type=float, default=30.0)
    ap.add_argument("--test-days", type=float, default=10.0)
    ap.add_argument("--embargo-candles", type=int, default=None,
                    help="default: horizon_candles della config")
    ap.add_argument("--grid", default="small", choices=sorted(GRIDS))
    ap.add_argument("--fast", action="store_true",
                    help="64 percorsi/simulatore invece di 256 (solo esplorazione)")
    ap.add_argument("--stress-costs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-trades", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(HERE / args.config)
    if args.fast:
        cfg = replace(cfg, ensemble=replace(cfg.ensemble, paths_per_simulator=64))

    ohlcv = np.loadtxt(args.csv, delimiter=",", ndmin=2)
    per_day = 86_400 // TIMEFRAME_SECONDS[cfg.exchange.timeframe]
    train_c = int(args.train_days * per_day)
    test_c = int(args.test_days * per_day)

    days_available = len(ohlcv) / per_day
    print(f"storico: {len(ohlcv)} candele (~{days_available:.0f} giorni) | "
          f"griglia {args.grid} ({len(GRIDS[args.grid])} config) | "
          f"train {train_c} / test {test_c} candele | "
          f"costi x{args.stress_costs:.1f}"
          + (" | FAST (64 percorsi)" if args.fast else ""))
    if days_available < 180:
        print("NOTA: meno di 6 mesi di dati — il risultato avrà poco valore "
              "statistico. Scarica più storico con run_download.py --days 1460.")

    report = run_walkforward(
        cfg, ohlcv,
        train_candles=train_c,
        test_candles=test_c,
        embargo_candles=args.embargo_candles,
        grid=args.grid,
        seed=args.seed,
        min_trades=args.min_trades,
        cost_multiplier=args.stress_costs,
        progress=True,
    )

    print("\n=== WALK-FORWARD REPORT ===")
    print(report.summary())
    if args.fast:
        print("\nNOTA: run in modalità --fast (64 percorsi). Prima di trarre "
              "conclusioni, ripetere la configurazione finale senza --fast.")

    out = HERE / "runtime" / "walkforward_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "grid": args.grid,
        "fast": args.fast,
        "stress_costs": args.stress_costs,
        "train_candles": report.train_candles,
        "test_candles": report.test_candles,
        "embargo_candles": report.embargo_candles,
        "oos_return_pct": report.oos_return_pct,
        "oos_sharpe_annualized": report.oos_sharpe_annualized,
        "oos_max_drawdown_pct": report.oos_max_drawdown_pct,
        "oos_trades": report.oos_trades,
        "is_return_pct_mean": report.is_return_pct_mean,
        "oos_return_pct_mean": report.oos_return_pct_mean,
        "overfitting_gap_pct": report.overfitting_gap_pct,
        "param_stability": report.param_stability,
        "windows": [
            {"index": w.index, "train_range": w.train_range,
             "test_range": w.test_range, "chosen": w.chosen,
             "is_return_pct": w.train_return_pct,
             "oos_return_pct": w.test_return_pct,
             "oos_trades": w.test.n_trades}
            for w in report.windows
        ],
        "oos_equity_curve": report.oos_equity_curve,
        "verdict": report.verdict(),
    }, indent=2))
    print(f"\nreport salvato in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
