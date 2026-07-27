#!/usr/bin/env python3
"""Diagnostica: perché il sistema fa (o non fa) operazioni?

Esegue il motore su uno storico e, invece del solo PnL, mostra il "polso" del
consenso: quante volte i voti hanno raggiunto 20, 22, 24, 26, 28... Così si
capisce a colpo d'occhio se il sistema è semplicemente conservativo (voti che
si fermano a 24-26, sotto la soglia di 28) o se qualcosa non va (voti sempre a
zero = bug o dati piatti).

Uso:
    python3 run_diagnostic.py --csv data/BTCUSDT_15m.csv
    python3 run_diagnostic.py --csv data/BTCUSDT_15m.csv --entry 26   # simula soglia più bassa
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.config import load_config
from engine.mirofish import DOWN, MiroFishEngine, UP


def bar(count: int, total: int, width: int = 40) -> str:
    filled = int(width * count / total) if total else 0
    return "█" * filled + "·" * (width - filled)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--entry", type=int, default=None,
                    help="soglia da simulare (default: quella in config)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-candles", type=int, default=2000,
                    help="limita le candele analizzate (default 2000, ~21 giorni a 15m)")
    args = ap.parse_args()

    cfg = load_config(HERE / args.config)
    if args.fast:
        from dataclasses import replace
        cfg = replace(cfg, ensemble=replace(cfg.ensemble, paths_per_simulator=64))
    entry = args.entry if args.entry is not None else cfg.ensemble.entry_votes

    ohlcv = np.loadtxt(args.csv, delimiter=",", ndmin=2)
    closes = ohlcv[:, 4].astype(float)
    lookback = cfg.ensemble.lookback_candles
    n_total = cfg.ensemble.n_simulators
    if len(closes) <= lookback + 10:
        raise SystemExit(f"servono più di {lookback + 10} candele, trovate {len(closes)}")

    eng = MiroFishEngine(cfg.ensemble, cfg.costs, rng=np.random.default_rng(42))

    start = max(lookback, len(closes) - args.max_candles)
    consensus_hist = Counter()          # bucket del consenso massimo per candela
    max_consensus_seen = 0
    would_enter = 0
    n_eval = 0
    directions = Counter()

    print(f"analizzo {len(closes) - start} candele "
          f"(soglia simulata: {entry}/{n_total})...\n")
    for i in range(start, len(closes)):
        d = eng.decide(closes[: i + 1])
        top = max(d.votes_up, d.votes_down)
        max_consensus_seen = max(max_consensus_seen, top)
        # bucket a step di 2
        consensus_hist[(top // 2) * 2] += 1
        if top >= entry:
            would_enter += 1
            directions[UP if d.votes_up >= d.votes_down else DOWN] += 1
        n_eval += 1

    print("DISTRIBUZIONE DEL CONSENSO MASSIMO PER CANDELA")
    print("(quanti voti concordi raggiunge, nella direzione più forte)\n")
    for bucket in range(0, n_total + 1, 2):
        c = consensus_hist.get(bucket, 0)
        marker = "  <- soglia ingresso" if bucket <= entry < bucket + 2 else ""
        print(f"  {bucket:2d}-{bucket+1:2d} voti | {bar(c, n_eval)} "
              f"{c:5d} ({100*c/n_eval:4.1f}%){marker}")

    print("\nRIEPILOGO")
    print(f"  candele analizzate:       {n_eval}")
    print(f"  consenso massimo visto:   {max_consensus_seen}/{n_total}")
    print(f"  candele sopra soglia {entry}: {would_enter} "
          f"({100*would_enter/n_eval:.2f}%)")
    if would_enter:
        print(f"    di cui LONG:  {directions[UP]}")
        print(f"    di cui SHORT: {directions[DOWN]} "
              f"(eseguiti solo se allow_short=true)")

    print("\nLETTURA:")
    if max_consensus_seen == 0:
        print("  ⚠️  consenso sempre a ZERO: probabile problema di dati (prezzi "
              "costanti/NaN) o finestra troppo corta. NON è normale.")
    elif would_enter == 0:
        gap = entry - max_consensus_seen
        print(f"  Il motore VOTA regolarmente ma non raggiunge mai {entry}/{n_total} "
              f"(picco {max_consensus_seen}). È il comportamento previsto: la barra "
              f"del consenso è alta apposta.")
        if gap <= 2:
            print(f"  Sei vicino: mancano {gap} voti. Con --entry {max_consensus_seen} "
                  "vedresti operazioni (a qualità inferiore) — utile solo per test.")
        else:
            print(f"  Su questi dati il segnale è debole (gap {gap}). Prova un "
                  "periodo con trend più marcato, o valuta l'orizzonte 1h-4h.")
    else:
        print(f"  Tutto regolare: {would_enter} ingressi potenziali su {n_eval} "
              "candele. Poche operazioni ma selettive è il progetto.")
    print("\nNB: abbassare --entry aumenta le operazioni ma NON crea edge; "
          "va deciso guardando il walk-forward (run_walkforward.py), non qui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
