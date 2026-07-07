#!/usr/bin/env python3
"""Runner paper/live: ciclo continuo su candele 5-min + server per la dashboard.

Uso:
    python3 run_live.py                  # paper trading (default, nessun rischio)
    python3 run_live.py --dashboard      # come sopra + dashboard su http://localhost:8787

Per il LIVE con capitale reale (sconsigliato senza settimane di validazione):
    1. config.yaml -> mode: live  (e testnet: true per provare sul testnet!)
    2. export MIROFISH_API_KEY=... MIROFISH_API_SECRET=...
    3. export MIROFISH_I_UNDERSTAND_THE_RISKS=YES
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import socketserver
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.broker import PaperBroker
from engine.config import load_config
from engine.data_feed import DataFeed
from engine.orchestrator import Orchestrator
from engine.risk import RiskManager


def serve_dashboard(port: int, state_file: Path) -> None:
    """Server statico minimale: dashboard/ + /state.json."""
    dash_dir = HERE / "dashboard"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(dash_dir), **kw)

        def do_GET(self):
            if self.path in ("/state.json", "/state"):
                try:
                    body = state_file.read_bytes()
                except FileNotFoundError:
                    body = json.dumps({"error": "state non ancora disponibile"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

        def log_message(self, *a):
            pass

    # Solo loopback: lo stato del conto non deve essere esposto sulla rete.
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.allow_reuse_address = True
        httpd.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dashboard", action="store_true", help="avvia la dashboard web")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    cfg = load_config(HERE / args.config)

    log_path = HERE / cfg.engine.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    log = logging.getLogger("runner")

    if cfg.mode == "live":
        from engine.broker import LiveBroker
        broker = LiveBroker(cfg)
        log.warning("MODE LIVE%s: ordini reali su %s",
                    " (TESTNET)" if cfg.exchange.testnet else "",
                    cfg.exchange.id)
    else:
        broker = PaperBroker(cash_usd=cfg.initial_capital_usd,
                             fee_bps=cfg.costs.fee_bps,
                             slippage_bps=cfg.costs.slippage_bps)
        log.info("MODE PAPER: nessun ordine reale, capitale simulato %.2f USD",
                 cfg.initial_capital_usd)

    feed = DataFeed(cfg.exchange.id, cfg.exchange.symbol,
                    cfg.exchange.timeframe, cfg.ensemble.lookback_candles)
    log.info("warmup: scarico %d candele %s di %s...",
             cfg.ensemble.lookback_candles, cfg.exchange.timeframe, cfg.exchange.symbol)
    feed.warmup()
    log.info("warmup completato: %d candele, ultimo prezzo %.2f",
             len(feed.candles), feed.last_price)

    risk = RiskManager(cfg.risk, broker.equity(feed.last_price))
    orch = Orchestrator(cfg, broker, risk)

    if args.dashboard:
        state_file = HERE / cfg.engine.state_file
        t = threading.Thread(target=serve_dashboard, args=(args.port, state_file),
                             daemon=True)
        t.start()
        log.info("dashboard su http://localhost:%d", args.port)

    # Prima decisione subito sul warmup, poi a ogni candela chiusa.
    report = orch.on_candle(time.time(), feed.closes, feed.data_age_seconds)
    orch.write_state()
    log.info("primo tick: %s (%s)", report.action, report.reason)

    # Circuit breaker: dopo N errori consecutivi nel ciclo decisionale il
    # motore smette di far finta di niente — tenta il flatten d'emergenza
    # della posizione e si ferma, invece di restare cieco con rischio aperto.
    MAX_CONSECUTIVE_FAILURES = 5
    failures = 0
    try:
        while True:
            time.sleep(cfg.engine.poll_seconds)
            try:
                new_candle = feed.poll()
            except Exception as e:
                log.error("errore feed dati: %s", e)
                continue
            if new_candle is None:
                continue
            try:
                report = orch.on_candle(new_candle.ts + feed.tf_seconds,
                                        feed.closes, feed.data_age_seconds)
                orch.write_state()
                failures = 0
                log.info("candela %.2f -> %s | equity %.2f | voti %d↑ %d↓ | %s",
                         new_candle.close, report.action, report.equity,
                         report.votes_up, report.votes_down, report.reason)
            except Exception as e:
                failures += 1
                log.exception("errore nel ciclo decisionale (%d/%d): %s",
                              failures, MAX_CONSECUTIVE_FAILURES, e)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    log.critical("circuit breaker: %d errori consecutivi, "
                                 "flatten d'emergenza e arresto", failures)
                    if broker.position.side != 0:
                        try:
                            rec = broker.close_position(
                                feed.last_price, time.time(), "circuit_breaker")
                            log.critical("posizione chiusa d'emergenza, pnl=%.2f",
                                         rec.pnl_usd)
                        except Exception:
                            log.exception(
                                "FLATTEN FALLITO: chiudi la posizione MANUALMENTE "
                                "sull'exchange (side=%d qty=%.8f)",
                                broker.position.side, broker.position.qty)
                    return 1
    except KeyboardInterrupt:
        log.info("arresto richiesto dall'utente")
        if broker.position.side != 0:
            log.warning("ATTENZIONE: posizione ancora aperta (side=%d qty=%.8f). "
                        "In live va gestita manualmente o riavviando il motore.",
                        broker.position.side, broker.position.qty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
