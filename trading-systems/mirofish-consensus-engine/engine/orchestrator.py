"""Orchestratore: il ciclo decisionale su ogni candela chiusa.

Regole (identiche in backtest, paper e live):
  1. MiroFish vota su 31 simulatori.
  2. FLAT  -> apre solo se consensus >= entry_votes (28) e il risk manager approva.
  3. IN POSIZIONE -> chiude appena i voti a favore della direzione scendono
     sotto exit_votes (26), o se il risk manager impone il flatten.
  4. La taglia è Kelly frazionario, cappata e verificata sul notional minimo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .broker import FLAT, LONG, SHORT, append_trade_csv
from .config import Config
from .kelly import kelly_size
from .mirofish import DOWN, NEUTRAL, UP, EnsembleDecision, MiroFishEngine
from .risk import RiskManager

log = logging.getLogger("mirofish")


@dataclass
class TickReport:
    """Esito dell'elaborazione di una candela (per log, stato e test)."""
    ts: float
    price: float
    equity: float
    votes_up: int
    votes_down: int
    consensus: int
    direction: int
    action: str            # "open_long" | "open_short" | "close" | "hold" | "flat"
    reason: str
    position_side: int
    decision: EnsembleDecision | None = field(default=None, repr=False)


class Orchestrator:
    def __init__(self, cfg: Config, broker, risk: RiskManager,
                 mirofish: MiroFishEngine | None = None, persist: bool = True):
        self.cfg = cfg
        self.broker = broker
        self.risk = risk
        self.mirofish = mirofish or MiroFishEngine(cfg.ensemble, cfg.costs)
        self.persist = persist
        self.candle_index = 0
        self.last_report: TickReport | None = None

    # ------------------------------------------------------------------ #
    def on_candle(self, ts: float, closes: np.ndarray,
                  data_age_seconds: float = 0.0) -> TickReport:
        self.candle_index += 1
        price = float(closes[-1])
        equity = self.broker.equity(price)
        self.risk.on_new_candle(ts, equity, self.candle_index)

        pos = self.broker.position

        # Kill switch: chiudi tutto, sempre.
        if self.risk.must_flatten() and pos.side != FLAT:
            rec = self.broker.close_position(price, ts, "kill_switch")
            self.risk.on_trade_closed(rec.pnl_usd, self.candle_index)
            self._persist_trade(rec)
            return self._report(ts, price, None, "close", self.risk.kill_reason)

        decision = self.mirofish.decide(closes)

        if pos.side == FLAT:
            return self._maybe_open(ts, price, decision, data_age_seconds)
        return self._maybe_close(ts, price, decision)

    # ------------------------------------------------------------------ #
    def _maybe_open(self, ts: float, price: float,
                    d: EnsembleDecision, data_age: float) -> TickReport:
        ens = self.cfg.ensemble
        if d.direction == NEUTRAL or d.consensus < ens.entry_votes:
            return self._report(ts, price, d, "flat",
                                f"consenso {d.consensus}/{ens.n_simulators} "
                                f"< soglia {ens.entry_votes}")

        check = self.risk.can_open(self.broker.equity(price), self.candle_index,
                                   data_age, d.direction)
        if not check.allowed:
            return self._report(ts, price, d, "flat", f"risk: {check.reason}")

        p_win = d.p_up if d.direction == UP else 1.0 - d.p_up
        sizing = kelly_size(
            equity_usd=self.broker.equity(price),
            p_win=p_win,
            win_loss_ratio=d.win_loss_ratio,
            kelly_fraction=self.cfg.sizing.kelly_fraction,
            max_position_pct=self.cfg.sizing.max_position_pct,
            min_notional_usd=self.cfg.sizing.min_notional_usd,
        )
        if sizing.fraction <= 0:
            return self._report(ts, price, d, "flat", f"sizing: {sizing.reason}")

        side = LONG if d.direction == UP else SHORT
        self.broker.open_position(side, sizing.notional_usd, price, ts, d.consensus)
        self.risk.on_trade_opened()
        action = "open_long" if side == LONG else "open_short"
        log.info("%s @ %.2f notional=%.2f (kelly=%.3f, voti=%d/%d)",
                 action, price, sizing.notional_usd, sizing.fraction,
                 d.consensus, d.n)
        return self._report(ts, price, d, action,
                            f"consenso {d.consensus}/{d.n}, "
                            f"notional {sizing.notional_usd:.2f} USD")

    def _maybe_close(self, ts: float, price: float, d: EnsembleDecision) -> TickReport:
        pos = self.broker.position
        favor = d.votes_up if pos.side == LONG else d.votes_down
        if favor < self.cfg.ensemble.exit_votes:
            rec = self.broker.close_position(
                price, ts, f"voti {favor} < soglia uscita {self.cfg.ensemble.exit_votes}")
            self.risk.on_trade_closed(rec.pnl_usd, self.candle_index)
            self._persist_trade(rec)
            log.info("close @ %.2f pnl=%.2f (%s)", price, rec.pnl_usd, rec.reason)
            return self._report(ts, price, d, "close", rec.reason)
        return self._report(ts, price, d, "hold",
                            f"voti a favore {favor} >= {self.cfg.ensemble.exit_votes}")

    # ------------------------------------------------------------------ #
    def _persist_trade(self, rec) -> None:
        if self.persist:
            append_trade_csv(self._resolve(self.cfg.engine.trades_file), rec)

    def _resolve(self, rel: str) -> Path:
        return Path(__file__).resolve().parent.parent / rel

    def _report(self, ts, price, d, action, reason) -> TickReport:
        r = TickReport(
            ts=ts, price=price, equity=self.broker.equity(price),
            votes_up=d.votes_up if d else 0,
            votes_down=d.votes_down if d else 0,
            consensus=d.consensus if d else 0,
            direction=d.direction if d else 0,
            action=action, reason=reason,
            position_side=self.broker.position.side,
            decision=d,
        )
        self.last_report = r
        return r

    # ------------------------------------------------------------------ #
    def write_state(self, extra: dict | None = None) -> None:
        """Snapshot JSON per la dashboard."""
        r = self.last_report
        if r is None:
            return
        d = r.decision
        state = {
            "ts": r.ts,
            "price": r.price,
            "equity": r.equity,
            "initial_equity": self.risk.initial_equity,
            "peak_equity": self.risk.peak_equity,
            "mode": self.cfg.mode,
            "symbol": self.cfg.exchange.symbol,
            "action": r.action,
            "reason": r.reason,
            "position": {
                "side": self.broker.position.side,
                "qty": self.broker.position.qty,
                "entry_price": self.broker.position.entry_price,
                "entry_votes": self.broker.position.entry_votes,
            },
            "votes": {
                "up": r.votes_up, "down": r.votes_down,
                "neutral": (d.votes_neutral if d else 0),
                "consensus": r.consensus,
                "entry_threshold": self.cfg.ensemble.entry_votes,
                "exit_threshold": self.cfg.ensemble.exit_votes,
                "n": self.cfg.ensemble.n_simulators,
            },
            "ensemble": [
                {"name": s.name, "vote": s.vote, "median_return": s.median_return,
                 "p_up": s.p_up, "q05": s.q05, "q95": s.q95}
                for s in (d.results if d else [])
            ],
            "p_up": d.p_up if d else 0.5,
            "expected_return": d.expected_return if d else 0.0,
            "risk": {
                "killed": self.risk.killed,
                "kill_reason": self.risk.kill_reason,
                "consecutive_losses": self.risk.consecutive_losses,
                "trades_today": self.risk.trades_today,
            },
            "trades": [asdict(t) for t in self.broker.trades[-50:]],
        }
        if extra:
            state.update(extra)
        path = self._resolve(self.cfg.engine.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(path)
