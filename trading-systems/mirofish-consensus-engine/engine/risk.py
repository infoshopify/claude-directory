"""Risk manager: i limiti che tengono in vita il conto.

Ogni trade deve passare TUTTI i controlli. Il risk manager può inoltre
imporre la chiusura immediata (kill switch) indipendentemente dal segnale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import RiskConfig


@dataclass
class RiskCheck:
    allowed: bool
    reason: str


@dataclass
class RiskManager:
    cfg: RiskConfig
    initial_equity: float
    peak_equity: float = field(init=False)
    day_start_equity: float = field(init=False)
    current_day: str = field(init=False)
    consecutive_losses: int = 0
    cooldown_until_candle: int = -1
    trades_today: int = 0
    killed: bool = False
    kill_reason: str = ""

    def __post_init__(self) -> None:
        self.peak_equity = self.initial_equity
        self.day_start_equity = self.initial_equity
        self.current_day = self._utc_day(time.time())

    @staticmethod
    def _utc_day(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ #
    def on_new_candle(self, ts: float, equity: float, candle_index: int) -> None:
        """Aggiorna lo stato a ogni candela: rollover giornaliero, picco, kill switch."""
        day = self._utc_day(ts)
        if day != self.current_day:
            self.current_day = day
            self.day_start_equity = equity
            self.trades_today = 0

        self.peak_equity = max(self.peak_equity, equity)
        dd = 1.0 - equity / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd >= self.cfg.max_drawdown_pct and not self.killed:
            self.killed = True
            self.kill_reason = (
                f"drawdown {dd:.1%} >= limite {self.cfg.max_drawdown_pct:.1%}: "
                "kill switch attivato, riavvio manuale richiesto"
            )

    def on_trade_closed(self, pnl: float, candle_index: int) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.cooldown_until_candle = candle_index + self.cfg.cooldown_candles
        else:
            self.consecutive_losses = 0

    def on_trade_opened(self) -> None:
        self.trades_today += 1

    # ------------------------------------------------------------------ #
    def can_open(self, equity: float, candle_index: int,
                 data_age_seconds: float, direction: int) -> RiskCheck:
        if self.killed:
            return RiskCheck(False, self.kill_reason)
        if direction < 0 and not self.cfg.allow_short:
            return RiskCheck(False, "short non abilitato (spot long/flat)")
        if data_age_seconds > self.cfg.stale_data_seconds:
            return RiskCheck(False, f"dati stantii ({data_age_seconds:.0f}s)")
        if candle_index < self.cooldown_until_candle:
            return RiskCheck(False,
                             f"cooldown dopo {self.consecutive_losses} perdite consecutive")
        if self.trades_today >= self.cfg.max_trades_per_day:
            return RiskCheck(False, f"limite {self.cfg.max_trades_per_day} trade/giorno raggiunto")

        daily_loss = 1.0 - equity / self.day_start_equity if self.day_start_equity > 0 else 0.0
        if daily_loss >= self.cfg.max_daily_loss_pct:
            return RiskCheck(False,
                             f"perdita giornaliera {daily_loss:.1%} >= "
                             f"limite {self.cfg.max_daily_loss_pct:.1%}")
        return RiskCheck(True, "ok")

    def must_flatten(self) -> bool:
        """True se ogni posizione aperta va chiusa subito (kill switch)."""
        return self.killed
