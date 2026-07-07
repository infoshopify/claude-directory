"""Caricamento e validazione della configurazione."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class ExchangeConfig:
    id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    testnet: bool = True

    def validate(self) -> None:
        from .data_feed import TIMEFRAME_SECONDS
        if self.timeframe not in TIMEFRAME_SECONDS:
            raise ConfigError(
                f"timeframe {self.timeframe!r} non supportato: "
                f"validi {sorted(TIMEFRAME_SECONDS)}"
            )
        if "/" not in self.symbol:
            raise ConfigError(f"symbol {self.symbol!r} non valido (atteso BASE/QUOTE)")


@dataclass(frozen=True)
class EnsembleConfig:
    n_simulators: int = 31
    entry_votes: int = 28
    exit_votes: int = 26
    paths_per_simulator: int = 256
    horizon_candles: int = 12
    lookback_candles: int = 288
    vote_threshold_bps: float = 12.0
    seed: int | None = None

    def validate(self) -> None:
        if not (0 < self.exit_votes <= self.entry_votes <= self.n_simulators):
            raise ConfigError(
                "richiesto 0 < exit_votes <= entry_votes <= n_simulators, "
                f"trovato exit={self.exit_votes} entry={self.entry_votes} n={self.n_simulators}"
            )
        if self.horizon_candles < 1 or self.lookback_candles < 32:
            raise ConfigError("horizon_candles >= 1 e lookback_candles >= 32 richiesti")


@dataclass(frozen=True)
class SizingConfig:
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.20
    min_notional_usd: float = 25.0

    def validate(self) -> None:
        if not (0 < self.kelly_fraction <= 1.0):
            raise ConfigError("kelly_fraction deve essere in (0, 1]")
        if not (0 < self.max_position_pct <= 1.0):
            raise ConfigError("max_position_pct deve essere in (0, 1]")


@dataclass(frozen=True)
class RiskConfig:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_consecutive_losses: int = 5
    cooldown_candles: int = 24
    max_trades_per_day: int = 20
    stale_data_seconds: int = 90
    allow_short: bool = False

    def validate(self) -> None:
        if not (0 < self.max_daily_loss_pct <= 1.0):
            raise ConfigError("max_daily_loss_pct deve essere in (0, 1]")
        if not (0 < self.max_drawdown_pct <= 1.0):
            raise ConfigError("max_drawdown_pct deve essere in (0, 1]")
        if self.max_consecutive_losses < 1 or self.cooldown_candles < 0:
            raise ConfigError("max_consecutive_losses >= 1 e cooldown_candles >= 0 richiesti")
        if self.max_trades_per_day < 1 or self.stale_data_seconds <= 0:
            raise ConfigError("max_trades_per_day >= 1 e stale_data_seconds > 0 richiesti")


@dataclass(frozen=True)
class CostsConfig:
    fee_bps: float = 10.0
    slippage_bps: float = 3.0

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps)

    def validate(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ConfigError("fee_bps e slippage_bps non possono essere negativi")


@dataclass(frozen=True)
class EngineConfig:
    poll_seconds: int = 5
    state_file: str = "runtime/state.json"
    log_file: str = "runtime/engine.log"
    trades_file: str = "runtime/trades.csv"


@dataclass(frozen=True)
class Config:
    mode: str = "paper"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    initial_capital_usd: float = 10_000.0
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ConfigError(f"mode deve essere 'paper' o 'live', trovato {self.mode!r}")
        if self.initial_capital_usd <= 0:
            raise ConfigError("capital.initial_usd deve essere positivo")
        self.exchange.validate()
        self.ensemble.validate()
        self.sizing.validate()
        self.risk.validate()
        self.costs.validate()
        if self.mode == "live":
            # Doppio consenso esplicito prima di toccare capitale reale.
            if os.environ.get("MIROFISH_I_UNDERSTAND_THE_RISKS") != "YES":
                raise ConfigError(
                    "mode: live richiede la variabile d'ambiente "
                    "MIROFISH_I_UNDERSTAND_THE_RISKS=YES. Il trading con capitale "
                    "reale può azzerare il conto: usa paper/testnet finché il "
                    "sistema non è stato validato per settimane."
                )
            if not os.environ.get("MIROFISH_API_KEY") or not os.environ.get("MIROFISH_API_SECRET"):
                raise ConfigError(
                    "mode: live richiede MIROFISH_API_KEY e MIROFISH_API_SECRET nell'ambiente"
                )


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    def sect(name: str) -> dict:
        v = raw.get(name) or {}
        if not isinstance(v, dict):
            raise ConfigError(f"sezione '{name}' non valida")
        return v

    cfg = Config(
        mode=str(raw.get("mode", "paper")).lower(),
        exchange=ExchangeConfig(**{k: v for k, v in sect("exchange").items()
                                   if k in ExchangeConfig.__dataclass_fields__}),
        initial_capital_usd=float(sect("capital").get("initial_usd", 10_000.0)),
        ensemble=EnsembleConfig(**{k: v for k, v in sect("ensemble").items()
                                   if k in EnsembleConfig.__dataclass_fields__}),
        sizing=SizingConfig(**{k: v for k, v in sect("sizing").items()
                               if k in SizingConfig.__dataclass_fields__}),
        risk=RiskConfig(**{k: v for k, v in sect("risk").items()
                           if k in RiskConfig.__dataclass_fields__}),
        costs=CostsConfig(**{k: v for k, v in sect("costs").items()
                             if k in CostsConfig.__dataclass_fields__}),
        engine=EngineConfig(**{k: v for k, v in sect("engine").items()
                               if k in EngineConfig.__dataclass_fields__}),
    )
    cfg.validate()
    return cfg
