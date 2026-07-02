"""Backtester event-driven: replay di candele storiche nella stessa pipeline
usata in paper e live (nessuna differenza di codice = nessun bias di percorso).

Nota di onestà metodologica: un backtest su dati passati NON dimostra che la
strategia guadagnerà. Serve a scartare le strategie che perdono anche sul
passato e a stimare l'ordine di grandezza di drawdown e turnover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .broker import PaperBroker
from .config import Config
from .mirofish import MiroFishEngine
from .orchestrator import Orchestrator
from .risk import RiskManager


@dataclass
class BacktestResult:
    n_candles: int
    n_trades: int
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    sharpe_annualized: float
    equity_curve: list[float] = field(repr=False, default_factory=list)
    trades: list = field(repr=False, default_factory=list)

    def summary(self) -> str:
        return (
            f"candele:        {self.n_candles}\n"
            f"trade chiusi:   {self.n_trades}\n"
            f"equity finale:  {self.final_equity:,.2f} USD\n"
            f"rendimento:     {self.total_return_pct:+.2f}%\n"
            f"max drawdown:   {self.max_drawdown_pct:.2f}%\n"
            f"win rate:       {self.win_rate:.1%}\n"
            f"Sharpe (ann.):  {self.sharpe_annualized:.2f}"
        )


def run_backtest(cfg: Config, ohlcv: np.ndarray,
                 seed: int | None = 42) -> BacktestResult:
    """ohlcv: array (N, 6) = [ts_ms, open, high, low, close, volume]."""
    closes_all = ohlcv[:, 4].astype(float)
    ts_all = ohlcv[:, 0].astype(float) / 1000.0
    lookback = cfg.ensemble.lookback_candles
    if len(closes_all) <= lookback + 2:
        raise ValueError(f"servono più di {lookback + 2} candele, trovate {len(closes_all)}")

    rng = np.random.default_rng(seed)
    broker = PaperBroker(cash_usd=cfg.initial_capital_usd,
                         fee_bps=cfg.costs.fee_bps,
                         slippage_bps=cfg.costs.slippage_bps)
    risk = RiskManager(cfg.risk, cfg.initial_capital_usd)
    mirofish = MiroFishEngine(cfg.ensemble, cfg.costs, rng=rng)
    orch = Orchestrator(cfg, broker, risk, mirofish, persist=False)

    equity_curve = []
    for i in range(lookback, len(closes_all)):
        window = closes_all[: i + 1]
        orch.on_candle(ts_all[i], window[-lookback:], data_age_seconds=0.0)
        equity_curve.append(broker.equity(closes_all[i]))

    # Chiusura forzata a fine periodo per valutare il PnL completo.
    if broker.position.side != 0:
        broker.close_position(closes_all[-1], ts_all[-1], "fine_backtest")

    eq = np.array(equity_curve)
    final = broker.equity(closes_all[-1])
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(1.0 - eq / peak)) * 100 if len(eq) else 0.0

    rets = np.diff(eq) / eq[:-1]
    # 288 candele 5-min al giorno, 365 giorni.
    ann = np.sqrt(288 * 365)
    sharpe = float(np.mean(rets) / np.std(rets) * ann) if len(rets) > 2 and np.std(rets) > 0 else 0.0

    wins = sum(1 for t in broker.trades if t.pnl_usd > 0)
    return BacktestResult(
        n_candles=len(equity_curve),
        n_trades=len(broker.trades),
        final_equity=final,
        total_return_pct=(final / cfg.initial_capital_usd - 1.0) * 100,
        max_drawdown_pct=max_dd,
        win_rate=wins / len(broker.trades) if broker.trades else 0.0,
        sharpe_annualized=sharpe,
        equity_curve=eq.tolist(),
        trades=broker.trades,
    )
