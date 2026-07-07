"""Walk-forward validation con purging/embargo (Fase 1 del piano, voce D1).

Il problema che risolve: qualunque parametro "ottimo" trovato guardando un
periodo storico è, in parte, rumore memorizzato. Il walk-forward separa
rigidamente la scelta dei parametri dalla loro valutazione:

    [---- TRAIN ----][embargo][-- TEST --]
                     [---- TRAIN ----][embargo][-- TEST --]  ->- scorre

Su ogni finestra TRAIN si prova una griglia (piccola!) di configurazioni e si
sceglie la migliore; quella configurazione viene poi valutata SOLO sulla
finestra TEST successiva, mai vista durante la scelta. L'embargo tra train e
test copre l'orizzonte di previsione, così nessuna decisione presa nel train
"tocca" candele del test (purging del leakage).

Il verdetto onesto è la somma dei soli risultati out-of-sample concatenati.
Il divario tra performance in-sample e out-of-sample misura quanto stavamo
overfittando.

Nota: la finestra TEST usa come warmup le ultime `lookback` candele precedenti
(che possono appartenere al train). Non è leakage: al momento del test quei
prezzi sono passato noto; il leakage sta nello scegliere i parametri guardando
il futuro, ed è esattamente ciò che questo schema impedisce.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .backtester import BacktestResult, run_backtest
from .config import Config

# ---------------------------------------------------------------- griglie --
# Le griglie sono volutamente piccole: ogni combinazione in più è un grado di
# libertà per l'overfitting (e va dichiarata quando si interpreta il risultato).

GRIDS: dict[str, list[dict]] = {
    "small": [
        {"entry_votes": 28, "exit_votes": 26, "kelly_fraction": 0.25},
        {"entry_votes": 28, "exit_votes": 26, "kelly_fraction": 0.15},
        {"entry_votes": 26, "exit_votes": 24, "kelly_fraction": 0.25},
        {"entry_votes": 26, "exit_votes": 24, "kelly_fraction": 0.15},
    ],
    "medium": [
        {"entry_votes": ev, "exit_votes": xv, "horizon_candles": h,
         "kelly_fraction": k}
        for ev, xv in ((30, 28), (28, 26), (26, 24))
        for h in (6, 12, 24)
        for k in (0.15, 0.25)
    ],
}

ENSEMBLE_KEYS = {"entry_votes", "exit_votes", "horizon_candles",
                 "paths_per_simulator", "vote_threshold_bps", "seed"}
SIZING_KEYS = {"kelly_fraction", "max_position_pct", "min_notional_usd"}


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Crea una variante della config con i parametri della griglia."""
    ens = {k: v for k, v in overrides.items() if k in ENSEMBLE_KEYS}
    siz = {k: v for k, v in overrides.items() if k in SIZING_KEYS}
    unknown = set(overrides) - ENSEMBLE_KEYS - SIZING_KEYS
    if unknown:
        raise ValueError(f"override sconosciuti: {unknown}")
    new = replace(
        cfg,
        ensemble=replace(cfg.ensemble, **ens) if ens else cfg.ensemble,
        sizing=replace(cfg.sizing, **siz) if siz else cfg.sizing,
    )
    new.ensemble.validate()
    new.sizing.validate()
    return new


def stress_costs(cfg: Config, multiplier: float) -> Config:
    """Moltiplica fees e slippage: se l'edge sopravvive a costi 2x, è più credibile."""
    if multiplier == 1.0:
        return cfg
    return replace(cfg, costs=replace(
        cfg.costs,
        fee_bps=cfg.costs.fee_bps * multiplier,
        slippage_bps=cfg.costs.slippage_bps * multiplier,
    ))


# ---------------------------------------------------------------- scoring --
def score(result: BacktestResult, min_trades: int) -> float:
    """Punteggio robusto: rendimento netto penalizzato dal drawdown.

    Con meno di `min_trades` il campione è troppo piccolo per fidarsi:
    punteggio fortemente negativo così la selezione non premia la fortuna
    di 1-2 trade.
    """
    if result.n_trades < min_trades:
        return -1e9 + result.n_trades  # comunque ordinabile tra loro
    return result.total_return_pct - 0.5 * result.max_drawdown_pct


# ---------------------------------------------------------------- risultati --
@dataclass
class WindowResult:
    index: int
    train_range: tuple[int, int]        # indici candela [start, end)
    test_range: tuple[int, int]
    chosen: dict
    train_score: float
    train_return_pct: float
    test: BacktestResult = field(repr=False)

    @property
    def test_return_pct(self) -> float:
        return self.test.total_return_pct


@dataclass
class WalkForwardReport:
    windows: list[WindowResult]
    grid_size: int
    train_candles: int
    test_candles: int
    embargo_candles: int
    cost_multiplier: float
    elapsed_seconds: float

    # -- aggregati out-of-sample ---------------------------------------- #
    @property
    def oos_return_pct(self) -> float:
        """Rendimento OOS composto concatenando le finestre di test."""
        r = 1.0
        for w in self.windows:
            r *= 1.0 + w.test_return_pct / 100.0
        return (r - 1.0) * 100.0

    @property
    def is_return_pct_mean(self) -> float:
        return float(np.mean([w.train_return_pct for w in self.windows])) \
            if self.windows else 0.0

    @property
    def oos_return_pct_mean(self) -> float:
        return float(np.mean([w.test_return_pct for w in self.windows])) \
            if self.windows else 0.0

    @property
    def oos_equity_curve(self) -> list[float]:
        """Curva OOS concatenata (ogni finestra rinormalizzata in continuità)."""
        curve: list[float] = [1.0]
        for w in self.windows:
            eq = np.asarray(w.test.equity_curve)
            if len(eq) < 2:
                continue
            seg = eq / eq[0] * curve[-1]
            curve.extend(seg[1:].tolist())
        return curve

    @property
    def oos_sharpe_annualized(self) -> float:
        eq = np.asarray(self.oos_equity_curve)
        if len(eq) < 20:
            return 0.0
        rets = np.diff(eq) / eq[:-1]
        sd = float(np.std(rets))
        if sd == 0:
            return 0.0
        return float(np.mean(rets) / sd * np.sqrt(288 * 365))

    @property
    def oos_max_drawdown_pct(self) -> float:
        eq = np.asarray(self.oos_equity_curve)
        if len(eq) < 2:
            return 0.0
        peak = np.maximum.accumulate(eq)
        return float(np.max(1.0 - eq / peak)) * 100.0

    @property
    def oos_trades(self) -> int:
        return sum(w.test.n_trades for w in self.windows)

    @property
    def overfitting_gap_pct(self) -> float:
        """Media IS - media OOS: più è grande, più la selezione era rumore."""
        return self.is_return_pct_mean - self.oos_return_pct_mean

    @property
    def param_stability(self) -> dict[str, int]:
        """Quante volte ogni combinazione è stata scelta (instabilità = sospetto)."""
        counts: dict[str, int] = {}
        for w in self.windows:
            key = str(sorted(w.chosen.items()))
            counts[key] = counts.get(key, 0) + 1
        return counts

    # -- verdetto --------------------------------------------------------- #
    def verdict(self) -> str:
        if not self.windows:
            return "Nessuna finestra valutata: dati insufficienti."
        lines = []
        if self.oos_return_pct <= 0:
            lines.append(
                "VERDETTO: nessun edge dimostrato out-of-sample. Con questi dati "
                "la strategia NON va portata in live; ha senso rivedere i segnali "
                "(fasi 4-5 del piano) o l'orizzonte, non ottimizzare ancora."
            )
        elif self.overfitting_gap_pct > max(2.0, 2 * abs(self.oos_return_pct_mean)):
            lines.append(
                "VERDETTO: OOS positivo ma con forte divario in-sample/out-of-sample: "
                "la selezione parametri sta catturando molto rumore. Servono più dati "
                "e/o griglia più piccola prima di trarre conclusioni."
            )
        else:
            lines.append(
                "VERDETTO: OOS positivo e coerente con l'in-sample su questi dati. "
                "NON è una garanzia: ripetere su più anni/regimi e validare in paper "
                "(fase D4) prima di qualunque live."
            )
        if self.oos_trades < 30 * max(1, len(self.windows)) / 10:
            lines.append(
                f"ATTENZIONE: solo {self.oos_trades} trade OOS totali — campione "
                "piccolo, la varianza domina qualunque conclusione."
            )
        if len(self.param_stability) > max(1, len(self.windows) // 2):
            lines.append(
                "ATTENZIONE: i parametri scelti cambiano quasi a ogni finestra — "
                "instabilità tipica di assenza di segnale robusto."
            )
        return "\n".join(lines)

    def summary(self) -> str:
        rows = [
            f"finestre:              {len(self.windows)}",
            f"griglia:               {self.grid_size} configurazioni",
            f"train/test/embargo:    {self.train_candles}/{self.test_candles}/"
            f"{self.embargo_candles} candele",
            f"moltiplicatore costi:  x{self.cost_multiplier:.1f}",
            f"trade OOS totali:      {self.oos_trades}",
            f"rendimento OOS:        {self.oos_return_pct:+.2f}% (composto)",
            f"rendimento IS medio:   {self.is_return_pct_mean:+.2f}% per finestra",
            f"rendimento OOS medio:  {self.oos_return_pct_mean:+.2f}% per finestra",
            f"gap overfitting:       {self.overfitting_gap_pct:+.2f} punti",
            f"Sharpe OOS (ann.):     {self.oos_sharpe_annualized:.2f}",
            f"max drawdown OOS:      {self.oos_max_drawdown_pct:.2f}%",
            f"tempo di calcolo:      {self.elapsed_seconds:.0f}s",
            "",
            "scelte per finestra:",
        ]
        for w in self.windows:
            rows.append(
                f"  W{w.index:02d} train[{w.train_range[0]}:{w.train_range[1]}) "
                f"test[{w.test_range[0]}:{w.test_range[1]}) -> {w.chosen} | "
                f"IS {w.train_return_pct:+.2f}% / OOS {w.test_return_pct:+.2f}% "
                f"({w.test.n_trades} trade)"
            )
        rows += ["", self.verdict()]
        return "\n".join(rows)


# ---------------------------------------------------------------- driver --
def run_walkforward(
    cfg: Config,
    ohlcv: np.ndarray,
    train_candles: int,
    test_candles: int,
    embargo_candles: int | None = None,
    grid: str | list[dict] = "small",
    seed: int = 42,
    min_trades: int = 3,
    cost_multiplier: float = 1.0,
    progress: bool = False,
) -> WalkForwardReport:
    """Esegue il walk-forward completo su un array OHLCV (N, 6)."""
    t0 = time.time()
    combos = GRIDS[grid] if isinstance(grid, str) else grid
    if not combos:
        raise ValueError("griglia vuota")
    base = stress_costs(cfg, cost_multiplier)
    lookback = base.ensemble.lookback_candles
    embargo = base.ensemble.horizon_candles if embargo_candles is None \
        else embargo_candles

    n = len(ohlcv)
    if train_candles <= lookback:
        raise ValueError(f"train_candles ({train_candles}) deve superare "
                         f"lookback ({lookback})")
    needed = lookback + train_candles + embargo + test_candles
    if n < needed:
        raise ValueError(f"servono almeno {needed} candele, trovate {n}")

    windows: list[WindowResult] = []
    # La prima finestra parte lasciando lookback candele di warmup al train.
    t_start = lookback
    w_idx = 0
    while t_start + train_candles + embargo + test_candles <= n:
        train_lo, train_hi = t_start, t_start + train_candles
        test_lo = train_hi + embargo
        test_hi = test_lo + test_candles

        # -- selezione sulla finestra train (stesso seed per confronto equo) --
        best_combo, best_res, best_score = None, None, -np.inf
        for combo in combos:
            c = apply_overrides(base, combo)
            res = run_backtest(c, ohlcv[train_lo - lookback: train_hi], seed=seed)
            s = score(res, min_trades)
            if s > best_score:
                best_combo, best_res, best_score = combo, res, s

        # -- valutazione out-of-sample, una sola volta, mai riottimizzata --
        chosen_cfg = apply_overrides(base, best_combo)
        test_res = run_backtest(chosen_cfg, ohlcv[test_lo - lookback: test_hi],
                                seed=seed + 1)

        windows.append(WindowResult(
            index=w_idx,
            train_range=(train_lo, train_hi),
            test_range=(test_lo, test_hi),
            chosen=best_combo,
            train_score=best_score,
            train_return_pct=best_res.total_return_pct,
            test=test_res,
        ))
        if progress:
            print(f"  W{w_idx:02d}: {best_combo} | IS "
                  f"{best_res.total_return_pct:+.2f}% -> OOS "
                  f"{test_res.total_return_pct:+.2f}%", flush=True)
        t_start += test_candles
        w_idx += 1

    return WalkForwardReport(
        windows=windows,
        grid_size=len(combos),
        train_candles=train_candles,
        test_candles=test_candles,
        embargo_candles=embargo,
        cost_multiplier=cost_multiplier,
        elapsed_seconds=time.time() - t0,
    )
