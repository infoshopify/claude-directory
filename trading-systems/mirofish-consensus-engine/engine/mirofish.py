"""MiroFish: ensemble di 31 simulatori Monte Carlo indipendenti.

Ogni simulatore riceve la stessa storia di candele 5-min ma è calibrato in modo
diverso (finestra di volatilità, code grasse, blocchi di bootstrap, tilt di
momentum o mean-reversion, condizionamento al regime). Ognuno proietta
`paths_per_simulator` percorsi di prezzo su `horizon_candles` candele e vota:

  UP      se la mediana del rendimento terminale supera +soglia
  DOWN    se scende sotto -soglia
  NEUTRAL altrimenti

La soglia è il massimo tra `vote_threshold_bps` e i costi di andata e ritorno:
un voto direzionale ha senso solo se l'edge atteso paga fees e slippage.
La diversità dei 31 modelli è il punto: quando 28 configurazioni diverse
vedono la stessa cosa, il segnale è robusto rispetto alla scelta del modello.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CostsConfig, EnsembleConfig

UP, DOWN, NEUTRAL = 1, -1, 0


@dataclass(frozen=True)
class SimulatorSpec:
    """Configurazione di un singolo simulatore dell'ensemble."""

    name: str
    kind: str            # "block_bootstrap" | "gbm_t" | "ewma_regime"
    vol_window: int      # finestra per la stima di volatilità
    drift_window: int    # finestra per la stima del drift
    block_size: int      # per il bootstrap a blocchi
    dof: float           # gradi di libertà Student-t (code grasse)
    momentum_tilt: float # >0 amplifica il trend recente, <0 mean-reversion
    ewma_lambda: float   # decadimento EWMA per la vol condizionale


def build_ensemble_specs(n: int) -> list[SimulatorSpec]:
    """Genera n specifiche deterministiche e diverse tra loro.

    Le tre famiglie si alternano; i parametri scorrono su griglie scelte per
    coprire orizzonti di memoria corti/lunghi, code sottili/grasse e tilt
    momentum/mean-reversion. Con n=31 nessun simulatore è identico a un altro.
    """
    kinds = ("block_bootstrap", "gbm_t", "ewma_regime")
    vol_windows = (24, 48, 96, 144, 288)
    drift_windows = (12, 36, 72, 144)
    block_sizes = (3, 6, 12, 24)
    dofs = (3.0, 4.0, 6.0, 10.0)
    tilts = (-0.5, -0.25, 0.0, 0.25, 0.5)
    lambdas = (0.90, 0.94, 0.97)

    specs = []
    for i in range(n):
        specs.append(SimulatorSpec(
            name=f"sim_{i:02d}",
            kind=kinds[i % len(kinds)],
            vol_window=vol_windows[i % len(vol_windows)],
            drift_window=drift_windows[i % len(drift_windows)],
            block_size=block_sizes[i % len(block_sizes)],
            dof=dofs[i % len(dofs)],
            momentum_tilt=tilts[i % len(tilts)],
            ewma_lambda=lambdas[i % len(lambdas)],
        ))
    return specs


@dataclass
class SimulatorResult:
    name: str
    vote: int                  # UP / DOWN / NEUTRAL
    median_return: float       # rendimento log mediano a orizzonte
    p_up: float                # frazione di percorsi con rendimento > 0
    q05: float                 # quantile 5% del rendimento terminale
    q95: float                 # quantile 95%


@dataclass
class EnsembleDecision:
    votes_up: int
    votes_down: int
    votes_neutral: int
    direction: int             # UP / DOWN / NEUTRAL (direzione di maggioranza)
    consensus: int             # voti nella direzione di maggioranza
    p_up: float                # media ensemble della probabilità di rialzo
    expected_return: float     # mediana ensemble del rendimento a orizzonte
    win_loss_ratio: float      # |guadagno medio se vince| / |perdita media se perde|
    results: list[SimulatorResult]

    @property
    def n(self) -> int:
        return len(self.results)


class MiroFishEngine:
    def __init__(self, cfg: EnsembleConfig, costs: CostsConfig,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.costs = costs
        self.specs = build_ensemble_specs(cfg.n_simulators)
        self.rng = rng or np.random.default_rng(cfg.seed)
        # Soglia di voto: mai sotto i costi di round-trip.
        self.threshold = max(cfg.vote_threshold_bps, costs.round_trip_bps) / 1e4

    # ------------------------------------------------------------------ #
    def decide(self, closes: np.ndarray) -> EnsembleDecision:
        """Esegue tutti i simulatori sull'ultima finestra di prezzi di chiusura."""
        closes = np.asarray(closes, dtype=float)
        if closes.ndim != 1 or len(closes) < self.cfg.lookback_candles // 2:
            raise ValueError(f"servono almeno {self.cfg.lookback_candles // 2} chiusure")

        window = closes[-self.cfg.lookback_candles:]
        log_ret = np.diff(np.log(window))
        if not np.all(np.isfinite(log_ret)):
            raise ValueError("prezzi non validi nella finestra (NaN/inf/<=0)")

        results = [self._run_simulator(spec, log_ret) for spec in self.specs]

        votes_up = sum(1 for r in results if r.vote == UP)
        votes_down = sum(1 for r in results if r.vote == DOWN)
        votes_neutral = len(results) - votes_up - votes_down

        if votes_up >= votes_down and votes_up > 0:
            direction, consensus = UP, votes_up
        elif votes_down > votes_up:
            direction, consensus = DOWN, votes_down
        else:
            direction, consensus = NEUTRAL, votes_neutral

        med = float(np.median([r.median_return for r in results]))
        p_up = float(np.mean([r.p_up for r in results]))

        # Rapporto vincita/perdita stimato dalla distribuzione dell'ensemble,
        # usato dal Kelly sizing: upside al q95 contro downside al q05.
        ups = float(np.median([r.q95 for r in results]))
        downs = float(np.median([r.q05 for r in results]))
        if direction == DOWN:
            ups, downs = -downs, -ups
        win_loss = abs(ups) / max(abs(downs), 1e-6)

        return EnsembleDecision(
            votes_up=votes_up, votes_down=votes_down, votes_neutral=votes_neutral,
            direction=direction, consensus=consensus,
            p_up=p_up, expected_return=med, win_loss_ratio=win_loss,
            results=results,
        )

    # ------------------------------------------------------------------ #
    def _run_simulator(self, spec: SimulatorSpec, log_ret: np.ndarray) -> SimulatorResult:
        h = self.cfg.horizon_candles
        m = self.cfg.paths_per_simulator

        if spec.kind == "block_bootstrap":
            paths = self._paths_block_bootstrap(spec, log_ret, m, h)
        elif spec.kind == "gbm_t":
            paths = self._paths_gbm_t(spec, log_ret, m, h)
        else:
            paths = self._paths_ewma_regime(spec, log_ret, m, h)

        terminal = paths.sum(axis=1)  # rendimento log cumulato a orizzonte
        med = float(np.median(terminal))
        p_up = float(np.mean(terminal > 0))
        q05 = float(np.quantile(terminal, 0.05))
        q95 = float(np.quantile(terminal, 0.95))

        if med > self.threshold:
            vote = UP
        elif med < -self.threshold:
            vote = DOWN
        else:
            vote = NEUTRAL
        return SimulatorResult(spec.name, vote, med, p_up, q05, q95)

    # -- famiglie di modelli ------------------------------------------- #
    def _drift(self, spec: SimulatorSpec, log_ret: np.ndarray) -> float:
        recent = log_ret[-spec.drift_window:]
        base = float(np.mean(recent))
        # tilt > 0: estrapola il momentum; tilt < 0: si aspetta il rientro.
        return base * (1.0 + spec.momentum_tilt)

    def _paths_block_bootstrap(self, spec, log_ret, m, h) -> np.ndarray:
        """Ricampionamento a blocchi: preserva l'autocorrelazione a breve."""
        b = max(1, min(spec.block_size, h))
        n_blocks = int(np.ceil(h / b))
        max_start = len(log_ret) - b
        starts = self.rng.integers(0, max_start + 1, size=(m, n_blocks))
        idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(m, -1)[:, :h]
        paths = log_ret[idx]
        drift_adj = self._drift(spec, log_ret) - float(np.mean(log_ret))
        return paths + drift_adj

    def _paths_gbm_t(self, spec, log_ret, m, h) -> np.ndarray:
        """Moto browniano geometrico con innovazioni Student-t (code grasse)."""
        vol = float(np.std(log_ret[-spec.vol_window:], ddof=1))
        drift = self._drift(spec, log_ret)
        t = self.rng.standard_t(spec.dof, size=(m, h))
        t *= np.sqrt((spec.dof - 2) / spec.dof)  # normalizza a varianza unitaria
        return drift + vol * t

    def _paths_ewma_regime(self, spec, log_ret, m, h) -> np.ndarray:
        """Vol condizionale EWMA + campionamento dal regime di vol corrente."""
        lam = spec.ewma_lambda
        var = np.empty_like(log_ret)
        var[0] = log_ret[0] ** 2
        for i in range(1, len(log_ret)):
            var[i] = lam * var[i - 1] + (1 - lam) * log_ret[i] ** 2
        sigma = np.sqrt(np.maximum(var, 1e-12))

        # Regime corrente: campiona i rendimenti storici con vol simile a oggi.
        current = sigma[-1]
        dist = np.abs(sigma - current)
        k = max(spec.vol_window, 32)
        pool = log_ret[np.argsort(dist)[:k]]
        samples = self.rng.choice(pool, size=(m, h), replace=True)
        drift_adj = self._drift(spec, log_ret) - float(np.mean(pool))
        return samples + drift_adj
