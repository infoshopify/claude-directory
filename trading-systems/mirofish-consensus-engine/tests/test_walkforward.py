"""Test della walk-forward validation (Fase 1: D1+D2)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import load_config
from engine.walkforward import (GRIDS, apply_overrides, run_walkforward, score,
                                stress_costs)

HERE = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    c = load_config(HERE / "config.yaml")
    # Percorsi ridotti: nei test conta la correttezza dello schema, non la
    # precisione statistica dei simulatori.
    return replace(c, ensemble=replace(c.ensemble, paths_per_simulator=32))


def make_ohlcv(n, seed=0, drift=0.0005, vol=0.002, start=75_000.0):
    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    ts = (np.arange(n) * 300 + 1_700_000_000) * 1000.0
    return np.column_stack([ts, closes, closes * 1.001, closes * 0.999,
                            closes, np.full(n, 100.0)])


# ---------------------------------------------------------------- griglie --
def test_grids_are_small_and_valid(cfg):
    assert 2 <= len(GRIDS["small"]) <= 6
    assert len(GRIDS["medium"]) <= 24
    for combo in GRIDS["small"] + GRIDS["medium"]:
        c = apply_overrides(cfg, combo)  # non deve sollevare
        assert c.ensemble.exit_votes <= c.ensemble.entry_votes


def test_apply_overrides_rejects_unknown(cfg):
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"parametro_inventato": 1})


def test_stress_costs_multiplies(cfg):
    c2 = stress_costs(cfg, 2.0)
    assert c2.costs.fee_bps == cfg.costs.fee_bps * 2
    assert c2.costs.slippage_bps == cfg.costs.slippage_bps * 2
    assert stress_costs(cfg, 1.0) is cfg


# ---------------------------------------------------------------- scoring --
def test_score_penalizes_small_samples():
    from engine.backtester import BacktestResult
    good = BacktestResult(100, 10, 10_500, 5.0, 2.0, 0.6, 1.0)
    few = BacktestResult(100, 1, 11_000, 10.0, 0.5, 1.0, 3.0)
    assert score(good, min_trades=3) > score(few, min_trades=3)


# ---------------------------------------------------- struttura finestre --
def test_windows_never_overlap_and_respect_embargo(cfg):
    lookback = cfg.ensemble.lookback_candles
    n = lookback + 3 * 400 + 200
    ohlcv = make_ohlcv(n, seed=1)
    rep = run_walkforward(cfg, ohlcv, train_candles=400, test_candles=200,
                          embargo_candles=12, grid=[GRIDS["small"][0]], seed=7)
    assert len(rep.windows) >= 2
    for w in rep.windows:
        train_lo, train_hi = w.train_range
        test_lo, test_hi = w.test_range
        # embargo rispettato: il test inizia dopo train_end + embargo
        assert test_lo == train_hi + 12
        assert test_hi > test_lo
    # le finestre di test scorrono senza sovrapporsi
    for a, b in zip(rep.windows, rep.windows[1:]):
        assert b.test_range[0] >= a.test_range[1]


def test_parallel_matches_sequential(cfg):
    """jobs>1 deve dare risultati identici a jobs=1 (seed fissi per task)."""
    lookback = cfg.ensemble.lookback_candles
    n = lookback + 3 * 400 + 200
    ohlcv = make_ohlcv(n, seed=5, drift=0.0008)
    seq = run_walkforward(cfg, ohlcv, train_candles=400, test_candles=200,
                          grid="small", seed=11, jobs=1)
    par = run_walkforward(cfg, ohlcv, train_candles=400, test_candles=200,
                          grid="small", seed=11, jobs=2)
    assert len(seq.windows) == len(par.windows)
    assert seq.oos_return_pct == pytest.approx(par.oos_return_pct, rel=1e-12)
    assert seq.oos_trades == par.oos_trades
    for a, b in zip(seq.windows, par.windows):
        assert a.chosen == b.chosen
        assert a.test_return_pct == pytest.approx(b.test_return_pct, rel=1e-12)


def test_insufficient_data_raises(cfg):
    ohlcv = make_ohlcv(cfg.ensemble.lookback_candles + 100)
    with pytest.raises(ValueError):
        run_walkforward(cfg, ohlcv, train_candles=400, test_candles=200)


def test_train_must_exceed_lookback(cfg):
    ohlcv = make_ohlcv(3000)
    with pytest.raises(ValueError):
        run_walkforward(cfg, ohlcv, train_candles=100, test_candles=200)


# ------------------------------------------------------------ end-to-end --
def test_walkforward_end_to_end_report(cfg):
    lookback = cfg.ensemble.lookback_candles
    n = lookback + 2 * (500 + 12 + 250)
    ohlcv = make_ohlcv(n, seed=2, drift=0.0008)
    rep = run_walkforward(cfg, ohlcv, train_candles=500, test_candles=250,
                          grid=GRIDS["small"][:2], seed=3)
    assert rep.grid_size == 2
    assert rep.oos_trades >= 0
    assert isinstance(rep.oos_return_pct, float)
    assert rep.oos_max_drawdown_pct >= 0
    txt = rep.summary()
    assert "VERDETTO" in txt
    assert "gap overfitting" in txt
    # ogni scelta viene dalla griglia
    for w in rep.windows:
        assert w.chosen in GRIDS["small"][:2]
    # la curva OOS è continua (nessun salto di rinormalizzazione)
    eq = rep.oos_equity_curve
    assert eq[0] == 1.0
    assert all(e > 0 for e in eq)
