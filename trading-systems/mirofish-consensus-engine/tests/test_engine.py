"""Test del motore: ensemble, consenso, Kelly, risk, broker, backtest end-to-end."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.backtester import run_backtest
from engine.broker import FLAT, LONG, PaperBroker
from engine.config import Config, ConfigError, EnsembleConfig, load_config
from engine.kelly import kelly_size
from engine.mirofish import DOWN, NEUTRAL, UP, MiroFishEngine, build_ensemble_specs
from engine.orchestrator import Orchestrator
from engine.risk import RiskManager

HERE = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> Config:
    c = load_config(HERE / "config.yaml")
    assert c.mode == "paper"
    return c


def synthetic_closes(n=600, drift=0.0, vol=0.002, seed=0, start=75_000.0):
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    return start * np.exp(np.cumsum(rets))


# ---------------------------------------------------------------- ensemble --
def test_ensemble_has_31_distinct_simulators(cfg):
    specs = build_ensemble_specs(cfg.ensemble.n_simulators)
    assert len(specs) == 31
    assert len({(s.kind, s.vol_window, s.drift_window, s.block_size,
                 s.dof, s.momentum_tilt, s.ewma_lambda) for s in specs}) == 31


def test_votes_sum_to_n(cfg):
    eng = MiroFishEngine(cfg.ensemble, cfg.costs, rng=np.random.default_rng(1))
    d = eng.decide(synthetic_closes())
    assert d.votes_up + d.votes_down + d.votes_neutral == 31
    assert 0.0 <= d.p_up <= 1.0


def test_strong_uptrend_produces_up_consensus(cfg):
    eng = MiroFishEngine(cfg.ensemble, cfg.costs, rng=np.random.default_rng(2))
    closes = synthetic_closes(drift=0.004, vol=0.001, seed=3)  # trend fortissimo
    d = eng.decide(closes)
    assert d.direction == UP
    assert d.votes_up >= cfg.ensemble.entry_votes


def test_strong_downtrend_produces_down_consensus(cfg):
    eng = MiroFishEngine(cfg.ensemble, cfg.costs, rng=np.random.default_rng(4))
    closes = synthetic_closes(drift=-0.004, vol=0.001, seed=5)
    d = eng.decide(closes)
    assert d.direction == DOWN
    assert d.votes_down >= cfg.ensemble.entry_votes


def test_flat_market_no_consensus(cfg):
    eng = MiroFishEngine(cfg.ensemble, cfg.costs, rng=np.random.default_rng(6))
    closes = synthetic_closes(drift=0.0, vol=0.0005, seed=7)
    d = eng.decide(closes)
    # In un mercato piatto l'edge non supera i costi: niente 28/31.
    assert max(d.votes_up, d.votes_down) < cfg.ensemble.entry_votes


def test_vote_threshold_at_least_round_trip_costs(cfg):
    eng = MiroFishEngine(cfg.ensemble, cfg.costs)
    assert eng.threshold >= cfg.costs.round_trip_bps / 1e4


# ------------------------------------------------------------------- kelly --
def test_kelly_positive_edge():
    s = kelly_size(10_000, p_win=0.60, win_loss_ratio=1.5,
                   kelly_fraction=0.25, max_position_pct=0.20, min_notional_usd=25)
    assert s.fraction > 0
    assert s.notional_usd <= 10_000 * 0.20


def test_kelly_no_edge_refuses():
    s = kelly_size(10_000, p_win=0.45, win_loss_ratio=1.0,
                   kelly_fraction=0.25, max_position_pct=0.20, min_notional_usd=25)
    assert s.fraction == 0.0


def test_kelly_caps_at_max_position():
    s = kelly_size(100_000, p_win=0.95, win_loss_ratio=5.0,
                   kelly_fraction=1.0, max_position_pct=0.20, min_notional_usd=25)
    assert s.fraction == pytest.approx(0.20)


def test_kelly_min_notional():
    s = kelly_size(100, p_win=0.60, win_loss_ratio=1.5,
                   kelly_fraction=0.25, max_position_pct=0.20, min_notional_usd=25)
    assert s.fraction == 0.0


# -------------------------------------------------------------------- risk --
def test_kill_switch_on_drawdown(cfg):
    rm = RiskManager(cfg.risk, 10_000)
    rm.on_new_candle(0, 10_000, 1)
    rm.on_new_candle(300, 8_900, 2)  # -11% > limite 10%
    assert rm.killed
    assert not rm.can_open(8_900, 3, 0, UP).allowed


def test_daily_loss_limit(cfg):
    rm = RiskManager(cfg.risk, 10_000)
    rm.on_new_candle(0, 10_000, 1)
    rm.on_new_candle(300, 9_650, 2)  # -3.5% nel giorno
    chk = rm.can_open(9_650, 3, 0, UP)
    assert not chk.allowed
    assert "giornaliera" in chk.reason


def test_consecutive_losses_cooldown(cfg):
    rm = RiskManager(cfg.risk, 10_000)
    for i in range(cfg.risk.max_consecutive_losses):
        rm.on_trade_closed(-10, candle_index=i)
    assert not rm.can_open(10_000, cfg.risk.max_consecutive_losses, 0, UP).allowed
    ok_index = cfg.risk.max_consecutive_losses + cfg.risk.cooldown_candles
    assert rm.can_open(10_000, ok_index, 0, UP).allowed


def test_short_blocked_when_disabled(cfg):
    # Con allow_short=false (conto spot) uno short deve essere rifiutato.
    risk = replace(cfg.risk, allow_short=False)
    rm = RiskManager(risk, 10_000)
    assert not rm.can_open(10_000, 1, 0, DOWN).allowed


def test_short_allowed_when_enabled(cfg):
    # Con allow_short=true (futures) uno short deve essere consentito.
    risk = replace(cfg.risk, allow_short=True)
    rm = RiskManager(risk, 10_000)
    assert rm.can_open(10_000, 1, 0, DOWN).allowed


def test_stale_data_blocks(cfg):
    rm = RiskManager(cfg.risk, 10_000)
    assert not rm.can_open(10_000, 1, cfg.risk.stale_data_seconds + 1, UP).allowed


# ------------------------------------------------------------------ broker --
def test_paper_broker_round_trip_costs():
    b = PaperBroker(cash_usd=10_000, fee_bps=10, slippage_bps=3)
    b.open_position(LONG, 1_000, ref_price=50_000, ts=0, votes=28)
    rec = b.close_position(ref_price=50_000, ts=300, reason="test")
    # Prezzo invariato: la perdita è esattamente costi (fees + slippage).
    assert rec.pnl_usd < 0
    assert abs(rec.pnl_usd) < 1_000 * 0.006  # < 60 bps sul notional
    assert b.position.side == FLAT
    assert b.equity(50_000) < 10_000


def test_paper_broker_rejects_over_leverage():
    b = PaperBroker(cash_usd=100, fee_bps=10, slippage_bps=3)
    with pytest.raises(ValueError):
        b.open_position(LONG, 1_000, ref_price=50_000, ts=0, votes=28)


def test_paper_broker_rejects_over_leverage_short():
    from engine.broker import SHORT
    b = PaperBroker(cash_usd=100, fee_bps=10, slippage_bps=3)
    with pytest.raises(ValueError):
        b.open_position(SHORT, 1_000, ref_price=50_000, ts=0, votes=28)


# ------------------------------------------------------------ orchestrator --
def test_orchestrator_full_cycle(cfg):
    """Trend forte -> apre; poi mercato piatto -> il consenso cade -> chiude."""
    rng = np.random.default_rng(8)
    broker = PaperBroker(cash_usd=10_000, fee_bps=10, slippage_bps=3)
    risk = RiskManager(cfg.risk, 10_000)
    eng = MiroFishEngine(cfg.ensemble, cfg.costs, rng=rng)
    orch = Orchestrator(cfg, broker, risk, eng, persist=False)

    up = synthetic_closes(600, drift=0.004, vol=0.001, seed=9)
    r1 = orch.on_candle(0.0, up)
    assert r1.action == "open_long"
    assert broker.position.side == LONG

    flat_tail = np.full(400, up[-1]) * np.exp(
        0.0002 * np.random.default_rng(10).standard_normal(400).cumsum())
    closes2 = np.concatenate([up, flat_tail])
    r2 = orch.on_candle(300.0, closes2)
    assert r2.action == "close"
    assert broker.position.side == FLAT
    assert len(broker.trades) == 1


def test_config_live_requires_env(tmp_path, cfg, monkeypatch):
    monkeypatch.delenv("MIROFISH_I_UNDERSTAND_THE_RISKS", raising=False)
    text = (HERE / "config.yaml").read_text().replace("mode: paper", "mode: live")
    p = tmp_path / "live.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError):
        load_config(p)


def test_config_invalid_timeframe_rejected(tmp_path):
    text = (HERE / "config.yaml").read_text().replace("timeframe: 15m", "timeframe: 7m")
    p = tmp_path / "tf.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError):
        load_config(p)


def test_config_negative_fees_rejected(tmp_path):
    text = (HERE / "config.yaml").read_text().replace("fee_bps: 10.0", "fee_bps: -1.0")
    p = tmp_path / "fees.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError):
        load_config(p)


def test_config_thresholds_validated(tmp_path):
    text = (HERE / "config.yaml").read_text().replace("entry_votes: 28", "entry_votes: 40")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError):
        load_config(p)


# ---------------------------------------------------------------- backtest --
def test_backtest_end_to_end(cfg):
    n = cfg.ensemble.lookback_candles + 120
    closes = synthetic_closes(n, drift=0.0008, vol=0.002, seed=11)
    ts = (np.arange(n) * 300 + 1_700_000_000) * 1000.0
    ohlcv = np.column_stack([ts, closes, closes * 1.001, closes * 0.999,
                             closes, np.full(n, 100.0)])
    res = run_backtest(cfg, ohlcv, seed=12)
    assert res.n_candles == 120
    assert res.final_equity > 0
    assert 0.0 <= res.win_rate <= 1.0
    assert res.max_drawdown_pct < 100
