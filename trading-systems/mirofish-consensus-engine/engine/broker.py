"""Esecuzione ordini: PaperBroker (default) e LiveBroker (ccxt).

Il PaperBroker simula fill a mercato applicando slippage e commissioni,
mantenendo equity e storico trade. Il LiveBroker invia ordini market reali
via ccxt e viene istanziato SOLO se la config live è stata validata
(vedi Config.validate: richiede MIROFISH_I_UNDERSTAND_THE_RISKS=YES).
"""

from __future__ import annotations

import csv
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

LONG, FLAT, SHORT = 1, 0, -1


@dataclass
class Position:
    side: int = FLAT           # LONG / FLAT / SHORT
    qty: float = 0.0           # quantità base (es. BTC)
    entry_price: float = 0.0
    entry_ts: float = 0.0
    entry_votes: int = 0


@dataclass
class TradeRecord:
    ts_open: float
    ts_close: float
    side: int
    qty: float
    entry_price: float
    exit_price: float
    pnl_usd: float
    reason: str


@dataclass
class Fill:
    price: float
    qty: float
    fee_usd: float


class Broker(ABC):
    @abstractmethod
    def buy(self, qty: float, ref_price: float) -> Fill: ...
    @abstractmethod
    def sell(self, qty: float, ref_price: float) -> Fill: ...
    @abstractmethod
    def equity(self, mark_price: float) -> float: ...


@dataclass
class PaperBroker(Broker):
    """Simulatore di esecuzione con fees e slippage; nessun ordine reale."""

    cash_usd: float
    fee_bps: float
    slippage_bps: float
    position: Position = field(default_factory=Position)
    trades: list[TradeRecord] = field(default_factory=list)

    def _fill_price(self, ref_price: float, is_buy: bool) -> float:
        slip = self.slippage_bps / 1e4
        return ref_price * (1 + slip) if is_buy else ref_price * (1 - slip)

    def buy(self, qty: float, ref_price: float) -> Fill:
        price = self._fill_price(ref_price, True)
        cost = qty * price
        fee = cost * self.fee_bps / 1e4
        if cost + fee > self.cash_usd + 1e-9:
            raise ValueError(f"cash insufficiente: serve {cost + fee:.2f}, ho {self.cash_usd:.2f}")
        self.cash_usd -= cost + fee
        return Fill(price, qty, fee)

    def sell(self, qty: float, ref_price: float) -> Fill:
        price = self._fill_price(ref_price, False)
        proceeds = qty * price
        fee = proceeds * self.fee_bps / 1e4
        self.cash_usd += proceeds - fee
        return Fill(price, qty, fee)

    def equity(self, mark_price: float) -> float:
        pos_value = self.position.qty * mark_price * self.position.side
        return self.cash_usd + pos_value

    # -- ciclo di vita della posizione (long/flat; short solo se abilitato) --
    def open_position(self, side: int, notional_usd: float, ref_price: float,
                      ts: float, votes: int) -> Position:
        if self.position.side != FLAT:
            raise ValueError("posizione già aperta")
        qty = notional_usd / ref_price
        if side == LONG:
            fill = self.buy(qty, ref_price)
        else:
            # Short paper a margine 1x: l'esposizione non può superare il cash,
            # simmetrico al vincolo di capitale del lato long.
            if notional_usd > self.cash_usd + 1e-9:
                raise ValueError(
                    f"margine insufficiente per short: notional {notional_usd:.2f} "
                    f"> cash {self.cash_usd:.2f}")
            fill = self.sell(qty, ref_price)  # short: vende allo scoperto (paper)
        self.position = Position(side=side, qty=fill.qty, entry_price=fill.price,
                                 entry_ts=ts, entry_votes=votes)
        return self.position

    def close_position(self, ref_price: float, ts: float, reason: str) -> TradeRecord:
        pos = self.position
        if pos.side == FLAT:
            raise ValueError("nessuna posizione da chiudere")
        if pos.side == LONG:
            fill = self.sell(pos.qty, ref_price)
            pnl = (fill.price - pos.entry_price) * pos.qty - fill.fee_usd
        else:
            fill = self.buy(pos.qty, ref_price)
            pnl = (pos.entry_price - fill.price) * pos.qty - fill.fee_usd
        rec = TradeRecord(pos.entry_ts, ts, pos.side, pos.qty,
                          pos.entry_price, fill.price, pnl, reason)
        self.trades.append(rec)
        self.position = Position()
        return rec


class LiveBroker(Broker):
    """Ordini market reali via ccxt. Richiede validazione esplicita in Config."""

    def __init__(self, cfg: Config):
        import ccxt  # import locale: dipendenza necessaria solo in live

        klass = getattr(ccxt, cfg.exchange.id)
        self.x = klass({
            "apiKey": os.environ["MIROFISH_API_KEY"],
            "secret": os.environ["MIROFISH_API_SECRET"],
            "enableRateLimit": True,
        })
        if cfg.exchange.testnet and hasattr(self.x, "set_sandbox_mode"):
            self.x.set_sandbox_mode(True)
        self.symbol = cfg.exchange.symbol
        self.x.load_markets()
        self.position = Position()
        self.trades: list[TradeRecord] = []

    def _order(self, side: str, qty: float,
               retries: int = 6, retry_wait_s: float = 1.0) -> Fill:
        qty = float(self.x.amount_to_precision(self.symbol, qty))
        try:
            order = self.x.create_market_order(self.symbol, side, qty)
        except Exception as e:
            raise RuntimeError(f"invio ordine {side} {qty} {self.symbol} fallito: {e}") from e

        # Rilegge l'ordine finché non ha un prezzo medio di fill reale: un Fill
        # a prezzo 0 corromperebbe PnL, risk state e log. Se l'ordine esiste
        # sull'exchange ma non riusciamo a leggerlo, meglio fermarsi con un
        # errore esplicito che proseguire con uno stato falso.
        last_err: Exception | None = None
        for _ in range(retries):
            try:
                fetched = self.x.fetch_order(order["id"], self.symbol)
                price = float(fetched.get("average") or fetched.get("price") or 0.0)
                filled = float(fetched.get("filled") or 0.0)
                if price > 0 and filled > 0:
                    fee = sum(float(f["cost"]) for f in (fetched.get("fees") or [])
                              if f.get("cost"))
                    return Fill(price, filled, fee)
            except Exception as e:
                last_err = e
            time.sleep(retry_wait_s)
        raise RuntimeError(
            f"ordine {order.get('id')} inviato ma fill non confermato dopo "
            f"{retries} tentativi (ultimo errore: {last_err}). Verifica "
            f"MANUALMENTE la posizione sull'exchange prima di riavviare."
        )

    def buy(self, qty: float, ref_price: float) -> Fill:
        return self._order("buy", qty)

    def sell(self, qty: float, ref_price: float) -> Fill:
        return self._order("sell", qty)

    def equity(self, mark_price: float, retries: int = 3) -> float:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                bal = self.x.fetch_balance()
                base, quote = self.symbol.split("/")
                return float(bal["total"].get(quote, 0.0)) + \
                    float(bal["total"].get(base, 0.0)) * mark_price
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"fetch_balance fallito dopo {retries} tentativi: {last_err}")

    def open_position(self, side: int, notional_usd: float, ref_price: float,
                      ts: float, votes: int) -> Position:
        if self.position.side != FLAT:
            raise ValueError("posizione già aperta")
        qty = notional_usd / ref_price
        fill = self.buy(qty, ref_price) if side == LONG else self.sell(qty, ref_price)
        self.position = Position(side=side, qty=fill.qty, entry_price=fill.price,
                                 entry_ts=ts, entry_votes=votes)
        return self.position

    def close_position(self, ref_price: float, ts: float, reason: str) -> TradeRecord:
        pos = self.position
        if pos.side == FLAT:
            raise ValueError("nessuna posizione da chiudere")
        fill = self.sell(pos.qty, ref_price) if pos.side == LONG else self.buy(pos.qty, ref_price)
        if pos.side == LONG:
            pnl = (fill.price - pos.entry_price) * pos.qty - fill.fee_usd
        else:
            pnl = (pos.entry_price - fill.price) * pos.qty - fill.fee_usd
        rec = TradeRecord(pos.entry_ts, ts, pos.side, pos.qty,
                          pos.entry_price, fill.price, pnl, reason)
        self.trades.append(rec)
        self.position = Position()
        return rec


def append_trade_csv(path: str | Path, rec: TradeRecord) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_open", "ts_close", "side", "qty",
                        "entry_price", "exit_price", "pnl_usd", "reason"])
        w.writerow([f"{rec.ts_open:.0f}", f"{rec.ts_close:.0f}", rec.side,
                    f"{rec.qty:.8f}", f"{rec.entry_price:.2f}",
                    f"{rec.exit_price:.2f}", f"{rec.pnl_usd:.2f}", rec.reason])
