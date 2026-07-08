"""Feed dati: candele OHLCV 5-min dall'exchange (REST pubblico, nessuna API key).

Usa ccxt se disponibile; in fallback chiama direttamente l'endpoint pubblico
klines di Binance con urllib (nessuna dipendenza). Le candele vengono tenute
in un buffer; solo le candele CHIUSE alimentano il motore decisionale.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

TIMEFRAME_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
                     "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


@dataclass
class Candle:
    ts: float      # apertura candela, secondi epoch UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed:
    def __init__(self, exchange_id: str, symbol: str, timeframe: str,
                 lookback: int):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.tf_seconds = TIMEFRAME_SECONDS[timeframe]
        self.candles: list[Candle] = []
        self._ccxt = None
        try:
            import ccxt
            self._ccxt = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        except Exception:
            if exchange_id != "binance":
                raise RuntimeError(
                    f"ccxt non disponibile e fallback REST supporta solo binance, "
                    f"non {exchange_id}: pip install ccxt"
                )

    # ------------------------------------------------------------------ #
    def fetch_ohlcv(self, limit: int) -> list[Candle]:
        if self._ccxt is not None:
            rows = self._ccxt.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        else:
            rows = self._binance_rest(limit)
        return [Candle(r[0] / 1000.0, float(r[1]), float(r[2]),
                       float(r[3]), float(r[4]), float(r[5])) for r in rows]

    @property
    def has_ccxt(self) -> bool:
        """True se ccxt è disponibile (necessario per il download paginato)."""
        return self._ccxt is not None

    def fetch_ohlcv_since(self, since_ms: int, limit: int = 1000) -> list[list]:
        """Pagina OHLCV grezza a partire da un timestamp (ms). Richiede ccxt.

        API pubblica per il downloader storico: evita di accoppiarlo ai
        dettagli interni del client. Ritorna righe [ts_ms, o, h, l, c, v].
        """
        if self._ccxt is None:
            raise RuntimeError("download paginato richiede ccxt (pip install ccxt)")
        return self._ccxt.fetch_ohlcv(self.symbol, self.timeframe,
                                      since=since_ms, limit=limit)

    def _binance_rest(self, limit: int) -> list[list]:
        symbol = self.symbol.replace("/", "")
        qs = urllib.parse.urlencode({"symbol": symbol, "interval": self.timeframe,
                                     "limit": min(limit, 1000)})
        url = f"https://api.binance.com/api/v3/klines?{qs}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"impossibile raggiungere {url}: {e}. Verifica connessione, "
                "firewall/proxy o eventuali restrizioni geografiche dell'exchange."
            ) from e
        return [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in data]

    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        """Carica lo storico iniziale (solo candele chiuse)."""
        candles = self.fetch_ohlcv(self.lookback + 2)
        self.candles = self._drop_open_candle(candles)

    def poll(self) -> Candle | None:
        """Ritorna la nuova candela chiusa se disponibile, altrimenti None."""
        last_ts = self.candles[-1].ts if self.candles else 0.0
        fresh = self._drop_open_candle(self.fetch_ohlcv(5))
        new = [c for c in fresh if c.ts > last_ts]
        if not new:
            return None
        self.candles.extend(new)
        self.candles = self.candles[-(self.lookback + 10):]
        return new[-1]

    def _drop_open_candle(self, candles: list[Candle]) -> list[Candle]:
        """Scarta la candela corrente non ancora chiusa."""
        now = time.time()
        return [c for c in candles if c.ts + self.tf_seconds <= now]

    # ------------------------------------------------------------------ #
    @property
    def closes(self) -> np.ndarray:
        return np.array([c.close for c in self.candles])

    @property
    def last_price(self) -> float:
        return self.candles[-1].close if self.candles else float("nan")

    @property
    def data_age_seconds(self) -> float:
        if not self.candles:
            return float("inf")
        last_close_time = self.candles[-1].ts + self.tf_seconds
        return max(0.0, time.time() - last_close_time)
