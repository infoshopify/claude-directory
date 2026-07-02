"""Position sizing con criterio di Kelly frazionario.

Kelly pieno: f* = p - (1-p)/b, con p = probabilità di vincita e b = rapporto
vincita/perdita. Nessun desk serio usa Kelly pieno: la stima di p e b è
rumorosa e Kelly pieno è violentemente sovra-ottimista sui parametri stimati.
Qui si usa una frazione (default 0.25) con cap duro sulla taglia massima.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizingDecision:
    fraction: float        # frazione del capitale da impegnare (0 se no-trade)
    notional_usd: float    # controvalore in USD
    kelly_full: float      # Kelly pieno stimato (diagnostica)
    reason: str


def kelly_size(
    equity_usd: float,
    p_win: float,
    win_loss_ratio: float,
    kelly_fraction: float,
    max_position_pct: float,
    min_notional_usd: float,
) -> SizingDecision:
    """Calcola la taglia della posizione. Ritorna fraction=0 se il trade non ha edge."""
    if equity_usd <= 0:
        return SizingDecision(0.0, 0.0, 0.0, "equity nulla o negativa")
    if p_win <= 0.0 or win_loss_ratio <= 0:
        return SizingDecision(0.0, 0.0, 0.0, "parametri fuori dominio")
    # Una p stimata al 100% è un artefatto del campionamento finito, non una
    # certezza: si tronca a 0.99 per non far esplodere il Kelly.
    p_win = min(p_win, 0.99)

    kelly_full = p_win - (1.0 - p_win) / win_loss_ratio
    if kelly_full <= 0:
        return SizingDecision(0.0, 0.0, kelly_full, "edge atteso non positivo")

    fraction = min(kelly_full * kelly_fraction, max_position_pct)
    notional = equity_usd * fraction
    if notional < min_notional_usd:
        return SizingDecision(0.0, 0.0, kelly_full,
                              f"notional {notional:.2f} sotto il minimo {min_notional_usd}")

    return SizingDecision(fraction, notional, kelly_full, "ok")
