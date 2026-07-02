"""MiroFish Consensus Engine.

Sistema di trading a consenso d'ensemble su candele BTC 5 minuti:
31 simulatori Monte Carlo indipendenti votano la direzione attesa;
il trade parte solo con >= 28/31 voti concordi e viene chiuso appena
il consenso scende sotto 26. Position sizing con Kelly frazionario,
risk management con limiti giornalieri, kill switch su drawdown e cooldown.
"""

__version__ = "1.0.0"
