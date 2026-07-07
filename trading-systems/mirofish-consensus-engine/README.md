# MiroFish Consensus Engine

Sistema di trading algoritmico su candele **BTC 5 minuti** basato su **consenso
d'ensemble**: 31 simulatori Monte Carlo indipendenti votano la direzione attesa
del prezzo; il trade parte **solo quando almeno 28 su 31 sono d'accordo** e
viene chiuso all'istante appena i voti a favore scendono **sotto 26**. La
taglia della posizione segue il **criterio di Kelly frazionario**, dentro un
risk management con limiti giornalieri, kill switch su drawdown e cooldown.

Include: motore decisionale, backtester event-driven, paper trading, modalità
live via ccxt (protetta da doppio consenso esplicito), dashboard web in tempo
reale e suite di test (21 test).

---

## ⚠️ Leggi prima di tutto — onestà sul rischio

Questo progetto replica l'**architettura** descritta in un post virale. Vale la
pena dirlo chiaramente:

- I numeri mostrati in quel post ($946K, Sharpe 4.21, 71% win rate) **non sono
  verificabili** e con ogni probabilità la dashboard era un mockup. Nessuno
  pubblica gratis una strategia che stampa denaro.
- **Nessun software garantisce profitti.** Il consenso di 31 modelli riduce il
  rischio di dipendere da un singolo modello sbagliato, ma i 31 modelli
  guardano tutti gli stessi dati: il consenso NON crea edge dal nulla.
- Sul BTC a 5 minuti i costi (fees + slippage) divorano la maggior parte degli
  edge apparenti. Per questo il motore **rifiuta di votare** una direzione se
  l'edge atteso non supera i costi di andata e ritorno.
- Il default è **paper trading**: zero ordini reali. La modalità live richiede
  di modificare la config **e** di impostare una variabile d'ambiente di
  consenso esplicito. Usa solo capitale che puoi permetterti di perdere
  interamente.
- Il trading in criptovalute può avere implicazioni fiscali e normative nel tuo
  paese: informati prima.

**Percorso corretto prima di un solo euro reale:**
`backtest` → `paper trading per ≥ 4 settimane` → `testnet` → `live con taglia minima`.

---

## Architettura

```
candele 5-min (exchange) 
        │
        ▼
┌─────────────────────┐   31 simulatori diversi tra loro:
│  MiroFish Ensemble  │   • block bootstrap (autocorrelazione breve)
│  31 simulatori ×    │   • GBM con innovazioni Student-t (code grasse)
│  256 percorsi ×     │   • EWMA + campionamento condizionato al regime di vol
│  12 candele (1h)    │   con finestre, code e tilt momentum/mean-rev diversi
└─────────┬───────────┘
          │  voti UP / DOWN / NEUTRAL (soglia ≥ costi round-trip)
          ▼
┌─────────────────────┐    FLAT:   apre solo se consenso ≥ 28/31
│  Regola di consenso │    APERTO: chiude appena voti a favore < 26
└─────────┬───────────┘
          ▼
┌─────────────────────┐    f = 0.25 × Kelly, cap 20% equity,
│  Kelly frazionario  │    p e b stimati dalla distribuzione dell'ensemble
└─────────┬───────────┘
          ▼
┌─────────────────────┐    limiti: -3% giorno, -10% drawdown (kill switch),
│  Risk Manager       │    5 perdite di fila → cooldown 2h, max 20 trade/g,
└─────────┬───────────┘    dati stantii → non si opera, short off su spot
          ▼
┌─────────────────────┐    PaperBroker (default) | LiveBroker (ccxt)
│  Esecuzione         │    stessa pipeline in backtest, paper e live
└─────────────────────┘
```

Il punto architetturale importante: **backtest, paper e live condividono lo
stesso codice decisionale** (`engine/orchestrator.py`) — niente doppie
implementazioni della strategia. Restano però diversi data feed, latenza ed
esecuzione reale: è esattamente il divario che il paper trading serve a
misurare prima del live.

## Struttura del progetto

```
mirofish-consensus-engine/
├── config.yaml            # tutta la configurazione (default: paper)
├── run_live.py            # runner paper/live + dashboard
├── run_backtest.py        # backtest su storico scaricato o CSV
├── run_download.py        # download incrementale storico pluriennale (D2)
├── run_walkforward.py     # walk-forward validation con purging (D1)
├── engine/
│   ├── config.py          # caricamento e validazione config
│   ├── data_feed.py       # candele 5m via ccxt o REST pubblico Binance
│   ├── mirofish.py        # ensemble 31 simulatori Monte Carlo
│   ├── kelly.py           # sizing Kelly frazionario
│   ├── risk.py            # limiti, kill switch, cooldown
│   ├── broker.py          # PaperBroker / LiveBroker (ccxt)
│   ├── orchestrator.py    # ciclo decisionale (identico ovunque)
│   ├── backtester.py      # replay event-driven
│   └── walkforward.py     # selezione in-sample / verifica out-of-sample
├── dashboard/index.html   # dashboard web (stile del post originale)
└── tests/test_engine.py   # 21 test
```

## Installazione

```bash
cd trading-systems/mirofish-consensus-engine
pip install -r requirements.txt
```

## Uso

### 1. Backtest

```bash
python3 run_backtest.py --days 30        # scarica 30 giorni di BTC/USDT 5m
python3 run_backtest.py --csv dati.csv   # oppure da CSV: ts_ms,o,h,l,c,v
```

Output: rendimento, max drawdown, win rate, Sharpe annualizzato, numero trade.

### 1b. Walk-forward validation (il guard-rail — Fase 1 del piano)

```bash
python3 run_download.py --days 1460            # scarica ~4 anni (incrementale)
python3 run_walkforward.py --csv data/BTCUSDT_5m.csv --grid small --fast
python3 run_walkforward.py --csv data/BTCUSDT_5m.csv --grid medium \
        --stress-costs 2.0                     # robustezza a costi doppi
```

Su ogni finestra TRAIN sceglie i parametri migliori da una griglia piccola e
li valuta SOLO sulla finestra TEST successiva (con embargo pari all'orizzonte
di previsione, per il purging del leakage). Il report dà: rendimento OOS
composto, Sharpe OOS, gap overfitting (IS − OOS), stabilità dei parametri e
un verdetto esplicito. **Regola della casa: qualunque modifica alla strategia
entra solo se migliora l'out-of-sample qui**, non il backtest semplice.

### 2. Paper trading + dashboard

```bash
python3 run_live.py --dashboard
# apri http://localhost:8787
```

Il motore scarica 24h di candele, poi a ogni candela 5-min chiusa fa girare i
31 simulatori e decide. La dashboard mostra equity, PnL, il grafo dei 31 nodi
votanti (verde=bull, rosso=bear), le probabilità dell'ensemble e il "pulse" del
prezzo. Aprendo `dashboard/index.html` senza motore parte una **modalità demo**
con dati simulati, chiaramente etichettata.

### 3. Testnet e live (solo dopo settimane di validazione)

```bash
# 1. API key SOLO-trade (mai withdrawal), con restrizione IP
export MIROFISH_API_KEY=...
export MIROFISH_API_SECRET=...

# 2. config.yaml:  mode: live   e  testnet: true  (prima il testnet!)

# 3. consenso esplicito ai rischi
export MIROFISH_I_UNDERSTAND_THE_RISKS=YES

python3 run_live.py --dashboard
```

Senza il punto 3 il motore **si rifiuta di partire** in live. Passa a
`testnet: false` solo come ultimo passo, con la taglia minima.

## Test

```bash
python3 -m pytest tests/ -v
```

Coprono: diversità dei 31 simulatori, consenso su trend forti/assenza di
consenso su mercati piatti, soglia voti ≥ costi, Kelly (edge positivo/nullo,
cap, notional minimo), kill switch, limite giornaliero, cooldown, blocco short
su spot, dati stantii, costi del PaperBroker, ciclo completo apri/chiudi
dell'orchestratore, validazioni di config e backtest end-to-end.

## Parametri principali (`config.yaml`)

| Parametro | Default | Significato |
|---|---|---|
| `ensemble.n_simulators` | 31 | simulatori paralleli |
| `ensemble.entry_votes` | 28 | voti concordi minimi per aprire |
| `ensemble.exit_votes` | 26 | sotto questa soglia la posizione si chiude |
| `ensemble.horizon_candles` | 12 | orizzonte di previsione (1h) |
| `sizing.kelly_fraction` | 0.25 | frazione di Kelly (mai 1.0) |
| `sizing.max_position_pct` | 0.20 | cap sulla singola posizione |
| `risk.max_daily_loss_pct` | 0.03 | stop giornaliero |
| `risk.max_drawdown_pct` | 0.10 | kill switch permanente |
| `costs.fee_bps` / `slippage_bps` | 10 / 3 | costi per lato usati ovunque |

## Licenza e responsabilità

Software fornito "così com'è", senza alcuna garanzia. L'uso con capitale reale
è a esclusivo rischio dell'utente. Questo non è un consiglio finanziario.
