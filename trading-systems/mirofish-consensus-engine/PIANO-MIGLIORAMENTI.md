# Piano di miglioramento — MiroFish Consensus Engine

> Obiettivo: massimizzare la probabilità di avere un edge **netto** positivo
> (dopo i costi) e minimizzare il rischio di rovina. "Profittevole al massimo"
> non è una manopola: è il risultato di (1) più informazione nei segnali,
> (2) meno costi di esecuzione, (3) validazione che impedisce di illudersi.
> Ogni voce ha impatto stimato, sforzo e priorità. Nessuna modifica è stata
> applicata: questo è solo il piano su cui decidere.

---

## Diagnosi onesta del sistema attuale

Il limite non è il meccanismo di voto: è **cosa** vota. Oggi i 31 simulatori
sono modelli diversi ma leggono tutti **lo stesso identico input** (le chiusure
5-min). La diversità dei modelli riduce il rischio di fidarsi del modello
sbagliato, ma non aggiunge informazione: se le chiusure 5-min da sole non
predicono il futuro (e da sole lo predicono pochissimo), 31 modi diversi di
guardarle non cambiano il risultato. In più:

- l'esecuzione è **market/taker**: ~13 bps di costo per lato è un macigno che
  su un orizzonte di 1 ora divora quasi ogni edge realistico;
- l'uscita è **solo a candela chiusa**: tra una candela e l'altra (5 minuti)
  un crash non trova nessuno stop;
- il polling REST arriva **secondi dopo** la chiusura candela: su segnali a
  5 minuti la latenza costa parte dell'edge;
- non esiste ancora una **validazione walk-forward**: senza, qualunque
  parametro "buono" trovato sul passato è probabilmente overfitting.

Le sezioni sotto sono ordinate per impatto atteso sul PnL netto.

---

## A. Più informazione nei segnali (impatto: ALTO)

### A1. Feature di microstruttura e derivati — priorità 1
Aggiungere input che le chiusure non contengono:
- **Order flow**: imbalance del book L2 (bid/ask depth ratio), delta volume
  (CVD), trade aggressivi buy vs sell;
- **Derivati**: funding rate dei perpetual, open interest e sue variazioni,
  basis spot-perp, liquidazioni long/short (cascate = mean reversion violenta);
- **Cross-market**: prezzo su 2-3 exchange (lead-lag), ETH/BTC come proxy di
  risk-on/off.

Come integrarle senza stravolgere l'architettura: ogni simulatore riceve, oltre
alle chiusure, un vettore di feature e **condiziona** drift e campionamento
(es. funding molto positivo → tilt ribassista sul drift; liquidazioni short a
cascata → vol regime alto). Il voto resta identico, il consenso resta 28/31.
- Sforzo: medio-alto (feed dati nuovi + calibrazione). Dipendenze: WebSocket (E1).

### A2. Filtro di regime multi-timeframe — priorità 2
Un layer sopra il consenso: si opera long solo se il regime 1h/4h non è
apertamente ribassista (e viceversa se short abilitato). Regime = trend
(EMA/ADX) + livello di volatilità. Elimina la classe di errori più costosa:
comprare consenso rialzista dentro un downtrend maggiore.
- Sforzo: basso. Impatto tipico: meno trade, PnL medio per trade migliore.

### A3. Un modello appreso come "32° votante" con potere di veto — priorità 3
Gradient boosting (LightGBM) o regressione logistica su feature A1 + tecniche,
addestrato **walk-forward** (mai sul futuro), che stima P(rialzo | stato).
Non sostituisce l'ensemble: aggiunge un veto — se il modello appreso dissente
fortemente dal consenso, il trade non parte.
- Sforzo: alto. Da fare solo DOPO A1 e D1, altrimenti è overfitting industriale.

### A4. Simulatori realmente eterogenei — priorità 4
Oggi le 3 famiglie sono varianti dello stesso ceppo statistico. Aggiungere
famiglie con logica diversa: breakout/range (Donchian), mean-reversion su
VWAP, momentum cross-sezionale, stagionalità oraria (ore US/EU/Asia),
pattern di volume. Più la diversità è genuina, più il consenso 28/31 significa
qualcosa.
- Sforzo: medio.

---

## B. Tagliare i costi di esecuzione (impatto: ALTO, certo al 100%)

Questa è la sezione con il guadagno più certo: ogni bps di costo risparmiato è
un bps di PnL, a condizione che fill rate e gestione dei non-fill restino
sotto controllo (un limit che non viene eseguito ha un costo-opportunità).

### B1. Esecuzione maker invece che taker — priorità 1
Ordini **limit post-only** sul best bid/ask invece di market:
- taker: ~10 bps di fee + 3 di slippage per lato → ~26 bps round-trip
- maker: 0-2 bps (su molti exchange i maker pagano ~0, su alcuni rebate)

Serve gestione del "non fill": se il limit non viene eseguito entro N secondi,
chase del prezzo o fallback a market solo per le uscite d'emergenza (stop e
kill switch restano market: lì conta la certezza, non il costo).
- Sforzo: medio. Impatto: recupera ~20+ bps a round-trip — spesso è la
  differenza tra strategia perdente e vincente a questo timeframe.

### B2. Ridurre il churn attorno alla soglia — priorità 2
Con entry 28 / exit 26 il sistema può aprire-chiudere-riaprire in poche candele
pagando costi ogni volta. Rimedi: tempo minimo in posizione (es. 3 candele),
isteresi più larga (28/24 — da validare, non decidere a tavolino), e cooldown
di rientro sullo stesso segnale.
- Sforzo: basso.

### B3. Tier commissionali — priorità 3
Scelta dell'exchange per fee structure, sconto token exchange (es. BNB),
programmi maker. Banale ma sono bps veri.
- Sforzo: nullo (operativo, non di codice).

---

## C. Gestione della posizione (impatto: MEDIO-ALTO)

### C1. Stop-loss server-side sempre attivo — priorità 1
Oggi tra una candela e l'altra la posizione è nuda. Piazzare **sull'exchange**
(non nel bot) uno stop a k×ATR dall'ingresso appena aperta la posizione
(ordine OCO/stop-market). Protegge da crash intra-candela, da crash del bot,
della VPS e della connessione. Questo non aumenta il profitto medio: **taglia
la coda sinistra**, che è ciò che manda a zero i conti.
- Sforzo: basso-medio. Non negoziabile prima di qualunque live.

### C2. Take-profit e trailing per lasciare correre — priorità 2
Uscita attuale = solo caduta del consenso. Aggiungere: trailing stop a k×ATR
che si stringe col profitto, e/o take-profit parziale (es. 50% a +1R, resto
col trailing). Da validare in D1: sui trend il trailing di solito paga, sul
chop paga l'uscita a consenso.
- Sforzo: basso.

### C3. Sizing continuo invece di on/off — priorità 3
Il consenso è un'informazione ricca (28 vs 31 non sono uguali) ma oggi produce
una decisione binaria. Scala la taglia col consenso: es. 28 voti → 60% della
taglia Kelly, 31 voti → 100%; e riduci (invece di chiudere tutto) quando i voti
calano ma restano sopra l'uscita.
- Sforzo: medio.

### C4. Futures: short + funding (opzionale) — priorità 4
Su spot il sistema è long/flat: metà delle opportunità non esiste. Perpetual
futures abilitano lo short e incassare funding quando è a favore. Contro:
leva = rischio di liquidazione, va cappata a 1-2x massimo con margine isolato.
- Sforzo: medio. Solo dopo che il long/flat è validato profittevole.

---

## D. Validazione anti-illusione (impatto: ALTO — evita di perdere)

Metà della "profittabilità" è non bruciare capitale su una strategia che
sembrava buona solo per caso.

### D1. Walk-forward con purging — priorità 1, PRIMA di ogni altra modifica
Ogni parametro (soglie voti, orizzonte, Kelly fraction, isteresi B2, stop C1…)
va scelto così: ottimizza su finestra 1, testa out-of-sample su finestra 2,
scorri; mai valutare sul periodo usato per scegliere. Con embargo tra train e
test per il leakage. Questo è il guard-rail per TUTTE le voci di questo piano:
una modifica entra solo se migliora l'out-of-sample.
- Sforzo: medio. È il moltiplicatore di tutto il resto.

### D2. Backtest su anni e regimi, non giorni — priorità 1
Minimo 3-4 anni di 5-min (bull 2021, bear 2022, chop 2023-24, 2025): la
strategia deve sopravvivere a tutti i regimi o avere il filtro A2 che la
spegne in quelli ostili. Aggiungere stress: costi raddoppiati, fill peggiori,
dati mancanti.
- Sforzo: basso (pipeline dati) una volta che D1 esiste.

### D3. Statistica onesta — priorità 2
Deflated Sharpe ratio (corregge per quante configurazioni hai provato),
Monte Carlo sull'ordine dei trade per la distribuzione dei drawdown attesi,
stima di capacità (a che taglia lo slippage mangia l'edge).
- Sforzo: basso-medio.

### D4. Paper con tracking error — priorità 2
Il paper trading non serve "a vedere se guadagna": serve a misurare quanto il
live diverge dal backtest (fill, latenza, dati). Loggare per ogni trade il
delta tra prezzo teorico e reale; se il tracking error supera l'edge stimato,
il live è matematicamente inutile.
- Sforzo: basso.

---

## E. Infrastruttura di produzione (impatto: MEDIO, abilita A e B)

### E1. WebSocket al posto del polling REST — priorità 1
Candele e book in push, decisione pronta al millisecondo della chiusura invece
che secondi dopo. Prerequisito per A1 (book) e B1 (maker).
- Sforzo: medio.

### E2. Stato persistente e riconciliazione — priorità 2
Al riavvio il bot deve: rileggere la posizione REALE dall'exchange (non fidarsi
del proprio stato), riconciliare, riattaccare gli stop. Ordini idempotenti con
clientOrderId. Senza questo, un crash del processo in posizione è roulette.
- Sforzo: medio. Non negoziabile prima del live.

### E3. Watchdog e alerting — priorità 3
Heartbeat, alert Telegram/mail su: kill switch, stop colpito, feed morto,
riavvio, drawdown oltre soglia. VPS vicina all'exchange (AWS Tokyo per
Binance). Deploy con systemd/docker e riavvio automatico.
- Sforzo: basso-medio.

### E4. Trade database e dashboard di analisi — priorità 4
SQLite/Postgres dei trade con snapshot delle feature al momento della
decisione: è il dataset che alimenta A3 e le autopsie delle perdite.
- Sforzo: basso.

---

## F. Cosa NON fare (perché distrugge la profittabilità)

1. **Ottimizzare i parametri finché il backtest è bellissimo** — è il modo più
   rapido per perdere soldi con fiducia. Ogni parametro in più è un grado di
   libertà per l'overfitting. (Antidoto: D1, D3.)
2. **Aumentare la leva per "massimizzare"** — Kelly pieno o leva > 2x
   trasformano una strategia con edge in una macchina di rovina per varianza.
3. **Scendere sotto i 5 minuti** — sotto il minuto si compete con market maker
   con colocation e fee negative: guerra persa in partenza per un retail.
4. **Aggiungere asset/segnali a caso per "diversificare"** — ogni aggiunta va
   validata da sola in D1, o è solo rumore in più.
5. **Fidarsi di un mese buono di paper** — con ~20 trade/mese la varianza
   domina: servono centinaia di trade per distinguere edge da fortuna.

---

## Roadmap proposta (ordine di esecuzione)

| Fase | Voci | Perché in quest'ordine |
|------|------|------------------------|
| 1 ✅ | D1 + D2 (walk-forward, anni di dati) — **fatto**: `engine/walkforward.py`, `run_walkforward.py`, `run_download.py` (+ watchdog feed in `run_live.py`) | senza guard-rail ogni altra modifica è cieca |
| 2 | B1 + B2 (maker, anti-churn) + E1 (websocket) | guadagno certo sui costi, abilita il resto |
| 3 | C1 (stop server-side) + E2 (riconciliazione) | sicurezza: obbligatorie prima di ogni live |
| 4 | A2 (regime filter) + C2 (trailing) | miglior rapporto sforzo/impatto sui segnali |
| 5 | A1 (microstruttura/derivati) | il vero salto di qualità informativo |
| 6 | C3 (sizing continuo) + A4 (famiglie nuove) | raffinamenti sul sistema ormai validato |
| 7 | A3 (modello appreso) + C4 (futures) + D3 | solo a valle di tutto, con dati E4 accumulati |

Stima realistica: fasi 1-3 sono ~2-3 settimane di lavoro; è lì che si decide
se il sistema ha un edge. Se dopo la fase 4 l'out-of-sample netto è ancora
negativo, la risposta onesta è cambiare orizzonte (es. 1h-4h, dove i costi
pesano 10 volte meno) piuttosto che continuare a torturare i 5 minuti.

---

*Nessuna di queste modifiche garantisce profitti: alzano la probabilità di
avere un edge netto e abbassano quella di rovina. Qualsiasi numero andrà
dimostrato out-of-sample, non promesso.*
