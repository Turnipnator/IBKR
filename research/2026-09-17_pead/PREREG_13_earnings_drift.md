# Pre-registration — attempt 13: post-earnings announcement drift

Registered 2026-09-17, before any of this strategy's returns were computed.

## 1. Identity

- **Name:** Post-earnings announcement drift, long only, US shares
- **Date registered:** 2026-09-17
- **Attempt number:** 13 — threshold 5% ÷ 13 = **0.385%** (at most 3 of 1,000 random runs at least as good)
- **Registered by:** Claude, at the account owner's request
- **Why this idea:** every signal tested on this account so far — trend, reversal, intraday momentum, per-stock
  tuned rules — is built from **past prices only**, and all four failed the same way: no better than buying a
  random name from the same list. This card changes the input, not the arithmetic. The signal here is an
  **event**: a company reporting results, and the market's own immediate reaction to them.

## 2. The idea and why it should work

- **Hypothesis:** shares that jump on the day their results are announced keep drifting in that direction for
  weeks afterwards. Buying them the next morning and holding a month beats buying a random name from the same
  universe on the same days, after costs.
- **Who is on the other side:** investors who react slowly. The classic explanation is that the market
  under-reacts to earnings news — analysts revise estimates gradually, index and retail flows arrive late, and
  attention is limited, so the price walks to its new level over weeks instead of jumping to it in a day.
- **Why it hasn't been competed away:** it partly has (see below). What is left is attributed to limits on
  arbitrage — the drift is strongest in names that are costly or awkward to trade, and it requires holding a
  concentrated single-name position through a period where the next piece of news can arrive at any time.
- **Published evidence:** Ball & Brown (1968) first documented it; Bernard & Thomas (1989, 1990) established the
  60-day drift and the "delayed response" explanation; Chan, Jegadeesh & Lakonishok (1996) showed it survives
  alongside price momentum and works using the **announcement-day return** as the surprise measure, which is the
  version used here. **Disclosure, the honest part:** Chordia, Goyal, Sadka, Sadka & Shivakumar (2009) and others
  show the drift is concentrated in illiquid, low-priced, low-attention shares and has decayed substantially in
  large caps since the 2000s. This account can only trade large liquid names, which is the worst place to look
  for it. A pass here would be a weakened version of a documented effect; a fail is entirely plausible.
- **Claude's prior knowledge (disclosed):** familiarity with the literature above, including the decay finding.
  No backtest of this rule on this data has been run or viewed, and the earnings dates have never been joined to
  the price bars.

## 3. Exact rules

- **Universe (fixed now, the same 40 names as attempt 11):** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO,
  JPM, XOM, UNH, JNJ, V, PG, MA, HD, CVX, MRK, ABBV, PEP, KO, COST, WMT, BAC, CRM, MCD, CSCO, ACN, LIN, ADBE,
  TMO, ABT, DHR, WFC, VZ, TXN, NEE, PM, INTC, IBM. A name is eligible once it has 25 trading days of bars.
- **Survivorship, stated plainly:** this is today's list of large US companies. It flatters absolute returns and
  does **not** distort the pass mark, because the random-name comparison in §5 draws from the identical list.
- **Announcement dates:** every SEC **8-K tagged item 2.02 (Results of Operations and Financial Condition)** for
  that company's CIK, taken from EDGAR's own submissions files with their `acceptanceDateTime`. No estimates, no
  analyst data, no vendor — the filing itself.
- **Reaction day (day 0), fixed now:** if the filing was accepted **before 19:30 UTC** on a trading day, day 0 is
  that day; otherwise day 0 is the next trading day. (US market close is 20:00 or 21:00 UTC; this assigns any
  announcement made close to or after the bell to the following session, which is the conservative direction.)
- **Surprise measure:** the abnormal return on day 0, `AR = (close_0 / close_-1 − 1) − M`, where `M` is the
  equal-weight return of all eligible names in the universe that day. No external market index is needed.
- **Entry:** if `AR ≥ +2.0%`, buy at the **open of day 1** (the session after day 0).
- **Exit:** sell at the **open 20 trading days later**. A position still open at the end of the data is closed at
  the last available open and counted.
- **One position at a time** (cash account). Qualifying events that occur while a position is held are **skipped,
  not queued**. If several names qualify on the same day 0, take the largest `AR`; ties by alphabetical ticker.
- **Sizing:** £2,000 notional, whole shares, no leverage. Idle cash earns 0%.
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Surprise measure | day-0 abnormal return | Chan, Jegadeesh & Lakonishok 1996 |
  | Threshold | +2.0% | roughly the median absolute announcement-day move for large caps; fixed now |
  | Holding period | 20 trading days | Bernard & Thomas: about half the 60-day drift arrives in the first month |
  | Positions | 1 at a time | account size and the cash-account settlement limit |
  | Entry timing | the open after day 0 | day 0's close is the signal; it can never be used to trade |
  | Universe | the 40 names of attempt 11 | fixed before any returns; bars already fetched |

## 4. Costs and execution

| Item | Value used | Stress value |
|---|---|---|
| Commission | $1.00 per order (measured on this account, 2026-09-17) | same |
| Slippage, per side | 5 bps | 15 bps |
| Fill prices | the day's open, moved against by the slippage | |
| Position size | £2,000 notional, whole shares | |
| Idle cash | 0% | |

## 5. Test design

- **Tuning window:** none. Every setting is fixed in §3 above.
- **Test window:** 2006-10-02 → 2026-09-10, the daily bars already fetched for attempt 11.
- **Sub-periods:** the trades split into 4 equal blocks by count.
- **Overlap with data already seen:** the daily bars for these 40 names were used in attempts 11 and 12 (a
  five-day reversal rule and a tuned entry/exit grid). The **earnings dates are new data, never joined to these
  bars before.** The overlap is disclosed and is exactly why the verdict rests on the random-name comparison.
- **The two comparisons, both required:**
  1. **Random name (primary):** on each entry day, buy a random eligible name instead of the announcing one, held
     for the same 20 days, with the same costs and trade count. Cancels survivorship and the market's own return.
     1,000 runs, seeds 0–999.
  2. **Random timing:** the same names shifted by a random offset of 8 to n−8 events, breaking the alignment
     between the name and its announcement. 1,000 runs.
- **Pass-mark metric:** Sharpe ratio of daily returns, 0% risk-free, annualised with √252.
- **Benchmark:** equal-weight buy-and-hold of the same 40 names, rebalanced each January, same costs.
- **Reported, not pass conditions:** the average cumulative abnormal return from day 1 to day 60 for qualifying
  events (the drift curve — this is what the literature actually measures), the same curve for the mirror case
  `AR ≤ −2.0%` which this account cannot trade because it cannot short, and the drift curve split into
  2006–2016 and 2017–2026 to show whether the documented decay is present in this sample.
- **Code:** `research/2026-09-17_pead/`; commit hash recorded in §10.

## 6. Pass mark (every box must hold)

- [ ] **1 Beats random names:** at most 0.385% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **2 Beats random timing:** at most 0.385% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **3 Beats buy-and-hold:** higher Sharpe than the equal-weight 40, **or** a worst fall no more than half of
      it with a CAGR no more than 2 points below
- [ ] **4 Enough trades:** at least 100 completed round trips
- [ ] **5 Sub-periods:** positive after costs in at least 3 of 4
- [ ] **6 Stress slippage:** still ≤ 0.385% against random names at 15 bps a side
- [ ] **7 Neighbours:** all four variants — holding period 10 and 40 days, threshold +1.5% and +3.0% — above the
      random-name median. Declared now so that +2.0% and 20 days cannot be a lucky cell.
- [ ] **8 Fidelity and data quality:** two independent implementations agree on every event and trade; no trade
      uses a price from before day 0's close; **and** the announcement data passes its own checks — between 3 and
      6 item-2.02 filings per name per year (flagging any name that fails), and a hand-checked sample of ten
      dates against the known reporting calendar

## 7. Decisions, written now

- **Fails any box:** record in `research_notes.md` and the attempts ledger. No re-tuning on this data; a changed
  threshold or holding period is a new card with a stricter threshold.
- **Passes every box:** a forward-test card at minimum size, judged on implementation, before any live money.
- **Expected honestly:** the effect is real and heavily documented, but documented as **decaying in exactly the
  large liquid names this account is limited to**. The drift curve in §5 will show whether anything is left in
  this sample even if the tradeable rule fails its boxes — that diagnostic is worth having either way.

## 8. What live trading would require (not part of this test)

- A daily check for new item-2.02 filings after the US close, then a market order at the next open (14:30 UK).
- Live US quotes, and roughly £2,000 of settled cash.
- Holding a single name through a 20-day window, which is a concentrated position by this account's standards.

## 9. Ledgers

Attempt 13 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:** 2026-09-17, rules `9ce0b3a`, code `49c2654`; 1,000 runs for each comparison.
  Window 2006-10-02 → 2026-09-10. **3,738 item-2.02 filings → 3,172 events in window → 967 qualifying
  (AR ≥ +2.0%) → 176 traded** (the rest overlapped a position already held).
- **Headline:** CAGR **+14.6%**, Sharpe **0.65**, total +1,410%, worst fall −40.5%, 62% winners,
  **+1.93% net per trade**. Equal-weight buy-and-hold of the same 40: CAGR +18.5%, Sharpe 0.74, fall −43.8%.
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random names | ≤0.385% of runs as good | **43.70%** (random median 0.62, 5th/95th 0.34/0.87) | ✗ |
  | 2 Beats random timing | ≤0.385% | 38.10% (shifted median 0.59) | ✗ |
  | 3 Beats buy-and-hold | higher Sharpe, or half the fall with CAGR within 2 points | Sharpe 0.65 vs 0.74; fall −40.5% vs −43.8%; CAGR +14.6% vs +18.5% | ✗ |
  | 4 Enough trades | ≥100 | 176 | ✓ |
  | 5 Sub-periods | ≥3 of 4 positive | 4 of 4 (+81.6%, +240.3%, +47.5%, +65.5%) | ✓ |
  | 6 Stress slippage | ≤0.385% at 15 bps | 42.80% (Sharpe falls to 0.58) | ✗ |
  | 7 Neighbours | all four above the random-name median (0.62) | hold 10d **0.83**, hold 40d **0.64**, AR≥1.5% **0.52**, AR≥3.0% **0.79** — one below | ✗ |
  | 8 Fidelity and data | agree, and 3–6 filings per name-year | 0 implementation disagreements; **37 complete name-years outside 3–6** (CVX and ABBV file 7–8 item-2.02 releases a year; on a per-name median reading only TSLA fails) | ✗ |

- **The drift curve (§5) — the effect is real and it is small.** Average cumulative abnormal return after a
  +2% earnings jump, across 950 events: **+0.25% by day 5, +0.58% by day 20, +0.97% by day 60.** The mirror case
  (a −2% jump, which this account cannot trade because it cannot short) drifts **−0.61% by day 20** before
  recovering to −0.06% by day 60. Contrary to the decay literature, in this sample the drift is **larger in the
  recent era**: +1.21% at day 60 for 2017–2026 against +0.69% for 2006–2016.
- **Why a real effect still failed every comparison — the decisive arithmetic.** At day 20 the abnormal return
  averages **+0.598% with a standard deviation of 6.81%** across 964 events (t = 2.73, so the effect itself is
  statistically solid). The edge is one twelfth of the noise on a single trade. Seeing it at t = 2 takes roughly
  **519 events**; holding one position at a time produced 176 trades in twenty years. Each trade's outcome is
  therefore dominated by the ±6.8% of ordinary single-name movement, which is precisely why buying a random name
  on the same days matched it 44% of the time.
- **What the strategy actually earned.** +1.93% per trade against a +0.60% abnormal edge: **the bulk of the
  return was simply twenty days of market exposure**, not the earnings signal. That is also why it lost to
  holding all forty names — the same exposure, continuously, with no toll.
- **Decision:** **Fail** (boxes 1, 2, 3, 6, 7, 8). Not re-tuned on this data.
- **Costs were not the binding constraint again** (+1.93% a trade against an 18.5 bps toll), but for the first
  time cost *is* part of the structural verdict: harvesting a 0.6% edge needs hundreds of simultaneous small
  positions, and at £500 a position a $2 round trip is 0.3% — half the edge — before the account runs out of
  slots and settled cash.
- **Disclosure:** a 25-seed smoke printed a verdict before one fix to the §6 box 8 coverage check landed
  (`49c2654`: the check was counting the partial years 2006 and 2026 as if they were full ones, flagging every
  name). No rule, window, threshold or pass mark on returns was changed in response.
