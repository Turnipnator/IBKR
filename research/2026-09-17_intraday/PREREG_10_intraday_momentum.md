# Pre-registration — attempt 10: intraday momentum on US shares

Registered 2026-09-17, before any of this strategy's returns were computed.

## 1. Identity

- **Name:** Market intraday momentum (first half-hour predicts last half-hour), long-only
- **Date registered:** 2026-09-17
- **Attempt number:** 10 — threshold 5% ÷ 10 = **0.5%** (at most 5 of 1,000 random runs at least as good)
- **Registered by:** Claude, at the account owner's request
- **Why this idea:** the owner wants the bot trading actively for small, frequent profits rather than holding
  for years. Today's measurement makes that arithmetically possible for the first time: IBKR charges **$1.00**
  per order on US shares against **$4.00** on the UCITS ETFs the bot trades now, so a round trip costs about
  £1.50 instead of £6, and a £1,500 position breaks even on a 0.10% move instead of 0.40%.

## 2. The idea and why it should work

- **Hypothesis:** on days when the first half-hour of the US session is up, the last half-hour tends to be up
  too. Holding only that last half-hour on those days beats random timing of the same number of sessions, and
  beats holding the last half-hour every day regardless of the signal.
- **Who is on the other side:** Gao, Han, Li & Zhou (2018) attribute it to late-day flows that follow the
  morning: index funds and leveraged-ETF rebalancing, plus investors who react to the morning move only when
  they get to their desks. The other side is whoever has to trade into the close for reasons other than price.
- **Why it hasn't been competed away:** the window is 30 minutes a day, the effect is small per trade, and it
  cannot be scaled by a large fund without moving the close — but it is small enough to survive at £1,500 a trade.
- **Published evidence:** Gao, Han, Li & Zhou, "Market intraday momentum", *Journal of Financial Economics*
  (2018), on S&P 500 futures/ETF data from 1993. **Disclosure:** the paper's rule goes both long and short; a
  cash account cannot short, so this card tests the long half only (see §3). Published since 2018, so some of
  the test window post-dates publication.
- **Claude's prior knowledge (disclosed):** general familiarity with the paper's claim. No backtest of this
  rule on this data has been run or viewed; bars were fetched and checked only for coverage.

## 3. Exact rules

- **Instruments.** Primary, and the only one this account could trade: **AAPL**, chosen in advance as the most
  liquid US mega-cap. Reported but not part of the pass mark: **SPY** (the published version's instrument, which
  UK retail cannot buy) and nine other mega-caps (MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, JPM, XOM) for breadth.
- **Data:** IBKR 30-minute TRADES bars, regular trading hours only (13 bars a session, 09:30–16:00 New York).
- **Signal, each session:** r_first = the return of the 09:30–10:00 bar (its close ÷ its open − 1).
- **Entry:** if r_first > 0, buy at the **open of the 15:30 bar**; otherwise do not trade that day.
- **Exit:** sell at the **close of the 16:00 bar** (the session close), always. No position is ever held overnight.
- **Deviation from the paper, declared:** the published rule also shorts when r_first < 0. A cash account cannot
  short, so those days are spent in cash. This halves the opportunities and is not a tuning choice.
- **Sizing:** £1,500 of notional per trade, whole shares, no leverage, one position at a time. Idle cash earns 0%.
- **Settlement:** a same-session round trip uses settled cash and is permitted in a cash account; proceeds settle
  the next day, so at most one round trip per day. The simulation enforces one trade per session.
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Signal window | first 30 minutes | Gao et al. 2018 |
  | Trading window | last 30 minutes | Gao et al. 2018 |
  | Direction | long only | account constraint (no shorting in a cash account) |
  | Instrument | AAPL | chosen before any data: most liquid US mega-cap this account can trade |
  | Position size | £1,500 | what the account can fund once cash frees up |

## 4. Costs and execution

| Item | Value used | Stress value |
|---|---|---|
| Commission | $1.00 per order, measured on this account 2026-09-17 (what-if orders) | same |
| Slippage, per side | 3 bps | 10 bps |
| Fill prices | bar open on entry, bar close on exit, moved against by the slippage | |
| Currency | USD returns; commission converted at 0.75 GBP/USD for the cost ratio | |
| Idle cash | 0% | |

## 5. Test design

- **Tuning window:** none.
- **Test window:** every session for which all 13 regular-hours bars exist, from the first such session in the
  fetched history (about two years, 30-minute bars being IBKR's practical limit at this granularity) through
  the last complete session, 2026-09-16.
- **Sub-periods:** the sessions split into 4 equal blocks by count.
- **Overlap with data already seen:** none — no intraday data has been used in this project before.
- **Random-timing comparison:** the daily trade/no-trade sequence is circularly shifted by a random offset of
  20 to n−20 sessions, keeping the number of trading days, run lengths and costs identical while breaking the
  link to the morning move. 1,000 runs, seeds 0–999, at both slippage levels.
- **Pass-mark metric:** Sharpe ratio of daily returns (0% risk-free, the account's reality), annualised with √252.
- **Benchmarks:**
  1. **Always-long-the-close:** hold the last half-hour every session regardless of the signal, same costs. This
     is the benchmark that matters — it isolates the signal from simply being in the market at that time.
  2. Reported, not a box: buy-and-hold AAPL over the same window.
- **Neighbourhood check:** signal thresholds of r_first > +0.05% and r_first > −0.05%.
- **Code:** `research/2026-09-17_intraday/intraday_study.py`; commit hash recorded in §10.

## 6. Pass mark (every box must hold)

- [ ] **1 Beats random timing:** at most 0.5% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **2 Beats always-long-the-close:** higher Sharpe ratio **and** a higher total return after costs
- [ ] **3 Enough trades:** at least 100 completed round trips in the window
- [ ] **4 Sub-periods:** positive after costs in at least 3 of 4
- [ ] **5 Stress slippage:** still ≤ 0.5% of random runs at 10 bps a side
- [ ] **6 Neighbours:** both threshold variants above the random-timing median
- [ ] **7 Fidelity and data:** two independent signal implementations agree on every session; every used session
      has all 13 regular-hours bars; and the average measured gap between the 15:00 bar close and the 15:30 bar
      open is under 5 bps (a sanity check that entry prices are real)
- [ ] **8 Breadth (declared addition):** in at least 6 of the 10 mega-caps, the rule's Sharpe beats that name's
      own random-timing median

## 7. Decisions, written now

- **Fails any box:** record in `research_notes.md` and the attempts ledger. No re-tuning on this data; a changed
  rule is a new card.
- **Passes every box:** it still does not go live automatically. It would need a forward-test card of its own,
  because live trading this rule requires changes the bot does not have today (§8).
- **Two years of sessions is a short sample.** Even a pass is weak evidence; it says "worth a forward test", not
  "this makes money".

## 8. What live trading would require (not part of this test)

- The bot runs during LSE hours only; this rule trades at **20:30–21:00 UK time**, so its schedule would have to
  extend into the US session.
- Live US quotes need a market-data subscription (a few dollars a month); historical bars alone are not enough.
- One round trip per day is the cash-account limit, and the sleeve and momentum strategies already claim cash.

## 9. Ledgers

Attempt 10 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:** 2026-09-17, code `4834b17`, rules `812e667`; 1,000 random-timing runs at each
  slippage level. Window 2024-09-17 → 2026-09-16, **496 complete sessions** (5 dropped for missing bars).
- **Headline (AAPL):** Sharpe −3.85, total return **−26.9%**, worst fall −27.1%, 278 round trips, 33% winners,
  **−0.112% net per trade**. Always-long-the-close: −47.0%. SPY (the published instrument, not tradeable here):
  −32.6%. AAPL buy-and-hold over the same window: +54.1%.
- **Why it loses, in one line:** the average last-half-hour move on signal days was **+0.046%** (4.6 bps) while a
  round trip costs **16 bps** — 10 bps commission ($2 on a $2,000 position) plus 6 bps slippage. Gross 4.6 minus
  16 is −11.4 bps, which is the measured −0.112% per trade. **Even with zero commission the 6 bps of slippage
  alone exceeds the move.**
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random timing | ≤ 0.5% of runs at least as good | 23.1% | ✗ |
  | 2 Beats always-long-the-close | higher Sharpe and total return | −3.85 vs −5.81; −26.9% vs −47.0% | ✓ |
  | 3 Enough trades | ≥ 100 | 278 | ✓ |
  | 4 Sub-periods | ≥ 3 of 4 positive | 0 of 4 (−7.6%, −1.8%, −11.1%, −9.4%) | ✗ |
  | 5 Stress slippage | ≤ 0.5% at 10 bps | 30.2% | ✗ |
  | 6 Neighbours | above the random median (−4.27) | +0.05%: −3.70; −0.05%: −4.12 | ✓ |
  | 7 Fidelity and data | agreement, complete sessions, real entry prices | 0 disagreements in 496 sessions; entry gap 0.4 bps | ✓ |
  | 8 Breadth | ≥ 6 of 10 names above their own random median | 8 of 10 | ✓ |

- **Decision:** **Fail** (boxes 1, 4 and 5). Not re-tuned on this data.
- **What the passing boxes mean, and don't.** Boxes 2 and 8 say the morning move does carry a little information:
  trading only on up-mornings lost less than trading every day, and 8 of 10 names beat their own random timing.
  But box 1 says that edge is not distinguishable from chance (23% of random timings did as well), and it is far
  too small to pay for the trading. The signal isn't worthless; it's worth about 5 bps, against a 16 bps toll.
- **Caveats:** two years is a short sample; the whole window post-dates the 2018 paper, which is consistent with
  the effect being competed away; and a cash account can only trade the long half of the published rule.
