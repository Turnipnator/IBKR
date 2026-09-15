# Strategy test pre-registration

Copy this file to `research/<YYYY-MM-DD>_<topic>/PREREG.md`, fill in sections 1–9, and
**commit it before running anything that looks at returns**. The commit timestamp is the
proof that the rules came first.

Once committed, sections 1–9 do not change. Changing a rule after seeing any result makes
it a new attempt with a new card. A card filled in after the results are known is not a
pre-registration.

A worked example is at the bottom.

---

## 1. Identity

- **Name:**
- **Date registered:**
- **Attempt number:** (every idea and every variant counts; see the ledger in §9)
- **Registered by:**

## 2. The idea and why it should work

- **Hypothesis, in one sentence:**
- **Who is on the other side of these trades, and why do they keep losing to it?**
  (a behavioural bias, a risk the market pays you to carry, a constraint that stops others
  trading it). "The indicator crossed a level" is not an answer.
- **Why hasn't it been competed away at this account size and cost?**
- **Published evidence, if any:** source, markets, period, and whether results were net of costs.

## 3. Exact rules

Precise enough that two people coding it separately would produce the same trades.

- **Universe:** exact tickers, and how names join or leave.
  (UK retail: UCITS lines on LSE only. US-listed ETFs are PRIIPs-blocked for trading.)
- **Data:** source, bar size, adjusted or unadjusted, and which bars the rule can see at
  decision time. The live bot decides at 14:00 London on a part-day bar.
- **Signal:** formula, lookbacks, thresholds.
- **Entry:** when, order type, what is bought.
- **Exit:** stop type and distance, profit targets, time exits, signal exits.
- **Sizing:** per-position rule, caps, maximum positions, cash kept back.
- **Timing:** rebalance day and time (Europe/London), and what happens on holidays.
- **Where every setting came from:**

  | Setting | Value | Source (paper / chosen before any data / tuned on period X) |
  |---|---|---|
  | | | |

  Anything tuned must name the period it was tuned on, and that period is excluded from
  the test window in §5.

## 4. Costs and execution for this account

| Item | Value used | Stress value |
|---|---|---|
| Commission | max($4, 0.05% of notional) per order | |
| Spread + slippage, per side | 5 bps | 15 bps |
| Fill price | | |
| Settlement | T+2; buys limited to settled cash with a 6% buffer | |
| Share size | whole shares only (IBKR API blocks fractional orders) | |
| Starting capital | | |
| Pence-quoted lines | EQQQ and IJPN are priced in GBX | |
| Currency | GBP base; USD lines converted at the daily GBPUSD rate | |

## 5. Test design

- **Tuning window (if any):** from ___ to ___
- **Test window:** from ___ to ___ (run once)
- **Overlap with data already seen:** check the ledger in §9 and list every overlap. A
  period an earlier study used for a similar idea is not out of sample.
- **Random-pick comparison:** same timing, number of positions, stops, sizing and costs,
  with the names chosen at random. At least 500 runs, seeds fixed here: ___
- **Passive benchmark:** (e.g. 60/40 equities/bonds), including its own costs.
- **Sub-periods:** list them now (e.g. four equal blocks).
- **Neighbourhood check:** each setting moved one step either way. List the steps now.
- **Code location:** path, plus the commit hash of the code that produces the result (in §10).

## 6. Pass mark (every box must hold)

- [ ] **Beats random picks at live costs:** the share of random runs that do at least as
      well is no more than **5% ÷ attempt number** (attempt 4 needs 1.25% or less)
- [ ] **Beats the passive benchmark** on return per unit of risk (Sharpe), **or** meets this
      goal stated now: ___ (e.g. worst fall under half the benchmark's, with annual return
      within 2 points of it)
- [ ] **At least 100 completed trades** in the test window
- [ ] **Positive after costs in at least 3 of 4 sub-periods**
- [ ] **Still beats random picks at the stress slippage**
- [ ] **Every neighbouring setting stays above the random-pick median** (not knife-edge)
- [ ] **Replay fidelity checked:** the test's signals match the live engine's own code on
      sampled days

A note on scale: small edges take decades of data to show. A Sharpe ratio of 0.5 needs about
16 years to reach t ≈ 2. A test on a few years can reject an idea; it can only confirm a
large edge.

## 7. Decisions, written now

- **Fails any box:** write the result in `research_notes.md` and add it to the attempts
  ledger. No re-tuning on the same data. A changed rule is a new card with the next attempt
  number.
- **Passes every box:** forward test (§8). No live capital beyond minimum size until §8 passes.

## 8. Forward test

- **Size:** ___ (the smallest size that keeps commission a modest share of risk per trade)
- **Duration, fixed now:** ___ months or ___ trades, whichever comes later
- **Judged on whether live trading matches the test, not on profit:**
  - [ ] live signals match the replay on at least 90% of days
  - [ ] average slippage within the §4 stress value
  - [ ] commissions within 20% of what was modelled
- **Stop early if:** the drawdown is worse than 95% of the backtest's drawdowns, or any §4
  assumption proves wrong (e.g. fills routinely worse than the stress value).
- **Scaling steps, written now:** ___ (increase only while live results stay inside the
  backtest's range)

## 9. Ledgers (update before registering a new card)

### Attempts

| # | Date | Card | Result |
|---|---|---|---|
| 1 | 2026-09-15 | Live configuration, out of sample 2008–2023 (retro-filled, not pre-registered) | Fail |
| 2 | 2026-09-15 | Monthly trend timing, 5 asset classes (`2026-09-15_monthly_allocation/PREREG_2_gtaa.md`) | Fail (boxes 1, 2, 3, 5) |
| 3 | 2026-09-15 | Dual momentum (`2026-09-15_monthly_allocation/PREREG_3_dual_momentum.md`) | Fail (boxes 1, 2, 5) |
| 4 | 2026-09-15 | Volatility-targeted 60/40 (`2026-09-15_monthly_allocation/PREREG_4_vol_target_6040.md`) | Fail (boxes 1, 5) |

### Data already seen

| Period | Universe | Strategy family | Where |
|---|---|---|---|
| 2026-05-22 onward | live UCITS book | TSMOM + CSMOM, 3–8 slots, ATR trailing stops | live trade record |
| 2025-06 → 2026-08 | 23 UCITS lines | TSMOM + CSMOM, stop multiples 1–8×ATR, 3 vs 5 slots | 2026-08-24 study |
| 2023-09 → 2026-08 | 23 UCITS lines | TSMOM + CSMOM, class caps 40–100%, 3–5 slots | `2026-08-28_classcap/` |
| 2008-01 → 2026-09 | 23 US-ETF proxies of the UCITS lines | TSMOM + CSMOM, thresholds 0.3–0.7, 2–4×ATR, 3–5 slots, random-pick nulls | `2026-09-15_oos/` |
| 2008-05 → 2026-08 | SPY, EFA, IEF, VNQ, GSG, VEU, AGG, BIL | monthly trend timing (9–11 month averages), dual momentum (11–13 month lookbacks), volatility-targeted 60/40 (8–12%, 10–42 days), static 60/40, random-timing nulls | `2026-09-15_monthly_allocation/` |

IBKR serves at most 20 years of daily bars, so these proxies have no history before 2006.
For momentum or trend rules on this universe, every available period has been looked at.
The honest options are settings taken unchanged from published work and tested against
random picks over the full period, a different set of markets, or a forward test only.

## 10. Results (the only section written after the run)

- **Run date and code commit:**
- **Figure for each §6 box:**
- **Decision:**

---

## Worked example: the 2026-09-15 test as a card

Retro-filled to show the level of detail expected. This test was **not** pre-registered.

- **Hypothesis:** the live configuration beats random picks after costs on years it was not
  tuned on.
- **Who loses to it:** investors who under-react to trends lasting months (the argument in
  Moskowitz, Ooi & Pedersen, 2012).
- **Rules:** TSMOM on 21/63/252-day returns blended with a cross-sectional rank; buy the top 3
  names scoring at least 0.5 with volatility at least 8%; 30% per name, 60% per asset class;
  3×ATR trailing stop; top up below 70% of target; 10-day cooldown after a stop; daily
  rebalance at 14:00 London.
- **Settings source:** lookbacks from the paper; slots, caps and stop distance tuned on
  2024-09 → 2026-08.
- **Costs:** max($4, 0.05%) per order, 5 bps slippage (15 bps stress), £4,710 starting capital.
- **Test window:** 2008-01-09 → 2023-12-29 on US-ETF proxies; 500 random-pick runs; 60/40
  benchmark.
- **Result:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | Beats random picks | ≤ 5% of runs at least as good | 84% | ✗ |
  | Beats passive | Sharpe above 60/40 | −0.25 vs +0.89 | ✗ |
  | Trades | ≥ 100 | 582 | ✓ |
  | Sub-periods | positive in ≥ 3 of 4 | 1 of 4 | ✗ |
  | Stress slippage | still beats random picks | −7.2% a year | ✗ |
  | Neighbourhood | all above random median (−0.6%) | −25.8% to +3.3% a year | ✗ |
  | Fidelity | replay matches engine | exact on 400 sampled days | ✓ |

- **Decision:** fail on five of seven boxes. Do not run.
