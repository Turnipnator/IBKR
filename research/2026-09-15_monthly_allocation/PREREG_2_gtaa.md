# Pre-registration — attempt 2: monthly trend timing on five asset classes

Shared rules: `COMMON.md` (costs, window, random-timing comparison, threshold, benchmark, fidelity).

## 1. Identity

- **Name:** GTAA-5 (Faber timing model)
- **Date registered:** 2026-09-15
- **Attempt number:** 2 (registered with attempts 3 and 4; all three share the 1.25% threshold)
- **Registered by:** Claude, at the account owner's request

## 2. The idea and why it should work

- **Hypothesis:** holding each of five broad asset classes only while its month-end price is
  above its 10-month average beats random timing of the same holdings after this account's
  costs, and beats 60/40 on Sharpe ratio or drawdown.
- **Who is on the other side:** investors who under-react to news and then herd, which makes
  asset-class prices trend over months (Moskowitz, Ooi & Pedersen, 2012; Hurst, Ooi & Pedersen,
  2017). Most of the rule's value is meant to come from being out of an asset during long bear
  markets, while forced and panicked sellers keep pushing prices down.
- **Why it hasn't been competed away:** it gives up return in choppy markets and can lag
  buy-and-hold for years, which managers facing benchmark and career risk avoid. At five assets
  and monthly trades, capacity is irrelevant for this account.
- **Published evidence:** Faber, *A Quantitative Approach to Tactical Asset Allocation* (2007,
  updated 2013), US data from 1973. It reports returns similar to equal-weight buy-and-hold with
  much smaller drawdowns. The rule was published before this test window starts, so every tested
  month is after publication. Practitioner reviews describe weaker results since publication;
  not verified here.

## 3. Exact rules

- **Universe (test):** SPY (US equities), EFA (developed ex-US equities), IEF (US 7–10 year
  Treasuries), VNQ (US property), GSG (S&P GSCI commodities) — ETF versions of the paper's five
  indices.
- **Universe (possible live version, verified before any forward test; not part of the test):**
  CSPX; an MSCI EAFE or World ex-USA UCITS line (to be identified); IDTM; IDUP; a GSCI or broad
  commodity UCITS line.
- **Data:** dividend-adjusted month-end closes (`COMMON.md`).
- **Signal:** for each asset at month-end *m*, SMA10 = the average of the month-end closes for *m*
  and the 9 month-ends before it. The asset is **in** if its close at *m* is above SMA10,
  otherwise **out**.
- **Entry:** an asset that turns in is bought to a target of 20% of account value.
- **Exit:** an asset that turns out is sold in full; that 20% stays in cash.
- **Sizing:** a held asset is resized back to 20% only when its weight has drifted outside
  15%–25% of account value. This departs from the paper's monthly rebalancing and is declared
  for cost reasons (the $4 minimum per order).
- **No stops, no leverage.**
- **Timing:** `COMMON.md` (month-end close signal, trade at the next trading day's close).
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Moving-average length | 10 month-ends | Faber 2007 |
  | Assets and weights | 5 × 20% | Faber 2007 |
  | Rebalance band | 15%–25% of account value | chosen before any data, for costs |
  | Return on cash | 0% | account reality (the paper used T-bills) |

## 4. Costs and execution for this account

As `COMMON.md`.

## 5. Test design

- **Tuning window:** none.
- **Test window:** `COMMON.md`.
- **Overlap with data already seen:** SPY, IEF and VNQ bars for 2008–2026 were used in the
  2026-09-15 study for a daily momentum strategy; EFA and GSG bars were fetched but not used.
  The 60/40 benchmark's 2008–2023 results have been seen. This rule has not been run on any of it.
- **Random-timing comparison:** `COMMON.md`, with an independent offset per asset.
- **Passive benchmark:** `COMMON.md` (60/40). Also reported but not part of the pass mark:
  equal-weight buy-and-hold of the same five assets, rebalanced each January.
- **Sub-periods:** `COMMON.md`.
- **Neighbourhood check:** moving-average length of 9 and of 11 month-ends.
- **Code:** `COMMON.md`.

## 6. Pass mark (every box must hold)

- [ ] **Beats random timing:** at most 1.25% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **Beats 60/40:** higher Sharpe ratio, or worst fall no more than half of 60/40's with CAGR
      within 2 points
- [ ] **At least 100 completed round trips** (an asset bought and later sold) in the test window
- [ ] **Positive in at least 3 of 4 sub-periods**
- [ ] **Still beats random timing (≤ 1.25%) at 15 bps slippage**
- [ ] **Both neighbouring settings have a Sharpe ratio above the median random-timing run**
- [ ] **Fidelity:** the two signal implementations agree on every month

## 7. Decisions, written now

- **Fails any box:** record the result in `research_notes.md` and the attempts ledger. No
  re-tuning on this data; a changed rule is a new card.
- **Passes every box:** forward test (§8). The live account keeps its current strategy until the
  owner decides otherwise.

## 8. Forward test (only if §6 passes)

- **Size:** the five positions at the smallest size the owner agrees, on the live account.
- **Duration:** 12 months (12 signals).
- **Judged on:** live signals match the replay at all 12 month-ends; average slippage within
  15 bps; commissions within 20% of modelled.
- **Stop early if:** the drawdown is worse than 95% of the backtest's rolling 12-month drawdowns,
  or fills are routinely worse than 15 bps.
- **Scaling steps:** none in this card.

## 9. Ledgers

Attempts 2–4 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:** 2026-09-15, code `05f52b7`, rules `62c93ac`; 1,000 random-timing runs per slippage level. Window: signals 2008-05-30 → 2026-07-31 (219 months), trades from 2008-06-02, valued to 2026-08-31. IEF spliced with its ARCA-listed series before 2017-08-03 (overlap return correlation 0.999).
- **Headline:** CAGR +5.5% a year, Sharpe 0.67, worst fall −14.8%, volatility 8.6%; £4,710.57 → £12,464;
  217 orders, fees 0.45% of account value a year. 60/40 benchmark: CAGR +10.8%, Sharpe 0.93, worst fall
  −15.8%. Equal-weight buy-and-hold of the same five assets (reported, not a box): CAGR +7.4%, Sharpe 0.59,
  worst fall −28.1%.
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random timing | ≤ 1.25% of runs at least as good | 6.2% (random Sharpe median 0.49, 95th percentile 0.68) | ✗ |
  | 2 Beats 60/40 | Sharpe above 0.93, or worst fall ≤ 7.9% with CAGR ≥ 8.8% | Sharpe 0.67; worst fall −14.8%; CAGR +5.5% | ✗ |
  | 3 Round trips | at least 100 | 91 | ✗ |
  | 4 Sub-periods | at least 3 of 4 positive | 4 of 4 (+17.4%, +34.5%, +29.8%, +29.1%) | ✓ |
  | 5 Stress slippage | ≤ 1.25% at 15 bps | 5.9% | ✗ |
  | 6 Neighbours | Sharpe above the random median (0.49) | 9 months 0.75; 11 months 0.55 | ✓ |
  | 7 Fidelity | no disagreements | 0 in 239 months | ✓ |

- **Smoke run:** a 5-seed smoke run preceded the registered run to catch crashes. Five random runs cannot resolve a 1.25% threshold, so its box ticks are not results. No code changed between the two runs; `smoke_run_5_seeds.log` is kept for transparency.
- **Decision:** **Fail** (boxes 1, 2, 3 and 5). Not run live, and not re-tuned on this data.
  Observation, not a box: it halved the worst fall of holding the same five assets and beat 94% of
  randomly timed runs, which is suggestive of real drawdown control but well short of the bar. Its asset
  mix (ex-US equities, property, commodities) trailed a US 60/40 by about 5 points a year over this window.
