# Pre-registration — attempt 6: US 10-month trend timing, 1928–2007

Shared rules: `COMMON.md` (threshold, random-timing comparison, boxes, and Part A data, window,
costs and benchmark).

## 1. Identity

- **Name:** US 10-month trend timing (Faber's rule on a single asset)
- **Date registered:** 2026-09-15
- **Attempt number:** 6 (registered with attempts 5 and 7; all three share the 0.714% threshold)
- **Registered by:** Claude, at the account owner's request

## 2. The idea and why it should work

- **Hypothesis:** holding US stocks while their total-return index is above its 10-month average,
  and T-bills otherwise, beats random timing of the same holdings after costs, and beats a 60/40
  stock/bond portfolio on Sharpe ratio or drawdown, over 1928–2007.
- **Who is on the other side:** as attempt 5 — slow reaction and herding make market trends persist;
  the rule is meant to be out of stocks during long bear markets.
- **Why it hasn't been competed away:** as attempt 5 — whipsaw losses in choppy markets and long
  spells behind buy-and-hold.
- **Published evidence:** Faber, *A Quantitative Approach to Tactical Asset Allocation* (2007,
  updated 2013). Faber's published US stock results extend before 1973, so this period is new to
  this project but not to the literature (disclosed in `COMMON.md`).

## 3. Exact rules

- **Universe:** US stock market (Ken French market total return); 1-month T-bills.
- **Signal at month-end m:** build a total-return index from monthly stock returns. SMA10 = the
  average of the index at month-end m and the 9 month-ends before it. Hold **stocks** if the index at
  m is above SMA10, otherwise hold **T-bills**.
- **Positions:** 100% in the held asset. A switch is a sell and a buy at the next trading day's close.
- **No stops, no leverage.**
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Moving-average length | 10 month-ends | Faber 2007 |
  | Out-of-market asset | T-bills | Faber 2007 |
  | Index used for the average | total return | declared choice: the only month-end series in this data; Shiller's price index uses monthly averages, which would flatter timing |

- **Possible live version (not part of this test):** a pass would lead to a new card testing a UCITS
  implementation, not straight to live trading.

## 4. Costs and execution

As `COMMON.md`, Part A.

## 5. Test design

- **Tuning window:** none.
- **Test window and sub-periods:** `COMMON.md`, Part A.
- **Overlap with data already seen:** none of 1928–2007 has been used in this project; Faber has
  published US results over this period (disclosed).
- **Random-timing comparison:** `COMMON.md` (the monthly stocks/T-bills sequence shifted as one).
- **Passive benchmark:** 60/40 stocks/bonds (`COMMON.md`, Part A). Also reported, not a box:
  buy-and-hold stocks.
- **Neighbourhood check:** moving-average length of 9 and of 11 month-ends.
- **Code:** `COMMON.md`.

## 6. Pass mark

The seven boxes in `COMMON.md`, with box 7 including the Part A bond data check (bonds are part of
the benchmark).

## 7. Decisions, written now

- **Fails any box:** record the result in `research_notes.md` and the attempts ledger. No re-tuning
  on this data.
- **Data check fails:** withdrawn, recorded as such, not run.
- **Passes every box:** register a new card testing a UCITS version on data not yet used.

## 8. Forward test

Not applicable to this card (historical index data). See §7.

## 9. Ledgers

Attempts 5–7 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:** 2026-09-15, code `6fad2a8`, rules `c2eaa21`; 1,000 random-timing runs per slippage level. Window: signals 1927-12 → 2007-11 (960 months), trades from 1928-01-03, valued to 2007-12-31 (21,171 trading days, 80.0 years).
- **Data check:** calendar-year returns of the constructed bond series and Shiller's "Monthly Total Bond Returns" correlate 0.965 over 1928–2007 (960 months each). Passed.
- **Headline:** CAGR +9.1% a year, Sharpe 0.48 (daily returns over T-bills), worst fall −47.7%, volatility
  12.3%; 116 switches, 233 orders. 60/40 benchmark: CAGR +8.4%, Sharpe 0.50, worst fall −62.1%. Median
  random-timing run: CAGR +7.4%.
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random timing | ≤ 0.714% of runs at least as good | 2.5% (random Sharpe median 0.32, 95th percentile 0.45) | ✗ |
  | 2 Beats 60/40 | Sharpe above 0.50, or worst fall ≤ 31.1% with CAGR ≥ 6.4% | Sharpe 0.48; worst fall −47.7%; CAGR +9.1% | ✗ |
  | 3 Decisions and switches | ≥ 100 months and ≥ 15 switches | 960 months, 116 switches | ✓ |
  | 4 Sub-periods | at least 3 of 4 positive | 4 of 4 (+165%, +765%, +508%, +636%) | ✓ |
  | 5 Stress slippage | ≤ 0.714% at 15 bps | 2.5% | ✗ |
  | 6 Neighbours | Sharpe above the random median (0.32) | 9 months 0.43; 11 months 0.50 | ✓ |
  | 7 Fidelity and data | no disagreements; data check passes | 0 in 1,201 months; bond check 0.965 | ✓ |

- **Smoke run:** a 5-seed smoke run preceded the registered run to catch crashes; 5 random runs cannot resolve a 0.714% threshold, so its ticks are not results. No code changed between the two runs.
- **Decision:** **Fail** (boxes 1, 2 and 5). Not re-tuned on this data. It beat 97.5% of randomly timed runs,
  so the timing did something, but not enough for the batch threshold, and its Sharpe ratio fell just short of 60/40's.
