# Pre-registration — attempt 5: US absolute momentum, 1928–2007

Shared rules: `COMMON.md` (threshold, random-timing comparison, boxes, and Part A data, window,
costs and benchmark).

## 1. Identity

- **Name:** US absolute momentum (the "absolute" half of dual momentum)
- **Date registered:** 2026-09-15
- **Attempt number:** 5 (registered with attempts 6 and 7; all three share the 0.714% threshold)
- **Registered by:** Claude, at the account owner's request

## 2. The idea and why it should work

- **Hypothesis:** at each month-end, holding US stocks when their 12-month return beats T-bills,
  and 10-year Treasuries otherwise, beats random timing of the same holdings after costs, and beats
  a 60/40 stock/bond portfolio on Sharpe ratio or drawdown, over 1928–2007.
- **Who is on the other side:** time-series momentum (Moskowitz, Ooi & Pedersen, 2012): investors
  react slowly and then herd, so returns persist over months. The switch to bonds is meant to sit
  out sustained bear markets such as 1929–32, 1973–74 and 2000–02.
- **Why it hasn't been competed away:** it misses sharp rebounds and lags in V-shaped recoveries,
  which managers with career risk avoid.
- **Published evidence:** Antonacci, *Dual Momentum Investing* (2014), data 1974–2013; Moskowitz,
  Ooi & Pedersen (2012), futures from 1985; Hurst, Ooi & Pedersen (2017), trend following back to
  1880. Everything before 1974 is outside Antonacci's sample.

## 3. Exact rules

- **Universe:** US stock market (Ken French market total return), 10-year Treasury bonds (built
  from yields, `COMMON.md`), 1-month T-bills (hurdle only, never held).
- **Signal at month-end m:** R_stocks = product of (1 + monthly stock total return) over the 12
  months ending m, minus 1. R_bills = the same for T-bills. Hold **stocks** if R_stocks > R_bills,
  otherwise hold **bonds**.
- **Positions:** 100% in the held asset. A switch is a sell and a buy at the next trading day's close.
- **No stops, no leverage.**
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Lookback | 12 months | Antonacci 2014 |
  | Hurdle | T-bills | Antonacci 2014 |
  | Safe asset | 10-year Treasuries | Antonacci used US aggregate bonds, which have no index before 1976; declared stand-in |

- **Possible live version (not part of this test):** a pass would lead to a new card testing a UCITS
  implementation, not straight to live trading.

## 4. Costs and execution

As `COMMON.md`, Part A.

## 5. Test design

- **Tuning window:** none.
- **Test window and sub-periods:** `COMMON.md`, Part A.
- **Overlap with data already seen:** none of 1928–2007 has been used in this project. The published
  literature has studied trend rules over parts of it (disclosed in `COMMON.md`).
- **Random-timing comparison:** `COMMON.md` (the monthly stocks/bonds sequence shifted as one).
- **Passive benchmark:** 60/40 stocks/bonds (`COMMON.md`, Part A). Also reported, not a box:
  buy-and-hold stocks.
- **Neighbourhood check:** lookback of 11 and of 13 months.
- **Code:** `COMMON.md`.

## 6. Pass mark

The seven boxes in `COMMON.md`, with box 7 including the Part A bond data check.

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
- **Headline:** CAGR +11.0% a year, Sharpe 0.64 (daily returns over T-bills), worst fall −46.0%, volatility
  11.9%; 70 switches, 141 orders; in stocks 659 of 960 months. 60/40 benchmark: CAGR +8.4%, Sharpe 0.50, worst
  fall −62.1%. Stocks buy-and-hold (reported, not a box): CAGR +9.7%, Sharpe 0.42, worst fall −84.1%. Median
  random-timing run: CAGR +7.9%.
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random timing | ≤ 0.714% of runs at least as good | 0.3% (random Sharpe median 0.36, 95th percentile 0.51) | ✓ |
  | 2 Beats 60/40 | Sharpe above 0.50, or worst fall ≤ 31.1% with CAGR ≥ 6.4% | Sharpe 0.64; worst fall −46.0%; CAGR +11.0% | ✓ |
  | 3 Decisions and switches | ≥ 100 months and ≥ 15 switches | 960 months, 70 switches | ✓ |
  | 4 Sub-periods | at least 3 of 4 positive | 4 of 4 (+352%, +823%, +831%, +1,017% over the four 20-year blocks) | ✓ |
  | 5 Stress slippage | ≤ 0.714% at 15 bps | 0.3% | ✓ |
  | 6 Neighbours | Sharpe above the random median (0.36) | 11 months 0.67; 13 months 0.59 | ✓ |
  | 7 Fidelity and data | no disagreements; data check passes | 0 in 1,201 months; bond check 0.965 | ✓ |

- **Smoke run:** a 5-seed smoke run preceded the registered run to catch crashes; 5 random runs cannot resolve a 0.714% threshold, so its ticks are not results. No code changed between the two runs.
- **Decision:** **Pass** (all seven boxes). Under §7 the next step is a new card testing a UCITS version on data
  not yet used. It does not go to live trading.
- **Post-registration robustness (not part of the verdict).** `robustness_a.py`, commit `f2eaf5a`, was written
  after this pass was known but before its own results were seen. Concern: FRED's yields are monthly averages,
  so the bond return in the month after a switch could include part of the previous month's rally.

  | Variant | Sharpe | CAGR | Random runs ≥ |
  |---|---|---|---|
  | Registered data | 0.64 | +11.0% | 0.3% |
  | Month-end yields (FRED DGS10) from 1962 | 0.65 | +11.1% | 0.1% |
  | No bond return in the first month after each switch into bonds | 0.64 | +11.0% | 0.3% |
  | Both of the above | 0.64 | +11.0% | 0.1% |
  | T-bills instead of bonds as the safe asset | 0.52 | +9.5% | 2.3% |

  The averaging concern does not explain the result. Holding bonds rather than cash when out of stocks does
  matter: with T-bills the rule still beats 97.7% of random runs but would miss the 0.714% threshold.

  | Bear market | Stocks | Strategy | Months out of stocks |
  |---|---|---|---|
  | 1929–32 | −84.0% | −30.3% | 31 of 34 |
  | 1937–38 | −50.9% | −22.0% | 6 of 13 |
  | 1946–49 | −21.6% | −20.0% | 16 of 38 |
  | 1968–70 | −36.8% | −12.8% | 11 of 19 |
  | 1973–74 | −48.2% | −8.0% | 20 of 22 |
  | 1987 crash | −33.1% | −21.1% | 2 of 5 |
  | 2000–02 | −49.2% | +2.2% | 23 of 32 |

- **Caveats, stated with the result:** index returns in USD, before fund fees and taxes; trend rules on US stocks
  over this period are documented in the published literature; its advantage came mainly from slow bear markets,
  and a 12-month signal still took a −46% worst fall; the related dual-momentum rule showed no timing value over
  2008–2026 (attempt 3) and the UK version failed (attempt 7).
