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

- **Run date and code commit:**
- **Figure for each §6 box:**
- **Decision:**
