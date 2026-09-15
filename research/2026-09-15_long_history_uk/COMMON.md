# Shared rules for attempts 5–7

Registered 2026-09-15, together with `PREREG_5_us_absolute_momentum.md`,
`PREREG_6_us_trend_timing.md` and `PREREG_7_uk_dual_momentum.md`. All four files were committed
before any of the three strategies' returns were computed.

## Why this batch

Attempts 2–4 (US ETFs, 2008–2026) all failed. The account owner asked to keep researching dual
momentum. Its rules cannot be re-tested on 2008–2026, so this batch tests the same ideas,
unchanged, on data this project has not used:

- **Part A:** US stock, bond and T-bill history, 1928–2007 (attempts 5 and 6).
- **Part B:** a UK investor's version on UK-listed UCITS lines, 2008–2026 (attempt 7).

## Researcher's prior knowledge (disclosed)

- Everything in the template's data ledger, including attempt 3: dual momentum on US ETFs made
  +10.2% a year over 2008–2026, and its timing was no better than random.
- General knowledge that published studies report trend-following and absolute-momentum rules
  avoiding much of the 1929–32 and 1973–74 bear markets, and that Faber's published US results
  extend before 1973. For attempt 6 the pre-1973 period is new to this project but not to the
  literature.
- No backtest of these three rules on these data has been run or viewed. The data files were
  inspected only for column names and date coverage.

## Threshold

Attempts 5, 6 and 7 are registered together. Each must meet the strictest threshold in the
batch: 5% ÷ 7 = **0.714%**, i.e. at most 7 of 1,000 random runs at least as good.

## Random-timing comparison (all three cards)

Each strategy's monthly decision sequence (which asset to hold) is circularly shifted by a random
offset between 12 and n−12 months (n = months in the window). Time in each asset, number of
switches and run lengths are unchanged; only the timing becomes random. 1,000 runs, seeds 0–999,
with the same simulator and costs. "Share of random runs at least as good" = share of runs whose
Sharpe ratio is at least the strategy's.

## Pass mark boxes (all three cards)

1. **Beats random timing:** share ≤ 0.714%.
2. **Beats the part's 60/40 benchmark:** higher Sharpe ratio, **or** worst fall no more than half
   of the benchmark's **and** CAGR no more than 2 percentage points below it.
3. **At least 100 monthly decisions and at least 15 switches.** Declared departure from the
   template's 100-trade box, as for attempt 3: these are one-asset monthly rotations.
4. **Positive in at least 3 of 4 sub-periods** (account value at the end of each block above its
   value at the start, on the full run).
5. **Beats random timing at stress slippage:** share ≤ 0.714% at 15 bps.
6. **Both neighbouring settings** have a Sharpe ratio above the median random-timing run.
7. **Fidelity:** two independent signal implementations agree on every month, **and** the part's
   data checks below pass. A failed data check withdraws the card (not run) rather than failing it.

---

## Part A — US history, 1928–2007 (attempts 5 and 6)

### Data

- **Stocks:** Ken French data library, US market total return = Mkt-RF + RF (CRSP value-weighted,
  NYSE/AMEX/NASDAQ). The monthly file drives signals; the daily file drives account values.
  Files downloaded 2026-09-15, built from the CRSP 202607 database.
- **T-bills:** RF (1-month T-bill) from the same monthly and daily files.
- **10-year Treasury bonds:** monthly total return built from yields, using FRED `LTGOVTBD`
  (long-term government yield) before 1953-04 and FRED `GS10` from 1953-04:

  r_t = y_{t−1}/12 + P_t − 1, where P_t is the price (per 1 of face value) at the end of month t
  of a par bond bought at the end of month t−1 with coupon y_{t−1} and 10 years to maturity,
  repriced at yield y_t with 10 − 1/12 years left, semi-annual coupons.

  Each month's bond return is spread geometrically across that month's trading days.
- **Data check (box 7):** calendar-year returns of this bond series and of Shiller's "Monthly Total
  Bond Returns" (`ie_data.xls`) correlate at least 0.90 over 1928–2007. If not, attempts 5 and 6
  are withdrawn.

### Window

- Signals at month-ends from 1927-12 through 2007-11 (960 months). Each trade executes at the close
  of the next trading day; accounts are valued through 2007-12-31.
- **Sub-periods:** 1928–1947, 1948–1967, 1968–1987, 1988–2007 (240 signal months each).

### Simulation and costs

- Index returns, so no share rounding and no settlement. Accounts start at 100; results are USD
  percentages.
- Cost per order: 0.07% commission (about the $4 minimum on a $6,000 order at this account's size)
  plus 5 bps slippage; stress slippage 15 bps. A switch is two orders.
- **Benchmark A:** 60% stocks, 40% 10-year bonds, rebalanced at the close of the first trading day
  of each January, with the same costs on the traded amount.
- **Sharpe ratio (A):** daily returns in excess of the daily T-bill return, annualised by the square
  root of the average number of trading days a year in the window (the data include Saturday
  sessions before 1952).
- **Limitations, stated now:** yields before 1962 are monthly averages; the pre-1953 long-term yield
  stands in for a 10-year yield; bond returns are smoothed within each month.

---

## Part B — UK investor, 2008–2026 (attempt 7)

### Data

- **IBKR daily `ADJUSTED_LAST` bars on the LSE** (fetched 2026-09-15, read-only session):
  - ISF, iShares Core FTSE 100 (priced in pence);
  - IWRD, iShares MSCI World (priced in pence). No world-ex-UK line has usable history on IBKR (the
    only one found starts 2024), so MSCI World, about 4% UK, stands in;
  - IGLT, iShares Core UK Gilts (pounds).

  Pence prices are divided by 100.
- **Data check (box 7):** distributions are in the adjusted series. For each line, adjusted close
  ÷ raw close on its first bar must be below 0.95. If not, attempt 7 is withdrawn.
- **UK cash hurdle:** FRED `IR3TIB01GBM156N`, the UK 3-month interbank rate (monthly, % a year).
  Month t earns rate_t ÷ 1,200. The series ends 2026-01; later months repeat its last value.

### Window

- Signals from the first month-end at which all three lines and the hurdle have 12 months of
  history, through 2026-07-31. Trades at the close of the next LSE trading day; accounts valued at
  the last trading day of August 2026.
- **Sub-periods:** the signal months in 4 equal blocks by count (remainder to the last block).

### Costs and execution

As `../2026-09-15_monthly_allocation/COMMON.md`, except:
- currency is GBP throughout (no FX);
- commission is max(£3, 0.05% of notional) per order (IBKR UK fixed pricing).

So: slippage 5 bps (stress 15 bps), T+2 settlement with a 2% cash buffer, sells before buys, whole
shares, £4,710.57 starting capital, idle cash at 0%.

- **Benchmark B:** 60% IWRD, 40% IGLT, rebalanced on the first trading day of each January, same
  costs. Also reported, not a box: buy-and-hold ISF.
- **Sharpe ratio (B):** daily GBP returns, 0% risk-free rate, annualised with √252.
- **Overlap:** IWRD is about 70% US equities, so its 2008–2026 path overlaps information already seen
  in attempts 2–4. ISF, IGLT and the UK rate series have not been examined.

---

## Code

- Part A: `research/2026-09-15_long_history_uk/long_history_study.py`
- Part B: `research/2026-09-15_long_history_uk/uk_dual_momentum_study.py`

Both run in the bot's Docker image on the VPS. The commit hash of the code that produces each result
is recorded in §10 of each card.
