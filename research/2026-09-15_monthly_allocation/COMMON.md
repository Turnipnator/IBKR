# Shared rules for attempts 2–4

Registered 2026-09-15, together with `PREREG_2_gtaa.md`, `PREREG_3_dual_momentum.md` and
`PREREG_4_vol_target_6040.md`. These rules fill §4 and parts of §5–§6 for all three cards.
All four files were committed before any of the three strategies' returns were computed.

## Researcher's prior knowledge (disclosed)

- Claude has seen results for SPY and a 60/40 SPY/IEF portfolio over 2008–2023 in the
  2026-09-15 study (60/40: +10.5% a year, Sharpe 0.89, worst fall −15.9%, in GBP), and the
  momentum-strategy results listed in the template's data ledger.
- Claude has general prior knowledge from public commentary that trend-timing and
  dual-momentum portfolios lagged balanced portfolios for much of 2010–2020.
- None of the three rules below has been run or viewed on this data.

## Data

- IBKR daily `ADJUSTED_LAST` (dividend-adjusted) bars for US-listed ETFs, read-only session,
  20-year maximum. Where IBKR's default contract has truncated history (as found for
  TLT/IEF/SHY), the same ETF's ARCA-listed series is spliced on before the first bar, choosing
  the candidate with the highest overlap return correlation, exactly as in `2026-09-15_oos`.
- GBPUSD daily midpoint from IBKR. All results are in GBP.
- Calendar: US trading days. A missing bar is forward-filled for valuation only.

## Test window

- Signals use each month-end close (the last trading day of the month). Trades execute at the
  close of the next trading day.
- The window starts at the first month-end at which all three strategies have their full
  lookback, and ends with valuation at the 2026-08-31 close. The last signal used is 2026-07-31.
- **Sub-periods:** the window's signal months split into 4 equal blocks by count (any remainder
  goes to the last block). A block is positive if the full-run account value at the block's end
  is above its value at the block's start.

## Costs and execution (§4 for all three cards)

| Item | Value used | Stress value |
|---|---|---|
| Commission | max($4, 0.05% of notional) per order | same |
| Spread + slippage, per side | 5 bps | 15 bps |
| Fill price | close of the trade day, plus slippage on buys and minus it on sells | |
| Settlement | T+2 trading days. Buys are paid only from settled cash, keeping a 2% buffer (marketable limit orders). A buy that can't be fully funded buys what it can, then retries at each later close until funded or until the next signal replaces it. | |
| Order sequence | all sells first, then buys, on the same day | |
| Share size | whole shares only | |
| Starting capital | £4,710.57 | |
| Idle cash | earns 0% (IBKR pays no interest at this account size) | |
| Currency | GBP base; USD positions valued at the daily GBPUSD close | |

## Random-timing comparison (the null for all three)

Each strategy's decisions form a monthly sequence: in or out per asset (attempt 2), which asset
to hold (attempt 3), or the exposure level (attempt 4). A null run circularly shifts that
sequence by a random offset between 12 and n−12 months (n = months in the window). Time
invested, time in each asset, number of switches and run lengths are all unchanged; only the
timing becomes random. Attempt 2 shifts each asset's sequence by its own offset.

- 1,000 runs, seeds 0–999, executed with the same simulator, costs and settlement rules.
- **Pass-mark metric:** Sharpe ratio of daily GBP returns, annualised with √252, using a 0%
  risk-free rate (matching idle cash that earns nothing).
- **"Share of random runs at least as good"** = share of null runs whose Sharpe ratio is at
  least the strategy's.

## Threshold

The three cards are attempts 2, 3 and 4, registered together. Each must meet the strictest
threshold in the batch: 5% ÷ 4 = **1.25%**, i.e. at most 12 of 1,000 random runs at least as good.

## Passive benchmark (box 2 for all three)

60/40 SPY/IEF (IEF spliced as above), bought at the first trade date and rebalanced to 60/40 on
the first trading day of each January, with the same costs and settlement rules.

**Alternative goal (the same for all three, declared now):** worst fall no more than half of the
benchmark's, **and** annual return (CAGR) no more than 2 percentage points below the benchmark's.

## Fidelity (box 7 for all three)

Signals are computed two independent ways (a vectorised pandas version and a plain
month-by-month loop). They must agree on 100% of months. Three hand-check rows per strategy are
printed for inspection.

## Code

`research/2026-09-15_monthly_allocation/monthly_study.py`, run in the bot's Docker image on the
VPS. The commit hash of the code that produces each result is recorded in §10 of each card.
