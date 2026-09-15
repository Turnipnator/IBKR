# Pre-registration — attempt 7: dual momentum for a UK investor, 2008–2026

Shared rules: `COMMON.md` (threshold, random-timing comparison, boxes, and Part B data, window,
costs and benchmark).

## 1. Identity

- **Name:** UK dual momentum (UK shares / world shares / gilts)
- **Date registered:** 2026-09-15
- **Attempt number:** 7 (registered with attempts 5 and 6; all three share the 0.714% threshold)
- **Registered by:** Claude, at the account owner's request

## 2. The idea and why it should work

- **Hypothesis:** each month, if UK shares' 12-month return beats UK cash, hold whichever of UK shares
  or world shares has the higher 12-month return; otherwise hold UK gilts. This beats random timing of
  the same holdings after this account's costs, and beats a 60/40 world shares/gilts portfolio on
  Sharpe ratio or drawdown, over 2008–2026 in GBP.
- **Who is on the other side:** as dual momentum — relative momentum across markets plus
  time-series momentum; testing the home market against cash mirrors Antonacci's use of US shares
  for a US investor.
- **Why it hasn't been competed away:** a one-asset portfolio with large tracking error and whipsaw
  losses, which institutions rarely hold.
- **Published evidence:** Antonacci, *Dual Momentum Investing* (2014), for a US investor. Claude knows
  of no published test of this UK version; stated now.

## 3. Exact rules

- **Universe:** ISF (UK shares), IWRD (world shares), IGLT (UK gilts); UK 3-month interbank rate as
  the hurdle only (`COMMON.md`, Part B).
- **Signal at month-end m:** R(x) = adjusted close at m ÷ adjusted close 12 month-ends earlier − 1.
  Hurdle H = product of (1 + rate_t ÷ 1,200) over the 12 months ending m, minus 1.
  - If R(ISF) > H: hold ISF if R(ISF) ≥ R(IWRD), otherwise hold IWRD.
  - Otherwise hold IGLT.
- **Positions:** 100% of the account in the selected line (whole shares, 2% cash buffer). On a switch,
  sell the old holding in full and buy the new one as settled cash allows (T+2).
- **Rebalancing:** no trades unless the selection changes. A buy delayed by settlement completes as
  soon as the cash settles.
- **No stops, no leverage.**
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Lookback | 12 months | Antonacci 2014 |
  | Absolute test | home market (UK shares) vs cash | direct translation of Antonacci's US rule |
  | World shares | IWRD (MSCI World) | data availability: no world-ex-UK line with history |
  | Safe asset | IGLT (UK gilts) | UK analogue of US aggregate bonds; longest-history UK bond line |
  | Hurdle | UK 3-month interbank rate | data availability: UK cash ETFs start 2012 or later |
  | Return on idle cash | 0% | account reality |

- **Live version:** these exact lines trade in the live account in GBP, with no PRIIPs restriction.

## 4. Costs and execution

As `COMMON.md`, Part B.

## 5. Test design

- **Tuning window:** none.
- **Test window and sub-periods:** `COMMON.md`, Part B.
- **Overlap with data already seen:** IWRD is about 70% US equities, so its path overlaps information
  seen in attempts 2–4; ISF, IGLT and the UK rate series have not been examined (`COMMON.md`).
- **Random-timing comparison:** `COMMON.md` (the monthly holding sequence shifted as one).
- **Passive benchmark:** 60% IWRD, 40% IGLT (`COMMON.md`, Part B). Also reported, not a box:
  buy-and-hold ISF.
- **Neighbourhood check:** lookback of 11 and of 13 months. A month whose neighbouring signal has too
  little history holds cash.
- **Code:** `COMMON.md`.

## 6. Pass mark

The seven boxes in `COMMON.md`, with box 7 including the Part B distribution data check.

## 7. Decisions, written now

- **Fails any box:** record the result in `research_notes.md` and the attempts ledger. No re-tuning on
  this data.
- **Data check fails:** withdrawn, recorded as such, not run.
- **Passes every box:** forward test (§8). The live account keeps its current strategy until the owner
  decides otherwise.

## 8. Forward test (only if §6 passes)

- **Size:** the smallest size the owner agrees, on the live account.
- **Duration:** 12 months (12 signals).
- **Judged on:** live signals match the replay at all 12 month-ends; average slippage within 15 bps;
  commissions within 20% of modelled.
- **Stop early if:** the drawdown is worse than 95% of the backtest's rolling 12-month drawdowns, or
  fills are routinely worse than 15 bps.
- **Scaling steps:** none in this card.

## 9. Ledgers

Attempts 5–7 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:**
- **Figure for each §6 box:**
- **Decision:**
