# Pre-registration — attempt 3: dual momentum

Shared rules: `COMMON.md` (costs, window, random-timing comparison, threshold, benchmark, fidelity).

## 1. Identity

- **Name:** Dual momentum (Global Equities Momentum)
- **Date registered:** 2026-09-15
- **Attempt number:** 3 (registered with attempts 2 and 4; all three share the 1.25% threshold)
- **Registered by:** Claude, at the account owner's request

## 2. The idea and why it should work

- **Hypothesis:** each month, holding whichever of US or ex-US equities has the higher 12-month
  return — or US bonds when US equities have returned less than T-bills over 12 months — beats
  random timing of the same holdings after this account's costs, and beats 60/40 on Sharpe ratio
  or drawdown.
- **Who is on the other side:** relative momentum (Jegadeesh & Titman, 1993; Asness, Moskowitz
  & Pedersen, 2013) captures slow reaction to information across markets; the absolute-momentum
  switch to bonds is meant to sidestep sustained equity bear markets, as in time-series momentum.
- **Why it hasn't been competed away:** a one-asset portfolio carries large tracking error and
  suffers whipsaw losses when trends reverse, which institutional investors can rarely tolerate.
- **Published evidence:** Antonacci, *Dual Momentum Investing* (2014), data 1974–2013. The test
  window from 2008 to 2013 overlaps his sample; everything from 2014 onward is after
  publication. The post-2014 segment is reported separately (not a pass-mark box).

## 3. Exact rules

- **Universe (test):** SPY (US equities); VEU (FTSE All-World ex-US equities, standing in for
  the book's MSCI ACWI ex-US; ACWX is used instead only if IBKR has no usable VEU history); AGG
  (US aggregate bonds); BIL (1–3 month T-bills, used only as the hurdle and never held).
- **Universe (possible live version, verified before any forward test; not part of the test):**
  CSPX; an ACWI ex-US or All-World ex-US UCITS line (to be identified); a USD aggregate bond
  UCITS line; IBTA or IB01 as the hurdle.
- **Data:** dividend-adjusted month-end closes (`COMMON.md`).
- **Signal:** at month-end *m*, R(x) = close at *m* ÷ close 12 month-ends earlier − 1.
  - If R(SPY) > R(BIL): hold SPY if R(SPY) ≥ R(VEU), otherwise hold VEU.
  - Otherwise hold AGG.
- **Entry and exit:** 100% of the account in the selected asset (whole shares, 2% cash buffer).
  On a switch, sell the old holding in full and buy the new one as settled cash allows (T+2).
- **Rebalancing:** no trades unless the selection changes. A buy delayed by settlement completes
  as soon as the cash settles (`COMMON.md`).
- **No stops, no leverage.**
- **Timing:** `COMMON.md`.
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Lookback | 12 months | Antonacci 2014 |
  | Absolute-momentum hurdle | T-bills (BIL) | Antonacci 2014 |
  | Safe asset | US aggregate bonds (AGG) | Antonacci 2014 |
  | Ex-US equity proxy | VEU (ACWX if unavailable) | data availability, chosen before any data |
  | Return on cash | 0% | account reality |

## 4. Costs and execution for this account

As `COMMON.md`.

## 5. Test design

- **Tuning window:** none.
- **Test window:** `COMMON.md`.
- **Overlap with data already seen:** SPY bars 2008–2026 and the 60/40 benchmark's 2008–2023
  results have been seen. VEU, ACWX, AGG and BIL have not been examined. This rule has not been
  run on any of it.
- **Random-timing comparison:** `COMMON.md` (the monthly holding sequence shifted as one).
- **Passive benchmark:** `COMMON.md` (60/40).
- **Sub-periods:** `COMMON.md`. Also reported, not a box: the post-publication segment from
  2014-01 onward.
- **Neighbourhood check:** lookback of 11 and of 13 months.
- **Code:** `COMMON.md`.

## 6. Pass mark (every box must hold)

- [ ] **Beats random timing:** at most 1.25% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **Beats 60/40:** higher Sharpe ratio, or worst fall no more than half of 60/40's with CAGR
      within 2 points
- [ ] **At least 100 monthly decisions and at least 15 switches in the test window.**
      *Declared departure from the template's 100-trade box:* a one-asset monthly rotation cannot
      reach 100 round trips in under 20 years, so months are the decision unit and the
      random-timing comparison is what controls luck. This card's evidence is weaker as a result.
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

- **Size:** the smallest size the owner agrees, on the live account.
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
- **Ex-US proxy used:** VEU (IBKR history from 2007-03-08; ACWX not needed).
- **Headline:** CAGR +10.2% a year, Sharpe 0.67, worst fall −26.2%, volatility 16.5%; £4,710.57 → £27,800;
  97 orders, fees 0.21% of account value a year. Post-publication segment from 2014-01 (reported, not a box):
  CAGR +9.1%. 60/40 benchmark: CAGR +10.8%, Sharpe 0.93, worst fall −15.8%.
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random timing | ≤ 1.25% of runs at least as good | 35.7% (random Sharpe median 0.65, 95th percentile 0.79) | ✗ |
  | 2 Beats 60/40 | Sharpe above 0.93, or worst fall ≤ 7.9% with CAGR ≥ 8.8% | Sharpe 0.67; worst fall −26.2%; CAGR +10.2% | ✗ |
  | 3 Decisions and switches | ≥ 100 months and ≥ 15 switches | 219 months, 29 switches | ✓ |
  | 4 Sub-periods | at least 3 of 4 positive | 4 of 4 (+56.4%, +95.0%, +30.1%, +48.8%) | ✓ |
  | 5 Stress slippage | ≤ 1.25% at 15 bps | 37.2% | ✗ |
  | 6 Neighbours | Sharpe above the random median (0.65) | 11 months 0.68; 13 months 0.67 | ✓ |
  | 7 Fidelity | no disagreements | 0 in 239 months | ✓ |

- **Smoke run:** a 5-seed smoke run preceded the registered run to catch crashes. Five random runs cannot resolve a 1.25% threshold, so its box ticks are not results. No code changed between the two runs; `smoke_run_5_seeds.log` is kept for transparency.
- **Decision:** **Fail** (boxes 1, 2 and 5). Not run live, and not re-tuned on this data. Its timing was
  no better than chance: about a third of randomly timed runs did at least as well. Its return came close to
  60/40, but with a worst fall 1.7 times deeper.
