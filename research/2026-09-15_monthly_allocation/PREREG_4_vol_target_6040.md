# Pre-registration — attempt 4: volatility-targeted 60/40

Shared rules: `COMMON.md` (costs, window, random-timing comparison, threshold, benchmark, fidelity).

## 1. Identity

- **Name:** Volatility-targeted 60/40
- **Date registered:** 2026-09-15
- **Attempt number:** 4 (registered with attempts 2 and 3; all three share the 1.25% threshold)
- **Registered by:** Claude, at the account owner's request

## 2. The idea and why it should work

- **Hypothesis:** scaling a 60/40 SPY/IEF portfolio's exposure each month to 10% ÷ its
  last-month realised volatility, capped at 100%, earns a higher Sharpe ratio than random timing
  of the same exposure levels after this account's costs, and beats static 60/40 on Sharpe ratio
  or drawdown.
- **Who is on the other side:** volatility clusters and can be forecast a month ahead, but
  average returns do not rise enough in volatile spells to pay for the extra risk (Moreira & Muir,
  2017; Harvey et al., 2018). Investors who keep constant exposure through high-volatility periods
  carry those losses.
- **Why it hasn't been competed away:** cutting exposure after a volatility spike means missing
  sharp rebounds, and many mandates cannot de-risk on a formula.
- **Published evidence (mixed, stated now):** Moreira & Muir, "Volatility-Managed Portfolios",
  *Journal of Finance* (2017); Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & Van Hemert, "The
  Impact of Volatility Targeting", *Journal of Portfolio Management* (2018), which finds better
  risk-adjusted returns for equities and little effect for bonds and balanced portfolios;
  Cederburg, O'Doherty, Wang & Yan (2020), who find that volatility-managed portfolios often fail
  to beat unmanaged ones out of sample.

## 3. Exact rules

- **Universe:** SPY and IEF, split 60/40 within the invested part; the rest in cash.
- **Universe (possible live version, verified before any forward test; not part of the test):**
  CSPX and IDTM.
- **Data:** dividend-adjusted daily closes (`COMMON.md`).
- **Volatility:** at each month-end, take the daily returns of a 60/40 mix in USD
  (0.6 × SPY return + 0.4 × IEF return) over the last 21 trading days up to and including the
  month-end. σ = their standard deviation × √252.
- **Exposure:** e = min(1, 0.10 ÷ σ). Targets: SPY at 0.6 × e and IEF at 0.4 × e of account value.
- **Trading:** an asset is traded only if its target differs from its current weight by more than
  5 percentage points of account value (declared for costs).
- **No stops, no leverage** (cash account).
- **Timing:** `COMMON.md`.
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Target volatility | 10% a year | chosen before any data |
  | Volatility lookback | 21 trading days | previous month's daily returns, as in Moreira & Muir 2017 |
  | Scaling | 1 ÷ σ, capped at 100% | Harvey et al. 2018 scale by volatility (Moreira & Muir use variance); cap because it is a cash account |
  | Trade band | 5 percentage points | chosen before any data, for costs |
  | Return on cash | 0% | account reality |

## 4. Costs and execution for this account

As `COMMON.md`.

## 5. Test design

- **Tuning window:** none.
- **Test window:** `COMMON.md`.
- **Overlap with data already seen:** SPY and IEF bars for 2008–2026 and the static 60/40
  benchmark's 2008–2023 results have been seen. This rule has not been run on any of it.
- **Random-timing comparison:** `COMMON.md` (the monthly exposure sequence shifted as one; the
  60/40 split is unchanged).
- **Passive benchmark:** `COMMON.md` (static 60/40).
- **Sub-periods:** `COMMON.md`.
- **Neighbourhood check:** target volatility of 8% and of 12%; lookback of 10 and of 42 trading
  days (four neighbours).
- **Code:** `COMMON.md`.

## 6. Pass mark (every box must hold)

- [ ] **Beats random timing:** at most 1.25% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **Beats 60/40:** higher Sharpe ratio, or worst fall no more than half of 60/40's with CAGR
      within 2 points
- [ ] **At least 100 orders in the test window.** *Declared departure from the template's
      100-round-trip box:* this strategy resizes positions rather than entering and exiting them.
- [ ] **Positive in at least 3 of 4 sub-periods**
- [ ] **Still beats random timing (≤ 1.25%) at 15 bps slippage**
- [ ] **All four neighbouring settings have a Sharpe ratio above the median random-timing run**
- [ ] **Fidelity:** the two signal implementations agree on every month

## 7. Decisions, written now

- **Fails any box:** record the result in `research_notes.md` and the attempts ledger. No
  re-tuning on this data; a changed rule is a new card.
- **Passes every box:** forward test (§8). The live account keeps its current strategy until the
  owner decides otherwise.

## 8. Forward test (only if §6 passes)

- **Size:** the smallest size the owner agrees, on the live account.
- **Duration:** 12 months (12 signals).
- **Judged on:** live exposure matches the replay at all 12 month-ends; average slippage within
  15 bps; commissions within 20% of modelled.
- **Stop early if:** the drawdown is worse than 95% of the backtest's rolling 12-month drawdowns,
  or fills are routinely worse than 15 bps.
- **Scaling steps:** none in this card.

## 9. Ledgers

Attempts 2–4 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:** 2026-09-15, code `05f52b7`, rules `62c93ac`; 1,000 random-timing runs per slippage level. Window: signals 2008-05-30 → 2026-07-31 (219 months), trades from 2008-06-02, valued to 2026-08-31. IEF spliced with its ARCA-listed series before 2017-08-03 (overlap return correlation 0.999).
- **Headline:** CAGR +9.6% a year, Sharpe 0.95, worst fall −15.3%, volatility 10.2%; £4,710.57 → £24,916;
  127 orders, fees 0.16% of account value a year. Static 60/40 benchmark: CAGR +10.8%, Sharpe 0.93, worst
  fall −15.8%.
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random timing | ≤ 1.25% of runs at least as good | 9.9% (random Sharpe median 0.88, 95th percentile 0.96) | ✗ |
  | 2 Beats 60/40 | Sharpe above 0.93, or worst fall ≤ 7.9% with CAGR ≥ 8.8% | Sharpe 0.95 | ✓ |
  | 3 Orders | at least 100 | 127 | ✓ |
  | 4 Sub-periods | at least 3 of 4 positive | 4 of 4 (+44.8%, +81.3%, +49.8%, +34.6%) | ✓ |
  | 5 Stress slippage | ≤ 1.25% at 15 bps | 9.1% | ✗ |
  | 6 Neighbours | Sharpe above the random median (0.88) | 8%: 0.94; 12%: 0.93; 10 days: 0.92; 42 days: 0.93 | ✓ |
  | 7 Fidelity | no disagreements | 0 in 239 months | ✓ |

- **Smoke run:** a 5-seed smoke run preceded the registered run to catch crashes. Five random runs cannot resolve a 1.25% threshold, so its box ticks are not results. It printed PASS for this strategy; the registered 1,000-run result below is the one that counts. No code changed between the two runs; `smoke_run_5_seeds.log` is kept for transparency.
- **Decision:** **Fail** (boxes 1 and 5). Not run live, and not re-tuned on this data. Its Sharpe edge
  over static 60/40 is 0.02 for 1.2 points a year less return, and 1 in 10 randomly timed runs matched it.
  In practice it behaved like 60/40 with extra trades.
