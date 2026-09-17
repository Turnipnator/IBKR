# Pre-registration — attempt 12: tuned entry and exit rules, global versus per-stock

Registered 2026-09-17, before any of this strategy's returns were computed.

## 1. Identity

- **Name:** Tuned entry/exit on 10 US shares — one global rule versus a rule chosen per stock
- **Date registered:** 2026-09-17
- **Attempt number:** 12 — threshold 5% ÷ 12 = **0.417%** (at most 4 of 1,000 random runs at least as good)
- **Registered by:** Claude, at the account owner's request
- **Why this idea:** the owner asked, reasonably, why not hold a watchlist of ten stocks and work out the entry
  and exit rules for each one. That is how most hand-built systems are made, and it is also the most reliable
  way to fool yourself: 64 rule combinations across 10 stocks is 640 choices, and at a 5% false-positive rate
  roughly 32 will look excellent on past data through luck alone. This card tests the idea properly by choosing
  the rules on one decade and judging them on another the choice never saw.

## 2. The idea and why it should work

- **Hypothesis:** different shares have persistent differences in how they trend and revert, so a rule fitted
  to each share individually should beat a single rule applied to all of them, and both should beat buying a
  random name — when all are judged on data that the fitting never saw.
- **Who is on the other side:** the same liquidity-provision and under-reaction stories as attempts 10 and 11.
  If per-stock differences are real they should come from stable characteristics: a share's volatility, its
  investor base, how much index and options flow it carries.
- **Why it might not work:** those characteristics change, and the rule that fitted a share's past decade may
  describe that decade's events rather than the share. Measuring that gap is the point of this card.
- **Published evidence:** none for per-stock rule tuning as a strategy — the literature on it is mostly about
  why it fails (data snooping; Sullivan, Timmermann & White 1999; White's Reality Check). Momentum and reversal
  themselves are documented (Jegadeesh & Titman 1993; Lehmann 1990), and both directions are in the grid below.
- **Claude's prior knowledge (disclosed):** attempt 11 found that picking the worst five-day performer from a
  40-name list beat neither random names nor buy-and-hold, while clearing costs comfortably. Attempt 10 found
  intraday trading dead on costs. No version of this grid has been run or viewed.

## 3. Exact rules

- **Universe (the owner's ten):** AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, JPM, XOM. Daily
  dividend-adjusted IBKR bars already fetched for attempt 11. A share is eligible once it has 25 trading days
  of history.
- **Rule family (64 combinations per share), fixed now:**

  | Element | Values |
  |---|---|
  | Lookback N | 3, 5, 10, 20 trading days |
  | Direction | buy the winner (that share's own N-day return above zero) or buy the loser (below zero) |
  | Exit (8 options) | fixed hold of 3, 5, 10 or 20 trading days, **or** a trailing stop at 1, 2, 3 or 4 × ATR(20) |

  4 lookbacks × 2 directions × 8 exits = **64 combinations per share**, so 640 choices across the ten. Signals
  are per share, not cross-sectional: "winner" means that share's own N-day return is positive. A trailing-stop
  exit sets the stop that many ATRs below the highest close since entry (ATR fixed at entry, stop ratcheting up
  only), checks it against each day's low, fills at the stop price or at the day's open if the market gapped
  through it, and force-exits after 40 trading days so nothing is held indefinitely.
- **Entry:** at the **open of the day after** the signal day, never the signal bar's own close.
- **One position at a time** (cash account). When several shares signal on the same day, take the one with the
  largest absolute N-day move; ties by alphabetical ticker. Declared now, not chosen by results.
- **Sizing:** £2,000 notional, whole shares, no leverage. Idle cash earns 0%.
- **How the rules are chosen:**
  - **Global:** the single combination with the best training Sharpe, pooled across all ten shares.
  - **Per stock:** for each share, its own best-in-training combination; in live use the share trades only on
    its own rule.
  - Both are chosen using **training data only** and then applied unchanged to the test period.

## 4. Costs and execution

| Item | Value used | Stress value |
|---|---|---|
| Commission | $1.00 per order (measured on this account, 2026-09-17) | same |
| Slippage, per side | 5 bps | 15 bps |
| Fill prices | the day's open (entry and fixed-hold exit); stop price or the day's open if gapped (stop exit) | |
| Position size | £2,000 notional, whole shares | |
| Idle cash | 0% | |

## 5. Test design

- **Training window:** 2006-10-02 → 2016-12-30. Used only to pick the combinations. Its returns are reported
  purely to measure the in-sample-to-out-of-sample gap.
- **Test window:** 2017-01-03 → 2026-09-10, run **once**, with the combinations frozen.
- **Sub-periods:** the test window's trades split into 4 equal blocks by count.
- **Overlap with data already seen:** attempt 11 used these names' daily bars in a 40-name universe with a
  five-day reversal rule. This card uses the same bars with a different rule family and a train/test split;
  that overlap is disclosed and is why the pass mark rests on the random-name comparison.
- **Comparisons, all on the test window:**
  1. **Random name (primary):** a random eligible share on each entry day, same trade count and costs. 1,000 runs.
  2. **Random timing:** the chosen entries shifted by a random offset. 1,000 runs.
- **Pass-mark metric:** Sharpe ratio of daily returns, 0% risk-free, annualised with √252.
- **Benchmark:** equal-weight buy-and-hold of the ten shares, rebalanced each January, same costs.
- **Code:** `research/2026-09-17_tuned/tuned_study.py`; commit hash recorded in §10.

## 6. Pass mark (every box must hold)

- [ ] **1 Per-stock beats random names:** at most 0.417% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **2 Per-stock beats the global rule out of sample:** higher Sharpe on the test window — this is the
      owner's question, answered on data neither rule was fitted to
- [ ] **3 Beats buy-and-hold:** higher Sharpe than the equal-weight ten, **or** a worst fall no more than half
      of it with a CAGR no more than 2 points below
- [ ] **4 Enough trades:** at least 100 completed round trips in the test window
- [ ] **5 Sub-periods:** positive after costs in at least 3 of 4
- [ ] **6 Stress slippage:** still ≤ 0.417% against random names at 15 bps a side
- [ ] **7 Random timing:** at most 0.417% of shifted runs have a Sharpe ratio at least as high
- [ ] **8 Fidelity:** two independent implementations agree on every signal and trade; no entry ever uses a
      price from the signal day or earlier; every fill comes from a real bar

## 7. Reported, not pass conditions

- **The overfitting tax:** training Sharpe versus test Sharpe, for both the global and the per-stock versions.
  If per-stock tuning is mostly curve-fitting, its training Sharpe will be far higher than the global rule's
  and its test Sharpe will not be.
- Which combination each share picked, and how many shares picked the same one.

## 8. Decisions, written now

- **Fails any box:** record in `research_notes.md` and the attempts ledger. No re-tuning on this data, and in
  particular no second grid — a different rule family is a new card with a stricter threshold.
- **Passes every box:** a forward-test card at minimum size, judged on implementation, before any live money.
- **Expected honestly:** with 640 in-sample choices, large training Sharpes are near-certain and mean nothing.
  The test window is the only number that counts, and the literature on data snooping says it usually collapses.

## 9. Ledgers

Attempt 12 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:** 2026-09-17, rules `0fbd7b1`, code `8a174f9`; 1,000 runs for each comparison.
  Training 2006-10-27 → 2016-12-30 (the card said 2006-10-02; the 25-day eligibility rule makes the first
  tradeable day three weeks later), test 2017-01-03 → 2026-09-10, **169 round trips**.
- **Headline:** the per-stock version made **CAGR +26.1%, Sharpe 0.81**, total +840%, worst fall −42.5%,
  60% winners, **+1.68% net per trade**. The single global rule (3-day loser, hold 3 days) made CAGR +22.8%,
  Sharpe 0.79 over 475 trades with a −65.0% worst fall. **Equal-weight buy-and-hold of the same ten:
  CAGR +36.2%, Sharpe 1.29, worst fall −36.6%.**
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random names | ≤0.417% of runs as good | **31.20%** (random median 0.70, 95th 1.04) | ✗ |
  | 2 Per-stock beats the global rule | higher test Sharpe | 0.81 vs 0.79 | ✓ |
  | 3 Beats buy-and-hold | higher Sharpe, or half the fall with CAGR within 2 points | Sharpe 0.81 vs 1.29; fall −42.5% vs −36.6%; CAGR +26.1% vs +36.2% | ✗ |
  | 4 Enough trades | ≥100 | 169 | ✓ |
  | 5 Sub-periods | ≥3 of 4 positive | 4 of 4 (+144.4%, +30.1%, +164.4%, +11.7%) | ✓ |
  | 6 Stress slippage | ≤0.417% at 15 bps | 31.00% (Sharpe falls to 0.72) | ✗ |
  | 7 Random timing | ≤0.417% of shifted runs as good | **1.60%** (shifted median 0.16) | ✗ |
  | 8 Fidelity | two implementations agree | 0 disagreements over 169 trades; every entry the open after its signal | ✓ |

- **The overfitting tax (§7):** the global rule decayed the textbook way — **training Sharpe 0.95 → test 0.79**.
  The per-stock version went 0.62 → 0.81, which looks like the opposite but is not a good sign: its training
  figure is low because ten different rules compete for the single slot and crowd each other out, while each
  share's *own* training Sharpe was 0.50–1.08. What matters is where both landed: **0.79 and 0.81, against a
  random-name median of 0.70.** Fitting 640 combinations bought roughly a tenth of a Sharpe point over picking
  names out of a hat, and that tenth is well inside the noise (the random-name 95th percentile is 1.04).
- **What the ten shares chose:** five wanted to buy strength (AAPL, AMZN, META, GOOGL, AVGO) and five to buy
  weakness (MSFT, NVDA, TSLA, JPM, XOM) — eight distinct combinations among the ten, with only MSFT/JPM and
  AAPL/GOOGL agreeing. That disagreement was the thing being tested, and the test decade says it was ten
  separate accidents of 2006–2016, not ten personalities.
- **Decision:** **Fail** (boxes 1, 3, 6 and 7). Box 2 technically passed by 0.02 of a Sharpe point, which is
  noise; per-stock tuning did not beat one global rule in any meaningful sense. Not re-tuned on this data.
- **Costs were never the problem — again.** +1.68% net per trade against an 18.5 bps toll, the same finding as
  attempt 11. Every version of this made money in absolute terms. Every version also lost to simply holding the
  ten names, which returned +36.2% a year over the test decade with a smaller drawdown.
- **The survivorship point, stated plainly:** these are 2026's ten largest US companies looking backwards, so
  +36% a year for buy-and-hold is not an achievable benchmark either — it is what the winners did. That is
  exactly why box 1 compares against random names drawn from the same ten, and 31% of those runs matched.
- **Disclosure:** a 5-seed smoke run printed a verdict before two conformance fixes landed (`8a174f9`: the test
  window had run four days past the card's end date, and box 8's second implementation was not yet wired in).
  Both fixes make the test stricter. No rule, window, grid or pass mark was changed in response.
