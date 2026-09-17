# Pre-registration — attempt 11: short-term reversal on US shares

Registered 2026-09-17, before any of this strategy's returns were computed.

## 1. Identity

- **Name:** Weekly short-term reversal, long only
- **Date registered:** 2026-09-17
- **Attempt number:** 11 — threshold 5% ÷ 11 = **0.455%** (at most 4 of 1,000 random runs at least as good)
- **Registered by:** Claude, at the account owner's request
- **Why this idea:** attempt 10 established the cost floor on this account — about **0.16% per round trip** at
  £1,500 a position ($1 commission a side plus slippage). Intraday scalping died because it chases a 4.6 bps
  move. This rule targets moves of 1–3% over five trading days, so the same toll is roughly a tenth of the
  target instead of triple it, and it trades about once a week.

## 2. The idea and why it should work

- **Hypothesis:** the worst performer of the past five trading days, bought at the next Monday's open and held
  five trading days, beats a random name drawn from the same universe after costs, and beats holding the
  universe.
- **Who is on the other side:** whoever had to sell that share in a hurry. Short-term reversal is usually
  explained as payment for supplying liquidity to forced sellers — fund redemptions, margin calls, index
  changes and panic. Nagel (2012) shows the pay-off is largest when markets are stressed, which is exactly when
  liquidity is scarce.
- **Why it hasn't been competed away:** it is capacity-limited and unpleasant to hold. You are buying what
  everyone just sold, and the position is concentrated in one name for a week. Large funds cannot size it, and
  it can be badly wrong when the fall was news rather than flow.
- **Published evidence:** Lehmann (1990) and Jegadeesh (1990) documented weekly reversal in US shares;
  Nagel (2012), "Evaporating liquidity", ties the returns to liquidity provision. **Disclosure:** the effect is
  well known, has weakened since the 1990s, and is strongest in small illiquid shares — the opposite of the
  liquid names this account must trade. A pass here would be a modest version of a documented effect.
- **Claude's prior knowledge (disclosed):** familiarity with the above literature, including that reversal has
  decayed in large caps. No backtest of this rule on this data has been run or viewed.

## 3. Exact rules

- **Universe (fixed now, 40 names):** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, JPM, XOM, UNH, JNJ, V,
  PG, MA, HD, CVX, MRK, ABBV, PEP, KO, COST, WMT, BAC, CRM, MCD, CSCO, ACN, LIN, ADBE, TMO, ABT, DHR, WFC, VZ,
  TXN, NEE, PM, INTC, IBM.
- **Joining:** a name becomes eligible once it has 5 trading days of history, and never leaves. Names listed
  later (META 2012, TSLA 2010, V 2008, ABBV 2013, PM 2008, LIN 2018) simply join when their data starts.
- **Survivorship, stated plainly:** this is today's list of large US companies, so it excludes firms that failed
  or shrank out of the index. That flatters absolute returns. It does **not** distort the pass mark, because the
  random-name comparison in §5 draws from the identical universe.
- **Signal, each Friday close:** for every eligible name, r5 = close ÷ close five trading days earlier − 1.
- **Entry:** buy the name with the **lowest r5** at the **open of the next trading day** (normally Monday).
- **Exit:** sell at the **open five trading days later**, then immediately repeat with that week's signal.
- **Position:** the whole sleeve in one name, whole shares, no leverage, never more than one position at a time.
  Idle cash earns 0%.
- **Ties and gaps:** ties broken by the alphabetically first ticker (fixed now, not by results). A name without a
  price on the entry or exit day is skipped and that week's trade goes to the next-worst name.
- **Sizing:** £2,000 of notional per trade.
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Lookback | 5 trading days | Lehmann 1990 / Jegadeesh 1990 (weekly reversal) |
  | Holding period | 5 trading days | same |
  | Number of positions | 1 (the worst performer) | account size: each extra name adds $2 of round-trip cost |
  | Entry timing | next open after the signal | avoids using the signal bar's own close |
  | Universe | 40 large US names | chosen before any returns; liquidity this account can trade |

## 4. Costs and execution

| Item | Value used | Stress value |
|---|---|---|
| Commission | $1.00 per order (measured on this account, 2026-09-17) | same |
| Slippage, per side | 5 bps | 15 bps |
| Fill prices | the day's open, moved against by the slippage | |
| Settlement | one position at a time; proceeds settle before the next weekly trade (T+1) | |
| Position size | £2,000 notional, whole shares | |
| Idle cash | 0% | |

## 5. Test design

- **Tuning window:** none.
- **Test window:** IBKR daily bars, dividend-adjusted, 20 years — from the first Monday on which at least 20
  names are eligible, through the last complete week before 2026-09-17.
- **Sub-periods:** the weeks split into 4 equal blocks by count.
- **Overlap with data already seen:** the ten mega-caps were used intraday in attempt 10; daily bars for this
  universe have not been examined. SPY daily bars were used in earlier attempts.
- **The two comparisons, both required:**
  1. **Random name (primary):** each week, buy a random eligible name instead of the worst performer, with the
     same costs and the same number of trades. This controls for survivorship and for the market's own return.
     1,000 runs, seeds 0–999.
  2. **Random timing:** circularly shift the sequence of chosen names by a random offset of 8 to n−8 weeks, so
     the names and turnover are unchanged but their alignment with the weeks is broken. 1,000 runs.
- **Pass-mark metric:** Sharpe ratio of weekly returns, 0% risk-free, annualised with √52.
- **Benchmarks:** equal-weight buy-and-hold of the same 40 names, rebalanced each January, same costs. Reported,
  not a box: SPY buy-and-hold.
- **Neighbourhood check:** lookback and holding period of 4 and of 10 trading days (both changed together, since
  the rule is "hold as long as you look back").
- **Code:** `research/2026-09-17_reversal/reversal_study.py`; commit hash recorded in §10.

## 6. Pass mark (every box must hold)

- [ ] **1 Beats random names:** at most 0.455% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **2 Beats random timing:** at most 0.455% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **3 Beats buy-and-hold:** higher Sharpe than the equal-weight universe, **or** a worst fall no more than
      half of it with a CAGR no more than 2 points below
- [ ] **4 Enough trades:** at least 100 completed round trips
- [ ] **5 Sub-periods:** positive after costs in at least 3 of 4
- [ ] **6 Stress slippage:** still ≤ 0.455% against random names at 15 bps a side
- [ ] **7 Neighbours:** both the 4-day and 10-day variants above the random-name median
- [ ] **8 Fidelity and data:** two independent signal implementations agree on every week; no week uses a price
      that does not exist; and the average gap between the signal Friday close and the entry Monday open is
      reported (a sanity check on entry prices, not a pass condition)

## 7. Decisions, written now

- **Fails any box:** record in `research_notes.md` and the attempts ledger. No re-tuning on this data; a changed
  rule is a new card.
- **Passes every box:** it earns a forward-test card of its own, at minimum size, judged on implementation —
  not an immediate move of live money.
- **Expected honestly:** the literature says this effect has decayed in large caps, and one name at a time is a
  concentrated way to hold it. A fail would be unsurprising; the test is cheap and the arithmetic finally fits.

## 8. What live trading would require (not part of this test)

- One trade a week at the US open (14:30 UK time), so the bot's schedule would need to cover it.
- Live US quotes (a small monthly data subscription).
- Roughly £2,000 of settled cash, which the account does not currently have free.

## 9. Ledgers

Attempt 11 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:**
- **Figure for each §6 box:**
- **Decision:**
