# Research notes — IBKR trend-following bot

Protocol: `RESEARCH.md`. Newest study first. Scripts/results live under `research/`.

---

## 2026-08-28 — Does the 40% asset-class cap suit a 3-slot / 30% book?

**Trigger.** Since the 08-24 change (3 slots × 30% per name, `max_asset_class_pct` 0.40) every
rebalance has logged `Scaled commodity from ~89% to 40.0%`: the three targets are all commodity, so
the class cap — not the slot count or the per-name cap — sets deployment (~40% of NLV, ~£627 per
name, i.e. the same size as the old 5-slot/18% book). Question: is the 40% cap the right ceiling for
a 3-slot book, or should it rise?

### 1. Decomposition
- Q1 How often does the class cap bind, and in which situations?
- Q2 What does it do to *intended* deployment (target gross) vs *actual* (the book, incl. legacy holdings)?
- Q3 Return / drawdown / fee profile across caps 40–100% at 3, 4 and 5 slots, on three windows.
- Q4 Does the cap actually bound the book's realised single-class exposure?
- Q5 Side effects on churn (trades, top-ups) — and anything the top-up path does that we had not noticed.

### 2. Competing hypotheses
- **H1 — linear scaling.** A looser cap deploys more; return and drawdown scale together; Sharpe unchanged.
- **H2 — the cap protects.** Single-class momentum regimes reverse sharply; the cap improves DD-adjusted return.
- **H3 — nothing matters.** The signal has no measured edge (08-24 study); differences between caps are noise; only fee drag is real.
- **H4 (added after the first pass) — the cap causes churn.** Targets oscillate between 20% and 30% as the class mix of the top-3 changes, generating top-ups and their side effects.

### 3. Method
`research/2026-08-28_classcap/classcap_study.py`, run inside the deployed bot image (so it is the
deployed engine, GBX pence handling included). Real IBKR daily bars, 3 years, all 23 live-universe
names + GBPUSD (read-only probe). Per day: the real `TrendFollowingAnalyzer` → `rank_cross_sectional`
→ `compute_combined_signal` on a trailing 256-bar window, then the real
`DecisionEngine._calculate_target_positions` (via `__new__`) with the variant's config. A simulator
then mirrors the bot's mechanics: enter at the close when a name enters the target set; top up when
held < 70% of target (and, as the bot does, re-arm the stop at price − 3×ATR); exit only on a 3×ATR
trailing stop ratcheting on the daily high (gap fills at the open); settled cash with T+2; 6% buffer
and 50% partial-entry floor; 10-day re-entry cooldown after a stop-out; commission max($4, 0.05%) per
order. Grid: class cap ∈ {40, 50, 60, 80, 100%} × (3 slots/30%, 4/22.5%, 5/18%). Windows: 2Y
(2024-09-02 → 2026-08-28), last 12M, LIVE (2026-05-22 →). Start NLV £4,710.57.

**Limitations (all shared across variants, so relative comparisons are cleaner than absolute
numbers):** fills at the close rather than 13:00 UTC; no daily-loss halt or drawdown brakes;
AIGS/RTWO/AIGA have zero-trade days with no bar (AIGS listed 2025-04); the universe was chosen in
2026 (survivorship in absolute returns); 15 variants × 3 windows invites snooping — the conclusion
below rests on the mechanical findings, not on picking the best cell. A first pass had a
mark-to-market bug (missing bar ⇒ position dropped for a day) that produced phantom ±25% days; fixed
by carrying the last close forward — worth remembering for any future replay.

### 4. Evidence

**E1 — the cap binds on most days, and not for the reason assumed (HIGH).** At 3 slots × 30%, two
targets in one class already total 60% > 40%, so the cap fires whenever *two* of the top-3 share a
class: **88% of days** at 3 slots (4 slots: 90%; 5 slots/18%: 70%).
The genuinely single-class top-3 — the case the cap was reasoned about — occurs on only
**72/505 days (14%)** (top-4 single-class 45; top-5 19). The top signal was
equity on 405 of 505 days, commodity on 91. At a 60% cap (= 2 × per-name cap) the bind rate
falls to 13% — i.e. exactly the single-class case.

**E2 — intended vs actual deployment (HIGH).** Average *target* gross at 3 slots: 40% cap
63% → 60% 81% → 100% 85%. Actual book deployment is ~88% in every variant
because non-target holdings persist until their stops fire — the cap constrains *new money*, not the book.

**E3 — the cap does not bound realised class exposure (HIGH, mechanical).** Worst single-class share
of NLV over 2Y at 3 slots: 82% (40% cap), 100% (60%), 98% (100%). Same soft-cap
property as the slot count: holdings bought under one regime stay while the target set moves on.

**E4 — churn (MEDIUM).** 2Y at 3 slots: 40% cap → 216 trades, 45 top-ups, £661 fees (7.0% of NLV/yr);
60% → 157 / 18 / £480 (5.1%); 100% → 160 / 18 / £490. The mechanism is H4: a name
entered at 20% (capped) becomes a 30% target the day the class mix changes → 67% of target → top-up.

**E5 — returns and drawdowns (LOW–MEDIUM; not statistically distinguishable).**

3 slots / 30% per name:

| Window | Class cap | Return | CAGR | Max DD | Sharpe | Target gross | Cap binds | Trades | Top-ups (ratchet↓) | Fees (£, %/yr) | Worst class exp. | W/L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2Y | 40% | -1.1% | -0.5% | -17.3% | +0.01 | 63% | 88% | 216 | 45 (38) | £661 (7.0%) | 82% | 27/56 |
| 2Y | 50% | +5.1% | +2.5% | -11.7% | +0.29 | 72% | 87% | 168 | 25 (23) | £515 (5.5%) | 100% | 24/45 |
| 2Y | 60% | +10.8% | +5.3% | -12.4% | +0.51 | 81% | 13% | 157 | 18 (18) | £480 (5.1%) | 100% | 24/43 |
| 2Y | 80% | +4.9% | +2.4% | -12.9% | +0.26 | 83% | 13% | 166 | 22 (21) | £507 (5.4%) | 98% | 26/44 |
| 2Y | 100% | +4.8% | +2.4% | -12.8% | +0.26 | 85% | 0% | 160 | 18 (16) | £490 (5.2%) | 98% | 22/47 |
| 12M | 40% | +16.1% | +15.9% | -8.6% | +1.36 | 64% | 89% | 103 | 22 (19) | £308 (6.5%) | 100% | 19/19 |
| 12M | 50% | +20.1% | +19.8% | -9.4% | +1.49 | 73% | 89% | 79 | 9 (7) | £236 (5.0%) | 78% | 15/17 |
| 12M | 60% | +20.3% | +20.0% | -12.1% | +1.54 | 82% | 15% | 64 | 6 (6) | £191 (4.0%) | 97% | 13/14 |
| 12M | 80% | +23.4% | +23.1% | -14.7% | +1.54 | 84% | 15% | 61 | 5 (5) | £182 (3.8%) | 99% | 12/14 |
| 12M | 100% | +26.2% | +25.9% | -15.0% | +1.62 | 86% | 0% | 59 | 5 (4) | £176 (3.7%) | 97% | 14/11 |
| LIVE | 40% | -2.3% | -7.9% | -7.3% | -0.81 | 57% | 84% | 36 | 7 (7) | £108 (8.2%) | 71% | 1/11 |
| LIVE | 50% | -1.2% | -4.2% | -3.6% | -0.46 | 67% | 84% | 33 | 4 (4) | £99 (7.5%) | 99% | 3/9 |
| LIVE | 60% | -5.2% | -17.4% | -7.1% | -2.33 | 76% | 27% | 32 | 4 (4) | £96 (7.3%) | 98% | 2/10 |
| LIVE | 80% | -6.1% | -20.2% | -8.0% | -2.73 | 80% | 24% | 32 | 4 (4) | £96 (7.3%) | 99% | 2/10 |
| LIVE | 100% | -5.9% | -19.6% | -8.3% | -2.38 | 84% | 0% | 31 | 5 (4) | £93 (7.1%) | 98% | 2/9 |

Paired 40% → 60% across window × slots (9 pairs): return better in **5/9**, drawdown shallower in
**4/9** — a coin flip. Loosening the cap does **not** buy a measurable return; it buys deployment
and fewer trades, with correspondingly larger absolute swings in single-class regimes.

| Window | Slots | Return (40% → 60%) | Max DD (40% → 60%) | Trades | Fees |
|---|---|---|---|---|---|
| 2Y | 3 | -1.1% → +10.8% ✓ | -17.3% → -12.4% ✓ | 216 → 157 | £661 → £480 |
| 2Y | 4 | +0.1% → +1.2% ✓ | -13.3% → -13.0% ✓ | 243 → 189 | £742 → £578 |
| 2Y | 5 | +5.0% → +2.4% ✗ | -12.5% → -14.0% ✗ | 269 → 255 | £822 → £779 |
| 12M | 3 | +16.1% → +20.3% ✓ | -8.6% → -12.1% ✗ | 103 → 64 | £308 → £191 |
| 12M | 4 | +12.3% → +24.4% ✓ | -12.0% → -7.2% ✓ | 124 → 97 | £371 → £291 |
| 12M | 5 | +18.8% → +18.2% ✗ | -6.7% → -8.4% ✗ | 140 → 132 | £418 → £395 |
| LIVE | 3 | -2.3% → -5.2% ✗ | -7.3% → -7.1% ✓ | 36 → 32 | £108 → £96 |
| LIVE | 4 | -2.0% → -2.5% ✗ | -6.0% → -6.1% ✗ | 40 → 39 | £120 → £117 |
| LIVE | 5 | -3.4% → -3.1% ✓ | -6.4% → -6.6% ✗ | 53 → 49 | £158 → £146 |

**E6 — new finding: top-ups give back the ratchet (HIGH, mechanical).** `engine.top_up_position`
calls `replace_trailing_stop(initial_stop_price=opportunity.stop_loss_price)` where
`stop_loss_price = price − 3×ATR` — with no reference to the old stop's ratcheted `trailStopPrice`.
In the sim (3 slots, 40% cap, 2Y) **38 of 45 top-ups re-armed the stop below its ratcheted level**,
by 1.1% of price on average. The 40% cap makes it worse because it manufactures the top-ups (E4).
Verifiable live: the next `protective stop replaced` log line whose `init=` is below the previous
`trailStopPrice`.

**E7 — the 08-24 "3 slots beat 5 in all 8 comparisons" does not reproduce here (MEDIUM).** 2Y at a 40%
cap: 5 slots +5.0% vs 3 slots -1.1%; at 100%: 3 slots +4.8% vs 5 slots +3.3%. This simulator includes
top-ups, the stop re-arm, T+2 and cooldowns; the 08-24 harness did not model all of these. Not
overturning the 08-24 decision on this — flagging that the 3-vs-5 result is sensitive to mechanics.
Fee drag is unambiguous either way: 5 slots costs ~8–9% of NLV/yr vs ~5–7% at 3.

### 5. Self-critique
- *What would disprove E1–E4?* Nothing in the data — they are properties of the engine's arithmetic
  (2 × 30% > 40%) replayed on real signals. They hold regardless of edge.
- *What would disprove the return story?* Any of it — 9 paired comparisons split 5/4 and 4/5; the 12M
  window is a single strong trend year (+16–26% for every variant); the LIVE window is 3 months. H3
  stands for returns.
- *Simpler explanation for E5?* Fees: the 40% variant pays ~£180 more over 2Y at 3 slots — that alone
  is ~4% of NLV, most of its gap to the 60% variant.
- *Snooping?* The 60% figure was not searched for — it is 2 × the per-name cap, the smallest cap
  that stops the two-names-one-class case from binding.

### 6. Conclusion
- **Most supported: H4 + E1–E3.** The 40% class cap was sized for an 8-slot/15% book. At 3 × 30% it
  fires on ~88% of days as a "two names in one class" limiter, holds intended deployment to ~63%,
  adds ~35% more trades and ~2 pts/yr of fee drag, manufactures ratchet-losing top-ups (E6) — and
  still does not bound the book's realised class exposure (E3). Setting it to **60% (= 2 × per-name
  cap)** restores its intended role: it then binds only on single-class top-3 days (~13%).
- **Ruled out:** H2 as a general property (no consistent DD protection across windows); H1 (churn
  cost breaks the linearity).
- **Not resolved:** whether the signal has any edge (08-24: none measurable; this 12M window says
  the last year was kind to every variant); the 3-vs-5 slot question (E7).
- **Expect from a 60% cap:** target gross ~81% instead of ~63%, ~25–30% fewer trades, larger absolute
  P&L swings in single-class regimes (max ~60% of NLV in one class *by target*; the book can already
  reach ~100% today). **Do not expect a return improvement** — that is noise-level.

### 7. Suggested actions — status
User decision 2026-08-28 (same day): **yes to all three** — (1) cap 0.60 applied, (2) IJPN re-added
(universe 24), (3) ratchet-preserving re-arm implemented in `orders.replace_trailing_stop`
(`_ratcheted_trigger`). All in one commit, deployed via bot-only rebuild that evening. (4) is the
standing follow-up.

### 7. Suggested actions
1. **User decision:** `max_asset_class_pct` 0.40 → 0.60 (config-only, bot-only rebuild, no 2FA).
2. **Code fix (P3, independent of 1):** in `top_up_position` / `replace_trailing_stop`, re-arm at
   `max(old ratcheted trailStopPrice, price − k×ATR)` so a top-up never lowers protection. Cheap;
   testable through the real `top_up_position` with a mock old stop.
3. Re-add IJPN to `data/watchlist.json` now the pence bug is fixed (universe decision; £18 ETF).
4. Re-run this study after ~6 more months of live data; the LIVE window is too short to weigh.
