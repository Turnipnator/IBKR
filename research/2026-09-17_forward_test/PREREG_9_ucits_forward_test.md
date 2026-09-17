# Pre-registration — attempt 9: UCITS forward test of US absolute momentum

Registered 2026-09-17, before the sleeve traded. This is a **live forward test at small size**, so it
carries no statistical threshold: it is judged on whether live trading matches the model, exactly as
§8 of the template describes. Registered together with attempt 8.

Card 5 §7 required "a new card testing a UCITS version on data not yet used". Live months from here
are, by definition, data nobody has seen.

## 1. Identity

- **Name:** US absolute momentum, UCITS forward test (the "sleeve")
- **Date registered:** 2026-09-17
- **Attempt number:** 9 (forward test; no statistical threshold)
- **Registered by:** Claude, at the account owner's request
- **Owner's decisions, 2026-09-17:** run alongside the existing momentum bot; about £2,500 of the
  account; the bot places the orders itself.

## 2. The idea and why it should work

As attempt 5, which passed all seven boxes on US data 1928–2007 (CAGR +11.0%, Sharpe 0.64 vs 60/40's
0.50; 0.3% of random-timing runs as good; robust to month-end yields and to removing the first month's
bond return). The open questions that only live trading can answer are whether the rule survives this
account's real fills, costs, settlement and whole-share rounding, and whether it can run without
disturbing the momentum bot it shares an account with.

**Stated now:** the related dual-momentum rule showed no timing value over 2008–2026 (attempt 3), the
UK version failed (attempt 7), and attempt 5's edge came mainly from long bear markets and from bonds
rallying while stocks fell. Twelve months of live trading cannot confirm or refute profitability.

## 3. Exact rules

- **Sleeve capital:** £2,500, set at the first sleeve rebalance. Sleeve equity = sleeve positions +
  sleeve cash reserve, tracked in the bot's database and never re-based by the momentum strategy.
- **Instruments** (none of the three is in the momentum bot's universe, so positions can never collide):

  | Role | Line | Currency | Notes |
  |---|---|---|---|
  | Stocks | VUAA (Vanguard S&P 500 UCITS, accumulating) | USD | about $147 a share, so roughly 20 shares |
  | Bonds | IBTM (iShares $ Treasury Bond 7–10yr UCITS, GBP-quoted line) | GBP | about £124 a share; same fund as the bot's IDTM, different listing |
  | Hurdle | IB01 (iShares $ Treasury Bond 0–1yr UCITS, accumulating) | USD | never held; stands in for US T-bills |

- **Signal**, on the first trading day of each month at 14:05 Europe/London (five minutes after the
  momentum rebalance, so the two never interleave): using IBKR daily `ADJUSTED_LAST` closes at the last
  12 month-ends, R_stocks = VUAA's total return over 12 months and R_bills = IB01's. **Hold VUAA if
  R_stocks > R_bills, otherwise hold IBTM.**
- **Positions:** 100% of sleeve equity in the selected line, whole shares, 2% cash buffer. On a switch,
  sell the old line in full, then buy the new one as settled cash allows (T+2), retrying at each later
  close until funded.
- **No stops, no leverage** — matching attempt 5.
- **Ring-fencing (the rules that keep the two strategies apart):**
  1. The momentum strategy's sizing capital excludes sleeve market value and the sleeve cash reserve.
  2. The momentum settled-cash gate subtracts the sleeve reserve, so it cannot spend the sleeve's money.
  3. Sleeve symbols are excluded from the momentum held-symbol filter, from protective-stop
     reconciliation and from the parity orphan check.
  4. The sleeve is exempt from the momentum daily-loss gate and drawdown brakes: those are the momentum
     strategy's risk controls, and the sleeve has its own in §7.
  5. The sleeve trades at most once per calendar month; the last executed month is stored in the
     database, so a restart cannot repeat it.
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Rule and lookback | absolute momentum, 12 months | attempt 5 (Antonacci 2014) |
  | Stocks / bonds lines | VUAA / IBTM | data availability; chosen to avoid the momentum bot's symbols |
  | Hurdle | IB01 | no UK-listed T-bill series is directly tradeable; IB01 is the closest UCITS equivalent |
  | Sleeve size | £2,500 | owner's decision, 2026-09-17 |
  | Timing | first trading day of the month, 14:05 London | monthly rule; after the momentum rebalance |

## 4. Costs and execution

IBKR commission max($4, 0.05%) on USD lines and max(£3, 0.05%) on GBP lines; T+2 settlement; whole
shares; GBP account base; idle sleeve cash earns 0%. Slippage is measured live rather than assumed.

## 5. Test design

Not a statistical test. Each month the bot records what the rule asked for and what actually happened,
and a replay recomputes the same signal from IBKR bars. The comparison is live against replay.

## 6. Pass mark, over 12 months (profit is deliberately **not** a criterion)

- [ ] Live signals match the replay at all 12 month-ends
- [ ] Every intended order is placed in the same session and filled that day, or as soon as cash settles
- [ ] Average slippage within 15 bps per side
- [ ] Commission within 20% of the modelled amount
- [ ] No month missed for want of cash
- [ ] **No interference with the momentum strategy:** order parity OK at every check, no sleeve position
      ever carries a protective stop, and momentum sizing capital excludes the sleeve at every rebalance

## 7. Decisions, written now

- **Stop the sleeve early if** its rolling 12-month drawdown is worse than **−21.6%** (the worst 5% of
  attempt 5's rolling 12-month drawdowns over 1928–2007; median −5.4%, worst −33.6%), or if any
  ring-fencing rule in §3 is breached in live trading.
- **A breach** pauses the sleeve, is recorded in `research_notes.md`, and the 12 months restart only
  after the cause is fixed and tested.
- **If the momentum strategy halts** on its own risk controls, the sleeve continues; they are separate.
- **After 12 months:** the owner decides whether to stop, continue, or scale. A pass here is evidence
  that the plumbing works, not that the strategy makes money.

## 8. Forward test mechanics

- **Duration:** 12 monthly signals from the first sleeve trade.
- **Reporting:** each month's signal, orders, fills, slippage and commission go to Telegram and the
  database, with a monthly line in `research_notes.md`.
- **Scaling:** none during the 12 months.

## 9. Ledgers

Attempts 8 and 9 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 9a. Amendments (dated, before the sleeve traded)

**2026-09-17, after attempt 13 — funding and context.** Two changes, both recorded before the sleeve
placed a single order:

1. **The momentum strategy is being wound down, so the sleeve is funded from its exits.** The owner's
   decision after attempt 13: momentum opens no new positions or top-ups
   (`TradingConfig.momentum_entries_enabled = False`), nothing is force-sold, and each position leaves
   on its own trailing stop. The sleeve's £2,500 reserve is therefore filled over weeks from settling
   proceeds rather than on day one — the account had £158 of settled cash when this was written. §3's
   rule ("buy as settled cash allows, retrying at each later close until funded") is unchanged; this
   is the situation it was written for, and it now gets exercised properly.
2. **Funding happens in tranches, and the minimum is stated.** IBKR charges a flat max($4, 0.05%) /
   max(£3, 0.05%) per order, so a £100 tranche costs 3–4% and a £500 one costs 0.6–0.8%. The sleeve
   therefore buys **£500 or more at a time**, except for a final tranche that leaves too little to buy
   another share, and never places an order below £150 (`SleeveConfig.min_topup_base` /
   `min_order_base`). Anything left over is swept into the next monthly switch, which sells the old
   line in full and rebuys with the whole reserve. This is an implementation choice on top of §3, not
   a change to the strategy, and it is judged under §6's implementation boxes like everything else.

**Bug found while making the above change (would have hit the first live switch).**
`Database.add_sleeve_reserve()` creates the reserve row from whatever delta it is handed, so if the
sleeve's *first* action had been a switch-sell, the reserve would have been seeded at the sale
proceeds instead of the £2,500 capital base — and the sleeve would have spent its life investing that
amount. The reserve is now read (and therefore seeded) before anything touches it, with a regression
test. The old code hid this: it bought one share with the mis-seeded reserve and the test passed.

## 10. Results (written as the months come in)

- **First trade date and code commit:**
- **Monthly log:** signal, orders, fills, slippage, commission, parity status
- **Figure for each §6 box:**
- **Decision:**
