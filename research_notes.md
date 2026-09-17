# Research notes — IBKR trend-following bot

Protocol: `RESEARCH.md`. Newest study first. Scripts/results live under `research/`.

---

## 2026-09-17 — Country absolute momentum (attempt 8, pass) and the live forward test (attempt 9, registered)

**Question.** Attempt 5 passed on US data 1928–2007. Is the effect US-only, or does the same rule show timing
value in other countries' markets? And can a UCITS version run live in this account without disturbing the
momentum strategy?

### 1–3. Hypotheses and method
Pre-registered before any returns (`research/2026-09-17_countries/PREREG_8…`, `research/2026-09-17_forward_test/
PREREG_9…`, commit `ff98f9f`; code `3e02c98`). **H8:** attempt 5's rule applied unchanged to each non-US market
(12-month dollar return vs US T-bills; US 10-year Treasuries when out) beats random timing and buy-and-hold.
**H0:** timing indistinguishable from random. Data: Ken French International Countries, section 1 (value-weighted
dollar returns), 21 countries, 1975-01 on; the US T-bill and 10-year Treasury series imported from attempt 5, so
they are identical. Equal-weight country sleeves reset each January, 0.07% + 5 bps per order, 1,000-run
circular-shift nulls at 5 and 15 bps, threshold 0.625% (attempt 8), plus a breadth box.

### 4. Evidence

| | CAGR | Sharpe | Worst fall | Random runs ≥ |
|---|---|---|---|---|
| **Country absolute momentum** | **+17.9%** | **0.96** | −24.2% | **0.00%** (0 of 1,000) |
| Equal-weight buy-and-hold, same countries | +15.3% | 0.63 | −36.8% | — |
| Random-timing runs (median) | +12.5% | 0.60 | — | — |

- **E1 (HIGH):** all eight boxes pass, including breadth — **21 of 21 countries** above their own random-timing
  median. Two signal implementations agree on all 6,922 country-months; pooled country returns correlate 0.68
  with the US market, confirming this is different data.
- **E2 robustness (HIGH, post-registration, `6d1f9b4`):** timing beats random in every variant (0.00% throughout):
  T-bills as safe asset 0.81, cash at 0% 0.61, a further month's execution delay 0.89, no January reset 0.92.
  **But** with cash at 0% the Sharpe (0.61) no longer beats buy-and-hold (0.63): the *timing* is real, while the
  *margin over buy-and-hold* leans on holding bonds when out — the same pattern as attempt 5's T-bill variant.
- **E3 regimes (MEDIUM):** 2000–02 +6.7% vs −30.5%; Japan 1990–92 +16.5% vs −6.1%; 1998 −1.8% vs −5.5%; but the
  fast 1987 crash −21.0% vs −21.7%, no help at all. In stocks 55% of country-months.
- **E4 conformance fix (HIGH):** the first run withdrew on its own data check (Malaysia eligible one month past
  the end of its data). Eligibility now requires a return for the execution month, matching §3. No returns were
  computed before the fix.

### 5. Self-critique
- The literature already documents trend/absolute momentum internationally, so this confirms published work
  rather than discovering something.
- Dollar returns for a dollar investor: no FX hedging cost, no fund fees, no tax. A GBP investor's version is
  untested here — and the only GBP test in this project (attempt 7) failed.
- The safe asset does a lot of the work. Whether bonds keep hedging equities is a regime question, and 2022
  showed they need not.
- Same-close execution is forced by monthly data; the delayed variant says that is not what makes it work.

### 6. Conclusion
**H8 holds.** The effect is not US-only: it shows up in all 21 countries, with the timing significant in every
robustness variant. Its edge over simply holding is materially smaller when the safe asset earns nothing.

### 7. Actions
Attempt 9, the live UCITS forward test, is registered and being built: a ring-fenced £2,500 sleeve
(VUAA/IBTM, IB01 as hurdle) running beside the momentum strategy, judged on implementation fidelity, not profit.

---

## 2026-09-15 — Pre-registered long-history and UK dual momentum tests (attempts 5–7)

**Question.** Dual momentum's rules could not be re-tested on 2008–2026. Do the same ideas, unchanged, show timing
value on data this project had not used: 80 years of US history (1928–2007) and a UK investor's version (2008–2026)?

### 1. Decomposition
- Q1 Can the rules be replicated on long index data with trustworthy bond returns, and on UK UCITS lines with dividends?
- Q2 Does the timing beat random timing of the same holdings at a batch-adjusted threshold (0.714%)?
- Q3 Does it beat the relevant 60/40?
- Q4 Is a pass robust to the known weakness of monthly-average yields?

### 2. Competing hypotheses
- **H5** US absolute momentum (stocks vs T-bills over 12 months, bonds when out) has timing value over 1928–2007.
- **H6** Faber's 10-month rule on US stocks has timing value over 1928–2007.
- **H7** UK dual momentum (ISF vs IWRD vs IGLT, UK cash hurdle) has timing value over 2008–2026.
- **H0** for each: timing indistinguishable from random; a passive 60/40 as good or better.

### 3. Method
Pre-registered (`research/2026-09-15_long_history_uk/`: COMMON.md and three cards, commit `c2eaa21`; code `6fad2a8`).
Part A: Ken French US market total return and T-bills (monthly for signals, daily for returns), 10-year Treasury
returns built from FRED LTGOVTBD/GS10 yields and checked against Shiller's bond returns (annual correlation 0.965),
next-day-close execution, 0.07% commission + 5 bps per order. Part B: IBKR adjusted bars for ISF/IWRD/IGLT (dividends
checked), FRED UK 3-month rate hurdle, the GBP cash-account simulator with IBKR UK commission. Both: circular-shift
random-timing nulls (1,000 runs at 5 and 15 bps), neighbouring lookbacks, four sub-periods.

### 4. Evidence

| | Window | CAGR | Sharpe | Worst fall | Random runs ≥ | Verdict |
|---|---|---|---|---|---|---|
| 60/40 stocks/bonds | 1928–2007 | +8.4% | 0.50 | −62.1% | — | benchmark |
| **US absolute momentum** | 1928–2007 | **+11.0%** | **0.64** | −46.0% | **0.3%** | **Pass, all 7 boxes** |
| US 10-month trend timing | 1928–2007 | +9.1% | 0.48 | −47.7% | 2.5% | Fail (1, 2, 5) |
| 60/40 IWRD/IGLT | 2008–2026 | +7.7% | 0.82 | −19.4% | — | benchmark |
| UK dual momentum | 2008–2026 | +6.6% | 0.57 | −18.6% | 24.0% | Fail (1, 2, 5) |

(Part A Sharpe ratios are over T-bills; Part B at a 0% risk-free rate.)

- **E1 fidelity (HIGH):** two signal implementations agree on every month in both parts; bond data check 0.965;
  UK adjusted/raw ratios 0.49–0.62 confirm dividends are included.
- **E2 absolute momentum (MEDIUM–HIGH):** passes every box. Post-registration robustness (`robustness_a.py`, commit
  `f2eaf5a`): month-end yields from 1962 → 0.1% of random runs as good; no bond credit in the first month after a
  switch → 0.3%; both → 0.1%. The monthly-average-yield concern does not explain it. With T-bills instead of bonds
  as the safe asset: Sharpe 0.52, 2.3% — timing still beats 97.7% of random runs, but part of the pass comes from
  bonds rallying while stocks trend down.
- **E3 where it earned it (HIGH):** long bear markets — 1929–32 stocks −84% vs −30%; 1973–74 −48% vs −8%; 2000–02
  −49% vs +2%. Fast crashes much less: 1987 −33% vs −21%. Worst fall still −46%.
- **E4 trend timing (MEDIUM):** beat 97.5% of random runs; not significant at 0.714% and Sharpe just under 60/40.
- **E5 UK dual momentum (HIGH):** timing no better than chance; world/gilts 60/40 better on return and Sharpe.

### 5. Self-critique
- *Literature overlap.* Trend and absolute-momentum rules on US stocks have been published over periods including
  1928–2007 (Hurst, Ooi & Pedersen back to 1880). The pass confirms known evidence more than it discovers an edge.
- *Regime dependence.* Its gains came from slow bear markets and from bonds hedging stocks. In 2008–2026 the related
  dual-momentum rule showed no timing value (attempt 3), and 2022 showed bonds and stocks falling together.
- *Implementation gap.* Index returns in USD before fund fees and taxes; the UCITS version for a GBP account is untested,
  and the only GBP-based test in this batch (attempt 7, a different rule) failed.
- *Multiple testing.* Seven registered attempts; the 0.714% threshold accounts for this batch. Box 1 at 0.3% is below
  it, but with 1,000 runs the estimate is coarse (3 runs).
- *Robustness was post hoc.* Its design was committed before its results were seen, and it cannot change the verdict.

### 6. Conclusion
- **Most supported:** H5 holds on 1928–2007: US absolute momentum with bonds as the safe asset showed real timing value
  and beat 60/40 on return and Sharpe, mainly by sidestepping long bear markets. H0 holds for H6 and H7.
- **Not shown:** that it works in the 2008–2026 regime, in GBP, through UCITS funds, or after fees and taxes.

### 7. Suggested actions (user decision)
1. Under card 5's §7, register a new card testing a UCITS version on data not yet used. Honest candidates: (a) a
   forward test at minimum size, judged on implementation, which will take years to say anything about profit;
   (b) the same rule on other countries' long index histories not yet examined, to test whether it generalises beyond
   the US.
2. Do not run it live on the strength of this backtest alone.

---

## 2026-09-15 — Pre-registered monthly allocation tests (attempts 2–4)

**Question.** After the live configuration failed out of sample, do three published, low-turnover
approaches that suit a £4.7k UK cash account beat random timing of their own decisions and a passive
60/40, after this account's costs?

### 1. Decomposition
- Q1 Can each rule be implemented exactly as published, with this account's costs and settlement?
- Q2 Does its timing beat the same holdings timed at random?
- Q3 Does it beat 60/40 on Sharpe, or halve its worst fall at a similar return?
- Q4 Is any result robust to sub-periods, slippage and neighbouring settings?

### 2. Competing hypotheses
- **H2 — trend timing (Faber 2007)** controls drawdowns without giving up much return.
- **H3 — dual momentum (Antonacci 2014)** picks the stronger equity market and dodges bear markets.
- **H4 — volatility targeting (Moreira & Muir 2017; Harvey et al. 2018)** raises the Sharpe ratio of 60/40.
- **H0 — for each:** its timing is indistinguishable from random timing, and passive 60/40 is as good or better.

### 3. Method
Pre-registered before any returns were computed: rules, costs, window, random-timing null, 1.25% threshold
(5% ÷ 4, shared by the batch) and seven-box pass marks in `research/2026-09-15_monthly_allocation/`
(`COMMON.md` and three `PREREG_*.md` cards, commit `62c93ac`); code `05f52b7`. Dividend-adjusted IBKR
bars; IEF spliced with its ARCA listing before 2017-08-03. Daily cash-account simulator in GBP: next-day
close fills ± 5 bps, max($4, 0.05%) commission, T+2 settlement with a 2% buffer, whole shares, idle cash at
0%. Null: each strategy's monthly decision sequence circularly shifted by a random 12 to n−12 months,
1,000 runs at 5 and at 15 bps. Window: signals 2008-05-30 → 2026-07-31 (219 months), valued to 2026-08-31.

### 4. Evidence

| | CAGR | Sharpe | Worst fall | Random runs ≥ | Boxes failed |
|---|---|---|---|---|---|
| 60/40 SPY/IEF, yearly rebalance | +10.8% | 0.93 | −15.8% | — | — |
| Trend timing, 5 assets | +5.5% | 0.67 | −14.8% | 6.2% | 1, 2, 3, 5 |
| Dual momentum | +10.2% | 0.67 | −26.2% | 35.7% | 1, 2, 5 |
| Volatility-targeted 60/40 | +9.6% | 0.95 | −15.3% | 9.9% | 1, 5 |
| Equal-weight 5 assets, yearly (info) | +7.4% | 0.59 | −28.1% | — | — |

- **E1 fidelity (HIGH):** both signal implementations agree on all 239 months for all three rules; hand-check
  rows in `results.json`.
- **E2 trend timing (MEDIUM):** beat 94% of randomly timed runs and halved the worst fall of holding the same
  five assets — suggestive of genuine drawdown control, not significant at the batch threshold (6.2% vs 1.25%)
  or even at an unadjusted 5%. Only 91 round trips. The asset mix, not the timing, cost ~5 points a year
  against a US 60/40.
- **E3 dual momentum (HIGH):** timing indistinguishable from random (35.7%); return near 60/40 with a worst
  fall 1.7× deeper; post-2014 CAGR +9.1%.
- **E4 volatility targeting (HIGH):** exposure was 100% most months, so it tracked 60/40: Sharpe +0.02,
  return −1.2 points a year, and 1 in 10 randomly timed runs matched it.
- **E5 costs (HIGH):** fees 0.16–0.45% of account value a year — monthly rules escape the fee wall that sank
  the daily bot; costs are not why these fail.

### 5. Self-critique
- *What would disprove the fails?* A strategy near the top of its random-timing distribution. Trend timing
  came closest (94th percentile); with three tests registered together, that is not enough.
- *Benchmark hindsight.* 2008–2026 favoured US equities and (until 2021) Treasuries, and the pound's fall lifted
  GBP returns; a US 60/40 is a strong benchmark in this window. A 1970s-style inflationary period might favour
  trend timing, but IBKR offers only 20 years of bars.
- *Null design.* Circular shifts keep time invested and run lengths, so a rule that is simply "mostly
  invested in a rising market" gains nothing over the null. That is intended: the question was whether the
  timing adds value.
- *Smoke run.* A 5-seed smoke run printed PASS for volatility targeting; 5 runs cannot resolve 1.25%. No code
  changed before the registered run.
- *Prior knowledge* of public commentary on these strategies was disclosed in `COMMON.md`.

### 6. Conclusion
- **Most supported: H0 for all three.** None clears its pre-registered bar. Dual momentum and volatility
  targeting show no timing value; trend timing shows a hint of drawdown control that is not significant.
- **The best practical result in this test was the passive benchmark:** 60/40 made +10.8% a year with a
  −15.8% worst fall and 0.04% a year in fees.
- **Ruled out, on this data:** each of the three as a way to beat 60/40. No re-tuning on this window.

### 7. Suggested actions (user decision)
1. If the aim is growing this money, the data-supported option is a low-cost passive allocation (e.g. a UCITS
   equity tracker plus a bond fund, rebalanced yearly); the bot's order and settlement plumbing could automate
   the yearly rebalance. That earns the market's return; it claims no edge.
2. Trend timing's near-miss can only be revisited on data not used here (other markets, or a multi-year
   forward test registered as a new card) — never by re-running this window with new settings.
3. Stop or wind down the live momentum bot (see the 2026-09-15 out-of-sample entry below).

---

## 2026-09-15 — Out-of-sample test of the frozen live config (2008–2023)

**Question.** Every structural parameter (3 slots x 30%, 60% class cap, 3xATR, 8% vol floor, top-up
gates) was chosen on 2024-09 -> 2026-08 data. Replayed unchanged over 2008-01-09 -> 2023-12-29, which
none of those decisions looked at, does the live configuration beat (a) random selection under
identical mechanics and costs, and (b) passive benchmarks?

### 1. Decomposition
- Q1 Can the engine and universe be replayed faithfully before 2024 (data, proxies, signal fidelity)?
- Q2 Net result at live capital and costs, with and without the drawdown brakes.
- Q3 Does momentum *selection* beat random selection with the same timing, stops, sizing and fees?
- Q4 Is there a gross edge that costs destroy (frictionless, £50k)?
- Q5 Regime behaviour: crises, calendar years, sub-periods.
- Q6 Parameter stability in the neighbourhood (reported, never selected on).

### 2. Competing hypotheses
- **H0 — no selection edge.** The strategy is indistinguishable from random picks with the same mechanics.
- **H1 — gross edge, eaten by fixed fees.** Frictionless and £50k runs would beat random picks.
- **H2 — net edge.** Survives live costs.
- **H3 — the value is crisis protection, not return.**

### 3. Method
`research/2026-09-15_oos/`: `fetch_bars.py`, `fetch_treasuries.py`, `oos_study.py`, `results.json`,
`curves_weekly.csv`, `trips_oos.csv`, `full_run.log` (raw bars stay on the VPS in
`/root/ibkr_research/oos/bars`). Read-only IBKR session (clientId 23, requests 12 s apart; the bot's
data probe stayed OK throughout). The 23 live UCITS lines map to the US ETFs they replaced, 20Y of
dividend-adjusted daily bars. AIGS (IBKR longName *WT SOFTS*) has no long proxy and is excluded; CMOD
tracks BCOM, so its proxy is DJP. IBKR's SMART-qualified TLT/IEF/SHY history starts 2016-02 / 2017-08;
the same ETFs pinned to ARCA return 20Y (overlap return corr 0.98–0.999) and are spliced on before the
first bar. Signals: vectorised TSMOM/ATR/vol, validated against the real `TrendFollowingAnalyzer`;
253-bar window (what live "1 Y" requests return); price-only TSMOM for proxies of distributing lines,
because the bot reads unadjusted bars. Targets come from the real
`DecisionEngine._calculate_target_positions`. Mechanics mirror live: fill at the signal close ± 5 bps;
3xATR trail ratcheting on the daily high, gap fills at the open; top-up below 70% of target with the
1xATR gate and ratchet-preserving re-arm; settled cash T+2 with the 6% buffer, per-cycle tally and 50%
floors; 10-day cooldown; £200 daily-loss gate; REDUCE halves targets at 10% below the all-time peak;
HALT flattens at 20% and stays halted (the stored peak never resets). Costs max($4, 0.05%) per order.
Nulls, 500 seeds each with the same mechanics and costs: **SEL** permutes scores among vol-eligible
names each month (the number of names passing the threshold — the timing — is kept; the names are
random); **ALL** gives every eligible name a random passing score (always invested, no timing).
Benchmarks in GBP: SPY buy-and-hold, 60/40 SPY/IEF rebalanced annually with fees, equal-weight
universe rebalanced monthly (frictionless).

### 4. Evidence

**E1 — fidelity (HIGH).** V0: 0/400 TSMOM or price mismatches, ATR/vol agree to 1e-13. V1 vs the
live `instrument_signals` (76 days): TSMOM exact 91.7% on the same-day bar (81.6% on the prior
close); the stored price equals the full-day close only 7% of the time, so the residual is the bot's
14:00 partial bar. V2, UCITS vs proxy since 2018-01-29: weekly return corr 0.69–0.97 (weakest CNYA/FXI
0.69 A- vs H-shares, IDUP/VNQ 0.72, AIGA/DBA 0.77); trade/no-trade agreement 91%; top-3 Jaccard 0.63.
Strategy level (brakes off): −1.7%/yr on proxies vs −2.0%/yr on UCITS bars, DD −38% vs −44%, 342 vs
349 trips, weekly corr 0.73.

**E2 — as live (HIGH).** The terminal HALT fires on **2009-06-22** at −20.3% after 65 trips; the bot
would have flattened and stopped. Without the terminal halt, REDUCE is on for **98% of days**: the
all-time peak never resets, half-size positions double the fee ratio, and £4,710 -> **£859** by 2023
(−10.1%/yr, DD −83.5%, fees 10.1% of NLV/yr) -> £129 by 2026-09.

**E3 — brakes off (HIGH).** −3.9%/yr, £4,710 -> £2,496, DD −57.5%, 582 trips, win 33%, payoff 1.70,
−0.57% per trip, fees 5.4% of NLV/yr. Sub-periods: 2008–12 +0.5%, 2013–17 −4.9%, 2018–23 −4.3%;
in-sample 2024–26 −0.2%. Benchmarks over the same OOS window: SPY +13.0%/yr (DD −32%), 60/40 +10.5%
(DD −16%, Sharpe 0.89), equal-weight universe +5.3%.

**E4 — vs random selection (HIGH that there is no large edge; MEDIUM on the sign).** Below the null
median in all six comparisons:

| Comparison (OOS unless noted) | Strategy CAGR | Null CAGR p5 / p50 / p95 | Share of nulls ≥ strategy |
|---|---|---|---|
| As live, no terminal halt vs SEL | −10.1% | −21.7% / −6.6% / +0.8% | 67% |
| Brakes off vs SEL | −3.9% | −6.5% / −0.6% / +4.7% | 84% |
| Brakes off vs ALL | −3.9% | −9.1% / −0.5% / +4.6% | 79% |
| Frictionless, brakes off vs SEL | +3.5% | +0.9% / +4.5% / +8.3% | 67% |
| £50k, brakes off vs SEL | +2.0% | −1.0% / +2.6% / +6.7% | 59% |
| In-sample 2024–26, brakes off vs SEL | −0.2% | −6.1% / +3.4% / +14.7% | 73% |

**E5 — costs (HIGH).** Frictionless +3.5%/yr; £50k (fees 0.9%/yr) +2.0%/yr; 15 bps slippage −7.2%/yr.
The gross positive exists, but random picks with the same stops earn it too (E4).

**E6 — null turnover (MEDIUM).** SEL nulls trade less: 66 vs 80 orders/yr, 42 vs 109 top-ups (daily
re-ranking moves targets between 20% and 30% under the class cap). That is worth ~1%/yr at live
costs, but the frictionless comparison, where turnover costs nothing, still favours random picks per
trip (+0.49% vs +0.35%). Conclusion unchanged; the null is mildly favoured by construction.

**E7 — crises and years (MEDIUM).** Brakes off: GFC from 2008-01 +6.2% (SPY −29.4%, 60/40 +1.9%),
COVID −2.8% (SPY −25.9%), 2022 +21.3% (SPY −8.3%, 60/40 −3.9%); but 2015–16 −7.6%, and negative in
9 of 15 calendar years 2009–2023 against one negative year for SPY.

**E8 — neighbourhood, brakes off (LOW; choosing parameters from this table would snoop on the OOS).**
Threshold 0.3 −2.0%, 0.7 −1.5% (both beat 0.5 — non-monotonic, i.e. noise); ATR 2x −25.8%, **4x
+3.3%** (fees 1.6%/yr, 45 orders/yr — the only positive live-cost cell, monotone in turnover and about
equal to the frictionless result); 5 slots/18% −6.0% (3 slots better, consistent with 08-24);
total-return signals −2.8%.

**E9 — reconciliation with 08-28 (MEDIUM).** That study's 2Y window (2024-09-02 -> 2026-08-28, no
slippage) gives +11.0% on proxies (08-28 reported +10.8%) and +5.0% on UCITS bars — inside the path
spread the nulls show. Not a simulator bias.

### 5. Self-critique
- *What would disprove H0?* The strategy near the top of the null distribution in any cost regime. It
  never gets there; its best showing beats 41% of nulls (£50k).
- *Null design.* Monthly-fixed permutations lower null turnover (E6); the frictionless comparison
  removes that advantage and H0 still holds.
- *Proxy error.* CNYA/FXI, IDUP/VNQ and AIGA/DBA are imperfect; AIGS is excluded. The universe was
  chosen in 2026 (survivorship in absolute returns), which affects the nulls equally.
- *Execution.* Close fills vs the bot's 14:00 partial bars; no intraday stop ordering; FX on USD cash
  not modelled; no interest on idle cash (IBKR pays none at this size); LSE vs US holidays ignored.
- *Snooping.* One pass of the frozen config over periods fixed before the results; the neighbourhood is
  reported, not optimised.
- *Simpler explanation for E2?* The brake design, not the signal — but brakes-off also loses (E3).

### 6. Conclusion
- **Most supported: H0.** Over 16 unseen years the live configuration's selection is indistinguishable
  from random picks, and below their median every time. The small gross positive (+3.5%/yr
  frictionless) is not the momentum ranking's doing, and fixed fees at £4.7k turn it negative
  (−3.9%/yr). With the live brakes it halts in June 2009; without the terminal halt it bleeds in
  permanent REDUCE. A passive 60/40 made +10.5%/yr with a −16% drawdown.
- **H3 partly:** crisis behaviour is real (GFC, COVID, 2022), but a 60/40 also protected in the GFC.
- **Ruled out:** H2 (no sub-period above +0.5%/yr); H1 in its "the signal is fine, it's the fees" form,
  since the frictionless and £50k runs still sit below random.
- **Open:** a different design (lower turnover, broad-asset TSMOM, monthly cadence) is untested, and
  testing it on this same window would snoop — it needs a pre-registered hypothesis and acceptance
  test. Partial-bar timing; in-sample path sensitivity.

### 7. Suggested actions (user decision — nothing changed live)
1. Decide whether to keep trading live on a configuration with no edge in 16 years out of sample:
   stop, move to a passive allocation, or keep it as an engineering project at minimum size.
2. If it keeps running: the REDUCE brake's never-resetting peak is a trap (98% of days once tripped).
3. Any new design: write the hypothesis and acceptance test first (e.g. beat SEL nulls at p < 0.05 at
   live costs), then test.
4. P3 cosmetic: `contracts.py` labels AIGS "Broad Commodities (DBC proxy)"; IBKR's longName is WT SOFTS.

---

## 2026-09-15 — Assessment of an external "is this bot sound?" review

**Question.** A third-party review (written from the old Jan-2026 "momentum scalper" README found
via search; the rewrite 2a9be05 landed on origin today) lists upgrades: prove net edge, independent
risk governor, volatility sizing, backtest hygiene, execution realism. Which apply to the live
daily UCITS trend bot, which are already done, and which are real gaps?

**Hypotheses.** H1 the review is mostly obsolete (wrong strategy); H2 its core point — no proven
net edge — holds regardless of strategy; H3 its control checklist exposes gaps in the risk shell.

**Evidence.**
- E1 (HIGH) Scalping-specific items (5-min bars, lunch/opening filters, earnings/FOMC, borrow,
  point-in-time constituents) do not apply to a daily-rebalanced long-only ETF book.
- E2 (HIGH) Live tape 05-22 -> 09-15: 31 round-trips, 5W/26L (16%), realized -£243.78 after both
  commissions = **-£7.86/trip**; payoff 1.61x needs 38% win rate. NLV £5,008 -> £4,724 (-5.7%).
  Rank IC over the live period -0.027 (08-24 study). Net expectancy is negative on the record.
- E3 (HIGH, arithmetic) The live tape cannot settle the edge question: t = SR x sqrt(years), so a
  Sharpe-0.5 strategy needs ~16 years of live data to reach t=2. Validation has to be an
  out-of-sample backtest with FROZEN parameters; the live tape's job is checking costs/mechanics.
- E4 (MEDIUM) Parameter stability is already shaky: the 08-24 "3 slots beat 5 in 8/8" did not
  reproduce under the fuller 08-28 simulator (E7 there). Structural params (slots, caps, 3xATR)
  were chosen on 2024-09 -> 2026-08 data; pre-2024 is untouched and usable as OOS.
- E5 (HIGH) Sizing is equal-NOTIONAL, not equal-risk: the 30% cap binds for every name, so
  risk-to-stop at the cap (3xATR, 09-14 signals, vol >= 8% names) ranges £26 (RTWO) -> £148 (ISLN);
  current targets CMOD £42 (0.9% NLV) vs CRUD £97 (2.1%). README's "equal risk per slot" misleads.
- E6 (HIGH) Entries are market orders (engine.py place_market_order); no bid/ask/mid is logged,
  so spread/slippage on thin ETC lines is unmeasured.
- E7 (HIGH) Controls already present: server-side GTC trails, quantity-aware parity at
  start/reconnect/risk-check, daily loss on IBKR BASE realized+unrealized, 10%/20% drawdown brakes,
  per-name/class/gross caps, cooldown, settled-cash gate + per-cycle tally, fill notifier, ledger.
- E8 (LOW risk) Small gaps: no Telegram "pause new entries" command; no last-bar-date check before
  signals; `_last_rebalance_date` is in-memory, so a restart inside 14:00-14:04 re-runs the
  rebalance and the engine does not net open (unfilled) BUYs.
- E9 Pushback: a consecutive-loss breaker would fight a low-win-rate trend strategy; a kill switch
  that cancels working orders would strip the stops (pause entries, keep stops instead).

**Conclusion.** H1 partly (~40% of the review is obsolete); **H2 holds and is the one that
matters**; H3 yields only small gaps. At £4.7k the 08-28 sim's best 2Y cell made +5.3%/yr after
~5.1%/yr of fees — the fee wall is the size of any plausible edge.

**Next steps.** (1) Frozen-parameter OOS backtest pre-2024 on the US-ETF proxies in
`data/phase1_ucits_mapping.csv` (unverified: whether IBKR serves that history to a UK retail
account, and several proxies start 2007-2014). (2) Log bid/ask/mid at order time to measure
implementation shortfall before considering marketable limits. (3) Optional: entry-pause command,
bar-date check, persisted rebalance date. Sizing refinements wait on (1).

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
