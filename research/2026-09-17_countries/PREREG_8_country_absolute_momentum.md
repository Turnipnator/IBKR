# Pre-registration — attempt 8: absolute momentum across non-US country markets, 1975–2007

Registered 2026-09-17, before any of this strategy's returns were computed. Registered together with
attempt 9 (`../2026-09-17_forward_test/PREREG_9_ucits_forward_test.md`), which is a live forward test
and carries no statistical threshold; attempt 8 is the only statistical test in this batch.

## 1. Identity

- **Name:** Country absolute momentum (attempt 5's rule applied to non-US markets)
- **Date registered:** 2026-09-17
- **Attempt number:** 8 — threshold 5% ÷ 8 = **0.625%** (at most 6 of 1,000 random runs at least as good)
- **Registered by:** Claude, at the account owner's request

## 2. The idea and why it should work

- **Hypothesis:** applying attempt 5's rule unchanged to each non-US developed market — hold that
  country's stock market while its 12-month dollar return beats US T-bills, otherwise hold 10-year US
  Treasuries — beats random timing of the same holdings after costs, across an equally weighted pooled
  portfolio of those countries, 1975–2007; and beats buy-and-hold of the same countries on Sharpe ratio
  or drawdown.
- **Who is on the other side:** as attempt 5 — investors react slowly and then herd, so markets trend
  over months, and the switch to bonds is meant to sit out long bear markets. If that is a real feature
  of markets rather than a quirk of US history, it should show up outside the US too.
- **Why it hasn't been competed away:** as attempt 5 — it gives up return in choppy markets, lags
  buy-and-hold for years at a time, and concentrates in one asset per country sleeve.
- **Published evidence:** Antonacci (2014); Moskowitz, Ooi & Pedersen (2012), 58 futures markets;
  Hurst, Ooi & Pedersen (2017), trend following across markets back to 1880. **Disclosure:** the
  literature already documents trend and absolute momentum in international markets, so a pass here
  confirms published work rather than discovering something new.
- **Claude's prior knowledge (disclosed):** attempt 5 passed on US data 1928–2007 (Sharpe 0.64, 0.3% of
  random runs as good); attempts 3, 6 and 7 failed. No backtest of this country version has been run or
  viewed; the data files were inspected only for layout and coverage.

## 3. Exact rules

- **Data:** Ken French's "F-F International Countries" archive, downloaded 2026-09-17. One file per
  country; **section 1** of each file ("Value-Weight Dollar Returns — All 4 Data Items Not Reqd"),
  column **`Mkt`**: monthly value-weighted total returns in US dollars, in percent, with −99.99 marking
  missing months. The archive holds 21 countries and **no US file**, so it does not overlap attempt 5's
  equity data.
- **Countries and first months:** UK, Australia, Belgium, France, Germany, Hong Kong, Italy, Japan,
  Netherlands, Norway, Singapore, Spain, Sweden and Switzerland from 1975-01; Canada 1977-01;
  Austria 1987-01; Finland 1988-01; New Zealand 1988-01; Denmark 1989-01; Ireland 1991-01; Malaysia
  1994-01 to 2001-10 (its data then ends).
- **Joining and leaving:** a country enters at the first month-end where it has 12 consecutive months of
  data, and leaves when its data ends. No country is added or dropped for any other reason.
- **Signal at month-end m for country c:** R_c = the product of (1 + monthly return) over the 12 months
  ending m, minus 1. R_bills = the same for the US 1-month T-bill (Ken French RF). Hold country c's
  market if R_c > R_bills, otherwise hold **10-year US Treasuries** (the series built in attempt 5 from
  FRED LTGOVTBD/GS10 yields).
- **Portfolio:** one equally weighted sleeve per eligible country, each following its own signal. Sleeve
  weights are reset to equal on the first month of each calendar year.
- **Execution:** monthly data only, so a switch happens at the same month-end close that produced the
  signal. Attempt 5's next-day execution cannot be reproduced here; this is a declared limitation and is
  shared by the random-timing runs.
- **No stops, no leverage.**
- **Where every setting came from:**

  | Setting | Value | Source |
  |---|---|---|
  | Lookback | 12 months | Antonacci 2014, as in attempt 5 |
  | Hurdle | US 1-month T-bills | attempt 5 |
  | Safe asset | US 10-year Treasuries | attempt 5 |
  | Country weights | equal, reset each January | chosen before any data |
  | Returns used | dollar returns (section 1, `Mkt`) | the widest-coverage series in the file; a dollar investor's view, matching attempt 5's currency |

## 4. Costs

0.07% commission (about the $4 minimum on a $6,000 order at this account's size) plus 5 bps slippage per
order; stress slippage 15 bps. A sleeve switch is two orders. Index returns, so no share rounding and no
settlement.

## 5. Test design

- **Tuning window:** none.
- **Test window:** signals at month-ends 1975-12 → 2007-11; valued to 2007-12.
- **Sub-periods:** four equal blocks of 96 signal months.
- **Overlap with data already seen:** none of this country data has been used in this project. The US
  T-bill and 10-year Treasury series come from attempt 5 and have been seen; the equity data has not.
- **Random-timing comparison:** each country sleeve's monthly holding sequence is circularly shifted by
  its own random offset of 12 to n−12 months, so each sleeve keeps its time in stocks, run lengths and
  number of switches. 1,000 pooled runs, seeds 0–999, same costs.
- **Pass-mark metric:** Sharpe ratio of the pooled portfolio's monthly returns in excess of the US
  T-bill, annualised with √12.
- **Benchmark:** equally weighted buy-and-hold of the same countries, reset each January, same costs.
  Also reported, not a box: 60% of that benchmark and 40% US 10-year Treasuries.
- **Neighbourhood check:** lookback of 11 and of 13 months.
- **Code:** `research/2026-09-17_countries/countries_study.py`; commit hash recorded in §10.

## 6. Pass mark (every box must hold)

- [ ] **1 Beats random timing:** at most 0.625% of 1,000 runs have a Sharpe ratio at least as high
- [ ] **2 Beats the benchmark:** higher Sharpe ratio, or worst fall no more than half the benchmark's with
      CAGR no more than 2 points below it
- [ ] **3 Enough decisions:** at least 100 months, and at least 15 switches per country on average
- [ ] **4 Sub-periods:** positive in at least 3 of 4
- [ ] **5 Stress slippage:** still ≤ 0.625% at 15 bps
- [ ] **6 Neighbours:** both 11 and 13 months above the random-timing median
- [ ] **7 Fidelity and data:** two independent signal implementations agree on every country-month; no
      −99.99 inside any country's used window; and the pooled country return series correlates below 0.95
      with attempt 5's US market returns over 1975–2007 (a check that this really is different data)
- [ ] **8 Breadth (declared addition):** in at least two-thirds of countries, that country's own strategy
      Sharpe ratio is above its own random-timing median

## 7. Decisions, written now

- **Fails any box:** record in `research_notes.md` and the attempts ledger. No re-tuning on this data.
- **Data check fails:** withdrawn, recorded as such, not run.
- **Passes every box:** it strengthens attempt 5's case that the effect is not US-only. It does **not**
  by itself move money: the only live exposure is the £2,500 forward-test sleeve registered as attempt 9,
  and any scaling is a separate decision by the account owner.

## 8. Forward test

Not applicable (historical index data). See §7.

## 9. Ledgers

Attempts 8 and 9 added to `research/PREREGISTRATION_TEMPLATE.md` on registration.

## 10. Results (written after the run)

- **Run date and code commit:** 2026-09-17, code `3e02c98`, rules `ff98f9f`; 1,000 random-timing runs per
  slippage level. Window: signals 1975-12 → 2007-11, executed over 384 months (1976-01 → 2007-12), 21 countries.
- **Conformance fix before any returns existed:** the first run withdrew on its own data check
  (`Malaysia: 1`). Malaysia's signal is valid at its last data month (2001-10) but the execution month
  2001-11 has no return, so the sleeve was still "eligible" after the data ended. §3 says a country leaves
  when its data ends, so eligibility now requires a return for the execution month, and the data check looks
  for gaps *inside* a country's window rather than at its end (commit `3e02c98`). No strategy returns were
  computed before that fix.
- **Headline:** CAGR +17.9% a year, Sharpe 0.96, worst fall −24.2%, volatility 11.9%; 100 → 19,402.
  Equal-weight buy-and-hold of the same countries: CAGR +15.3%, Sharpe 0.63, worst fall −36.8%, 100 → 9,529.
  Random-timing runs: Sharpe median 0.60 (5th/95th 0.52/0.70), CAGR median +12.5%. Average 32.9 switches per
  country; sleeves were in stocks 55% of country-months.
- **Figure for each §6 box:**

  | Box | Needed | Got | |
  |---|---|---|---|
  | 1 Beats random timing | ≤ 0.625% of runs at least as good | 0.00% (0 of 1,000) | ✓ |
  | 2 Beats buy-and-hold | Sharpe above 0.63, or worst fall ≤ 18.4% with CAGR ≥ 13.3% | Sharpe 0.96; worst fall −24.2%; CAGR +17.9% | ✓ |
  | 3 Enough decisions | ≥ 100 months and ≥ 15 switches per country | 384 months, 32.9 average | ✓ |
  | 4 Sub-periods | at least 3 of 4 positive | 4 of 4 (+274.7%, +369.2%, +245.2%, +219.7%) | ✓ |
  | 5 Stress slippage | ≤ 0.625% at 15 bps | 0.00% | ✓ |
  | 6 Neighbours | above the random median (0.60) | 11 months 0.98; 13 months 0.91 | ✓ |
  | 7 Fidelity and data | agreement, no interior gaps, pooled vs US corr < 0.95 | 0 disagreements in 6,922 country-months; corr 0.68 | ✓ |
  | 8 Breadth | ≥ 14 of 21 countries above their own random median | 21 of 21 | ✓ |

- **Decision:** **Pass** (all eight boxes). Under §7 this strengthens attempt 5's case that the effect is not
  US-only. No money moves on it: the only live exposure remains the £2,500 forward-test sleeve (attempt 9).
- **Post-registration robustness** (`robustness_8.py`, commit `6d1f9b4`, written before its results were seen):

  | Variant | Sharpe | CAGR | Random runs ≥ |
  |---|---|---|---|
  | As registered | 0.96 | +17.9% | 0.00% |
  | T-bills as the safe asset | 0.81 | +15.2% | 0.00% |
  | Cash at 0% as the safe asset | 0.61 | +12.7% | 0.00% |
  | Switch delayed one further month | 0.89 | +17.1% | 0.00% |
  | No January reset | 0.92 | +17.4% | 0.00% |

  The timing beats random in every variant. But the **margin over buy-and-hold depends on what it holds when
  out**: with cash at 0% its Sharpe is 0.61 against buy-and-hold's 0.63. The declared monthly-execution
  limitation does not drive it — a further month's delay still gives 0.89.

  | Window | Strategy | Buy-and-hold | Sleeves in stocks |
  |---|---|---|---|
  | 1987 crash | −21.0% | −21.7% | 52% |
  | Japan bust 1990–92 | +16.5% | −6.1% | 41% |
  | 1998 LTCM | −1.8% | −5.5% | 56% |
  | 2000–02 bear | +6.7% | −30.5% | 26% |

- **Caveats, stated with the result:** dollar returns for a dollar investor, before fund fees, tax and any
  currency hedging a GBP investor would face; the effect is already documented in the published literature;
  it earns its keep in slow bear markets and did nothing in the fast 1987 crash; and the same rule's live-era
  relatives failed (attempt 3 over 2008–2026, attempt 7 in the UK).
