# Research Brief — Real-Data Validation for a Swiggy Instamart Quick-Commerce Strategy Case Study

**To:** Claude (web research)
**From:** the analyst building the case study
**Goal:** find **real, citable, public** figures to validate / refine / replace the numbers below, and return them in the exact format in Section 4 so they can be dropped into a data table.

---

## 0. Your task in one paragraph

This is a competitive-strategy case study on **Swiggy Instamart** (Indian quick commerce), benchmarked against **Blinkit** (owned by **Eternal**, formerly Zomato) and **Zepto** (Kiranakart). It contains quantitative simulations calibrated to a mix of company-disclosed figures and analyst **estimates**. Several load-bearing numbers are currently estimates with **no cited source**. For each figure in Section 3, find the best real public number, cite it with a URL and date, label it primary or secondary, and give a verdict (confirms / contradicts / refines / no data found). **Do not invent or approximate any number** — "no public data found" is a valid, useful answer that keeps a figure as an honest estimate.

## 1. Context you need

- **Players:** Swiggy Instamart (marketplace-led); Blinkit / Eternal (inventory-led, the profitability benchmark); Zepto (highest order density, IPO-bound).
- **Fiscal year:** Indian FY. **FY26 = Apr 2025 – Mar 2026**, so **Q4 FY26 = Jan–Mar 2026**. Prefer the most recent disclosures (latest is roughly the Q4/FY26 results cycle, ~May 2026).
- **Today:** mid-2026. Use figures dated through ~mid-2026.
- **Units:** Indian rupees; "cr" = crore = 10 million; GOV = Gross Order Value, NOV = Net Order Value, NRV = Net Revenue, MTU = Monthly Transacting Users, AOV = Average Order Value, "orders/store/day" = dark-store order density.
- **Watch the blend trap:** companies often mix **food delivery** and **quick commerce** in the same disclosure. I need **quick-commerce-specific** figures wherever the metric is about Instamart/Blinkit/Zepto, and **food-delivery-specific** figures where noted (Group C).

## 2. How to source (please follow)

- **Primary sources first**, and label them as primary:
  - Swiggy Ltd: quarterly **shareholder letters**, investor presentations, earnings releases, exchange filings (BSE/NSE), annual report.
  - Eternal Ltd (Blinkit): shareholder letters / investor presentations / earnings (Blinkit is reported as a segment).
  - Zepto: its **IPO prospectus (DRHP / UDRHP)** and any filings.
- **Secondary sources** are acceptable but must be labelled secondary: credible analysts (JM Financial, Nomura, Motilal Oswal, Bernstein, Goldman Sachs, Jefferies) and reputable press (Reuters, Economic Times, Moneycontrol, Business Standard, Entrackr, Inc42, Datum Intelligence, The Ken).
- **Cite every number** with: source name, **direct URL**, and **publication date**.
- **Flag contradictions** loudly — where a real figure differs from the "current value" below, that is the most valuable thing you can return.
- Distinguish a **disclosed** company number from an **analyst estimate** in your verdict.

## 3. Figures to find

Legend: **[D]** = company-disclosed already (just confirm/refresh + give the primary citation), **[E]** = analyst-estimated / unsourced (the ones I most need real data for), **★** = highest priority.

### Group A — Market structure & player economics
| Figure | Current value in study | Tag | What I need |
|---|---|---|---|
| QC market share, Jan 2026 | Blinkit 46% / Instamart 24% / Zepto 22% / Others 8% | [E] ★ | A primary or named-analyst source (e.g. Datum Intelligence) with date |
| Dark stores | Instamart 1,143 · Blinkit 2,243 | [D] | Confirm latest counts + citation |
| Orders/store/day (density) | Instamart 1,025 · Blinkit ~1,337 · Zepto ~2,140 | [E] ★ | Real per-store order density; Blinkit & Zepto are currently *derived/estimated* |
| AOV | Instamart ₹700 · Blinkit ₹525 · Zepto ₹387 | [D] | Confirm + citation (note if "Net AOV") |
| MTU | Instamart 13.3M · Blinkit ~27.2M · Swiggy food-delivery 18.3M | [E] for Instamart/Blinkit ★ | Instamart QC MTU and Blinkit MTU are estimates — find disclosed values |
| Per-order loss/contribution (FY26) | Instamart ₹85.18 · Blinkit ₹3.02 · Zepto ₹78.75 | [E] ★ | These are analyst-derived; find any disclosed per-order economics |
| Instamart Q4FY26 GOV / NOV / contribution margin | GOV ₹7,881cr · NOV ₹5,675cr · CM −1.8% | [D] | Confirm + primary citation |
| Blinkit Q4FY26 NOV / adj. EBITDA / margin | NOV ₹14,386cr · EBITDA +₹37cr · +0.3% | [D] | Confirm + citation |
| Top-3 Instamart order share | FY24 34.3% → FY25 24.5% → FY26 20.9% | [D] (Zepto DRHP) | Confirm from the DRHP + citation |

### Group B — Inventory-led model (feeds the RL simulation, 06a)
| Figure | Current value | Tag | What I need |
|---|---|---|---|
| Blinkit inventory-led NOV share | 90% | [D] | Confirm + citation |
| Inventory-model margin benefit | "~60 bps (CFO-stated 50–70 bps)" | [D] ★ | Find the actual Swiggy/CFO statement and exact wording/range |
| **Dark-store build cost (capex per store)** | ₹3.5 cr/store (and ₹80 cr per 5pp transition) — **both unsourced** | [E] ★★ | Any real per-store build/fit-out capex for Blinkit/Instamart/Zepto, or marketplace→inventory transition cost |
| Swiggy QIP raise / Instamart earmark | ₹10,000 cr raised · ₹4,475 cr to Instamart | [D] | Confirm + citation |
| IOCC inventory-model vote | 72.35% in favour vs 75% needed (failed) | [D] ★ | Find the resolution result / filing |

### Group C — Cross-sell economics (feeds the uplift + Monte Carlo, 06b)
| Figure | Current value | Tag | What I need |
|---|---|---|---|
| **Food-delivery → quick-commerce cross-sell conversion** | assumed range **8–25%** | [E] ★★★ | THE single biggest unsourced assumption. Any benchmark for cross-app conversion in Indian super-apps / Swiggy disclosure of FD↔Instamart overlap or crossover |
| Instamart take rate | ~19.2% (attributed to JM Financial) | [E] ★ | Confirm/replace with a disclosed QC take rate |
| Swiggy food-delivery take rate | ~25.8% | [D] | Confirm + citation |
| Food-delivery orders per user / month | ~4.2 | [E] | Any disclosed FD frequency / MTU-to-orders figure |
| Food-delivery AOV | ~₹480 | [E] | Any disclosed FD AOV |
| Advertising revenue | Zepto ₹1,636cr · Blinkit ~₹1,000cr+ | [D]/[E] | Confirm Zepto (DRHP), find Blinkit ad revenue |

### Group D — Density economics (feeds system dynamics, 06c)
| Figure | Current value | Tag | What I need |
|---|---|---|---|
| QC store breakeven density | ~1,552 orders/store/day | [E] ★ | Any analyst estimate of orders/day a QC dark store needs to break even |
| Store maturation time | ~2 quarters (assumed) | [E] | Any benchmark for how long a new dark store takes to mature |
| New-store starting density | ~400 orders/day | [E] | Any benchmark for a freshly-opened store's day-one volume |
| Instamart capacity ceiling | "2,000+" orders/store/day (Swiggy-stated) | [D] | Find the Swiggy statement + citation |
| Blinkit mature-market EBITDA margin | Gurgaon/Noida 5.0% · Delhi NCR 3.5% | [D] | Confirm + citation |

### Group E — Competitive dynamics (feeds the synthesis, 06d)
| Figure | Current value | Tag | What I need |
|---|---|---|---|
| Flipkart Minutes | ~800 stores, adding ~100/month | [E] ★ | Real store count + growth rate + date |
| Amazon Now | target 100 cities, 1,000+ micro-fulfilment centres | [E] | Confirm the stated plan + citation |
| Any FY26 entrant share / scaling data | — | [E] | Useful colour on the closing competitive window |

## 4. The exact output format I need back

For **every** figure, return one block in this shape (a markdown table is ideal):

| Figure | Current value | Found value | Source name | URL | Date | Primary/Secondary | Verdict |
|---|---|---|---|---|---|---|---|
| (e.g.) Zepto orders/store/day | ~2,140 (estimate) | 1,890 | Zepto DRHP | https://… | 2026-0x-xx | Primary | CONTRADICTS |

Then three short sections:

1. **Key contradictions to address** — figures where your finding differs from the current value (most important).
2. **Still genuinely unsourced** — figures you could NOT find public data for (these legitimately stay as estimates).
3. **Suggested rows for `master_metrics.csv`** — formatted in this exact CSV schema so I can paste them in:
   `row_id,company,segment,period,period_type,metric,value,unit,yoy_pct,qoq_pct,confidence,source_id,notes`
   - `confidence` ∈ {`disclosed`, `analyst_estimate`, `derived`}
   - `source_id` = a short tag you assign (e.g. `S26`, `S27`) and define in the `notes`, with the URL in `notes`
   - leave `row_id` blank (I'll renumber)

## 5. Guardrails (please re-read before finishing)

- **Never fabricate a number or a citation.** If unsure, say "no public data found."
- Every figure needs a **URL + date**.
- Label **primary vs secondary** for each.
- Keep **quick-commerce** figures separate from **food-delivery** figures.
- Where you only find a *blended* number, say so explicitly.
- Prioritise the **★★ / ★★★** items (cross-sell conversion, dark-store capex, densities, market shares) — those drive the simulations hardest.

Thank you — the entire credibility of this project rests on traceable sourcing, so careful citations matter more than completeness.
