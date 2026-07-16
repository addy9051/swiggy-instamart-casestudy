# Swiggy Instamart Case Study — Improvement Brief for Claude Code

> **Source materials used:**
> - Fact-check research markdown (Q4 FY26 data from primary filings)
> - IIMA Consult Club Case Book 2025-26
> - Primary sources: Swiggy Q4 FY26 Shareholder Letter, Eternal Q4 FY26 Shareholder Letter, Zepto UDRHP
>
> **How to use this file:** Work through each section in order. Each section has a `WHAT TO DO` block with concrete file-level instructions, followed by the supporting rationale and exact numbers to use.

---

## Section 1 — Fix Data Errors (Priority: Critical)

These are direct contradictions with primary filings. Any interviewer or analyst who has read the Swiggy/Eternal shareholder letters will catch these immediately. Fix before anything else.

### 1.1 Orders per store per day — stale quarter

**WHAT TO DO:**
- Search the entire project for `1,025` or `1025` (the old OPD figure)
- Replace every instance with `1,093`
- Add a footnote or inline comment: `Q4 FY26 figure per Swiggy Shareholder Letter (May 2026); 1,025 was Q2 FY26`

**Why:** Swiggy's Q4 FY26 shareholder letter (primary source) discloses 1,093 orders/store/day. The 1,025 figure is from Q2 FY26 — two quarters stale. The metric actually dipped from 1,190 in Q4 FY25 as the network expanded rapidly, then partially recovered to 1,093 by Q4 FY26.

**Correct values by player (Q4 FY26):**
- Instamart: **1,093** OPD (primary — Swiggy letter)
- Blinkit: **~1,425** OPD (derived: 273.9M orders ÷ ~2,135 avg stores ÷ 90 days)
- Zepto: **~2,045** OPD (derived: 23.3 lakh/day ÷ 1,139 stores)

---

### 1.2 Dark-store capex per store — overstated by 3.5×

**WHAT TO DO:**
- Search for `3.5` or `3.5 crore` or `3.5cr` wherever dark-store capex is mentioned
- Replace with `~₹1 crore per store`
- Add explanatory note: `Blinkit disclosed ₹370 cr capex for 368 stores in H1 FY25 (~₹1 cr/store); implied ROCE ~40% at ₹26 cr annual NOV/store`
- If any payback period or capex recovery calculations exist, recalculate them using ₹1 cr, not ₹3.5 cr

**Why:** The ₹3.5 cr figure conflates three separate costs — setup capex (~₹20–50 lakh), multi-year lease commitments, and inventory working capital — into a single number. Zomato's actual filings show ₹370 cr for 368 stores. Using ₹3.5 cr overstates payback periods and makes Blinkit's economics look worse than they are.

**Correct setup cost breakdown to add:**
- Hard capex (fit-out, shelving, tech): **₹20–50 lakh per store**
- Total invested capital per store (including working capital): **~₹1 cr**
- Annual NOV per mature Blinkit store: **~₹26 cr**
- Implied ROCE at maturity: **~40%**

---

### 1.3 Inventory model margin benefit — wrong number, wrong company

**WHAT TO DO:**
- Search for `60 bps` wherever inventory model benefit is mentioned
- Replace with `~100 bps EBITDA accretion + ~300 bps gross-margin lift`
- **Critically:** Check whether this benefit is attributed to Swiggy/Instamart anywhere. It must NOT be. Re-label as a Blinkit/Eternal figure throughout.
- Add context: `Eternal CFO Akshant Goyal, Q2 FY26 earnings call: "overall margin accretion of ~1 percentage point"; gross margin already up ~300 bps by Q3 FY26`
- Add a separate note wherever Instamart's inventory model is discussed: `Instamart has NOT transitioned to inventory model — IOCC resolution failed at 72.36% (required 75%) in postal ballot closing May 20, 2026`

**Why:** Two separate errors here. First, the magnitude is wrong: the CFO stated ~1 ppt (100 bps), not 60 bps, of EBITDA accretion, with an additional ~300 bps of gross-margin improvement. Second, Swiggy attempted the same shift via an IOCC (Increase in Other than Cash Consideration) vote to restructure its foreign ownership and enable inventory purchasing — it failed by 2.64 percentage points. Attributing this benefit to Instamart is factually wrong and will immediately discredit the analysis.

---

### 1.4 Breakeven density — wrong benchmark

**WHAT TO DO:**
- Search for `1,552` wherever QC store breakeven density appears
- Replace with `~1,200–1,250 orders/store/day (mature metros)`
- Cite: `Redseer "Dark Store Blind Spot" report, 2026; PAN-India average OPD ~1,255 (Jan 2026), top-3 leaders ~1,350`
- Add context for non-metros: `non-metro breakeven band ~850 OPD (Redseer), reflecting lower competition and lower baseline demand`

**Why:** The 1,552 figure is significantly higher than what Redseer — the most cited independent QC analyst firm — publishes. Using an overstated breakeven makes stores look less viable than they are, distorting your investment thesis.

---

### 1.5 Swiggy food-delivery take rate — unconfirmed figure

**WHAT TO DO:**
- Search for `25.8%` wherever food delivery take rate is mentioned
- Replace with `~23%`
- Cite: `JM Financial Q4 FY26 preview; Motilal Oswal estimate 22.8%; Swiggy does not disclose a clean food take-rate in its shareholder letters`
- Do not present this as a primary/disclosed figure — label it as `analyst estimate`

**Why:** No primary source confirms 25.8%. The analyst consensus (JM Financial at ~23%, Motilal Oswal at 22.8%) clusters around 23%. Presenting an unconfirmed take rate as fact will get caught in any serious review.

---

### 1.6 Cross-sell conversion — label as assumption, not data

**WHAT TO DO:**
- Search for `8-25%` or `8% to 25%` wherever food-to-QC cross-sell conversion appears
- Do NOT delete the range — it's a valid modeling assumption
- Change the label from any implication of a sourced figure to: `[MODELED ASSUMPTION — no public disclosure]`
- Add the following note wherever cross-sell appears:
  > Both Swiggy management (Q2 FY26 call: "we won't be able to share specific numbers as these are sensitive") and Eternal (Deepinder Goyal: Zomato and Blinkit run as separate apps, "super brands not a super app") have explicitly declined to disclose cross-sell data. The nearest public proxy is BofA's Nov 2025 survey showing ~20% of users use both food-delivery apps — but that measures cross-app overlap within food delivery, not food-to-grocery conversion. Model with sensitivity bounds.

---

## Section 2 — Add IIMA Framework Structure

The case needs IIMA-standard analytical scaffolding. The casebook explicitly warns against "forcing frameworks" but rewards candidates who structure their analysis via a coherent issue tree. Add the following frameworks either as a new section or as structural headings within existing sections.

### 2.1 Add a Profitability Waterfall

**WHAT TO DO:**
- Create a revenue waterfall section (can be a table, a diagram, or a written breakdown) that walks Instamart's P&L from top to bottom using Q4 FY26 actuals:

```
Gross Order Value (GOV):    ₹7,881 cr  (+68.8% YoY)
 Less: discounts/returns
Net Order Value (NOV):      ₹5,675 cr  (+60.3% YoY, +4% QoQ)
 × Take rate (~19.5%):       ~₹1,107 cr platform revenue
 Less: delivery costs
 Less: dark store costs
Contribution Margin (CM):   -1.8% of NOV  (improving: Mar-26 was -1.1%)
 Less: central overhead
Adj. EBITDA:                Loss (exact not disclosed for Instamart standalone)
```

- Compare to Blinkit's Q4 FY26: NOV ₹14,386 cr, adj. EBITDA +₹37 cr (0.3% of NOV — its first profitable scaled quarter)
- Frame the gap: Instamart is ~2.5 ppt behind Blinkit at the CM level, and Blinkit is now EBITDA-positive while Instamart is still EBITDA-negative

**Why:** This is the core analytical structure for any profitability case per the IIMA casebook. "Revenues vs Costs" broken by unit → volume → mix is the expected starting point. Without this waterfall, the case lacks a spine.

---

### 2.2 Add Porter's Five Forces (QC-specific)

**WHAT TO DO:**
- Add a Porter's Five Forces section. Use the following pre-built analysis (sourced from IIMA casebook QC industry overview and primary data):

| Force | Intensity | Key drivers |
|---|---|---|
| Competitive rivalry | **Very High** | Blinkit (46% share), Zepto (22%), Instamart (24%), plus Flipkart Minutes, Amazon Now, JioMart all funded and expanding |
| Buyer bargaining power | **High** | Low switching costs; same SKUs, similar delivery times across platforms; users install multiple apps |
| Supplier bargaining power | **Low→Moderate** | Inventory-model players (Blinkit) negotiate directly with brands, reducing leverage; marketplace-model players (Instamart currently) face higher COGS |
| Threat of new entrants | **Moderate** | Model is replicable; dark store capex ~₹1 cr/store is not prohibitive for well-funded entrants; BUT achieving density + operational efficiency takes 12–18 months |
| Threat of substitutes | **Moderate** | Kirana stores, modern retail (DMart, Reliance), scheduled delivery (BigBasket), and traditional e-commerce all serve the same underlying need with different speed/price tradeoffs |

**Why:** The IIMA casebook's QC industry page explicitly maps these forces. Any consulting interview on this topic will expect a forces analysis. It also frames Instamart's structural disadvantage: high rivalry + low switching costs means market share is won purely on operational execution and unit economics.

---

### 2.3 Reframe the Problem Statement Around Market Share Loss

**WHAT TO DO:**
- Revise your case's problem statement section. The current framing (likely "how does Instamart grow?") should be refactored to lead with the share loss:

**Suggested problem statement:**
> Swiggy Instamart's order share among top-3 QC players has collapsed from 34.3% (FY24) to 20.9% (FY26) — a 13.4 percentage point loss in two years — despite 68.8% YoY GOV growth in Q4 FY26. The business is growing in absolute terms but losing the relative race. What structural changes does Instamart need to make to stop the share erosion and achieve EBITDA profitability?

- Source note to add inline: `Order share data from Zepto UDRHP (filed 2026) via Entrackr; Swiggy has publicly called Zepto's competitive data "baseless and unreliable" — present as competitor-sourced, not independently verified`

**Why:** The IIMA casebook's e-grocery case frames exactly this diagnostic — a business can show healthy absolute growth while losing relative position, and the case question should reflect that tension. Leading with share loss creates a sharper problem statement and sets up the structural analysis.

---

### 2.4 Add a Market Sizing Section

**WHAT TO DO:**
- Add or expand a market sizing section with the following structure:

```
Indian QC market:
  GMV FY24:        ~USD 3 billion
  Projected FY30:  ~USD 40 billion (CAGR ~45%)
  
Addressable users (Tier-1 cities):
  India urban population:       ~500M
  Tier-1 city population:       ~120M  
  QC-relevant demographic       
  (18–45, smartphone, disposable income): ~30–35M households
  Current penetration (MTU):
    Blinkit: 27.2M, Instamart: 13.3M, Zepto: ~20M+
    (significant overlap; total unduplicated ~35–40M active users)

Instamart's share of this:
  MTU: 13.3M  →  opportunity to reach 25–30M with store density + cross-sell
  
Dark store density math:
  Instamart: 1,143 stores / ~129 cities = 8.9 stores/city average
  Blinkit:   2,243 stores = ~17.4 stores/city average
  Gap: ~2× store density → primary driver of OPD and MTU gap
```

---

## Section 3 — Build a Per-Store Unit Economics Model

This is the quantitative core that most case studies treat superficially. Add or significantly expand a unit economics section using the following structure.

### 3.1 Per-Store P&L at Different Maturity Stages

**WHAT TO DO:**
- Add a table or model showing per-store economics at three stages:

| Metric | Day 1 | Month 6 (breakeven) | Mature (12–18 mo) |
|---|---|---|---|
| Orders/store/day | ~400 | ~1,200–1,250 | 1,500–2,000+ |
| Gross AOV | ₹700 | ₹700 | ₹700+ |
| Daily GOV/store | ₹2.8 lakh | ₹8.75 lakh | ₹10.5–14 lakh |
| Annual GOV/store | ₹10.2 cr | ₹31.9 cr | ₹38–51 cr |
| Annual NOV/store (~72% of GOV) | ₹7.4 cr | ₹23 cr | ₹27–37 cr |
| Contribution Margin | Deeply negative | ~0% | +3.5–5% of NOV |
| Annual CM at maturity | — | — | ₹0.95–1.85 cr |
| Capex invested | ₹1 cr | ₹1 cr | ₹1 cr |
| Payback period | — | — | ~7–13 months post-maturity |

- Source notes: AOV from Swiggy Q4 FY26 letter; breakeven OPD from Redseer; mature CM % from Eternal Q3 FY26 (Delhi NCR 3.5%, Gurgaon/Noida 5%); capex from Blinkit/Zomato filings

---

### 3.2 Store Ramp Timeline — Add to Model

**WHAT TO DO:**
- Add a ramp curve description or chart showing:
  - **Day 1:** ~400 OPD (no precise disclosure; industry anecdote range 150–500)
  - **Month 3–6:** Mature density in Tier-1 (Blinkit: 3–6 months; Bernstein estimate)
  - **Month 6–12:** Mature density in Tier-2/non-metro (Bernstein: 6–12 months)
  - **Network average today:** Instamart 1,093 OPD — below breakeven (~1,225) at network level, meaning the majority of the 1,143-store fleet is still loss-making

**Implication to add explicitly:**
> At 1,093 OPD network average vs. 1,200–1,250 breakeven, Instamart's network is not yet contribution-margin neutral in aggregate. Reaching breakeven requires either: (a) increasing orders on existing stores to ~1,225 OPD, or (b) waiting for newer stores (the ~316 added in Q4 FY25 and 7 in Q4 FY26) to ramp up naturally. The ₹4,475 cr QIP earmark for QC adds more stores, which temporarily pushes the network OPD average down before it recovers.

---

### 3.3 Advertising Revenue as a Margin Lever

**WHAT TO DO:**
- Add a section on QC advertising revenue as a margin improvement path:

```
Zepto ad revenue FY26:  ₹1,636 cr (+151% YoY)
  As % of NRV:          ~7.8%
  Brand partners:       2,468

Instamart ad revenue:   Not disclosed
  Estimated:            Likely ₹300–500 cr range (analyst inference)
  Opportunity:          If Instamart reaches Zepto's 7.8% ad-take on NOV
                        → ₹5,675 cr NOV × 7.8% = ₹443 cr in ad revenue
                        → equivalent to ~7.8 ppt improvement in effective take rate

Blinkit ad revenue:     Not disclosed by Eternal
  Management quote:     "Ad revenue as % of NOV is higher than food delivery"
                        (no absolute number given)
```

- Label Instamart estimate clearly as derived/analyst inference, not a disclosed figure

---

## Section 4 — Strengthen Competitive Positioning

### 4.1 Diagnose the Three Root Causes of Share Loss

**WHAT TO DO:**
- Add an explicit root-cause section with three distinct structural explanations. Do not conflate them.

**Root Cause 1 — Business model gap (most important):**
> Blinkit operates ~90% of NOV on its own inventory (Q3 FY26, Eternal letter). This gives it direct brand procurement, ~300 bps better gross margins, and better in-stock rates. Instamart remains marketplace-model. The IOCC vote that would have enabled Instamart's transition failed at 72.36% (needed 75%) — a 2.64 ppt gap. Until this is resolved, Instamart structurally cannot match Blinkit's unit economics.

**Root Cause 2 — Store density gap:**
> Blinkit has 2,243 stores vs Instamart's 1,143 — roughly 2× the footprint. This drives: (a) better delivery coverage (fewer dead zones), (b) higher OPD as stores become familiar in their catchments, (c) more data for inventory optimization. The density gap compounds over time.

**Root Cause 3 — User flywheel gap:**
> Blinkit MTU is 27.2M vs Instamart's 13.3M. Blinkit's higher OPD (~1,425) relative to Instamart (~1,093) suggests higher repeat purchase frequency per user, not just more users. This creates a compounding advantage: more orders → better inventory freshness → better in-stock rates → more orders.

---

### 4.2 Add Entrant Threat Quantification

**WHAT TO DO:**
- Add a competitive entrant section with current (Q1 CY26) data:

| Entrant | Stores | Scale | Threat Level |
|---|---|---|---|
| Flipkart Minutes | ~800 stores | Adding ~100/month; target 1,200–1,500 by end-2026 | High — Walmart/Flipkart's distribution + deep pockets |
| Amazon Now | ~1,000+ MFCs | Expanding to 100 cities; ₹2,800 cr investment | High — Prime bundle cross-sell |
| JioMart | ~800 dark stores | 1.6M orders/day (+53% QoQ); claims CM-positive via Reliance sourcing | Medium-High — FMCG sourcing advantage |
| BigBasket (Tata) | — | 5–7% market share | Medium — stronger in scheduled delivery |

- Source: Inc42 (Amazon/Flipkart), Reliance Q3 FY26 investor presentation (JioMart), Datum Intelligence via Reuters (BigBasket share)

---

### 4.3 Add Zepto's Structural Advantage

**WHAT TO DO:**
- Add a note on Zepto as the operational benchmark, not just a share competitor:

> Zepto at ~2,045 OPD (Q4 FY26, derived) is the density leader — 87% above Instamart's 1,093 OPD and 43% above Blinkit's ~1,425 OPD. Zepto's EDLP (everyday low price) model and lower AOV (~₹387 implied vs Instamart's ₹700 gross) means it serves higher-frequency, lower-basket users — a different demand segment. Zepto's FY26 adj. EBITDA loss per order (₹78.75, UDRHP-disclosed) is higher than Blinkit's (₹3.02) but declining rapidly from ₹136.15 in FY25. Its ad revenue model (₹1,636 cr, 2,468 brand partners) is the most advanced in the sector.

---

## Section 5 — Add Three-Horizon Recommendation Structure

The IIMA casebook's growth and DD cases consistently reward time-bound, tiered recommendations. Replace any single-point recommendation with this structure.

### 5.1 Now (0–6 months) — Stabilize and Redeploy Capital

**WHAT TO DO:**
- Add the following as your immediate recommendations section:

1. **Recalibrate the capex model** using ₹1 cr/store (not ₹3.5 cr). This changes the math on the ₹4,475 cr QIP earmark: at ₹1 cr/store capex, Swiggy can fund ~4,475 incremental stores in theory — but the binding constraint is operational capacity (the network grew by only 7 net stores in Q4 FY26 despite the QIP close). The real bottleneck is site identification, staff hiring, and supply chain setup, not capital.

2. **Prioritize OPD improvement on existing stores** over new store additions in the near term. Each OPD point gained on 1,143 existing stores is higher-ROI than opening stores that start at ~400 OPD and take 3–6 months to ramp. Target: push network OPD from 1,093 to the 1,200–1,250 breakeven band.

3. **Fix the contribution margin trajectory.** Mar-26 monthly CM was -1.1% vs Q4 FY26 average of -1.8% — the trend is improving. Model what gets Instamart to CM breakeven and by when.

---

### 5.2 Next (6–18 months) — Structural Fixes

**WHAT TO DO:**
- Add the following as your medium-term recommendations:

1. **Re-run the IOCC vote.** The 72.36% approval missed 75% by 2.64 ppt. A targeted shareholder engagement campaign (particularly with the FII/DII base that opposed it) could close this gap. The inventory model transition is the only structural path to closing the 300 bps gross-margin gap with Blinkit — no amount of operational improvement achieves this without it.

2. **Build the ad revenue platform.** Zepto's ₹1,636 cr advertising revenue (7.8% of NRV) with 2,468 brand partners shows the monetization ceiling. Instamart's current ad-take is undisclosed but likely significantly lower. A structured brand-partnership program — now viable given the QIP-funded store expansion — could add 3–5 ppt to effective margins with minimal incremental cost.

3. **Model cross-sell explicitly but conservatively.** Swiggy's food-delivery MTU is 18.3M vs Instamart's QC MTU of 13.3M — an overlap gap of at least 5M users who use Swiggy for food but not yet for grocery. Model this as a funnel: if 20–30% of incremental food users can be converted to QC (using the BofA cross-app usage proxy as a ceiling), that's 1–1.5M incremental QC MTUs. Label as sensitivity analysis.

---

### 5.3 Later (18 months+) — Growth and Monetization

**WHAT TO DO:**
- Add the following as long-term strategic options:

1. **Tier-2 expansion thesis.** Per the IIMA QC fulfillment acquisition case, Tier-2 cities are still largely untapped by the top-3 players (most operate in ~30–130 cities). The non-metro store maturation timeline is 6–12 months (vs 3–6 for metros), so capex-to-cash payback is longer — model separately. First-mover advantage in Tier-2 is significant (the casebook's T2 fulfillment case cites 80% city overlap as the key acquisition trigger).

2. **Private label / in-house brands.** The IIMA casebook identifies private labels as a QC revenue driver. At scale, in-house brands can add 10–15 ppt of gross margin on relevant SKU categories. Requires inventory model first (see 5.2, point 1).

3. **Management's own target to anchor your projection:** Swiggy has guided for >₹1 lakh cr NOV with 4–5% EBITDA margin as a medium-term target. Model what store count, OPD, and AOV are required to achieve this, and what the implied timeline is given current ramp rates.

---

## Section 6 — Data Table: Correct Metrics Reference

Use this table as the single source of truth for all numbers in the project. Every figure listed as "primary" comes from official shareholder letters or DRHP filings.

```
METRIC                          VALUE           SOURCE              CONFIDENCE
─────────────────────────────────────────────────────────────────────────────
Instamart dark stores (Q4 FY26) 1,143           Swiggy Q4 FY26 letter   PRIMARY
Blinkit dark stores (Q4 FY26)   2,243           Eternal Q4 FY26 letter  PRIMARY
Zepto dark stores (Q4 FY26)     1,139           Zepto UDRHP             PRIMARY

Instamart OPD (Q4 FY26)         1,093           Swiggy Q4 FY26 letter   PRIMARY
Blinkit OPD (Q4 FY26)           ~1,425          Derived from Eternal    DERIVED
Zepto OPD (Q4 FY26)             ~2,045          Derived from Zepto UDRHP DERIVED

Instamart AOV gross (Q4 FY26)   ₹700            Swiggy Q4 FY26 letter   PRIMARY
Instamart AOV net (Q4 FY26)     ~₹485           Swiggy Q4 FY26 letter   PRIMARY
Blinkit AOV net (Q4 FY26)       ₹525            Eternal Q4 FY26 letter  PRIMARY
Zepto AOV implied (Q4 FY26)     ~₹387           Derived from UDRHP      DERIVED

Instamart MTU (Q4 FY26)         13.3M           Swiggy Q4 FY26 letter   PRIMARY
Blinkit MTU (Q4 FY26)           27.2M           Eternal Q4 FY26 letter  PRIMARY
Swiggy food MTU (Q4 FY26)       18.3M           Swiggy Q4 FY26 letter   PRIMARY

Instamart GOV (Q4 FY26)         ₹7,881 cr       Swiggy Q4 FY26 letter   PRIMARY
Instamart NOV (Q4 FY26)         ₹5,675 cr       Swiggy Q4 FY26 letter   PRIMARY
Instamart CM (Q4 FY26)          -1.8% of NOV    Swiggy Q4 FY26 letter   PRIMARY
Blinkit NOV (Q4 FY26)           ₹14,386 cr      Eternal Q4 FY26 letter  PRIMARY
Blinkit adj. EBITDA (Q4 FY26)   +₹37 cr (0.3%) Eternal Q4 FY26 letter  PRIMARY

Instamart order share (FY24)     34.3%           Zepto UDRHP*            PRIMARY*
Instamart order share (FY25)     24.5%           Zepto UDRHP*            PRIMARY*
Instamart order share (FY26)     20.9%           Zepto UDRHP*            PRIMARY*
(* competitor-sourced; Swiggy disputes this data)

QC market share (Jan 2026):
  Blinkit:   46%                Datum Intelligence/Reuters  SECONDARY
  Instamart: 24%                Datum Intelligence/Reuters  SECONDARY
  Zepto:     22%                Datum Intelligence/Reuters  SECONDARY

Loss per order FY26:
  Instamart: ₹85.18             INDmoney from filings       DERIVED
  Blinkit:   ₹3.02              INDmoney from filings       DERIVED
  Zepto:     ₹78.75             Zepto UDRHP                 PRIMARY

Instamart take rate (Q4 FY26)   ~19.5% on NOV               JM Financial est  ANALYST
Swiggy food take rate (Q4 FY26) ~23% on GOV                 JM/Motilal est    ANALYST
Instamart take rate (alt)        ~12.8% on GOV               Motilal Oswal     ANALYST

QC breakeven density             ~1,200–1,250 OPD            Redseer 2026      SECONDARY
Dark store capex/store           ~₹1 cr                      Blinkit filings   DERIVED
Store maturation (Tier-1)        3–6 months                  Bernstein         ANALYST
Store maturation (Tier-2)        6–12 months                 Bernstein         ANALYST
Dark store capacity ceiling      2,000+ OPD (megapods higher) Swiggy Q2 FY26   PRIMARY

Blinkit inventory model (Q3 FY26) ~90% of NOV own inventory  Eternal letter    PRIMARY
Inventory EBITDA accretion       ~100 bps (Blinkit only)     CFO Q2 FY26 call  PRIMARY
Inventory GM accretion           ~300 bps (Blinkit only)     CFO Q2 FY26 call  PRIMARY
IOCC vote result (May 2026)      72.36% (needed 75%)         SEBI filing       PRIMARY

Swiggy QIP raised (FY26)         ₹10,000 cr                  Regulatory filing PRIMARY
QIP earmark for QC               ₹4,475 cr                   Regulatory filing PRIMARY

Zepto ad revenue (FY26)          ₹1,636 cr (+151% YoY)       Zepto UDRHP       PRIMARY
Zepto ad brand partners          2,468                        Zepto UDRHP       PRIMARY
Blinkit ad revenue               Not disclosed                —                 N/A

Swiggy platform frequency (Q4)   4.01 orders/user/month      Swiggy Q4 FY26    PRIMARY
(platform-wide, not food-only; 4.22 was Q4 FY25)

Cross-sell conversion (FD→QC)    8–25% [MODELED — no source] —                 ASSUMPTION
```

---

## Section 7 — Source Citations to Add Throughout

When Claude Code adds or modifies any section, use these citations consistently:

| Source | Citation format |
|---|---|
| Swiggy Q4 FY26 shareholder letter | `Swiggy Q4 FY26 Shareholder Letter (May 2026)` |
| Eternal Q4 FY26 shareholder letter | `Eternal Q4 FY26 Shareholder Letter (Apr 2026)` |
| Zepto UDRHP | `Zepto UDRHP (2026)` |
| Redseer breakeven | `Redseer "Dark Store Blind Spot" (2026)` |
| CFO inventory call | `Eternal CFO Akshant Goyal, Q2 FY26 earnings call (Oct 2025)` |
| Market share (Datum) | `Datum Intelligence (founder Satish Meena) via Reuters (Jan 2026) — secondary estimate, not company disclosure` |
| JM Financial take rate | `JM Financial Q4 FY26 preview (May 2026) — analyst estimate` |
| Blinkit capex | `Derived: Zomato H1 FY25 capex filing (₹370 cr / 368 stores)` |
| IOCC vote | `Swiggy SEBI postal ballot filing, closing May 20 2026` |
| Bernstein maturation | `Bernstein research (via Storyboard18) — analyst estimate` |
| Zepto order share | `Zepto UDRHP (competitor-sourced; Swiggy disputes accuracy)` |

---

## Quick Checklist for Claude Code

Use this as a task list when working through the project:

- [ ] Replace all instances of `1,025` OPD with `1,093` (Q4 FY26)
- [ ] Replace all instances of `₹3.5 cr` dark-store capex with `~₹1 cr`
- [ ] Replace `60 bps` inventory benefit with `~100 bps EBITDA + 300 bps GM`
- [ ] Remove any attribution of inventory model benefit to Swiggy/Instamart
- [ ] Add IOCC vote failure note wherever Instamart's inventory model is mentioned
- [ ] Replace `1,552` breakeven density with `~1,200–1,250`
- [ ] Replace `25.8%` food take rate with `~23%` and label as analyst estimate
- [ ] Label cross-sell conversion (8–25%) as `[MODELED ASSUMPTION]` everywhere
- [ ] Add profitability waterfall (GOV → NOV → CM → EBITDA) for Instamart vs Blinkit
- [ ] Add Porter's Five Forces table (QC-specific)
- [ ] Reframe problem statement around 34.3% → 20.9% order share loss
- [ ] Add per-store unit economics table (Day 1 / Breakeven / Mature stages)
- [ ] Add store ramp timeline with breakeven implications
- [ ] Add advertising revenue as margin lever (Zepto ₹1,636 cr benchmark)
- [ ] Add root-cause analysis: model gap + density gap + flywheel gap
- [ ] Add competitive entrant table (Flipkart Minutes, Amazon Now, JioMart)
- [ ] Replace single-point recommendation with 3-horizon structure
- [ ] Add metric confidence labels (PRIMARY / DERIVED / ANALYST / ASSUMPTION)
- [ ] Ensure all numbers match the Section 6 data table

---

*Generated from: Swiggy Q4 FY26 Shareholder Letter, Eternal Q4 FY26 Shareholder Letter, Zepto UDRHP, IIMA Consult Club Case Book 2025-26, Redseer QC research, JM Financial / Motilal Oswal analyst estimates, Datum Intelligence via Reuters.*
