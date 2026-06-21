"""
Generates 01_master_data.ipynb using nbformat.
Run with: python3 build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md(r"""# 01 — Master Data Notebook
### Swiggy Instamart Strategy Case Study — Why Instamart's Path to Profitability Lags Blinkit

**Purpose of this notebook**

Every other notebook in this case study (unit economics & density, customer behavior &
monetization, competitive market structure, logistics) reads from the three files this
notebook produces. Building the dataset once here, instead of inside each branch notebook,
means there is exactly one place to fix an error and exactly one place to add a new quarter
of data as it gets disclosed.

**Design decision: long ("tidy") format, not wide**

Swiggy, Blinkit (Eternal), and Zepto report different metrics on different bases (GOV vs.
NOV vs. NRV vs. Revenue), at different periodicities (some figures are quarterly, some are
only disclosed annually, some are one-off analyst estimates with no clean quarter attached),
and with different confidence levels (company-disclosed vs. analyst-estimated vs. derived
by back-calculation). Forcing this into one wide table (rows = quarters, columns = metrics)
would mean either fabricating numbers to fill gaps or littering the table with `NaN`. A long
table — one row per (company, segment, period, metric) observation — handles this
heterogeneity cleanly and stays easy to extend.

**Outputs written to `data/processed/`**
- `sources.csv` — every citation used, with a short `source_id` referenced elsewhere
- `data_dictionary.csv` — what each metric means and where definitions are *not* comparable across companies
- `master_metrics.csv` — the actual long-format dataset, ~80 observations as of this build

**Honesty flag up front:** this dataset is built entirely from what was already gathered in
the research phase (shareholder letters, earnings call coverage, analyst notes cited in the
issue-tree report). It is a *starting scaffold*, not a complete quarterly time series. Section
6 at the bottom lists exactly what is missing and where to go get it.
""")

code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 140)

RAW = Path("../data/raw")
PROCESSED = Path("../data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
""")

# ---------------------------------------------------------------------------
md(r"""## 1. Source log

Every number in `master_metrics.csv` traces back to a `source_id` here. Where the original
filing or shareholder letter URL is known, it's used directly. Where only secondary coverage
(a news site or analyst-note summary) was available, that's marked honestly in
`primary_or_secondary` rather than implied to be a primary filing.
""")

code(r"""
sources = [
    dict(source_id="S01", org="Swiggy", title="Q2 FY2026 Shareholder Letter",
         url="https://www.swiggy.com/corporate/wp-content/uploads/2025/10/Q2-FY2026-Shareholder-letter.pdf",
         primary_or_secondary="primary"),
    dict(source_id="S02", org="Swiggy", title="Investor Relations — Financial Results hub",
         url="https://www.swiggy.com/corporate/investor-relations/financial-results/",
         primary_or_secondary="primary"),
    dict(source_id="S03", org="Groww", title="Swiggy Q4 FY26 Results blog",
         url="https://groww.in/blog/swiggy-q4-fy26-results", primary_or_secondary="secondary"),
    dict(source_id="S04", org="Storyboard18", title="Swiggy Instamart expands to 1,143 dark stores, Q4 FY26",
         url="https://www.storyboard18.com/brand-marketing/swiggy-instamart-gov-jumps-69-percent-as-losses-narrow-in-q4-fy26-ws-l-97590.htm",
         primary_or_secondary="secondary"),
    dict(source_id="S05", org="INDmoney", title="Swiggy Q4 FY26: Revenue up 45%, stock 44% below peak",
         url="https://www.indmoney.com/blog/stocks/swiggy-q4-fy26-results-stock-falling-explained",
         primary_or_secondary="secondary"),
    dict(source_id="S06", org="Whalesbook", title="Swiggy's Instamart Prioritizes Profit Over Growth",
         url="https://www.whalesbook.com/news/English/consumer-products/Swiggys-Instamart-Prioritizes-Profit-Over-Growth-Amid-Competition/6a014780707d23e8442d3efa",
         primary_or_secondary="secondary"),
    dict(source_id="S07", org="Whalesbook", title="Swiggy Q4: Revenue jumps 45%, Instamart losses mount; Jefferies cuts target",
         url="https://www.whalesbook.com/news/English/tech/Swiggy-Q4-Revenue-Jumps-45percent-Instamart-Losses-Mount-Jefferies-Cuts-Target/6a0433139444064ed0161b9d",
         primary_or_secondary="secondary"),
    dict(source_id="S08", org="MarketsMojo", title="Swiggy Q4 FY26: losses narrow, path to profitability uncertain",
         url="https://www.marketsmojo.com/news/result-analysis/swiggy-q4-fy26-losses-narrow-sharply-as-revenue-momentum-builds-but-path-to-profitability-remains-uncertain-3982428",
         primary_or_secondary="secondary"),
    dict(source_id="S09", org="Multibagg", title="Swiggy Q4 Results FY26: loss narrows, revenue +45%",
         url="https://www.multibagg.ai/market-pulse/articles/swiggy-q4-fy26-loss-revenue-cmowttr5q01tmmn0j134e33zk",
         primary_or_secondary="secondary"),
    dict(source_id="S10", org="Univest", title="Swiggy Q4 results FY26 loss 800cr revenue 6383cr",
         url="https://univest.in/blogs/swiggy-q4-results-fy26-loss-800-crore-revenue-6383-crore-2",
         primary_or_secondary="secondary"),
    dict(source_id="S11", org="Alpha Spread", title="Swiggy Ltd investor relations summary",
         url="https://www.alphaspread.com/security/nse/swiggy/investor-relations", primary_or_secondary="secondary"),
    dict(source_id="S12", org="Screener.in", title="Swiggy Ltd company page (consolidated financials)",
         url="https://www.screener.in/company/SWIGGY/consolidated/", primary_or_secondary="secondary"),
    dict(source_id="S13", org="Inc42", title="Eternal Q4: Blinkit's adjusted EBITDA jumps 9x QoQ to ₹37 Cr",
         url="https://inc42.com/buzz/eternal-q4-blinkits-adjusted-ebitda-jumps-9x-qoq-to-%E2%82%B937-cr/",
         primary_or_secondary="secondary"),
    dict(source_id="S14", org="Sahi", title="Eternal Q4 FY26: Profit +346%, Blinkit EBITDA positive for first time",
         url="https://www.sahi.com/blogs/eternal-limited-zomato-parent-q4-fy26-results-analysis",
         primary_or_secondary="secondary"),
    dict(source_id="S15", org="Quash", title="Blinkit in 2026: Revenue, Market Share, Profitability, What Comes Next",
         url="https://quashbugs.com/blog/blinkit-surpasses-zomato-in-quick-commerce", primary_or_secondary="secondary"),
    dict(source_id="S16", org="BSE India", title="Swiggy corporate presentation (exchange filing)",
         url="https://www.bseindia.com/xml-data/corpfiling/AttachHIS/5bec0774-d901-48d2-b848-12c8a37842a8.pdf",
         primary_or_secondary="primary"),
    dict(source_id="S17", org="Trendlyne", title="Swiggy Ltd. investor presentation mirror",
         url="https://trendlyne.com/investor-presentations/2755533/SWIGGY/swiggy-ltd/", primary_or_secondary="secondary"),
    dict(source_id="S18", org="TechCrunch", title="Zomato raises $1B ahead of rival Swiggy India IPO",
         url="https://techcrunch.com/2024/10/22/zomato-to-raise-1b-ahead-of-rival-swiggy-india-ipo",
         primary_or_secondary="secondary"),
    dict(source_id="S19", org="Zepto", title="Zepto UDRHP (cited via secondary analyst/press coverage — exact filing URL not yet sourced)",
         url="UNCONFIRMED — locate on SEBI/exchange filings or Zepto IPO press kit", primary_or_secondary="secondary"),
    dict(source_id="S20", org="Nomura", title="Quick commerce density research note (cited via secondary press coverage)",
         url="UNCONFIRMED — original Nomura note not directly accessed", primary_or_secondary="secondary"),
    dict(source_id="S21", org="Bernstein", title="Quick commerce dark-store profitability research note (cited via secondary press coverage)",
         url="UNCONFIRMED — original Bernstein note not directly accessed", primary_or_secondary="secondary"),
    dict(source_id="S22", org="Jefferies", title="Swiggy/Instamart earnings estimate revision note (cited via Whalesbook)",
         url="UNCONFIRMED — original Jefferies note not directly accessed", primary_or_secondary="secondary"),
    dict(source_id="S23", org="Datum Intelligence / Reuters", title="Quick commerce market share snapshot, Jan 2026",
         url="UNCONFIRMED — cited via secondary press coverage (e.g. startupfeed.in)", primary_or_secondary="secondary"),
    dict(source_id="S24", org="JM Financial", title="Take-rate research note (cited via secondary press coverage)",
         url="UNCONFIRMED — original JM Financial note not directly accessed", primary_or_secondary="secondary"),
]

sources_df = pd.DataFrame(sources)
sources_df.to_csv(PROCESSED / "sources.csv", index=False)
print(f"sources.csv written — {len(sources_df)} sources")
sources_df
""")

# ---------------------------------------------------------------------------
md(r"""## 2. Data dictionary — and where comparability breaks

This is the single most important file for not embarrassing yourself in front of an MBB
recruiter. Eternal/Blinkit reports on an **inventory model** (it owns the goods it sells, so
its topline "Revenue" includes the full retail price). Swiggy Instamart and Zepto run closer
to a **marketplace/managed-marketplace model** for most of their volume, so their topline
revenue numbers are not on the same basis as Blinkit's. **GOV (Gross Order Value), NOV (Net
Order Value), and NRV (Net Revenue Value)** are three *different* metrics used by three
different companies as their headline "size of business" number — they are related but not
identical, and treating them as interchangeable is the single easiest mistake to make with
this dataset.
""")

code(r"""
data_dictionary = [
    dict(metric="GOV", full_name="Gross Order Value", definition="Total value of orders transacted on the platform before deductions (discounts, etc.)", comparability_note="Used by Swiggy. Roughly comparable to Blinkit's GOV in early disclosures, but Blinkit has shifted emphasis to NOV."),
    dict(metric="NOV", full_name="Net Order Value", definition="GOV net of customer-side discounts and cancellations", comparability_note="Used by both Swiggy Instamart and Blinkit in recent quarters — the more comparable cross-company metric where both report it."),
    dict(metric="NRV", full_name="Net Revenue Value", definition="Zepto's reported topline order-value metric", comparability_note="Conceptually closest to NOV but Zepto's exact deduction methodology is not fully disclosed publicly — treat NRV vs NOV comparisons as directionally indicative only."),
    dict(metric="Revenue from Operations", full_name="Statutory revenue", definition="Revenue recognized under accounting standards", comparability_note="NOT comparable across Blinkit and Swiggy/Zepto: Blinkit's inventory-led model means it recognizes the full retail sale as revenue, inflating it relative to marketplace-model peers. Do not chart Blinkit revenue next to Swiggy/Zepto revenue without this caveat."),
    dict(metric="Contribution Margin", full_name="Contribution margin (% of order value)", definition="Order economics after variable costs (delivery, packaging, etc.), before fixed overhead", comparability_note="Disclosed basis (GOV vs NOV) not always explicit in secondary coverage — verify against the actual KPI databook before using in a precise comparison."),
    dict(metric="Adjusted EBITDA Margin", full_name="Adjusted EBITDA as % of order value", definition="Company-defined non-GAAP profitability measure", comparability_note="Each company defines its own adjustments — 'Adjusted EBITDA' is not standardized across Swiggy, Blinkit, and Zepto."),
    dict(metric="AOV", full_name="Average Order Value", definition="Order value divided by number of orders", comparability_note="Comparable in spirit across companies, though the order-value basis (GOV/NOV/NRV) underneath differs."),
    dict(metric="Per Order Loss", full_name="Loss per order", definition="Total segment loss divided by order count", comparability_note="Useful cross-company comparator since it normalizes for scale — one of the cleaner comparable metrics in this dataset."),
    dict(metric="Orders per Store per Day", full_name="Daily order throughput per dark store", definition="Total orders / store count / days in period", comparability_note="Best available proxy for store-level density and capacity utilization — but watch for store age mix (a network adding many new, immature stores will show lower average throughput even if mature stores are at capacity)."),
    dict(metric="Take Rate", full_name="Platform monetization rate", definition="Adjusted revenue as % of GOV/NOV", comparability_note="Definition basis varies; treat trend direction within a company as more reliable than cross-company level comparison."),
]

data_dictionary_df = pd.DataFrame(data_dictionary)
data_dictionary_df.to_csv(PROCESSED / "data_dictionary.csv", index=False)
print(f"data_dictionary.csv written — {len(data_dictionary_df)} metric definitions")
data_dictionary_df
""")

# ---------------------------------------------------------------------------
md(r"""## 3. Master metrics table

Schema: `company | segment | period | period_type | metric | value | unit | yoy_pct | qoq_pct
| confidence | source_id | notes`

- **`period_type`** is one of `quarterly`, `annual`, `monthly`, or `snapshot` (a one-off
  point-in-time estimate, e.g. a market-share figure as of a specific month).
- **`confidence`** is one of `disclosed` (company-reported), `analyst_estimate` (a
  third-party research estimate), or `derived` (back-calculated by us from two disclosed
  figures — e.g. computing last quarter's margin from "improved 65 bps QoQ to X").
  `derived` rows say exactly how, in `notes`.
""")

code(r"""
rows = []

def add(company, segment, period, period_type, metric, value, unit,
        yoy_pct=None, qoq_pct=None, confidence="disclosed", source_id=None, notes=""):
    rows.append(dict(company=company, segment=segment, period=period, period_type=period_type,
                      metric=metric, value=value, unit=unit, yoy_pct=yoy_pct, qoq_pct=qoq_pct,
                      confidence=confidence, source_id=source_id, notes=notes))

# ---- Swiggy Instamart (Quick Commerce segment) -----------------------------
SI, SEG_QC = "Swiggy Instamart", "Quick Commerce"
add(SI, SEG_QC, "Q4FY25", "quarterly", "GOV", 4670, "INR_crore", source_id="S06")
add(SI, SEG_QC, "Q3FY26", "quarterly", "GOV", 7938, "INR_crore", source_id="S06",
    notes="Implied by reported sequential dip to 7881 in Q4FY26 (first-ever sequential GOV decline)")
add(SI, SEG_QC, "Q4FY26", "quarterly", "GOV", 7881, "INR_crore", yoy_pct=68.8, qoq_pct=-0.7, source_id="S06,S04")
add(SI, SEG_QC, "Q4FY26", "quarterly", "NOV", 5675, "INR_crore", yoy_pct=60.3, qoq_pct=4, source_id="S03,S04")
add(SI, SEG_QC, "Q4FY26", "quarterly", "Contribution Margin", -1.8, "percent", qoq_pct=None,
    source_id="S03,S04", notes="Improved 65 bps sequentially from Q3FY26; basis (GOV vs NOV) not explicit in secondary coverage — verify in KPI databook")
add(SI, SEG_QC, "Mar2026", "monthly", "Contribution Margin", -1.1, "percent", source_id="S03,S04")
add(SI, SEG_QC, "Q3FY26", "quarterly", "Contribution Margin", -2.45, "percent", confidence="derived",
    notes="Back-calculated: Q4FY26 value (-1.8%) minus the stated +65bps sequential improvement")
add(SI, SEG_QC, "Q4FY26", "quarterly", "Adjusted EBITDA", -858, "INR_crore", source_id="S07")
add(SI, SEG_QC, "Q4FY26", "quarterly", "Adjusted EBITDA Margin", -10.9, "percent", source_id="S03")
add(SI, SEG_QC, "Q3FY26", "quarterly", "Adjusted EBITDA Margin", -11.4, "percent", source_id="S03")
add(SI, SEG_QC, "Q4FY26", "quarterly", "AOV", 700, "INR", yoy_pct=32.8, source_id="S03,S04,S06")
add(SI, SEG_QC, "Q4FY26", "quarterly", "Dark Stores (count)", 1143, "count", source_id="S03,S04")
add(SI, SEG_QC, "Q4FY26", "quarterly", "Net Dark Store Adds", 7, "count", source_id="S04")
add(SI, SEG_QC, "Q4FY26", "quarterly", "Cities Covered", 129, "count", source_id="S04")
add(SI, SEG_QC, "Q4FY26", "quarterly", "Warehousing Area", 4.8, "million_sqft", yoy_pct=21.1, source_id="S04")
add(SI, SEG_QC, "Q2FY26", "quarterly", "Orders per Store per Day", 1025, "orders", qoq_pct=4, source_id="S01")
add(SI, SEG_QC, "Q4FY26", "quarterly", "MTU", 13.3, "million", confidence="analyst_estimate",
    notes="Cited in competitive-context analysis; cross-check against Swiggy's own MTU disclosure (Swiggy reports a consolidated/food-delivery MTU figure more prominently)")
add(SI, SEG_QC, "Q2FY26", "quarterly", "Store Profitability Share", 25, "percent_of_stores", source_id="S01")
add(SI, SEG_QC, "Q4FY25", "quarterly", "Store Profitability Share", 10, "percent_of_stores",
    confidence="analyst_estimate", notes="Described only as '<10%, two quarters before Q2FY26' — treat as approximate")
add(SI, SEG_QC, "FY26", "annual", "Per Order Loss", 85.18, "INR", confidence="analyst_estimate", source_id="S21,S22")
add(SI, SEG_QC, "FY26", "annual", "Capex Allocated from QIP", 4475, "INR_crore",
    notes="Earmarked for Instamart fulfillment infrastructure out of the broader ₹10,000cr QIP raise")

# ---- Swiggy Food Delivery segment (internal benchmark — already profitable) ----
SFD, SEG_FD = "Swiggy Food Delivery", "Food Delivery"
add(SFD, SEG_FD, "Q1FY26", "quarterly", "Adjusted EBITDA Margin", 2.4, "percent", confidence="derived",
    notes="Referenced as 'last quarter' in the Q2FY26 letter")
add(SFD, SEG_FD, "Q2FY26", "quarterly", "Adjusted EBITDA Margin", 2.8, "percent", qoq_pct=None, source_id="S01",
    notes="+44 bps QoQ per shareholder letter")
add(SFD, SEG_FD, "Q3FY26", "quarterly", "Adjusted EBITDA Margin", 3.04, "percent", confidence="derived",
    notes="Back-calculated: Q4FY26 (3.3%) minus the stated +26bps QoQ improvement")
add(SFD, SEG_FD, "Q4FY26", "quarterly", "Adjusted EBITDA Margin", 3.3, "percent", yoy_pct=None, qoq_pct=None,
    source_id="S03,S05", notes="+41bps YoY, +26bps QoQ per disclosure")
add(SFD, SEG_FD, "Q4FY26", "quarterly", "Adjusted EBITDA", 297, "INR_crore", yoy_pct=40, source_id="S03,S05")
add(SFD, SEG_FD, "Q4FY26", "quarterly", "GOV", 9005, "INR_crore", yoy_pct=22.6, source_id="S03,S05")
add(SFD, SEG_FD, "Q2FY26", "quarterly", "GOV", None, "INR_crore", yoy_pct=18.8, source_id="S01",
    notes="Absolute value not captured in source — YoY% only")
add(SFD, SEG_FD, "Q2FY26", "quarterly", "Take Rate", 25.8, "percent", source_id="S01", notes="+10bps QoQ")
add(SFD, SEG_FD, "Q1FY26", "quarterly", "Take Rate", 25.7, "percent", confidence="derived")
add(SFD, SEG_FD, "Q2FY26", "quarterly", "Contribution Margin", 7.3, "percent", source_id="S01", notes="Range-bound QoQ")
add(SFD, SEG_FD, "Q4FY26", "quarterly", "MTU", 18.3, "million", yoy_pct=21, source_id="S05")

# ---- Swiggy Out-of-Home (OOH) segment --------------------------------------
add("Swiggy OOH", "Out-of-Home Consumption", "Q4FY26", "quarterly", "GOV", None, "INR_crore", yoy_pct=43,
    notes="Absolute value not captured in source — YoY% only")
add("Swiggy OOH", "Out-of-Home Consumption", "FY26", "annual", "Full Year Profitability Achieved", 1, "boolean")

# ---- Swiggy Consolidated -----------------------------------------------------
SC = "Swiggy Consolidated"
add(SC, "Consolidated", "Q3FY25", "quarterly", "Net Loss", 799, "INR_crore", source_id="S03")
add(SC, "Consolidated", "Q3FY26", "quarterly", "Net Loss", 1065, "INR_crore", source_id="S03,S08")
add(SC, "Consolidated", "Q3FY25", "quarterly", "Revenue from Operations", 3993, "INR_crore", source_id="S03")
add(SC, "Consolidated", "Q3FY26", "quarterly", "Revenue from Operations", 6148, "INR_crore", source_id="S03")
add(SC, "Consolidated", "Q4FY25", "quarterly", "Net Loss", 1081, "INR_crore", source_id="S08")
add(SC, "Consolidated", "Q4FY26", "quarterly", "Net Loss", 800, "INR_crore", yoy_pct=-25.99, qoq_pct=-24.88,
    source_id="S08,S09,S10")
add(SC, "Consolidated", "FY25", "annual", "Net Loss", 3117, "INR_crore", source_id="S04")
add(SC, "Consolidated", "FY26", "annual", "Net Loss", 4154, "INR_crore", source_id="S04")
add(SC, "Consolidated", "Q4FY26", "quarterly", "Revenue from Operations", 6383, "INR_crore", yoy_pct=44.74,
    qoq_pct=3.82, source_id="S08,S09,S10")
add(SC, "Consolidated", "8May2026", "snapshot", "Stock Price", 280.80, "INR", source_id="S08")
add(SC, "Consolidated", "8May2026", "snapshot", "Stock Price YTD Change", -27.29, "percent", source_id="S08")
add(SC, "Consolidated", "8May2026", "snapshot", "Stock Price vs 52wk High", -40.63, "percent", source_id="S08")
add(SC, "Consolidated", "Q2FY26", "quarterly", "Cash Position", 2.0, "USD_billion", confidence="analyst_estimate",
    source_id="S11")
add(SC, "Consolidated", "post-QIP", "snapshot", "Cash Position", 17000, "INR_crore",
    notes="Approximate, post the ₹10,000cr QIP raise")
add(SC, "Consolidated", "QIP-event", "snapshot", "QIP Amount Raised", 10000, "INR_crore")
add(SC, "Consolidated", "QIP-event", "snapshot", "QIP Shares Issued", 26.67, "crore_shares")
add(SC, "Consolidated", "QIP-event", "snapshot", "QIP Price per Share", 375, "INR")

# ---- Blinkit (Eternal) -------------------------------------------------------
BL, SEG_QC2 = "Blinkit", "Quick Commerce"
add(BL, SEG_QC2, "Q4FY25", "quarterly", "GOV", 9421, "INR_crore", yoy_pct=134, source_id="S13,S14")
add(BL, SEG_QC2, "Q4FY26", "quarterly", "NOV", 14386, "INR_crore", yoy_pct=95.4, source_id="S13,S14,S15")
add(BL, SEG_QC2, "Q4FY26", "quarterly", "Adjusted EBITDA", 37, "INR_crore", source_id="S13,S14",
    notes="First quarter of positive adjusted EBITDA")
add(BL, SEG_QC2, "Q4FY26", "quarterly", "Dark Stores (count)", 2243, "count", source_id="S15")
add(BL, SEG_QC2, "Q4FY26", "quarterly", "Net Dark Store Adds", 216, "count", source_id="S15")
add(BL, SEG_QC2, "FY23", "annual", "Adjusted EBITDA Margin", -17.8, "percent", source_id="S14,S15")
add(BL, SEG_QC2, "FY24", "annual", "Adjusted EBITDA Margin", -3.7, "percent", source_id="S14,S15")
add(BL, SEG_QC2, "FY25", "annual", "Adjusted EBITDA Margin", -1.3, "percent", source_id="S14,S15")
add(BL, SEG_QC2, "FY26", "annual", "Adjusted EBITDA Margin", -0.6, "percent", source_id="S14,S15")
add(BL, SEG_QC2, "Q4FY26", "quarterly", "Adjusted EBITDA Margin", 0.3, "percent", source_id="S13,S14")
add(BL, SEG_QC2, "current", "snapshot", "Net AOV", 525, "INR", source_id="S15")
add(BL, SEG_QC2, "Q4FY25", "quarterly", "Delivery Cost per Order", 55, "INR", yoy_pct=-14, source_id="S15")
add(BL, SEG_QC2, "FY26", "annual", "Per Order Loss", 3.02, "INR", confidence="analyst_estimate", source_id="S21,S22")
add(BL, SEG_QC2, "Q3FY26", "quarterly", "Inventory-led NOV Share", 90, "percent", source_id="S15")
add(BL, SEG_QC2, "Jan2026", "snapshot", "Market Share", 46, "percent", confidence="analyst_estimate", source_id="S23")
add(BL, SEG_QC2, "FY26", "annual", "Revenue from Operations", 37779, "INR_crore", source_id="S14",
    notes="Inventory-led model inflates this vs marketplace-model peers — NOT comparable to Swiggy/Zepto revenue without adjustment")
add(BL, SEG_QC2, "current", "snapshot", "Mature Market EBITDA Margin (Delhi NCR aggregate)", 3.5, "percent", source_id="S15")
add(BL, SEG_QC2, "current", "snapshot", "Mature Market EBITDA Margin (Gurgaon/Noida)", 5.0, "percent", source_id="S15")
add(BL, SEG_QC2, "current", "snapshot", "Orders per Store per Day", 1337, "orders", confidence="derived",
    source_id="S20", notes="Derived from Nomura's relative framing: Zepto (~2140) is ~60% higher than Blinkit => Blinkit ≈ 2140/1.6")

# ---- Zepto --------------------------------------------------------------------
ZP, SEG_QC3 = "Zepto", "Quick Commerce"
add(ZP, SEG_QC3, "FY25", "annual", "Revenue", 11110, "INR_crore", source_id="S19")
add(ZP, SEG_QC3, "Jan2026", "snapshot", "Market Share", 22, "percent", confidence="analyst_estimate", source_id="S23")
add(ZP, SEG_QC3, "Q4FY26", "quarterly", "NRV", 8134, "INR_crore", yoy_pct=73, qoq_pct=28, source_id="S19")
add(ZP, SEG_QC3, "Q4FY26", "quarterly", "Orders", 210, "million", source_id="S19")
add(ZP, SEG_QC3, "FY26", "annual", "Orders (Full Year)", 640, "million", source_id="S19")
add(ZP, SEG_QC3, "FY25", "annual", "Orders (Full Year)", 332, "million", source_id="S19")
add(ZP, SEG_QC3, "current", "snapshot", "AOV", 387, "INR", source_id="S19")
add(ZP, SEG_QC3, "current", "snapshot", "Orders per Store per Day", 2140, "orders", confidence="analyst_estimate", source_id="S20")
add(ZP, SEG_QC3, "FY26", "annual", "Per Order Loss", 78.75, "INR", confidence="analyst_estimate", source_id="S21,S22")
add(ZP, SEG_QC3, "FY26", "annual", "Advertising Revenue", 1636, "INR_crore", source_id="S19", notes="~7.9% of NRV")

# ---- Market structure (cross-company) ----------------------------------------
add("Top3 Aggregate", "Market Structure", "FY24", "annual", "Instamart Order Share (of Top 3)", 34.3, "percent", source_id="S19")
add("Top3 Aggregate", "Market Structure", "FY25", "annual", "Instamart Order Share (of Top 3)", 24.5, "percent", source_id="S19")
add("Top3 Aggregate", "Market Structure", "FY26", "annual", "Instamart Order Share (of Top 3)", 20.9, "percent", source_id="S19")

# ---- Capital allocation / governance flags ------------------------------------
add(SC, "Governance", "current", "snapshot", "IOCC Resolution Vote Share", 72.35, "percent",
    notes="Required 75% threshold to pass — resolution failed by 2.65pp, blocking the inventory-model transition")
add(SC, "Governance", "current", "snapshot", "Inventory Model Margin Benefit (stated)", 60, "bps",
    notes="CFO-stated range was 50-70bps; midpoint used as a single point estimate")

master_metrics_df = pd.DataFrame(rows)
master_metrics_df.insert(0, "row_id", range(1, len(master_metrics_df) + 1))
master_metrics_df.to_csv(PROCESSED / "master_metrics.csv", index=False)
print(f"master_metrics.csv written — {len(master_metrics_df)} observations across {master_metrics_df['company'].nunique()} companies")
master_metrics_df.head(15)
""")

# ---------------------------------------------------------------------------
md(r"""## 4. Data completeness check

Before building anything on top of this, it's worth seeing exactly where the holes are —
which company/segment combinations have rich quarterly history versus a handful of
snapshot estimates.
""")

code(r"""
coverage = (master_metrics_df
            .groupby(["company", "confidence"])
            .size()
            .unstack(fill_value=0))

coverage = coverage[[c for c in ["disclosed", "derived", "analyst_estimate"] if c in coverage.columns]]
coverage
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
coverage.plot(kind="barh", stacked=True, ax=ax,
              color={"disclosed": "#185FA5", "derived": "#85B7EB", "analyst_estimate": "#B5D4F4"})
ax.set_xlabel("Number of observations")
ax.set_ylabel("")
ax.set_title("Observation count by company and confidence level")
ax.legend(title="Confidence", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PROCESSED / "chart_data_coverage.png", bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 5. Three illustrative cross-player charts

These are quick sanity-check charts to confirm the dataset is usable end to end — the real
analytical charts belong in the branch notebooks. Each one pulls straight from
`master_metrics_df`, which is the pattern every downstream notebook should follow.
""")

code(r"""
# Chart 1 — Blinkit's annual EBITDA margin trajectory FY23 -> Q4FY26, the clearest
# "this is achievable" reference line for the whole case study.
blinkit_margin = master_metrics_df[
    (master_metrics_df.company == "Blinkit") &
    (master_metrics_df.metric == "Adjusted EBITDA Margin")
].copy()

# enforce a sensible chronological order
order = ["FY23", "FY24", "FY25", "FY26", "Q4FY26"]
blinkit_margin["period"] = pd.Categorical(blinkit_margin["period"], categories=order, ordered=True)
blinkit_margin = blinkit_margin.sort_values("period")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(blinkit_margin["period"].astype(str), blinkit_margin["value"], marker="o",
        color="#0F6E56", linewidth=2)
ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--")
ax.set_ylabel("Adjusted EBITDA margin (%)")
ax.set_title("Blinkit: adjusted EBITDA margin, FY23 \u2192 Q4 FY26")
for x, y in zip(blinkit_margin["period"].astype(str), blinkit_margin["value"]):
    ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(PROCESSED / "chart_blinkit_margin_trajectory.png", bbox_inches="tight")
plt.show()
""")

code(r"""
# Chart 2 — Orders per store per day, cross-player density comparison.
density = master_metrics_df[master_metrics_df.metric == "Orders per Store per Day"].copy()
density = density.sort_values("value")

fig, ax = plt.subplots(figsize=(7, 3.5))
colors = ["#378ADD" if c == "disclosed" else "#85B7EB" for c in density["confidence"]]
ax.barh(density["company"], density["value"], color=colors)
for y, (val, conf) in enumerate(zip(density["value"], density["confidence"])):
    suffix = "" if conf == "disclosed" else "  (derived/estimate)"
    ax.text(val + 20, y, f"{val:,.0f}{suffix}", va="center", fontsize=9)
ax.set_xlabel("Orders per store per day")
ax.set_title("Store-level throughput: the density gap behind the margin gap")
plt.tight_layout()
plt.savefig(PROCESSED / "chart_density_comparison.png", bbox_inches="tight")
plt.show()
""")

code(r"""
# Chart 3 — Instamart's order-share erosion among the top 3 players, FY24 -> FY26.
share = master_metrics_df[master_metrics_df.metric == "Instamart Order Share (of Top 3)"].copy()
share["period"] = pd.Categorical(share["period"], categories=["FY24", "FY25", "FY26"], ordered=True)
share = share.sort_values("period")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(share["period"].astype(str), share["value"], marker="o", color="#993C1D", linewidth=2)
ax.set_ylabel("Instamart share of top-3 orders (%)")
ax.set_title("The erosion the margin story doesn't show:\nInstamart's order share, FY24 \u2192 FY26")
ax.set_ylim(0, 40)
for x, y in zip(share["period"].astype(str), share["value"]):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(PROCESSED / "chart_order_share_erosion.png", bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 6. Known gaps — what to fetch next to strengthen this dataset

This scaffold is enough to start the branch notebooks, but these gaps will limit how
rigorous the contribution-margin and density regressions in Branch 1 can be:

1. **Instamart's full quarterly contribution-margin history, Q1 FY25 through Q2 FY26.**
   Only Q3 FY26, Q4 FY26, and the March 2026 monthly figure are captured here. The actual
   numbers live in Swiggy's KPI databooks at
   `swiggy.com/corporate/investor-relations/financial-results/` — pull the databook PDF or
   XLS for each quarter rather than relying on secondary press summaries.
2. **Blinkit's quarterly (not just annual) GOV/NOV series for FY24-FY26.** Currently only
   two quarterly data points (Q4 FY25, Q4 FY26) plus the annual margin trend. Eternal's
   investor relations page publishes this on the same quarterly cadence as Swiggy.
3. **A verified Zepto UDRHP citation.** Several Zepto figures here (order share, NRV,
   advertising revenue) are sourced through secondary press coverage that cites the UDRHP
   without a direct link. If Zepto's IPO documents are public by the time this is finalized,
   replace `S19` with the actual filing.
4. **Orders-per-store-per-day for Blinkit and Zepto are currently derived/estimated**, not
   disclosed. If either company publishes this directly in a future shareholder letter,
   replace the derived row in `master_metrics.csv`.
5. **A consistent MTU (monthly transacting users) series for Instamart specifically.**
   Swiggy discloses MTU prominently at the consolidated/food-delivery level; the Instamart-only
   figure used here is a secondary citation and should be double-checked against the KPI
   databook.

**Pattern for every downstream notebook:**
```python
import pandas as pd
df = pd.read_csv("../data/processed/master_metrics.csv")
instamart_margin = df[(df.company == "Swiggy Instamart") & (df.metric == "Contribution Margin")]
```
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("notebooks/01_master_data.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/01_master_data.ipynb")
