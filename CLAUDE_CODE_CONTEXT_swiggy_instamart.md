# CLAUDE CODE CONTEXT — Swiggy Instamart Quick-Commerce Strategy Case Study

> **For Claude Code:** This document is the authoritative context file for the project at
> `D:\Downloads\swiggy-instamart-casestudy\`. Read it fully before touching any file.
> Every fact, file path, number, and decision recorded here reflects what has already been
> built and committed to disk. Do not invent, infer, or hallucinate anything not stated.

---

## 1. How this project started

The project began with a single question:

> *"Let us work on a consultant-style case study that we discussed earlier. Help me choose a
> company for this case study that will impress recruiters from McKinsey, BCG and Bain or top
> recruiters from top tech giants in this current market."*

### Why Swiggy Instamart / quick-commerce was chosen

After evaluating several sectors, Swiggy Instamart's quick-commerce vertical was selected for
four compounding reasons:

1. **Live, contested profitability race.** Blinkit (Eternal) turned adjusted EBITDA-positive for
   the first time in Q4 FY26 (₹37cr). Instamart is still negative at -10.9% adjusted EBITDA
   margin. The gap is real, current, and disclosed — not hypothetical.

2. **Genuine analytical complexity.** The problem is not "Instamart is losing." It's *why*, and
   the answer involves interacting drivers across unit economics, customer behavior, logistics,
   competitive structure, and monetization. That is a MECE issue-tree problem — exactly the
   structure MBB firms use in case interviews and real engagements.

3. **Rich public data.** All three players — Swiggy, Eternal (Blinkit), Zepto — are either
   listed or pre-IPO and publish quarterly shareholder letters, exchange filings, and KPI
   databooks. The case study could be built entirely on real disclosures rather than invented
   numbers, which is what MBB recruiters penalize most.

4. **A governance twist.** The single highest-leverage fix (moving to Blinkit's inventory-led
   fulfilment model) was put to a shareholder vote and failed by 2.65 percentage points
   (72.35% vs. the 75% needed). That is a story about business model, governance, and
   competitive positioning simultaneously — the kind of multi-layered insight that separates
   a strong case study from a basic financial analysis.

The chosen framing was deliberately MBB-style:
> *"Why does Swiggy Instamart's path to profitability lag Blinkit's?"*

---

## 2. The issue tree and the research phase

Before any notebook was written, a full MECE issue tree was built in a separate research
session. That research is captured in the attached file:

```
instamart_vs_blinkit_issue_tree_text_markdown.md
```

**Reference this file for the full hypothesis text, branch definitions, and source citations
from the research phase.** The document covers all seven branches of the issue tree.

### The seven-branch MECE structure

The root question was decomposed into seven mutually exclusive, collectively exhaustive branches:

| Branch | Status in notebooks |
|---|---|
| Branch 1 — Unit Economics & Density | **Tested** — Notebook 02 |
| Branch 2 — Customer Behavior | **Tested** — Notebook 03 (joint with Branch 5) |
| Branch 3 — Competitive Market Structure | **Tested** — Notebook 04 |
| Branch 4 — Logistics & Fulfillment | **Tested** — Notebook 05 |
| Branch 5 — Monetization | **Tested** — Notebook 03 (joint with Branch 2) |
| Branch 6 — Capital Allocation | **Context only** — no independently testable public data |
| Branch 7 — Regulatory & Policy | **Context only** — forward-looking risk, not retrospectively testable |

Branches 2 and 5 were combined in one notebook because customer behavior (frequency, AOV,
MTU) and monetization (take rate, advertising) share the same underlying disclosures and
the joint hypothesis set is small enough for one notebook.

---

## 3. The 25-source citation framework

All data in the project is traced back to `data/processed/sources.csv`. The framework
distinguishes three tiers of confidence used throughout all notebooks and the deck:

- **D (Disclosed):** directly from a company filing or shareholder letter
- **E (Analyst-estimated):** from a named research note (Nomura, Bernstein, JM Financial,
  Jefferies), cited via secondary press coverage
- **DV (Derived):** back-calculated by us from two disclosed figures

### Primary sources (direct company filings)

| ID | Organisation | Document |
|---|---|---|
| S01 | Swiggy | Q2 FY2026 Shareholder Letter (PDF) |
| S02 | Swiggy | Investor Relations Financial Results hub |
| S16 | BSE India | Swiggy corporate presentation (exchange filing) |

### Key secondary sources

| ID | Organisation | Content |
|---|---|---|
| S03 | Groww | Swiggy Q4 FY26 Results blog |
| S04 | Storyboard18 | Instamart dark-store expansion, Q4 FY26 |
| S06 | Whalesbook | Instamart prioritises profit over growth |
| S13 | Inc42 | Eternal Q4: Blinkit EBITDA jumps 9x QoQ |
| S14 | Sahi | Eternal Q4 FY26 full results analysis |
| S15 | Quash | Blinkit 2026: revenue, market share, profitability |
| S19 | Zepto | UDRHP (via secondary coverage — primary URL unconfirmed) |

### Sources tagged UNCONFIRMED

S19 (Zepto UDRHP), S20 (Nomura density note), S21 (Bernstein dark-store note),
S22 (Jefferies revision note), S23 (Datum Intelligence market-share snapshot),
S24 (JM Financial take-rate note) — all cited via secondary press coverage. The original
primary note or filing was not directly accessed. Every figure sourced from these is tagged
`analyst_estimate` in `master_metrics.csv`.

---

## 4. Key calibrated figures (the ground truth for all notebooks)

These are the numbers every notebook and the deck is built on. **Do not change these without
also updating `master_metrics.csv` and citing the new source.**

### Swiggy Instamart

| Metric | Value | Period | Confidence | Source |
|---|---|---|---|---|
| GOV | ₹7,881cr | Q4 FY26 | D | S06, S04 |
| GOV YoY growth | +68.8% | Q4 FY26 | D | S06 |
| GOV QoQ change | -0.7% | Q4 FY26 (first-ever sequential decline) | D | S06 |
| NOV | ₹5,675cr | Q4 FY26 | D | S03, S04 |
| Contribution margin | -1.8% | Q4 FY26 | D | S03, S04 |
| Contribution margin | -1.1% | March 2026 (monthly) | D | S03, S04 |
| Adjusted EBITDA | -₹858cr | Q4 FY26 | D | S07 |
| Adjusted EBITDA margin | -10.9% | Q4 FY26 | D | S03 |
| AOV | ₹700 | Q4 FY26 | D | S03, S04, S06 |
| AOV YoY growth | +32.8% | Q4 FY26 | D | S03 |
| Dark stores | 1,143 | Q4 FY26 | D | S03, S04 |
| Net store adds | 7 | Q4 FY26 | D | S04 |
| Cities covered | 129 | Q4 FY26 | D | S04 |
| Orders per store per day | 1,025 | Q2 FY26 | D | S01 |
| Orders per store per day QoQ | +4% | Q2 FY26 | D | S01 |
| Store profitability share | ~25% | Q2 FY26 | D | S01 |
| Per-order loss | ₹85.18 | FY26 | E | S21, S22 |
| MTU | 13.3M | Q4 FY26 | E | — |
| Take rate | 21.1% → 19.2% | Q2 FY26 → Q3 FY26 | E | S24 |
| Capex from QIP earmarked for Instamart | ₹4,475cr | FY26 | D | — |
| IOCC shareholder vote | 72.35% | Current | D | — |
| IOCC threshold needed | 75% | Current | D | — |
| Inventory model margin benefit (CFO-stated) | 50–70bps (midpoint 60bps) | Current | D | — |
| Instamart order share of top 3 | 34.3% → 24.5% → 20.9% | FY24 → FY25 → FY26 | E | S19 |

### Swiggy Food Delivery (internal benchmark)

| Metric | Value | Period | Confidence | Source |
|---|---|---|---|---|
| MTU | 18.3M | Q4 FY26 | D | S05 |
| MTU YoY growth | +21% | Q4 FY26 | D | S05 |
| GOV | ₹9,005cr | Q4 FY26 | D | S03, S05 |
| Adjusted EBITDA margin | +3.3% | Q4 FY26 | D | S03, S05 |
| Take rate | 25.8% | Q2 FY26 | D | S01 |

### Blinkit (Eternal)

| Metric | Value | Period | Confidence | Source |
|---|---|---|---|---|
| NOV | ₹14,386cr | Q4 FY26 | D | S13, S14, S15 |
| NOV YoY growth | +95.4% | Q4 FY26 | D | S13 |
| Adjusted EBITDA | +₹37cr | Q4 FY26 | D | S13, S14 |
| Dark stores | 2,243 | Q4 FY26 | D | S15 |
| Net store adds | 216 | Q4 FY26 | D | S15 |
| AOV | ₹525 (net) | Current | D | S15 |
| Per-order loss | ₹3.02 | FY26 | E | S21, S22 |
| Delivery cost per order | ₹55 | Q4 FY25 | D | S15 |
| Delivery cost YoY | -14% | Q4 FY25 | D | S15 |
| Inventory-led share of NOV | 90% | Q3 FY26 | D | S15 |
| Orders per store per day | 1,337 | Current | DV | S20 |
| Market share | 46% | January 2026 | E | S23 |
| Mature market adj EBITDA (Delhi NCR agg.) | ~3.5% | Current | E | S15 |
| Mature market adj EBITDA (Gurgaon/Noida) | ~5.0% | Current | E | S15 |
| Annual margin trend FY23→FY26 | -17.8% → -3.7% → -1.3% → -0.6% | Annual | D | S14, S15 |

### Zepto

| Metric | Value | Period | Confidence | Source |
|---|---|---|---|---|
| NRV | ₹8,134cr | Q4 FY26 | D | S19 |
| NRV YoY growth | +73% | Q4 FY26 | D | S19 |
| AOV | ₹387 | Current | D | S19 |
| Orders | 210M | Q4 FY26 | D | S19 |
| Full-year orders | 640M | FY26 | D | S19 |
| Per-order loss | ₹78.75 | FY26 | E | S21, S22 |
| Orders per store per day | ~2,140 | Current | E | S20 |
| Advertising revenue | ₹1,636cr | FY26 | D | S19 |
| Market share | 22% | January 2026 | E | S23 |

### Market structure

| Metric | Value | Source |
|---|---|---|
| Instamart market share | 24% | E, S23 |
| Others (Flipkart Minutes, Amazon Now, etc.) | 8% | E, S23 |
| Flipkart Minutes dark stores | ~800 | D |
| Flipkart Minutes monthly store adds | ~100/month | D |
| Amazon Now stated target | 100 cities, 1,000+ MFCs | D (stated plan, not operating) |

---

## 5. Repository structure

```
D:\Downloads\swiggy-instamart-casestudy\
├── .gitignore
├── requirements.txt
├── data\
│   ├── raw\                        (empty — data sourced from CSVs built in notebooks)
│   └── processed\
│       ├── master_metrics.csv      (79 observations, long-format tidy table)
│       ├── sources.csv             (25 sources, S01–S25)
│       ├── data_dictionary.csv     (10 metric definitions with comparability notes)
│       ├── tableau_all_verdicts.csv (22 hypothesis verdicts, all branches)
│       ├── tableau_master_metrics.csv
│       ├── b1_verdict_summary.csv
│       ├── b1_instamart_simulated_store_level_PROXY.csv  (n=1,143 synthetic stores)
│       ├── b2_b5_verdict_summary.csv
│       ├── b2_b5_supplementary_metrics.csv
│       ├── b3_verdict_summary.csv
│       ├── b3_supplementary_metrics.csv
│       ├── b4_verdict_summary.csv
│       ├── b6a_reward_weight_sensitivity.csv
│       └── [chart PNGs — b1_*, b2_b5_*, b3_*, b4_*, b6a_*, chart_*]
├── models\
│   └── strategy1\                  (PPO model checkpoints written by 06a)
├── notebooks\
│   ├── 01_master_data.ipynb
│   ├── 02_unit_economics_density.ipynb
│   ├── 03_customer_behavior_monetization.ipynb
│   ├── 04_competitive_market_structure.ipynb
│   ├── 05_logistics_cost_benchmarking.ipynb
│   ├── 06a_strategy1_rl_inventory_transition.ipynb
│   ├── 06b_strategy2_uplift_crosssell.ipynb
│   ├── 06c_strategy3_system_dynamics.ipynb
│   └── 06d_strategy_simulation_summary.ipynb
└── src\
    ├── __init__.py
    ├── build_notebook.py           (generates 01_master_data.ipynb)
    ├── build_notebook_06a.py       (generates 06a)
    ├── build_notebook_06b.py       (generates 06b)
    ├── build_notebook_06c.py       (generates 06c)
    └── build_notebook_06d.py       (generates 06d)
```

> **Important convention:** Notebooks 01–05 were built by running `src/build_notebook.py`
> via `nbformat`. Notebooks 06a–06d were built by running their respective
> `src/build_notebook_06*.py` scripts. The source of truth for each notebook's content
> is its build script. If you regenerate a notebook from its build script it will
> overwrite any manual edits made directly to the `.ipynb` file.

---

## 6. Python environment

- **Python version:** 3.11.2
- **Virtual environment:** `.venv\` (Windows, inside the repo root)
- **Activate:** `.venv\Scripts\activate` (PowerShell) or `.venv\Scripts\activate.bat` (cmd)

### `requirements.txt` — current state

```
# Core analysis (pinned to versions verified against all 5 branch notebooks)
pandas==3.0.2
numpy==2.4.4
matplotlib==3.10.8
scipy==1.17.1

# Jupyter
jupyterlab==4.5.8
notebook==7.5.7
ipykernel==7.3.0
jupyter==1.1.1
jupyter_client==8.9.1
jupyter_core==5.9.1

# Notebook generation
nbformat==5.10.4
nbconvert==7.17.1

# Strategy simulation notebooks (06a–06d)
gymnasium          # 06a — RL environment
stable-baselines3[extra]  # 06a — PPO agent (pulls PyTorch; large download)
scikit-learn       # 06b — X-Learner fallback + feature scoring
causalml           # 06b — X-Learner uplift model (optional; has sklearn fallback)
BPTK-Py            # 06c — System dynamics (optional; 06c has scipy ODE fallback)
ema-workbench      # 06c — Deep uncertainty analysis (optional)
tensorboard        # 06a — training monitoring
```

**Hardware context (for training decisions):**
- CPU: AMD Ryzen 5 4600H (6 cores / 12 threads)
- GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- Training device for 06a PPO: **CPU** (`device="cpu"` is set intentionally — PPO with
  MlpPolicy runs faster on CPU than on this GPU because the environment stepping is
  CPU-bound and the matmuls are tiny. GPU helps SB3 only for CnnPolicy on image observations.)

---

## 7. Notebook-by-notebook detail

### Notebook 01 — `01_master_data.ipynb`

**Purpose:** Single source of truth for all data. Every other notebook reads from what
this one writes. Never hard-code a number in a downstream notebook if it already lives here.

**Outputs written to `data/processed/`:**
- `sources.csv` — 25 sources S01–S25
- `data_dictionary.csv` — 10 metric definitions with cross-company comparability warnings
- `master_metrics.csv` — 79 observations, long-format schema:
  `row_id | company | segment | period | period_type | metric | value | unit |
   yoy_pct | qoq_pct | confidence | source_id | notes`

**Critical warning documented here:** Blinkit's inventory-led model means its "Revenue from
Operations" is not comparable to Swiggy/Zepto marketplace-model revenue. Never chart Blinkit
revenue next to Swiggy/Zepto revenue without this caveat. GOV and NOV are the only reasonably
comparable topline metrics, and even those are on different bases.

**Data completeness gaps documented in Section 6 of this notebook:**
1. Instamart contribution margin history Q1 FY25 through Q2 FY26 — only Q3, Q4 FY26 and
   March 2026 captured
2. Blinkit quarterly GOV/NOV series for FY24–FY26 — only Q4 FY25 and Q4 FY26 captured
3. Zepto UDRHP primary citation — still secondary
4. Density figures for Blinkit and Zepto — derived/estimated, not disclosed
5. Instamart-only MTU series — cited from secondary, should be verified in KPI databook

---

### Notebook 02 — `02_unit_economics_density.ipynb`

**Branch:** Branch 1 — Unit Economics & Density
**MECE question:** Is Instamart's density meaningfully below breakeven, and does density
explain the margin gap vs. Blinkit and Zepto?

**Hypotheses tested:**

| ID | Hypothesis | Verdict |
|---|---|---|
| H1.1 | Network density materially below breakeven/ceiling | Supported |
| H1.2 | Loss concentrated in a minority of stores | Supported |
| H1.3 | New stores take 6-12mo to mature, dragging blended margin | Inconclusive — not testable |
| H1.4 | Megapods improve throughput/margin via SKU breadth | Directionally supportive, thin |
| Cross-player (n=3) | Density alone explains the margin gap | Weak/inconclusive |
| Simulated (n=1,143) | Density predicts margin within a single network | Methodology demo |

**Key analytical decisions:**

- **Cross-player regression (n=3):** Swiggy (1,025, -₹85/order), Blinkit (1,337, -₹3/order),
  Zepto (2,140, -₹79/order). R²=0.03. Zepto has the highest density but near-worst per-order
  loss. Blinkit has mid-density and the best economics by far. Reported as weak/inconclusive —
  not dressed up. This is one of the three "honesty" highlights in the deck.

- **Simulated store network (n=1,143):** A synthetic dataset calibrated to match disclosed
  aggregates (mean ~1,025 orders/day, ~25% stores profitable). Used to demonstrate the
  regression method at a scale real data doesn't allow. Breakeven derived at ~1,552 orders/day
  (R²=0.41). Clearly flagged as a methodology exhibit, never as evidence about real stores.
  Saved at: `data/processed/b1_instamart_simulated_store_level_PROXY.csv`
  Schema: `store_id | is_megapod | city_tier | orders_per_day | contribution_margin_pct`

**Charts saved:**
- `b1_chart_density_vs_capacity.png`
- `b1_chart_cross_player_regression.png`
- `b1_chart_simulated_density_margin_regression.png`
- `b1_chart_store_profitability_share.png`

---

### Notebook 03 — `03_customer_behavior_monetization.ipynb`

**Branches:** Branch 2 (Customer Behavior) + Branch 5 (Monetization) — combined
**MECE questions:**
- Is Instamart's AOV advantage real and durable, and what is its actual frequency/MTU deficit?
- Is advertising/take-rate expansion currently working, and how does Instamart compare to peers?

**Hypotheses tested:**

| ID | Hypothesis | Verdict |
|---|---|---|
| H2.1 | Instamart leads on AOV — durable margin advantage | Supported |
| H2.2 | Deficit is frequency/MTU, not basket size | Directionally supported (estimate-dependent) |
| H2.3 | Non-grocery mix expansion raises AOV/margin | Directionally supported (2 data points) |
| H2.4 | Reduced promo intensity traded volume for economics | Supported |
| H5.1 | Advertising is currently driving take-rate expansion | Not supported |
| H5.2 | Ad revenue lags peers, leaving headroom | Supported + meta-finding |
| H5.3 | Take rate trails Blinkit's 20-22% guidance band | Supported |
| H5.4 | Non-grocery/private-label mix lifts ad + product margin | Partially supported (inferred) |

**Key findings:**
- Instamart ₹700 AOV clearly beats Blinkit ₹525 and Zepto ₹387 — the one unambiguous
  structural advantage Instamart holds.
- The take-rate thesis is broken: fell from 21.1% → 19.2% QoQ. Advertising-led expansion is
  not working yet. Zepto discloses ₹1,636cr ad revenue; Instamart does not disclose at all.
- Swiggy food delivery reaches 18.3M MTU. Instamart reaches ~13.3M. The cross-sell
  opportunity is literally inside the same company.

**Charts saved:**
- `b2_b5_chart_aov_vs_mtu_tension.png`
- `b2_b5_chart_take_rate_trend.png`
- `b2_b5_chart_ad_revenue_gap.png`
- `b2_b5_chart_gov_margin_tradeoff.png`
- `b2_b5_chart_nongrocery_aov_comovement.png`

---

### Notebook 04 — `04_competitive_market_structure.ipynb`

**Branch:** Branch 3 — Competitive Market Structure
**MECE question:** Is Instamart losing position relative to competitors as the market grows,
and are new entrants accelerating that loss?

**Hypotheses tested:**

| ID | Hypothesis | Verdict |
|---|---|---|
| H3.1 | Instamart losing share of incremental category growth | Strongly supported |
| H3.2 | Metro saturation caps orders/store and AOV | Not separately testable |
| H3.3 | New entrants fragmenting demand | Supported |
| H3.4 | Zepto's higher throughput proves frequency beats ticket size | Partially supported / overstated |

**Key findings:**
- Order share of the top 3 fell 34.3% → 24.5% → 20.9% FY24–FY26, a loss of 13.4pp —
  more than a third of the FY24 starting position. Sourced from Zepto UDRHP (S19) — flagged
  as a motivated source.
- Flipkart Minutes: ~800 stores, adding ~100/month. At that pace overtakes Zepto's footprint
  in ~3 months.
- Amazon Now: stated target 100 cities, 1,000+ MFCs — a plan, not an operating count.

**Charts saved:**
- `b3_chart_order_share_evolution.png`
- `b3_chart_market_share_donut.png`
- `b3_chart_dark_store_footprint_race.png`
- `b3_chart_aov_density_positioning.png`

---

### Notebook 05 — `05_logistics_cost_benchmarking.ipynb`

**Branch:** Branch 4 — Logistics & Fulfillment
**MECE question:** Does Instamart's marketplace model structurally prevent it from matching
Blinkit's logistics efficiency, and is there a path to close that gap?

**Hypotheses tested:**

| ID | Hypothesis | Verdict |
|---|---|---|
| H4.1 | Last-mile cost is large and density-sensitive | Directionally supported (1 firm, 2 points) |
| H4.2 | Tier-2 expansion raises blended delivery cost | Not independently testable |
| H4.3 | Marketplace model blocks Instamart's replenishment efficiency | Supported |
| H4.4 | Rider/regulatory cost inflation is a forward risk | Not testable retrospectively |

**Key findings:**
- 90% of Blinkit's NOV is inventory-led. Stated margin benefit: 50–70bps (midpoint 60bps).
- Swiggy's own board put this model to a shareholder vote. Vote: 72.35% in favor.
  Threshold: 75%. Failed by 2.65 percentage points. This is the governance finding that
  appears in the deck as the "single biggest lever blocked by a vote, not by economics."
- Blinkit delivery cost: ₹55/order, -14% YoY. No Instamart/Zepto public comparator.

**Charts saved:**
- `b4_chart_delivery_cost_trend.png`
- `b4_chart_inventory_model_opportunity.png`

---

### Notebook 06a — `06a_strategy1_rl_inventory_transition.ipynb`

**Strategy tested:** If Swiggy resolves the governance vote and transitions to an
inventory-led model, what is the optimal pace and sequencing?

**Method:** Reinforcement Learning — PPO (Stable-Baselines3)

**Environment class:** `InstamartTransitionEnv(gym.Env)`
- State space: 6 continuous variables
  `[density_norm, pct_inventory_led, contribution_margin, market_share, capex_norm, quarter]`
- Action space: Discrete(5)
  `0=Hold, 1=Slow+5pp, 2=Moderate+10pp, 3=Aggressive+15pp, 4=Retreat-5pp`
- Episode length: 12 quarters (3 fiscal years)
- Early termination: contribution_margin < -6% for 3 consecutive quarters
- Reward: `0.5 × margin + 0.3 × share_delta + 0.2 × capex_health`

**Causal relationships in the model (documented assumptions):**
1. Inventory-led share up → margin up at 60bps per full transition (stated 50–70bps range)
2. Density above breakeven (1,552) → margin up (log-linear, calibrated to n=1,143 sim)
3. Margin up → capex regenerates → density growth funded
4. Losing share vs Blinkit → competitive drag on margin
5. Each 5pp transition costs an estimated ₹80cr capex (**unsourced estimate — biggest
   assumption in the model**)

**Training configuration:**
- Algorithm: PPO, MlpPolicy
- `verbose=1` on PPO (shows rollout/train tables each update)
- `EvalCallback(verbose=1)` (periodic evaluation summaries every 5,000 steps)
- SB3 logger: stdout + CSV + TensorBoard (at `../models/strategy1/logs/` and `../models/strategy1/tb/`)
- `total_timesteps=400_000` (moderate — converges without locking up the laptop)
- `n_steps=1024`, `batch_size=64`, `n_epochs=10`, `gamma=0.97`, `learning_rate=3e-4`
- `device="cpu"` — intentional. PPO MlpPolicy is faster on CPU than GTX 1650 because
  env stepping (CPU-bound) dominates wall-clock; tiny matmuls make GPU overhead net-negative.
- Thread caps: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)` —
  leaves headroom for other laptop tasks.
- Both train and eval envs wrapped in `Monitor` (removes the "not wrapped with Monitor"
  UserWarning and gives correct episode stats to EvalCallback).

**Notebook sections:**
1. Setup & imports
2. Calibrated baseline (loaded from `master_metrics.csv`)
3. Environment design explanation (markdown)
4. `InstamartTransitionEnv` class definition
5. `check_env` + naive baseline (always-Hold vs always-Aggressive)
6. Train PPO (single training cell — `train00main`)
7. Evaluate learned policy — rollout 200 episodes
8. Reward-weight sensitivity analysis (3 weightings, 150k steps each, memory-cleaned between runs)
9. Verdict and honest limitations

**Charts saved:**
- `b6a_chart_learned_policy_trajectories.png` (margin + inventory-led share, median + IQR band)
- `b6a_chart_action_mix.png` (action frequency by quarter — stacked bar)
- `b6a_chart_breakeven_distribution.png` (histogram of quarters to positive margin)
- `b6a_chart_reward_sensitivity.png` (bar chart of mean final inv-share by reward weighting)

**CSVs saved:**
- `b6a_reward_weight_sensitivity.csv`

**Model checkpoints:** `../models/strategy1/best_model.zip` (written by EvalCallback)

---

### Notebook 06b — `06b_strategy2_uplift_crosssell.ipynb`

**Strategy tested:** Should Swiggy cross-sell its 18.3M food-delivery users into Instamart,
and if so, which users should be targeted first and how large is the prize?

**Method:** X-Learner uplift modeling (CausalML) + Monte Carlo revenue simulation

**Data:** Fully synthetic user base (n=500,000) calibrated to disclosed aggregates:
- Mean order frequency ~4.2/month (calibrated to Swiggy disclosures)
- AOV log-normal around ₹480 mean (food delivery baseline)
- City tier: Metro 40%, Tier-1 35%, Tier-2 25%
- Instamart penetration of the FD base: calibrated to 13.3M / 18.3M = 72.7%

**Calibration constants:**
- `FOOD_DELIVERY_MTU = 18.3M` (D — S05)
- `INSTAMART_MTU = 13.3M` (E)
- `INSTAMART_AOV = ₹700` (D)
- `INSTAMART_TAKE_RATE = 19.2%` (D — S24)
- `ORGANIC_CONVERSION = 4%` (E — assumed baseline crossover without nudge)
- `CONVERSION_LOW/MID/HIGH = 8% / 15% / 25%` (E — no public benchmark; wide range intentional)
- `TARGETING_REACH = 60%` (E — share of FD base in Instamart-eligible zones)

**Features (covariates):** `order_freq`, `aov`, `city_tier`, `days_since_last`,
`cuisine_home`, `swiggy_one`, `instamart_eligible`, `already_instamart`,
`high_frequency` (engineered), `recency_segment` (engineered), `aov_x_freq` (interaction)

**Uplift model:** `BaseXRegressor` from CausalML with `GradientBoostingRegressor` as base
learner. If CausalML is not installed, falls back to a transparent two-model T-learner
implemented with scikit-learn.

**Validation:** Because data is synthetic, true CATE is known. Spearman rank correlation
(true vs. estimated CATE) is computed. Target: rho > 0.60 indicates usable targeting ranking.

**Monte Carlo parameters (10,000 simulations):**
- Conversion rate: Uniform(8%, 25%)
- AOV: Uniform(₹630, ₹770)
- Take rate: Uniform(18.5%, 20.5%)

**Charts saved:**
- `b6b_chart_cate_validation.png` (scatter: true vs estimated CATE)
- `b6b_chart_uplift_curve.png` (cumulative incremental conversions vs population targeted)
- `b6b_chart_monte_carlo.png` (histograms of incremental MTU and GOV with P10/P50/P90)
- `b6b_chart_tornado.png` (correlation of input uncertainty to GOV variance)

**CSVs saved:**
- `b6b_cate_segments.csv` (quartile summary)
- `b6b_monte_carlo_summary.csv`

---

### Notebook 06c — `06c_strategy3_system_dynamics.ipynb`

**Strategy tested:** Should Swiggy hold new store expansion and let existing stores densify
first, rather than continuing aggressive footprint growth?

**Method:** System Dynamics — primary path uses a transparent `scipy.integrate.solve_ivp`
ODE implementation. Optional cells use BPTK-Py and EMA Workbench; both are wrapped in
try/except and the notebook runs end-to-end without them.

**State vector (4 stocks):** `[total_stores, avg_density, contribution_margin, capex_reserve]`

**Key causal loops:**
- R1 (Reinforcing — profitability engine): density up → margin up → capex regenerates →
  density investment → density up
- B1 (Balancing — expansion dilution): new stores open at ~400 orders/day, dragging average
  density down before maturation
- B2 (Balancing — maturation): new stores climb toward density ceiling over 2-4 quarters
- R2 (Reinforcing — competitive pressure): losing share → promo spend → margin drag

**Calibration constants (initial stock values):**
- `init_stores = 1,143` (D)
- `init_density = 1,025` (D — Q2 FY26)
- `init_margin = -0.018` (D — Q4 FY26)
- `init_capex = 500` (E)
- `density_ceiling = 2,000` (D)
- `breakeven_density = 1,552` (DV — Notebook 02)

**Uncertain parameters swept in deep-uncertainty analysis:**
- `maturation_qtr`: 1.0 – 4.0 quarters
- `margin_per_order`: 0.00008 – 0.00025 (pp per order/day)
- `capex_per_store`: ₹2.5cr – ₹5.0cr
- `new_store_density`: 300 – 500 (starting density of new stores)

**Scenarios compared:**
- `"aggressive"`: 60 new stores per quarter (current pace estimate)
- `"hold"`: 5 new stores per quarter (maintenance only)

**Deep-uncertainty analysis:** Latin-hypercube design, 2,000 experiments. RandomForest
feature importance used to rank which uncertain parameter most drives time-to-breakeven.
PRIM-style box-peeling finds the region of parameter space where Strategy 3 reliably succeeds.

**Optional EMA Workbench cell:** 1,000 experiments using the same wrapper function.
Wrapped in try/except — skips silently if `ema-workbench` is not installed.

**Charts saved:**
- `b6c_chart_scenario_comparison.png` (dual-line: density and margin, both scenarios)
- `b6c_chart_feature_importance.png` (bar chart: parameter importance to time-to-breakeven)

**CSVs saved:**
- `b6c_uncertainty_experiments.csv` (2,000 experiment outcomes)

---

### Notebook 06d — `06d_strategy_simulation_summary.ipynb`

**Purpose:** Synthesize 06a–06c into a single cross-strategy view. Reads artifact CSVs
from the other three notebooks — so **run 06a, 06b, 06c before running 06d**.

**Sections:**
1. Load outputs from 06a, 06b, 06c (safe_read with informative fallback)
2. Side-by-side comparison table (method, primary KPI, time to impact, capex need, blocker)
3. Cost vs. control vs. impact bubble chart (qualitative 1-5 scores, documented as assumptions)
4. Combined directional impact model (illustrative, not a forecast) — shows all three sequenced
5. Sequencing Gantt (S3 Q0-12, S2 starts Q2, S1 starts Q4)
6. Overall verdict and shared limitations

**Recommended sequencing (documented with dependency logic):**
- Q0-2: **S3** (hold expansion) — zero capex, fully controllable, stops dilution immediately
- Q2-4: **S2** (cross-sell campaign) — higher density from S3 lifts conversion
- Q4+: **S1** (revisit inventory vote) — improved margins + higher MTU strengthen the business
  case and vote prospects

**Charts saved:**
- `b6d_chart_strategy_space.png` (bubble chart)
- `b6d_chart_combined_impact.png` (combined margin trajectory)
- `b6d_chart_sequencing.png` (Gantt)

---

## 8. The strategy deck

**File:** `D:\Downloads\portfolio\assets\quick-commerce-strategy\Swiggy_Instamart_Strategy_Case_Study_v2.pdf`

**Format:** 18-slide MBB-style deck built with pptxgenjs (Node.js). The deck follows an
answer-first structure (executive summary on slide 2 before the evidence).

**Slide structure:**
1. Title
2. Executive summary — answer first, 4 stat cards
3. Market context — donut chart (4-player share: Blinkit 46%, Instamart 24%, Zepto 22%, Others 8%)
4. Problem statement — order-share line chart + sourced callout
5. MECE methodology tree (7 branches, 5 tested / 2 context-only)
6. Density findings
7. Inventory model / IOCC vote (the governance finding)
8. AOV vs. MTU tension
9. Monetization (take rate + ad revenue gap)
10. Footprint race (dark stores by player + new entrants)
11. Analytical rigor spotlight (3 teal cards: weak regression, labeled simulation, confidence tags)
12. Recommendations (3-part plan)
13. Risks (4 cards)
14. Closing / thank-you
15. Appendix divider
16–17. Full hypothesis verdict log tables (22 hypotheses)
18. Sources table (25 sources)

---

## 9. Portfolio integration

**Portfolio repo:** `D:\Downloads\portfolio\` (deployed via Netlify)

**Files modified for this project:**

### `index.html`
- "End-to-end projects shipped" stat: updated from 4 → 5
- 5th project card added to `.projects-sub-grid` after the Gaming card:
  - Tags: "Competitive Strategy · Hypothesis-Driven Analysis", "Quick Commerce"
  - Org: "Swiggy Instamart · Blinkit · Zepto"
  - Metric badges: "34.3% → 20.9% share", "22 hypotheses · 4 branches", "72.35% vote (75% needed)"
  - Stack tags: Python, pandas, NumPy, SciPy, matplotlib, Jupyter, pptxgenjs
  - GitHub link: `https://github.com/addy9051/swiggy_instamart_quick_commerce_strategy`
    (**NOTE: this repo does not yet exist remotely. GitHub link will 404 until you push.**)
  - Detail page link: `projects/quick-commerce-strategy.html`

### `projects/quick-commerce-strategy.html` (new file)
10-section project detail page with:
- Dark hero with 4 metric badges, role tags (BA + DS), action buttons
- Section 2: Executive summary
- **Deck embed section** (between sections 2 and 3): 18-slide PDF embedded via `<iframe>`
  pointing to `../assets/quick-commerce-strategy/Swiggy_Instamart_Strategy_Case_Study_v2.pdf`
- Section 3: Business context
- Section 4: Problem statement (order-share erosion)
- Section 5: MECE methodology (7-branch grid)
- Section 6: Findings
- Section 7: Analytical rigor (3 cards)
- Section 8: Recommendations (3-step list)
- Section 9: Reflections and limitations (2-column good/change grid + scope callout)
- Section 10: Sources and verdict table sample

---

## 10. Confidence and honesty conventions

These conventions are applied throughout every notebook, the deck, and the portfolio page.
Any future code written for this project must respect them.

### Three confidence tiers

| Tag | Meaning | How to identify in master_metrics.csv |
|---|---|---|
| **D (Disclosed)** | Directly from a company filing or shareholder letter | `confidence = "disclosed"` |
| **E (Analyst-estimated)** | From a named research note, cited via secondary coverage | `confidence = "analyst_estimate"` |
| **DV (Derived)** | Back-calculated by us from two disclosed figures | `confidence = "derived"` |

### The three honesty highlights (appear in slide 11 and the portfolio page)

1. **R²=0.03 reported honestly.** The cross-player density-vs-margin regression on n=3 points
   is weak. The deck says so plainly rather than dropping the chart or overstating it.

2. **Synthetic store simulation clearly labeled.** The n=1,143 simulated dataset (Notebook 02)
   is described as "calibrated proxy data" and "methodology exhibit" everywhere it appears.
   It is never presented as evidence about Instamart's real stores.

3. **Every figure confidence-tagged.** Every number in the deck traces back to a source_id in
   `sources.csv` and a confidence level in `master_metrics.csv`.

### What is and is not real evidence

Real evidence (from disclosures):
- The order-share erosion (34.3% → 20.9%) — from Zepto UDRHP (motivated source, flagged)
- The IOCC vote (72.35% vs 75%) — public record
- Instamart's AOV (₹700) — disclosed by Swiggy
- Blinkit's 90% inventory-led NOV and 50–70bps margin benefit — disclosed by Eternal
- The take-rate decline (21.1% → 19.2%) — JM Financial (UNCONFIRMED primary, via press)

Simulated/estimated (clearly labeled everywhere):
- The n=1,143 synthetic store network and its R²=0.41 regression
- The RL environment's transition cost (₹80cr per 5pp) — unsourced estimate
- The synthetic user base in 06b (n=500,000) — calibrated to disclosures but not real
- The SD model coefficients in 06c

---

## 11. What is not yet done

The following items are explicitly incomplete as of this context document:

1. **GitHub repo for the case study does not exist remotely.** The local repo at
   `D:\Downloads\swiggy-instamart-casestudy\` has no `remote` configured (confirmed via
   `.git/config`). The portfolio card links to
   `https://github.com/addy9051/swiggy_instamart_quick_commerce_strategy` — this will 404
   until you `git remote add origin` and push.

2. **Portfolio not pushed to Netlify** since the project card was added. Run
   `git add . && git commit -m "Add quick commerce strategy project" && git push`
   from `D:\Downloads\portfolio\` to make the card and PDF embed live.

3. **Notebooks 06a–06d have not been executed in the current state.** 06a was partially
   run (training was interrupted with KeyboardInterrupt). The fixed version of 06a exists
   on disk but needs a fresh top-to-bottom run. 06b, 06c, 06d have not been executed yet.

4. **No chart PNGs exist for 06b, 06c, 06d.** They will be created when those notebooks
   are executed. The portfolio detail page currently uses CSS-based stat displays rather
   than embedded chart images because the PNG files were not available at the time the
   page was built.

5. **Strategy deck watermark "IOCC" is decorative.** The large background text in the hero
   section of the portfolio page is a design element, not a claim.

6. **Branches 6 and 7 have no notebooks.** Capital Allocation (Branch 6) and Regulatory
   & Policy (Branch 7) are documented as context-only in the deck and the issue tree.
   No analysis was attempted because no independently testable public data exists for them.

7. **Tableau dashboard is in progress.** The closing slide of the deck says
   "interactive Tableau dashboard (in progress)." It does not exist yet.

---

## 12. Critical rules for any future code written on this project

1. **Never hard-code a number that lives in `master_metrics.csv`.** Always use
   `pd.read_csv("../data/processed/master_metrics.csv")` and look it up.

2. **Never remove the confidence tags.** Every figure must carry D / E / DV. If a
   number cannot be sourced to one of S01–S25, it must be flagged as an estimate
   and documented in a note.

3. **Never present the synthetic store network as real evidence.** The n=1,143
   dataset in `b1_instamart_simulated_store_level_PROXY.csv` is a methodology
   demonstration. Its `_PROXY` suffix is intentional.

4. **Do not use em-dashes (—) in any resume or career document for Ankit Addya.**
   Use commas, colons, semicolons, or restructure the sentence. This rule applies only
   to resume/career docs, not to analysis writeups.

5. **The build scripts in `src/` are the source of truth for notebooks 01–05 and 06a–06d.**
   If you regenerate a notebook from its build script it overwrites manual edits.
   If significant manual edits have been made to a `.ipynb` file, update the build
   script to match before regenerating.

6. **`verbose=1` is correct and intentional** on the PPO model and EvalCallback in 06a.
   Do not set it back to 0.

7. **`device="cpu"` in 06a is correct and intentional.** Do not change it to `"cuda"`
   without understanding why it is set to CPU (PPO MlpPolicy is faster on CPU here).
