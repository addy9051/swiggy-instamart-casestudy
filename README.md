# Swiggy Instamart Strategy Case Study

![Swiggy Instamart vs Blinkit vs Zepto](https://img.shields.io/badge/Strategy-Quick_Commerce-FC8019?style=for-the-badge) ![Methodology](https://img.shields.io/badge/Methods-RL_%7C_CausalML_%7C_System_Dynamics-0C9D61?style=for-the-badge) 

This repository contains a comprehensive, data-driven strategy case study analyzing Swiggy Instamart's path to profitability compared to its primary competitors, Blinkit (Eternal) and Zepto. 

Rather than relying on qualitative takes, this project grounds its analysis entirely in primary financial filings (Q4 FY26 shareholder letters, UDRHPs) and rigorously models strategic levers using advanced data science methods: **Reinforcement Learning**, **Uplift Modeling (CausalML)**, and **System Dynamics**.

---

## 📌 Executive Summary & Key Findings

The dominant industry narrative suggests that Instamart simply needs to "build more stores" to catch Blinkit's density. The data reveals a different, more structural reality. 

Here are the primary insights from the analysis:

1. **Density is necessary, but the fulfillment model dictates the ceiling.** A cross-player regression between density and per-order loss across the Big 3 shows almost no correlation (R² = 0.03). Blinkit's profitability stems from its owned-inventory model (~100 bps EBITDA accretion). Instamart's margin gap is currently a governance and FDI regulatory problem (blocked by a failed IOCC vote), not just an operational one.
2. **The highest-ROI lever today is a targeted cross-sell.** Using an X-Learner Uplift model, targeting the ~5M non-converting Swiggy Food Delivery MTUs based on frequency and grocery-affinity yields a projected ~₹1,255 Cr incremental GOV/year at a 25x revenue-to-cost ratio. Targeted uplift delivers **+27% more incremental revenue** than a mass campaign.
3. **The Contrarian Reframe:** Instamart's strongest strategic option is *not* to fight Blinkit in a hyper-local commodity frequency war. With a ₹700 gross AOV (vs. Blinkit's net ₹525 and Zepto's implied ₹387) and integration within a super-app of 18.3M users, Instamart should lean into **high-AOV planned grocery** bundled with food delivery.

### The Recommended Execution Sequence
1. **Spend first on Density** (Hold expansion, densify current stores).
2. **Fund the Cross-Sell** (To feed density without massive CAC).
3. **Treat the Inventory Model as a conditional, secondary bet** (Pending regulatory and cost clearance).

---

## 🔬 Methodology & Rigor

To size the strategic levers without hallucinating certainty, this project utilizes heavy computational methods across 9 Jupyter notebooks:

* **Hypothesis Testing (Notebooks 01–05):** 22 hypotheses tested across a MECE issue tree using validated data. A strict data governance framework tags all metrics as Disclosed (D), Analyst-Estimated (E), or Derived (DV).
* **Reinforcement Learning (Notebook 06a):** Proximal Policy Optimization (PPO) was used to optimize a capex-constrained inventory transition, discovering a 3-rule policy that acts as a solvency brake.
* **Uplift Modeling (Notebook 06b):** CausalML (X-Learner) was deployed to isolate the *incremental* effect of cross-selling campaigns on synthetic user profiles.
* **System Dynamics (Notebook 06c):** Modeled feedback loops of network expansion. A Latin Hypercube sample of 2,000 experiments proved that a "Holding & Densifying" strategy beats "Aggressive Expansion" in 97% of possible futures.

---

## 📂 Project Structure

```text
├── data/
│   ├── processed/      # Validated data outputs, metrics (master_metrics.csv), and simulation results
│   └── raw/            # Raw data extracts
├── notebooks/          # Core analytical and simulation notebooks
│   ├── 01_master_data.ipynb
│   ├── 02_unit_economics_density.ipynb
│   ├── 03_customer_behavior_monetization.ipynb
│   ├── 04_competitive_market_structure.ipynb
│   ├── 05_logistics_cost_benchmarking.ipynb
│   ├── 06a_strategy1_rl_inventory_transition.ipynb
│   ├── 06b_strategy2_uplift_crosssell.ipynb
│   ├── 06c_strategy3_system_dynamics.ipynb
│   └── 06d_strategy_simulation_summary.ipynb
├── src/                # Python scripts used to build and generate notebooks programmatically
├── models/             # Directory for trained ML/RL policies
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## 🚀 Installation & Setup

To run the notebooks locally and explore the simulations, use a Python virtual environment:

1. **Navigate to the project root:**
   ```bash
   cd path/to/swiggy-instamart-casestudy
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: If you encounter dependency resolution issues with BPTK-Py, manually install `parsimonious==0.10.0` as noted in the requirements file).*

4. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

*(Note: The synthetic datasets in the strategy notebooks, such as the n=1,143 simulated store network and the n=500,000 uplift model user base, are calibrated methodology exhibits and are not real proprietary data).*

---

## 🤖 AI Assistance Disclosure
This project was conceptualized, structured, and directed by the author (with brainstorming support from Google AI). **Claude Code** acted as the primary coding assistant to build the entire Python/ML codebase. **Claude** was utilized as a deep web research assistant to surface and validate primary financial filings and correct data discrepancies prior to modeling.
