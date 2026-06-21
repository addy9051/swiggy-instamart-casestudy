"""
Generates 06d_strategy_simulation_summary.ipynb using nbformat.
Run with: python3 build_notebook_06d.py  (from the src/ directory)

Strategy 6d — Cross-strategy comparison, combined impact, and sequencing.
Reads the artifact CSVs written by 06a / 06b / 06c so the summary always
reflects the latest run of each strategy notebook.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md(r"""# 06d — Strategy Simulation Summary
### Cross-strategy comparison, combined impact, and sequencing

**What this notebook does**

Notebooks 06a-06c each tested one strategy with a different method. This notebook pulls their
saved outputs together into the single view a decision-maker actually needs: how the three
compare, whether they stack, and in what order to run them. It reads the artifact CSVs the other
three notebooks wrote to `data/processed/`, so re-running any strategy notebook and then this one
keeps everything in sync.

**The three strategies, one line each:**

1. **S1 — Inventory-model transition** (RL): the biggest structural lever, blocked by a governance
   vote, not by economics.
2. **S2 — Cross-sell food-delivery users** (uplift + Monte Carlo): the most *controllable* lever —
   Swiggy already owns the 18.3M-user base.
3. **S3 — Hold expansion, grow density** (system dynamics): the *cheapest* lever — no capex, just
   discipline.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("../data/processed")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
SWIGGY = "#FC8019"; BLINKIT = "#0C9D61"; ZEPTO = "#8025FB"; NAVY = "#1B2A4A"

def safe_read(path, label):
    try:
        df = pd.read_csv(path)
        print(f"loaded {label}: {path.name}")
        return df
    except FileNotFoundError:
        print(f"!! {label} not found ({path.name}) — run its notebook first. Using None.")
        return None

mc_summary   = safe_read(PROCESSED / "b6b_monte_carlo_summary.csv", "S2 Monte Carlo summary")
mc_segments  = safe_read(PROCESSED / "b6b_cate_segments.csv",       "S2 CATE segments")
sd_experiments = safe_read(PROCESSED / "b6c_uncertainty_experiments.csv", "S3 uncertainty sweep")
s1_sensitivity = safe_read(PROCESSED / "b6a_reward_weight_sensitivity.csv", "S1 reward sensitivity")
""")

# ---------------------------------------------------------------------------
md(r"""## 1. Side-by-side comparison

The qualitative attributes are fixed properties of each strategy; the quantitative cells pull from
the saved artifacts where available so the table reflects the actual latest runs.
""")

code(r"""
# Pull a couple of live numbers from the artifacts where present
s2_gov_p50 = (mc_summary.loc[mc_summary.iloc[:,0]=="50%", "incremental_gov_cr"].values[0]
              if mc_summary is not None and "incremental_gov_cr" in mc_summary.columns else None)
s3_success = (sd_experiments["reached_breakeven"].mean()
              if sd_experiments is not None else None)

comparison = pd.DataFrame([
    dict(Strategy="S1 — Inventory-model transition",
         Method="PPO Reinforcement Learning",
         **{"Primary KPI": "Contribution margin / order",
            "Time to impact": "4-6 quarters",
            "Capex need": "High",
            "Key blocker": "75% shareholder vote (failed at 72.35%)",
            "Quant result": "Aggressive front-loaded transition beats do-nothing"}),
    dict(Strategy="S2 — Cross-sell food-delivery base",
         Method="X-Learner uplift + Monte Carlo",
         **{"Primary KPI": "Monthly transacting users",
            "Time to impact": "2-3 quarters",
            "Capex need": "Low",
            "Key blocker": "Targeting + conversion uncertainty",
            "Quant result": (f"P50 incremental GOV ~Rs.{s2_gov_p50:,.0f}cr/yr"
                             if s2_gov_p50 is not None else "see 06b")}),
    dict(Strategy="S3 — Hold expansion, grow density",
         Method="System Dynamics + deep uncertainty",
         **{"Primary KPI": "Avg density / margin",
            "Time to impact": "3-5 quarters",
            "Capex need": "None",
            "Key blocker": "Discipline to pause expansion",
            "Quant result": (f"Breakeven in {s3_success:.0%} of swept scenarios"
                             if s3_success is not None else "see 06c")}),
])
comparison.set_index("Strategy")
""")

# ---------------------------------------------------------------------------
md(r"""## 2. Cost vs. control vs. impact

A simple way to see why these three are complementary rather than competing: they sit at different
corners of a cost / control / structural-impact space. The bubble chart plots each on
*execution cost* (x) against *how much of the lever Swiggy directly controls* (y), with bubble size
standing in for structural impact ceiling.
""")

code(r"""
# Qualitative 1-5 scores (documented assumptions, not disclosed data)
strat = pd.DataFrame([
    dict(name="S1 Inventory model", cost=5, control=2, impact=5, color=ZEPTO),
    dict(name="S2 Cross-sell",      cost=2, control=5, impact=3, color=SWIGGY),
    dict(name="S3 Hold & densify",  cost=1, control=4, impact=4, color=BLINKIT),
])

fig, ax = plt.subplots(figsize=(8, 5.5))
for _, r in strat.iterrows():
    ax.scatter(r["cost"], r["control"], s=r["impact"] * 900, color=r["color"], alpha=0.55,
               edgecolor=NAVY, linewidth=1.5)
    ax.annotate(r["name"], (r["cost"], r["control"]), ha="center", va="center",
                fontsize=9, fontweight="bold")
ax.set_xlabel("Execution cost / capex  (1 = cheap, 5 = expensive)")
ax.set_ylabel("How directly Swiggy controls the lever  (1-5)")
ax.set_title("Bubble size = structural impact ceiling\nThe three strategies occupy different corners — they complement, not compete")
ax.set_xlim(0, 6); ax.set_ylim(0, 6)
ax.grid(alpha=0.15)
plt.tight_layout()
plt.savefig(PROCESSED / "b6d_chart_strategy_space.png", bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 3. Do they stack? A combined directional model

The three strategies aren't independent — they reinforce each other:

- **S3 → S1:** holding expansion lifts density and margin, which makes the inventory-model business
  case (and the shareholder vote) easier.
- **S2 → S1:** more users from cross-sell means the inventory model serves more orders at scale,
  improving its payback.
- **S3 → S2:** higher density means a better Instamart delivery experience, lifting cross-sell
  conversion.

This is a *directional* illustration of the combined contribution-margin path when the three run
together vs. each alone — built on the same calibrated assumptions, and explicitly not a forecast.
""")

code(r"""
quarters = np.arange(0, 13)
start_margin = -1.8   # % — Q4 FY26 disclosed

# Per-quarter margin contributions (pp) — documented illustrative assumptions
def ramp(total, lag, length, n=len(quarters)):
    out = np.zeros(n)
    for i in range(n):
        if i >= lag:
            out[i] = total * min((i - lag) / length, 1.0)
    return out

s3_path = ramp(1.6, 0, 5)    # cheap, fast
s2_path = ramp(1.1, 2, 5)    # starts q2, medium
s1_path = ramp(2.4, 4, 6)    # starts q4, biggest but slowest

combined = start_margin + s1_path + s2_path + s3_path
s3_only  = start_margin + s3_path

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(quarters, np.full_like(quarters, start_margin, dtype=float), color="#AEB6C2",
        linewidth=1.5, linestyle=":", label="Do nothing")
ax.plot(quarters, s3_only, color=BLINKIT, linewidth=2, label="S3 only (hold & densify)")
ax.plot(quarters, combined, color=SWIGGY, linewidth=2.5, label="All three, sequenced")
ax.axhline(0, color=NAVY, linestyle="--", linewidth=1)
ax.fill_between(quarters, 0, combined, where=(combined >= 0), color=SWIGGY, alpha=0.12)
ax.set_xlabel("Quarter"); ax.set_ylabel("Instamart contribution margin (%)")
ax.set_title("Directional combined impact (illustrative, not a forecast)")
ax.legend()
plt.tight_layout()
plt.savefig(PROCESSED / "b6d_chart_combined_impact.png", bbox_inches="tight")
plt.show()

be = quarters[np.argmax(combined >= 0)] if (combined >= 0).any() else None
print(f"Illustrative combined breakeven: Q{be}" if be is not None else "no breakeven in window")
""")

# ---------------------------------------------------------------------------
md(r"""## 4. Recommended sequencing

The dependency structure implies an order. Run the cheap, fully-controllable lever first; use the
margin and experience it buys to unlock the harder, higher-impact levers.
""")

code(r"""
seq = pd.DataFrame([
    dict(Phase="Quarter 0-2", Move="S3 — Hold expansion",
         Why="Zero capex, fully in Swiggy's control. Stops density dilution immediately and "
             "builds the stable base the other two need."),
    dict(Phase="Quarter 2-4", Move="S2 — Cross-sell campaign",
         Why="Higher density from S3 improves the Instamart delivery experience, lifting cross-sell "
             "conversion. Grows MTU from the existing base at low cost."),
    dict(Phase="Quarter 4+",  Move="S1 — Revisit inventory vote",
         Why="Improved margin (S3) and higher order volume (S2) strengthen the business case and the "
             "shareholder-vote odds for the biggest structural lever."),
])
seq.set_index("Phase")
""")

code(r"""
# Simple Gantt-style visual of the sequencing
fig, ax = plt.subplots(figsize=(9, 3))
bars = [("S3 Hold & densify", 0, 12, BLINKIT),
        ("S2 Cross-sell",      2, 10, SWIGGY),
        ("S1 Inventory vote",  4, 8,  ZEPTO)]
for i, (name, start, dur, color) in enumerate(bars):
    ax.barh(i, dur, left=start, height=0.5, color=color, alpha=0.8)
    ax.text(start + 0.2, i, name, va="center", ha="left", fontsize=10, color="white", fontweight="bold")
ax.set_yticks([]); ax.set_xlabel("Quarter"); ax.set_xlim(0, 12)
ax.set_title("Recommended sequencing: cheap & controllable first, structural & slow last")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PROCESSED / "b6d_chart_sequencing.png", bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 5. Overall verdict and shared limitations

**Overall:** the three strategies are complementary, not competing. The cheapest and most
controllable lever (S3) is also the right *first* move, because it buys the margin and experience
that make the harder levers (S2, then S1) easier. Sequenced this way, the directional model shows a
materially faster path to contribution-margin breakeven than any single strategy alone — while each
individual notebook is honest about the conditions under which its strategy works.

**Shared limitation across all three (state this plainly to any reviewer):** every simulation here is
built on data calibrated to public disclosures, not Swiggy's internal operational data. The models
correctly capture the *direction* and *relative magnitude* of effects and the *conditions* under which
each strategy works — they are not operationally precise forecasts. What would sharpen them:

- **S1 (RL):** real store-level margin/density data → replace the calibrated density→margin form and
  the unsourced transition-cost estimate.
- **S2 (uplift):** real user-level order history → replace the synthetic user base; run a live pilot to
  pin down the conversion rate that dominates the Monte Carlo.
- **S3 (SD):** real quarterly capex and store-vintage data → tighten the four swept coefficients from
  wide ranges to narrow ones.

This is standard scenario-modelling practice: build with the best available data, be explicit about
every assumption, and update as better data arrives. The combination of three different methods
(RL, causal ML, system dynamics) on one coherent business problem — each reported with its own honest
limitations — is the actual deliverable.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("notebooks/06d_strategy_simulation_summary.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/06d_strategy_simulation_summary.ipynb")
