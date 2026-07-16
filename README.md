# Swiggy Instamart Strategy Case Study

This repository contains a comprehensive data-driven case study analyzing Swiggy Instamart's path to profitability compared to competitors like Blinkit (Eternal) and Zepto. The project utilizes data analysis, machine learning (Reinforcement Learning, Uplift Modeling), and System Dynamics simulations to evaluate unit economics, customer behavior, logistics, and potential business strategies.

## Project Structure

- `data/`: Contains raw and processed datasets used across the notebooks.
- `notebooks/`: Contains the generated Jupyter notebooks for exploratory data analysis, modeling, and simulation.
- `src/`: Contains Python scripts (e.g., `build_notebook.py`) used to build and generate the analytical notebooks from source code.
- `models/`: Directory for storing trained machine learning or simulation models.
- `requirements.txt`: Python dependencies required to run the project.

## Analysis Modules (Notebooks)

The analysis is broken down into a series of interconnected Jupyter notebooks:

1. **`01_master_data.ipynb`**: Data loading, cleaning, and preprocessing. Constructs the core long-format dataset encompassing financial metrics, order volumes, and dark store counts from various primary and secondary sources.
2. **`02_unit_economics_density.ipynb`**: Deep dive into unit economics and dark store density, comparing Instamart's efficiency against its peers.
3. **`03_customer_behavior_monetization.ipynb`**: Analysis of customer behavior patterns and monetization strategies.
4. **`04_competitive_market_structure.ipynb`**: Evaluation of the competitive landscape in the quick commerce sector.
5. **`05_logistics_cost_benchmarking.ipynb`**: Benchmarking logistics and delivery costs.
6. **`06a_strategy1_rl_inventory_transition.ipynb`**: Strategy simulation utilizing Reinforcement Learning (using Stable Baselines3) for optimizing inventory transitions.
7. **`06b_strategy2_uplift_crosssell.ipynb`**: Strategy simulation utilizing Uplift Modeling (using Causal ML) to optimize cross-selling campaigns.
8. **`06c_strategy3_system_dynamics.ipynb`**: Modeling complex business interactions using System Dynamics (BPTK-Py) and handling deep uncertainty.
9. **`06d_strategy_simulation_summary.ipynb`**: Executive summary consolidating findings and recommendations from all the strategy simulations.

## Installation & Setup

To run the notebooks locally, it is recommended to use a Python virtual environment.

1. Navigate to the project root:
   ```bash
   cd path/to/swiggy-instamart-casestudy
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you run into dependency resolution issues with BPTK-Py, you may need to install `parsimonious==0.10.0` manually as noted in the `requirements.txt` file.*

## Usage

You can launch Jupyter Lab to interact with the notebooks:

```bash
jupyter lab
```

If you wish to regenerate the notebooks from their source Python scripts (which use `nbformat` to construct the `.ipynb` files programmatically), you can run the build scripts in the `src/` directory:

```bash
cd src
python build_notebook.py
python build_notebook_06a.py
# ... and so on
```
