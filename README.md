# Italian COVID-19 Mobility Network Model

Code for the epidemiological model in "[Targeted policies and household consumption dynamics: Evidence from high-frequency transaction data](https://www.sciencedirect.com/science/article/pii/S0167268124001926)" (Bonaccorsi et al., *JEBO*, 2024).

This codebase adapts the mobility network model from "[Mobility network models of COVID-19 explain inequities and inform reopening](https://www.nature.com/articles/s41586-020-2923-3)" (Chang et al., *Nature*, 2021) for Italian regions and provinces. 

The original study used US Metropolitan Statistical Areas (MSAs) and SafeGraph mobility data; this version is fitted to Italian mobility and official case count data, covering regions such as Lombardia, Veneto, and Toscana. The mobility data is obtained from high-frequency transaction data from a major Italian bank.

## How the Analysis Works

### The Mobility Network

The core of the model is a **bipartite network** linking two types of nodes:

- **Municipalities**: small geographic units representing residential areas and their populations, originally Census block groups (CBGs). 
- **Points of interest (POIs)**: locations that people visit — shops, restaurants, workplaces, etc.

Each edge in the network carries a visit count: how many residents from a given municipality visited a given POI in a given hour. These hourly visit matrices are the main input to the epidemic model.

### The Disease Model (SEIR)

The epidemic model is a compartmental **SEIR** model (Susceptible → Exposed → Infectious → Removed), run on top of the mobility network. At each hourly timestep, the model:

1. **Computes infection rates at each POI** based on the fraction of visitors who are currently infectious. The per-POI transmission rate scales with visit density and is proportional to `psi / area`, where `psi` is a fitted transmission parameter and `area` is the POI's floor area.
2. **Computes a baseline home transmission rate** (`home_beta`) representing infections that occur regardless of mobility (e.g., within households).
3. **Draws new infections** per CBG by combining the POI-mediated and home-mediated infection rates using a Poisson approximation. New infections enter the **Latent** compartment.
4. **Advances compartments**: after a latency period (~4 days), latent individuals become infectious; after an infectious period (~3.5 days), they move to Removed.

The model is **stochastic**: each parameter configuration is run with multiple random seeds (default: 90) to account for demographic randomness, and results are averaged across seeds.

### Parameter Fitting via Grid Search

Three key parameters govern transmission:

| Parameter | Meaning |
|---|---|
| `home_beta` | Per-hour probability of infection at home |
| `poi_psi` | Transmission scaling at POIs (higher = more transmission per visit) |
| `p_sick_at_t0` | Fraction of the population initially infected at simulation start |

Fitting proceeds in two steps:

1. **R0 calibration** (`calibrate_r0`): a coarse sweep to identify the plausible range of `home_beta` and `poi_psi` values that produce a biologically realistic basic reproduction number (R0 between ~0.1 and ~2). This narrows the search space before the full grid search.

2. **Grid search** (`normal_grid_search`): an exhaustive sweep over all combinations of `home_beta` (10 values), `poi_psi` (15 values), and `p_sick_at_t0` (13 values). Each combination is evaluated by comparing simulated case trajectories against real reported case counts. The configurations whose simulated curves best match the data (within a loss tolerance) are retained as **best-fit models**.

### Downstream Experiments

Once best-fit models are identified, the same fitted parameters can be used to run counterfactual simulations:

- **`test_interventions`**: Re-runs the model with visits to specific POI categories set to zero, quantifying the infection reduction attributable to closing each category.
- **`test_retrospective_counterfactuals`**: Simulates alternative mobility reductions that could have been applied in the past, estimating how many infections they would have prevented.
- **`test_max_capacity_clipping` / `test_uniform_proportion_of_full_reopening`**: Simulate partial reopening by scaling POI visits down to a fraction of capacity or pre-lockdown levels.
- **`rerun_best_models_and_save_cases_per_poi`**: Re-runs best-fit models and records per-POI infection counts on each day, enabling attribution analysis of which locations drove the most transmission.

## Installation

Tested on Windows Subsystem for Linux (WSL).

1. Create and activate a conda environment:

   ```bash
   conda create -n <env_name> python=3.7.11 pip ipython ipykernel numpy scipy matplotlib pandas seaborn networkx dask statsmodels pystan h5py pytables xlrd jupyterlab
   conda activate <env_name>
   ```

2. Install additional packages with pip:

   ```bash
   pip install argparse tqdm fbprophet
   ```

## Initial Setup

1. Run the setup script to create the required data and results folders:

   ```bash
   bash setup_folder.sh
   ```

2. Download and place your Italian mobility and case data in the appropriate folders (see `covid_constants_and_util.py` for path configuration).

3. By default, experiments will use 2 CPU cores. To increase this, register your machine in `run_parallel_models.py` inside `get_computer_and_resources_to_run`.

## Running the Pipeline

### 1. Prepare input data

Use the relevant notebook (`export_italian_data.ipynb` or `get_data_for_shorter_period_type.ipynb`) to generate the mobility matrices and provincial population files.

### 2. Calibrate R0

```bash
python model_experiments.py run_many_models_in_parallel calibrate_r0 -msa <region>
```

### 3. Run grid search

```bash
python model_experiments.py run_many_models_in_parallel normal_grid_search -msa <region>
```

Examples:

```bash
python model_experiments.py run_many_models_in_parallel normal_grid_search -msa Veneto
python model_experiments.py run_many_models_in_parallel normal_grid_search -msa Toscana
```

### 4. Run additional experiments (optional, require completed grid search)

| Experiment | Description |
|---|---|
| `test_interventions` | Simulate the effect of reopening each POI subcategory |
| `test_retrospective_counterfactuals` | Simulate counterfactual mobility reductions |
| `test_max_capacity_clipping` | Partial reopening via capacity clipping |
| `test_uniform_proportion_of_full_reopening` | Partial reopening via uniform visit reduction |
| `rerun_best_models_and_save_cases_per_poi` | Save per-POI infection counts for best-fit models |

Set `min_timestring_to_load_best_fit_models_from_grid_search` before running experiments that depend on grid search results.

### 5. Analyze results

Use the figure generation scripts (`make_figures_*.py`) or notebooks to visualize model output.

## Key Parameters

| Parameter | Location | Description |
|---|---|---|
| `BIGGEST_MSAS` | `covid_constants_and_util.py` | Italian regions/provinces to run |
| `num_seeds` | `run_parallel_models.py` | Number of random seeds per model |
| `MAX_MODELS_TO_TAKE_PER_MSA` | `covid_constants_and_util.py` | Number of best-fit models to retain |
| `ACCEPTABLE_LOSS_TOLERANCE` | `covid_constants_and_util.py` | Grid search tolerance |
| Grid search ranges | `run_parallel_models.py` | Parameter ranges for beta and psi |

## File Reference

| File | Description |
|---|---|
| `model_experiments.py` | Main entry point; manager/worker CLI for running experiments |
| `disease_model.py` | SEIR-like epidemic model on the mobility network |
| `run_parallel_models.py` | Parallel job manager; generates and dispatches model configs |
| `run_one_model.py` | Single-model fitting and saving logic |
| `covid_constants_and_util.py` | Global constants, paths, and utility functions |
| `mobility_processing.py` | Processes mobility data into hourly visit matrices |
| `model_evaluation.py` | Evaluates model fit against real case counts |
| `model_evaluation_subset.py` | Evaluation broken down by province subsets |
| `model_results.py` | Loads and aggregates saved model results |
| `search_model_results.py` | CLI tool to search and inspect saved model results |
| `helper_methods_for_aggregate_data_analysis.py` | Data loading and aggregation helpers |
| `helper_methods_for_plotting.py` | Plotting utilities |
| `helper_method_for_census_data.py` | Census data processing helpers |
| `make_figures_regions_paper.py` | Figures for regional analysis |
| `make_figures_ita_provinces_phase2.py` | Figures for provincial analysis (phase 2) |
| `make_figures_provinces_single_region.py` | Figures for a single region's provinces |
| `plot_contributions_to_infections.py` | Plots per-POI infection contributions |
| `online_reopening.py` | Online reopening scenario simulations |
| `utilities.py` | Shared utility functions |

## Notes

- Jobs launched via `run_many_models_in_parallel` run automatically in the background using `nohup`.
- To search or inspect previously saved model results, use `search_model_results.py`.
- See `documentation.md` for a detailed walkthrough of the internal code structure.
- See `changelog.md` for version history.

## References
[[1](https://www.sciencedirect.com/science/article/pii/S0167268124001926)] Bonaccorsi, G., Scotti, F., Pierri, F., Flori, A., & Pammolli, F. (2024). Targeted policies and household consumption dynamics: Evidence from high-frequency transaction data. Journal of Economic Behavior & Organization, 224, 111-134.

[[2](https://www.nature.com/articles/s41586-020-2923-3)] Chang, S., Pierson, E., Koh, P. W., Gerardin, J., Redbird, B., Grusky, D., & Leskovec, J. (2021). Mobility network models of COVID-19 explain inequities and inform reopening. Nature, 589(7840), 82-87.
