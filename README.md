# Topology-Agnostic Autonomous Traffic Light Controller Using Reinforcement Learning

Code, configurations, trained models and evaluation data.
*"Topology-Agnostic Autonomous Traffic Light Controller Using Reinforcement
Learning"* (Murin, Klein, Kainz, Semanova, Michalko).

A single Proximal Policy Optimisation (PPO) agent controls heterogeneous signalised
intersections through a **canonical intersection model** (13 vehicle lane groups + 14
pedestrian crossing segments) paired with a **deterministic phase-matching safety
layer**. The agent is evaluated in SUMO against three baselines: a deployed fixed-time
cycle, a vehicle-actuated controller, and a max-pressure controller.

## How it works

The architecture abstracts the physical SUMO details away from the
reinforcement-learning agent. A Stable-Baselines3 agent is wrapped in a custom
Gymnasium environment (`SumoIntersectionEnv`). On each decision tick:

1. The agent outputs a 27-dimensional continuous action array -- a preference
   score in [0, 1] for every canonical slot.
2. `SumoIntersectionEnv` passes the array to the `TrafficLightController`.
3. The controller scores it against a list of strictly non-colliding phase
   definitions (loaded from YAML) and selects the best matching phase by dot
   product (`scoring="linear"`), then decides whether a phase change is needed.
4. If a shift is initiated, a multi-step transition state machine
   (yellow -> all-red -> optional pedestrian extra-red -> red-yellow -> green)
   prevents hazardous instantaneous changes.
5. Commands are issued to SUMO through the TraCI API, the simulation advances,
   and the next 147-dimensional observation is built from lane densities, lead
   gaps, accumulated waits and signal ages.

### Canonical model
The `TrafficLightController` translates a standardised canonical space -- 13
vehicular slots and 14 pedestrian slots (4 approaches x 3 lane positions plus a
rail track) -- onto arbitrary physical SUMO layouts defined by the user. The
network output never reaches the lights directly: only a valid, non-conflicting
phase from the per-intersection library can be selected, so a collision is
impossible by construction. Inactive slots are left empty in the YAML and never
receive green.

### Reward and training
The dense reward combines trip quality, the step change in total waiting time,
and moving-vehicle and pedestrian terms. Training (`train_model.py`) runs
several SUMO environments in parallel through SB3's `SubprocVecEnv`, with a
curriculum that raises difficulty as training progresses.

### Evaluation
Each controller is evaluated on three real intersections in Kosice under four
demand scales x three variance levels, ten paired runs each, with a paired
Wilcoxon signed-rank test.

## Requirements

- Python 3.10
- SUMO 1.26.0. Install SUMO and set `SUMO_HOME`,
  or `pip install eclipse-sumo` to obtain `sumo`, `traci` and `libsumo`.
- Python packages: see `requirements.txt`
  (`pip install -r requirements.txt`). Key versions: Stable-Baselines3 2.8.0, PyTorch.

The reported model was trained on a Linux workstation with an AMD GPU (ROCm build of
PyTorch). CPU or CUDA also work; evaluation is CPU-bound (single-threaded SUMO).

### Tested environment
- SUMO 1.26.0 (networks built with netedit 1.26.0, net format 1.20)
- Python 3.10.19
- Stable-Baselines3 2.8.0, PyTorch 2.11.0 (ROCm build), Gymnasium 1.2.3, NumPy 2.2.6
- Linux (Ubuntu 24.04), AMD GPU (ROCm). CPU and CUDA also work; evaluation is
  CPU-bound (single-threaded SUMO).

## Repository structure

The layout is flat (scripts and model checkpoints in the root), matching the
relative paths the scripts use:

```
*.py            Python sources (canonical model, env, controllers, training, evaluation, stats)
models/         Trained checkpoints (see below)
configs/        Per-intersection YAML configurations (canonical + real layouts)
simulation/     SUMO networks (.net.xml), configs (.sumocfg), routes, and demand profiles
                (TrafficDensity/*.csv = five-minute vehicle/pedestrian counts)
results/        Per-run evaluation CSVs and paired-statistics output reproducing the tables
```

Run every command from the repository root.

### Scripts

Entry points (see *Reproducing the results*):

- `evaluate_strategy.py` — main evaluation: fixed-time, vehicle-actuated and PPO.
- `evaluate_maxpressure.py` — the max-pressure baseline (standalone).
- `compute_statistics.py` — paired Wilcoxon statistics and result tables.
- `evaluate_ablation.py`, `evaluate_lead_ablation.py` — reward/algorithm and lead-distance ablations.
- `train_model.py` — train a model from scratch (also defines the `SumoIntersectionEnv`).

Core modules (imported by the above):

- `TrafficLightController.py` — canonical model and deterministic phase-matching layer.
- `intersection_config.py` — parses the per-intersection YAML into a controller.
- `IntersectionRoutes.py` — canonical route-slot to SUMO lane mapping.
- `actuated_controller.py`, `max_pressure_controller.py` — the two non-learning baselines.
- `generate_density_routes.py`, `generate_routes.py`, `interpolate_traffic_density.py` — demand/route generation from the count profiles.

Utilities (optional, not needed to reproduce the tables):

- `eval_graphs.py` — plots evaluation overviews from the result CSVs.

### Models
- `PPO-FINAL.zip` — the deployed model reported in the paper. It is the best-performing
  checkpoint of a single training run (random seed not fixed; ~4.43M steps).
- `ATLAS5_seed*_PPO_full_Flight1.zip` — sample seeds from the ten-seed training-variance
  study. The full ten-seed evaluation outputs are in `results/` and are summarised by
  `reproduce_variance_generalisation.py`.
- `ATLAS5_PPO_trip / PPO_dwait / SAC_full` — reward/algorithm ablation checkpoints
  (development checkpoints; see the indicative ablation in the paper).

## Reproducing the results

All commands are run from the repository root with SUMO available and the packages installed.
Per-run route files and varied-demand CSVs are generated deterministically from the
scenario identifier, so every controller faces identical demand (paired comparison).

1. **Main evaluation** (fixed-time, vehicle-actuated, PPO) across four demand scales,
   three variance levels, ten paired runs per cell:
   ```
   python evaluate_strategy.py
   ```
   Writes `evaluation_results_<intersection>.csv`.

2. **Max-pressure baseline** (training-free; same canonical phase library and demand):
   ```
   python evaluate_maxpressure.py            # all intersections
   python evaluate_maxpressure.py Tory 0.2   # one slice (intersection, scale)
   ```
   Writes `evaluation_maxpressure_<intersection>[_s<scale>].csv`.

3. **Paired statistics and result tables** (means, SD, paired Wilcoxon signed-rank):
   ```
   python compute_statistics.py
   ```
   Writes `stats_out/` (summaries, tests, LaTeX tables).

4. **Ablation** (reward components, algorithm, lead-distance feature) on Toryska:
   ```
   python evaluate_ablation.py
   python evaluate_lead_ablation.py
   ```

5. **Training from scratch** (optional):
   ```
   python train_model.py
   ```
   Environment variables control the recipe (e.g. `REWARD_MODE`, `ALGO`, `TRAIN_STEPS`,
   `SEED`, `DEVICE`, `USE_LIBSUMO`). The reported model uses the full multi-component
   reward and a mixed-difficulty distribution of the canonical and real layouts.

## Training-seed variability and cross-topology generalisation

This covers the seed-variance study and the two foreign held-out intersections from the paper.

Training is seeded through the environment, so a ten-seed sweep is the same command with a
different `SEED`. Each run keeps its own `logs_seed<N>/` and `ATLAS5_seed<N>_PPO_full_Flight1.zip`,
so several can run at once without clobbering each other (`USE_LIBSUMO=1` is much faster):

```
set "SEED=1" & set "TRAIN_STEPS=4430320" & set "USE_LIBSUMO=1" & python train_model.py
```

To score one seed on the three Kosice intersections, point `PPO_MODEL` at it. Only PPO needs
re-running here; the baselines are seed-independent and come from the main evaluation:

```
set "PPO_MODEL=ATLAS5_seed1_PPO_full_Flight1.zip" & set "EVAL_TAG=_seed1" & set "PPO_ONLY=1" & set "EVAL_ONLY=Aupark,Galeria,Herlianska" & python evaluate_strategy.py
```

The deployed model is chosen on the Monaco validation intersection, which never appears in the
Kosice test set, so selection stays off the test data. Score the seeds with `EVAL_ONLY=Monaco`
and keep the one with the lowest wait. Luxembourg and Herlianska are then evaluated only as
independent held-out tests. Their Static/Actuated baselines run with `SKIP_PPO=1`:

```
set "SKIP_PPO=1" & set "EVAL_TAG=_baseline" & set "EVAL_ONLY=Herlianska,Monaco,LuST" & python evaluate_strategy.py
```

Both intersections are cropped from public SUMO scenarios and mapped onto the canonical model
(`configs/config_monaco.yaml`, `configs/config_lust.yaml`). Monaco is intersection 134872 from
MoST. LuST is intersection -3976 from the Luxembourg scenario; its sub-network was cut with
`netconvert --keep-edges.explicit`, and real demand was pulled from the LuST `DUARoutes` with
`cutRoutes.py`, trimmed to the evening peak (17:30-18:30) and shifted to start at t=0 so it
fits one 2000-step episode. Demand is scaled at evaluation time with SUMO's `--scale`, and the
phases in each config are a direct translation of the intersection's own collision-free program.

With the per-seed and `_baseline` CSVs sitting in `results/`, the tables come straight from:

```
python reproduce_variance_generalisation.py results
```

## Note on reproducibility and variance

The deployed checkpoint is the favourable end of a wide run-to-run distribution. Across
ten independently seeded retrainings, the per-vehicle wait on the test intersections had
a coefficient of variation of 16.4% (mean 67.3 s, SD 11.0 s, range 54.5-93.4 s); the
deployed checkpoint reaches a 73% reduction at the poorly-adapted intersection, which is
the best run rather than a typical one. Single-checkpoint ablation effects should be read
as indicative; see the paper.

## Use of AI assistance

The authors used Claude (Anthropic, Inc.) as an auxiliary tool for language editing and manuscript restructuring. The reinforcement-learning model, its training, the simulation environment, the experimental design, the data, the analyses, and the interpretation of the results are the authors’ own. All content and computational outputs were reviewed and verified by the authors, who take full responsibility for the final article.

## License

MIT (see `LICENSE`).

## Contact

Miroslav Murin (corresponding author): miroslav.murin@tuke.sk
