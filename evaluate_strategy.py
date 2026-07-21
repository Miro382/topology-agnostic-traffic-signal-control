import os
import time
import csv
import glob
import random
import zlib

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

# Optional in-process backend: USE_LIBSUMO=1 swaps traci for libsumo (faster, identical results)
import sys as _sys
if os.environ.get("USE_LIBSUMO", "0") == "1":
    try:
        import libsumo as _libsumo
        if not hasattr(_libsumo, "exceptions"):
            import types as _types
            _ex = _types.ModuleType("traci.exceptions")
            _ex.TraCIException = getattr(_libsumo, "TraCIException", Exception)
            _ex.FatalTraCIError = getattr(_libsumo, "FatalTraCIError", Exception)
            _libsumo.exceptions = _ex
            _sys.modules["traci.exceptions"] = _ex
        _sys.modules["traci"] = _libsumo
        print("[backend] Using libsumo (in-process)")
    except Exception as _e:
        print(f"[backend] libsumo unavailable ({_e}); falling back to traci")
from train_model import SumoIntersectionEnv
from intersection_config import load_intersection_config
from actuated_controller import ActuatedController
import traci

# Import the density route generator
from generate_density_routes import generate_density_routes
from interpolate_traffic_density import process_traffic_data

# Configuration
SCENARIOS_DIR = "simulation/TrafficDensity"

# Per-intersection model/static configs;
INTERSECTION_CONFIG_MAP = {

    "Aupark": [
        {
            "id": "Jantárova - Palackého",
            "config": "config_aupark.yaml",
            "models": {
                "PPO": "models/PPO-FINAL.zip",
            },
            "static_cycle": [
                {"duration": 24, "signals": [0, 0, 0,  0, 0, 0,  0, 0, 0,  1, 1, 1,  0,    0, 1, 0, 0,  1, 1, 0,  0, 1, 0, 1,  0, 0, 1]},
                {"duration": 12, "signals": [0, 0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0,    0, 1, 0, 1,  0, 0, 1,  0, 1, 0, 0,  1, 1, 0]},
                {"duration": 6,  "signals": [0, 1, 1,  1, 0, 0,  0, 0, 0,  0, 0, 0,  0,    0, 0, 0, 1,  0, 1, 0,  0, 1, 0, 0,  1, 1, 1]},
                {"duration": 8,  "signals": [0, 0, 0,  0, 0, 0,  0, 1, 1,  1, 0, 0,  0,    0, 1, 0, 0,  1, 1, 1,  0, 0, 0, 1,  0, 1, 0]},
            ]
        },
    ],
    "Galeria": [
        {
            "id": "Toryská - Trieda SNP",
            "config": "config_galeria.yaml",
            "models": {
                "PPO": "models/PPO-FINAL.zip",
            },
            "static_cycle": [
                {"duration": 20, "signals": [1, 1, 0,  0, 0, 0,  1, 1, 0,  0, 0, 0,  1,    0, 0, 0, 0,  0, 1, 1,  0, 0, 0, 0,  0, 1, 1]},
                {"duration": 17, "signals": [1, 1, 1,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0,    0, 0, 1, 1,  0, 1, 0,  1, 1, 1, 0,  0, 1, 1]},
                {"duration": 25, "signals": [0, 0, 0,  0, 1, 1,  1, 0, 0,  0, 0, 0,  0,    1, 1, 1, 1,  0, 0, 1,  0, 1, 1, 0,  0, 1, 0]},
                {"duration": 21, "signals": [1, 0, 0,  0, 0, 0,  0, 0, 0,  1, 1, 0,  0,    0, 1, 1, 0,  0, 1, 0,  1, 1, 1, 1,  0, 0, 1]},
                {"duration": 17, "signals": [0, 0, 0,  0, 0, 0,  1, 1, 1,  0, 0, 0,  0,    1, 1, 1, 0,  0, 1, 1,  0, 0, 1, 1,  0, 1, 0]},
            ]
        },
    ],
    "Herlianska": [
        {
            "id": "Herlianska - Zelená Stráň",
            "config": "config_shopbox.yaml",
            "models": {
                "PPO": "models/PPO-FINAL.zip",
            },
            "static_cycle": [
                {"duration": 14, "signals": [1, 1, 1,  1, 0, 0,  0, 0, 0,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 14, "signals": [0, 0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 29, "signals": [1, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 1,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 38, "signals": [0, 0, 0,  0, 0, 0,  1, 1, 1,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 50, "signals": [1, 1, 0,  0, 0, 0,  1, 1, 0,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
            ]
        },
    ],
    "Monaco": [
        {
            "id": "Monaco - 134872 (MoST)",
            "config": "config_monaco.yaml",
            "sumocfg": "simulation/Monaco.sumocfg",
            "models": {
                "PPO": "models/PPO-FINAL.zip",
            },
            "static_cycle": [
                {"duration": 22, "signals": [0,0,0, 1,0,1, 0,0,0, 0,0,0, 0,   0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]},
                {"duration": 22, "signals": [0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,   0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]},
                {"duration": 22, "signals": [0,0,0, 0,0,0, 1,1,0, 0,0,0, 0,   0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]},
            ],
        },
    ],
    "LuST": [
        {
            "id": "LuST - -3976 (Luxembourg)",
            "config": "config_lust.yaml",
            "sumocfg": "simulation/LuST.sumocfg",
            # Held-out, real demand: fixed cut route file; demand level via SUMO --scale.
            "real_route_file": "simulation/LuST_rush.rou.xml",
            "begin": 0,
            "models": {
                "PPO": "models/PPO-FINAL.zip",
            },
            "static_cycle": [
                {"duration": 30, "signals": [0,1,1, 0,0,0, 1,1,0, 0,0,0, 0,   0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]},
                {"duration": 10, "signals": [0,0,1, 0,0,0, 0,0,0, 0,0,0, 0,   0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]},
                {"duration": 30, "signals": [0,0,0, 1,0,1, 0,0,0, 1,1,1, 0,   0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]},
                {"duration": 10, "signals": [0,0,0, 0,0,1, 0,0,0, 0,0,1, 0,   0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]},
            ],
        },
    ],
}

"""
    "Galeria": [
        {
            "id": "Toryská - Trieda SNP",
            "config": "config_galeria.yaml",
            "models": {
                "PPO": "logs/sumo_ppo_4430320_steps-DIAMOND.zip",
            },
            "static_cycle": [
                {"duration": 20, "signals": [1, 1, 0,  0, 0, 0,  1, 1, 0,  0, 0, 0,  1,    0, 0, 0, 0,  0, 1, 1,  0, 0, 0, 0,  0, 1, 1]},
                {"duration": 17, "signals": [1, 1, 1,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0,    0, 0, 1, 1,  0, 1, 0,  1, 1, 1, 0,  0, 1, 1]},
                {"duration": 25, "signals": [0, 0, 0,  0, 1, 1,  1, 0, 0,  0, 0, 0,  0,    1, 1, 1, 1,  0, 0, 1,  0, 1, 1, 0,  0, 1, 0]},
                {"duration": 21, "signals": [1, 0, 0,  0, 0, 0,  0, 0, 0,  1, 1, 0,  0,    0, 1, 1, 0,  0, 1, 0,  1, 1, 1, 1,  0, 0, 1]},
                {"duration": 17, "signals": [0, 0, 0,  0, 0, 0,  1, 1, 1,  0, 0, 0,  0,    1, 1, 1, 0,  0, 1, 1,  0, 0, 1, 1,  0, 1, 0]},
            ]
        },
    ]
    "Aupark": [
        {
            "id": "Jantárova - Palackého",
            "config": "config_aupark.yaml",
            "models": {
                "PPO": "logs/sumo_ppo_4430320_steps-DIAMOND.zip",
            },
            "static_cycle": [
                {"duration": 24, "signals": [0, 0, 0,  0, 0, 0,  0, 0, 0,  1, 1, 1,  0,    0, 1, 0, 0,  1, 1, 0,  0, 1, 0, 1,  0, 0, 1]},
                {"duration": 12, "signals": [0, 0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0,    0, 1, 0, 1,  0, 0, 1,  0, 1, 0, 0,  1, 1, 0]},
                {"duration": 6,  "signals": [0, 1, 1,  1, 0, 0,  0, 0, 0,  0, 0, 0,  0,    0, 0, 0, 1,  0, 1, 0,  0, 1, 0, 0,  1, 1, 1]},
                {"duration": 8,  "signals": [0, 0, 0,  0, 0, 0,  0, 1, 1,  1, 0, 0,  0,    0, 1, 0, 0,  1, 1, 1,  0, 0, 0, 1,  0, 1, 0]},
            ]
        },
    ],
    "Herlianska": [
        {
            "id": "Herlianska - Zelená Stráň",
            "config": "config_shopbox.yaml",
            "models": {
                "PPO": "logs/sumo_ppo_4430320_steps-DIAMOND.zip",
            },
            "static_cycle": [
                {"duration": 14, "signals": [1, 1, 1,  1, 0, 0,  0, 0, 0,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 14, "signals": [0, 0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 29, "signals": [1, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 1,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 38, "signals": [0, 0, 0,  0, 0, 0,  1, 1, 1,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
                {"duration": 50, "signals": [1, 1, 0,  0, 0, 0,  1, 1, 0,  0, 0, 0,  0,    0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0]},
            ]
        },
    ],
"""

SCALES = [0.2, 0.6, 1.0, 1.4]
VARIANCES = [0.3, 0.5, 0.8]
RUNS_PER_TEST = 10
RESULTS_FILE = "evaluation_results.csv"

def run_simulation(env, model=None, static_cycle=None, actuated=None, steps=2000):
    obs, info = env.reset()
    if actuated is not None:
        actuated.reset()
    
    total_reward = 0
    cycle_index = 0
    time_in_phase = 0
    arrived_vehicles = 0
    
    for _ in range(steps):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        elif actuated is not None:
            # Vehicle-actuated control: gap-based extension over the canonical phase library
            action = actuated.decide()
        else:
            # Static cycle logic
            current_phase = static_cycle[cycle_index]
            action = current_phase["signals"]
            time_in_phase += 1
            if time_in_phase >= current_phase["duration"]:
                time_in_phase = 0
                cycle_index = (cycle_index + 1) % len(static_cycle)
                
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        arrived_vehicles += traci.simulation.getArrivedNumber()
        
        if done or truncated:
            break
            
    # Extract authentic metrics from environment and TraCI
    total_wait = env.total_cumulative_wait_time if hasattr(env, 'total_cumulative_wait_time') else 0
    max_wait_sec = env.max_wait_time_sec if hasattr(env, 'max_wait_time_sec') else 0
    vehicles_seen = len(env.vehicles_seen) if hasattr(env, 'vehicles_seen') else arrived_vehicles
    
    stats = {
        "vehicles_total": vehicles_seen,
        "total_wait": int(total_wait),
        "avg_wait": round(total_wait / vehicles_seen, 2) if vehicles_seen > 0 else 0,
        "max_wait": int(max_wait_sec),
        "total_reward": round(total_reward, 2)
    }
        
    return stats

def evaluate():
    fieldnames = ["intersection", "variance", "agent", "scale", "run", "vehicles_total", "total_wait", "avg_wait", "max_wait", "total_reward"]
    
    # Iterate through all CSV files in the scenarios directory
    csv_files = glob.glob(os.path.join(SCENARIOS_DIR, "*.csv"))

    for csv_file in csv_files:
        name = os.path.splitext(os.path.basename(csv_file))[0]
        if name not in INTERSECTION_CONFIG_MAP:
            print(f"Skipping {name}: No configuration found in INTERSECTION_CONFIG_MAP.")
            continue

        _only = os.environ.get("EVAL_ONLY")
        if _only and name not in [x.strip() for x in _only.split(",")]:
            print(f"Skipping {name}: not in EVAL_ONLY.")
            continue
            
        configs = INTERSECTION_CONFIG_MAP[name]
        if not isinstance(configs, list):
            configs = [configs]
            
        for data in configs:
            config_id = data.get("id", name)
            print(f"Evaluating intersection: {name} (Config: {config_id})")
            
            # Per-seed eval overrides (env): PPO_MODEL = model path, EVAL_TAG = suffix.
            _ppo_override = os.environ.get("PPO_MODEL")
            _eval_tag = os.environ.get("EVAL_TAG", "")

            # Start results file specific to this intersection config
            result_file = f"evaluation_results_{config_id}{_eval_tag}.csv"
            file_exists = os.path.exists(result_file)
            
            # Open file in append mode - safe against overwriting on rerun
            with open(result_file, 'a', newline='', encoding='utf-8') as current_f:
                current_writer = csv.DictWriter(current_f, fieldnames=fieldnames)
                if not file_exists:
                    current_writer.writeheader()
            
                controller, route_slots, bounds = load_intersection_config(data["config"])
                
                # 1. Load trained models
                loaded_models = {}
                # SKIP_PPO=1 -> baseline-only run (Static + Actuated), no model loaded.
                if "models" in data and os.environ.get("SKIP_PPO") != "1":
                    for m_name, m_path in data["models"].items():
                        is_ppo = "ppo" in m_path.lower() or "ppo" in m_name.lower()
                        if is_ppo and _ppo_override:
                            m_path = _ppo_override          # evaluate this seed's model
                        key = m_name + _eval_tag             # e.g. PPO_seed1 in the agent column
                        if is_ppo:
                            loaded_models[key] = PPO.load(m_path)
                        elif "sac" in m_path.lower() or "sac" in m_name.lower():
                            loaded_models[key] = SAC.load(m_path)
                        else:
                            try:
                                loaded_models[key] = PPO.load(m_path)
                            except:
                                loaded_models[key] = SAC.load(m_path)

                # Determine number of data rows in CSV (Row index 4+)
                df = pd.read_csv(csv_file, header=None)
                num_data_rows = range(4, len(df))

                # Loop through each row in the CSV
                for row_idx in num_data_rows:
                    
                    # 2. Iterate through scales
                    for scale in SCALES:
                        
                        # 3. Iterate through variances
                        for variance in VARIANCES:
                            print(f"  Scale: {scale*100}% | Variance: {variance} for Row Index {row_idx}")
                            
                            # 4. Build agent list (Models + Static + Actuated)
                            agents = list(loaded_models.keys())
                            if "static_cycle" in data and os.environ.get("PPO_ONLY") != "1":
                                agents.append("Static")
                                agents.append("Actuated")
                            
                            # Runs loop outside agents so all agents share the route file and seed (paired samples)
                            for run in range(RUNS_PER_TEST):
                                # Deterministic per-scenario seed (reproducible across runs, unlike hash())
                                seed = zlib.crc32(
                                    f"{name}|{row_idx}|{scale}|{variance}|{run}".encode()
                                ) % (2**31 - 1)
                                random.seed(seed)
                                np.random.seed(seed)
                                
                                # --- demand for this scenario+run ---
                                real_rf = data.get("real_route_file")
                                if real_rf:
                                    temp_route_file = real_rf   # held-out: fixed real-demand file, scaled via --scale
                                else:
                                    # generate ONE shared route file for this scenario+run
                                    temp_csv_file = f"simulation/TrafficDensity/{name}-varied-{variance}.csv"
                                    if variance > 0.0:
                                        process_traffic_data(csv_file, variance, temp_csv_file)
                                    else:
                                        temp_csv_file = csv_file

                                    temp_route_file = f"simulation/TrafficDensity/{name}-{variance}-density-scaled{scale}-run{run}.rou.xml"
                                    generate_density_routes(temp_csv_file, temp_route_file, row_index=row_idx, scale=scale)

                                    # Cleanup temp varied CSV
                                    if variance > 0.0 and os.path.exists(temp_csv_file):
                                        os.remove(temp_csv_file)
                                
                                # --- every agent runs the SAME route file and SAME seed ---
                                for agent_name in agents:
                                    print(f"    Run {run} | Agent: {agent_name} | Scale: {scale} | Variance: {variance}")
                                    
                                    sumo_cmd = [
                                        "sumo",
                                        "-c", data.get("sumocfg", f"simulation/{name}.sumocfg"),
                                        "-r", temp_route_file,
                                        "--start",
                                        "--quit-on-end",
                                        "--ignore-route-errors",
                                        "--step-length", "1.0",
                                        "--time-to-teleport", "-1",
                                        "--waiting-time-memory", f"{2000}",
                                        "--seed", str(seed),
                                        "--no-warnings", "true",
                                        "--message-log", "/dev/null"
                                    ]
                                    # Real-demand scenarios: rush-hour begin + demand scaling.
                                    if real_rf:
                                        sumo_cmd += ["--begin", str(data.get("begin", 0)),
                                                     "--scale", str(scale)]

                                    env = SumoIntersectionEnv(
                                        controller=controller,
                                        bounds=bounds,
                                        route_slots=route_slots,
                                        sumo_cmd=sumo_cmd, 
                                        gui=False
                                    )
                                    
                                    if agent_name == "Static":
                                        stats = run_simulation(env, static_cycle=data["static_cycle"])
                                    elif agent_name == "Actuated":
                                        act = ActuatedController(
                                            controller,
                                            route_slots,
                                            [p["signals"] for p in data["static_cycle"]],
                                        )
                                        stats = run_simulation(env, actuated=act)
                                    else:
                                        stats = run_simulation(env, model=loaded_models[agent_name])
                                        
                                    # Save result
                                    result = {
                                        "intersection": config_id,
                                        "variance": variance,
                                        "agent": agent_name,
                                        "scale": scale,
                                        "run": run,
                                        **stats
                                    }
                                    current_writer.writerow(result)
                                    current_f.flush()
                                    traci.close()
                                
                                # cleanup shared route file (skip the fixed real-demand file)
                                if not real_rf and os.path.exists(temp_route_file):
                                    os.remove(temp_route_file)
    
    print("All intersection evaluations complete.")

if __name__ == "__main__":
    evaluate()
