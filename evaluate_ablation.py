#!/usr/bin/env python3
"""Ablation evaluation: run algorithm/reward variants as paired agents on the Galeria intersection."""

import os
import csv
import evaluate_strategy as E  # importing first performs the libsumo swap and shared imports

INTERSECTION = {
    "name": "Galeria",
    "id": "Toryská - Trieda SNP",
    "config": "config_galeria.yaml",
    "csv": "simulation/TrafficDensity/Galeria.csv",
    "sumocfg": "simulation/Galeria.sumocfg",
}

# label -> checkpoint path (relative to prilohy/). PPO full is the reference.
MODELS = {
    # Reward / algorithm ablation (PPO full is the reference; large effects in the paper).
    "PPO full": "models/PPO-FINAL.zip",
    "SAC full": "models/ATLAS5_SAC_full_Flight5.zip",
    "trip":     "models/ATLAS5_PPO_trip_Flight1.zip",
    "dwait":    "models/ATLAS5_PPO_dwait_Flight3.zip",
    # Three independent retrainings of the full model -- for the between-run variance result.
    "seed1 full": "models/ATLAS5_seed1_PPO_full_Flight1.zip",
    "seed2 full": "models/ATLAS5_seed2_PPO_full_Flight1.zip",
    "seed3 full": "models/ATLAS5_seed3_PPO_full_Flight1.zip",
    # The paper's "wait-delta weight 1.0" and "flow reward" checkpoints were not archived
}

SCALES = E.SCALES
VARIANCES = E.VARIANCES
RUNS_PER_TEST = 10  # 4 scales x 3 variance x 10 = 120 runs/cell, as in the paper; lower to speed up


def resolve(p):
    if os.path.exists(p):
        return p
    if os.path.exists(p + ".zip"):
        return p + ".zip"
    return None


def load_model(path):
    algo = E.SAC if "sac" in os.path.basename(path).lower() else E.PPO
    return algo.load(path, device="cpu")


def main():
    import pandas as pd

    loaded = {}
    for label, raw in MODELS.items():
        path = resolve(raw)
        if path is None:
            print(f"[skip] missing checkpoint for '{label}': {raw}")
            continue
        loaded[label] = load_model(path)
        print(f"[ok]   loaded '{label}'")

    if not loaded:
        raise SystemExit("No models loaded; check paths.")

    controller, route_slots, bounds = E.load_intersection_config(INTERSECTION["config"])
    name = INTERSECTION["name"]
    cid = INTERSECTION["id"]
    csv_file = INTERSECTION["csv"]

    fieldnames = ["intersection", "variance", "agent", "scale", "run",
                  "vehicles_total", "total_wait", "avg_wait", "max_wait", "total_reward"]
    out = f"evaluation_ablation_{cid}.csv"
    write_header = not os.path.exists(out)

    df = pd.read_csv(csv_file, header=None)
    rows = range(4, len(df))

    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row_idx in rows:
            for scale in SCALES:
                for variance in VARIANCES:
                    for run in range(RUNS_PER_TEST):
                        # Shared demand realisation across all variants (paired).
                        seed = E.zlib.crc32(f"{name}|{row_idx}|{scale}|{variance}|{run}".encode()) % (2**31 - 1)
                        E.random.seed(seed)
                        E.np.random.seed(seed)

                        tmp_csv = f"simulation/TrafficDensity/{name}-abl-varied-{variance}.csv"
                        if variance > 0.0:
                            E.process_traffic_data(csv_file, variance, tmp_csv)
                        else:
                            tmp_csv = csv_file

                        rou = f"simulation/TrafficDensity/{name}-abl-{variance}-s{scale}-r{run}.rou.xml"
                        E.generate_density_routes(tmp_csv, rou, row_index=row_idx, scale=scale)
                        if variance > 0.0 and os.path.exists(tmp_csv):
                            os.remove(tmp_csv)

                        for label, model in loaded.items():
                            print(f"  row{row_idx} scale {scale} var {variance} run {run} | {label}")
                            sumo_cmd = [
                                "sumo", "-c", INTERSECTION["sumocfg"], "-r", rou,
                                "--start", "--quit-on-end", "--ignore-route-errors",
                                "--step-length", "1.0", "--time-to-teleport", "-1",
                                "--waiting-time-memory", "2000", "--seed", str(seed),
                                "--no-warnings", "true", "--message-log", "/dev/null",
                            ]
                            env = E.SumoIntersectionEnv(
                                controller=controller, bounds=bounds,
                                route_slots=route_slots, sumo_cmd=sumo_cmd, gui=False,
                            )
                            stats = E.run_simulation(env, model=model)
                            writer.writerow({
                                "intersection": cid, "variance": variance, "agent": label,
                                "scale": scale, "run": run, **stats,
                            })
                            f.flush()
                            E.traci.close()

                        if os.path.exists(rou):
                            os.remove(rou)

    print(f"\nAblation evaluation complete -> {out}")


if __name__ == "__main__":
    main()
