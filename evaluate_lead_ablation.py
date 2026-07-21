#!/usr/bin/env python3
"""Lead-distance feature ablation: evaluate PPO-FINAL on Galeria with the feature masked (set ABLATE_NO_LEAD=1)."""

import os
import csv
import evaluate_strategy as E  # libsumo swap + shared imports

INTERSECTION = {
    "name": "Galeria",
    "id": "Toryská - Trieda SNP",
    "config": "config_galeria.yaml",
    "csv": "simulation/TrafficDensity/Galeria.csv",
    "sumocfg": "simulation/Galeria.sumocfg",
}
MODEL_LABEL = "PPO lead-masked"
MODEL_PATH = "models/PPO-FINAL.zip"

SCALES = E.SCALES
VARIANCES = E.VARIANCES
RUNS_PER_TEST = 10


def main():
    if os.environ.get("ABLATE_NO_LEAD") != "1":
        print("WARNING: ABLATE_NO_LEAD is not set to 1. This run will NOT mask "
              "the lead-distance feature, so it will not produce the ablation. "
              "Set it first:  set ABLATE_NO_LEAD=1")
    import pandas as pd

    model = E.PPO.load(MODEL_PATH, device="cpu")
    controller, route_slots, bounds = E.load_intersection_config(INTERSECTION["config"])
    name, cid, csv_file = INTERSECTION["name"], INTERSECTION["id"], INTERSECTION["csv"]

    fieldnames = ["intersection", "variance", "agent", "scale", "run",
                  "vehicles_total", "total_wait", "avg_wait", "max_wait", "total_reward"]
    out = f"evaluation_leadmask_{cid}.csv"
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
                        seed = E.zlib.crc32(f"{name}|{row_idx}|{scale}|{variance}|{run}".encode()) % (2**31 - 1)
                        E.random.seed(seed)
                        E.np.random.seed(seed)
                        tmp_csv = f"simulation/TrafficDensity/{name}-lead-varied-{variance}.csv"
                        if variance > 0.0:
                            E.process_traffic_data(csv_file, variance, tmp_csv)
                        else:
                            tmp_csv = csv_file
                        rou = f"simulation/TrafficDensity/{name}-lead-{variance}-s{scale}-r{run}.rou.xml"
                        E.generate_density_routes(tmp_csv, rou, row_index=row_idx, scale=scale)
                        if variance > 0.0 and os.path.exists(tmp_csv):
                            os.remove(tmp_csv)
                        print(f"  row{row_idx} scale {scale} var {variance} run {run} | {MODEL_LABEL}")
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
                        writer.writerow({"intersection": cid, "variance": variance,
                                         "agent": MODEL_LABEL, "scale": scale, "run": run, **stats})
                        f.flush()
                        E.traci.close()
                        if os.path.exists(rou):
                            os.remove(rou)
    print(f"\nLead-distance masking evaluation complete -> {out}")


if __name__ == "__main__":
    main()
