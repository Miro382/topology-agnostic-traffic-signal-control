"""Evaluate only the max-pressure baseline over the same scale x variance x run grid as the main run."""

from __future__ import annotations

import os
import sys
import csv
import glob
import random
import zlib

import numpy as np
import pandas as pd
import traci

from evaluate_strategy import (
    SumoIntersectionEnv,
    load_intersection_config,
    generate_density_routes,
    process_traffic_data,
    run_simulation,
    SCENARIOS_DIR,
    INTERSECTION_CONFIG_MAP,
    SCALES,
    VARIANCES,
    RUNS_PER_TEST,
)
from max_pressure_controller import MaxPressureController


def evaluate_maxpressure(inter_filter=None, scale_filter=None):
    fieldnames = ["intersection", "variance", "agent", "scale", "run",
                  "vehicles_total", "total_wait", "avg_wait", "max_wait", "total_reward"]

    scales = [s for s in SCALES if scale_filter is None or abs(s - scale_filter) < 1e-9]
    if scale_filter is not None and not scales:
        print(f"Scale {scale_filter} not in {SCALES}; nothing to do.")
        return

    csv_files = glob.glob(os.path.join(SCENARIOS_DIR, "*.csv"))

    for csv_file in csv_files:
        name = os.path.splitext(os.path.basename(csv_file))[0]
        if name not in INTERSECTION_CONFIG_MAP:
            continue

        configs = INTERSECTION_CONFIG_MAP[name]
        if not isinstance(configs, list):
            configs = [configs]

        for data in configs:
            if "static_cycle" not in data:
                print(f"Skipping {name}: no static_cycle (no phase library to use).")
                continue

            config_id = data.get("id", name)

            if inter_filter and inter_filter.lower() not in name.lower() \
                    and inter_filter.lower() not in str(config_id).lower():
                continue

            print(f"Max-pressure evaluation: {name} (Config: {config_id})"
                  + (f" | scale {scale_filter}" if scale_filter is not None else ""))

            tag = f"_s{scale_filter}" if scale_filter is not None else ""
            result_file = f"evaluation_maxpressure_{config_id}{tag}.csv"
            file_exists = os.path.exists(result_file)

            with open(result_file, "a", newline="", encoding="utf-8") as current_f:
                current_writer = csv.DictWriter(current_f, fieldnames=fieldnames)
                if not file_exists:
                    current_writer.writeheader()

                controller, route_slots, bounds = load_intersection_config(data["config"])
                cycle_phases = [p["signals"] for p in data["static_cycle"]]

                df = pd.read_csv(csv_file, header=None)
                num_data_rows = range(4, len(df))

                for row_idx in num_data_rows:
                    for scale in scales:
                        for variance in VARIANCES:
                            print(f"  Scale: {scale} | Variance: {variance} | Row {row_idx}")
                            for run in range(RUNS_PER_TEST):
                                seed = zlib.crc32(
                                    f"{name}|{row_idx}|{scale}|{variance}|{run}".encode()
                                ) % (2**31 - 1)
                                random.seed(seed)
                                np.random.seed(seed)

                                _pid = os.getpid()
                                temp_csv_file = f"simulation/TrafficDensity/{name}-varied-{variance}-s{scale}-r{run}-p{_pid}.csv"
                                if variance > 0.0:
                                    process_traffic_data(csv_file, variance, temp_csv_file)
                                else:
                                    temp_csv_file = csv_file

                                temp_route_file = (
                                    f"simulation/TrafficDensity/{name}-{variance}-density-scaled{scale}-run{run}-p{_pid}.rou.xml"
                                )
                                generate_density_routes(temp_csv_file, temp_route_file,
                                                        row_index=row_idx, scale=scale)

                                if variance > 0.0 and os.path.exists(temp_csv_file):
                                    os.remove(temp_csv_file)

                                sumo_cmd = [
                                    "sumo",
                                    "-c", data.get("sumocfg", f"simulation/{name}.sumocfg"),
                                    "-r", temp_route_file,
                                    "--start",
                                    "--quit-on-end",
                                    "--ignore-route-errors",
                                    "--step-length", "1.0",
                                    "--time-to-teleport", "-1",
                                    "--waiting-time-memory", "2000",
                                    "--seed", str(seed),
                                    "--no-warnings", "true",
                                    "--message-log", "/dev/null",
                                ]

                                env = SumoIntersectionEnv(
                                    controller=controller,
                                    bounds=bounds,
                                    route_slots=route_slots,
                                    sumo_cmd=sumo_cmd,
                                    gui=False,
                                )

                                mp = MaxPressureController(controller, route_slots, cycle_phases)
                                stats = run_simulation(env, actuated=mp)

                                current_writer.writerow({
                                    "intersection": config_id,
                                    "variance": variance,
                                    "agent": "MaxPressure",
                                    "scale": scale,
                                    "run": run,
                                    **stats,
                                })
                                current_f.flush()
                                traci.close()

                                if os.path.exists(temp_route_file):
                                    os.remove(temp_route_file)

    print("Max-pressure evaluation complete.")


if __name__ == "__main__":
    _inter = sys.argv[1] if len(sys.argv) > 1 else None
    _scale = float(sys.argv[2]) if len(sys.argv) > 2 else None
    evaluate_maxpressure(inter_filter=_inter, scale_filter=_scale)
