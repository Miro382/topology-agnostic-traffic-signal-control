"""
Reproduce the training-seed variability and cross-topology generalisation
results
"""
import sys, glob, os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
KEY = ["intersection", "scale", "variance", "run"]
HELDOUT = ["Herlianska - Zelená Stráň",
           "Monaco - 134872 (MoST)",
           "LuST - -3976 (Luxembourg)"]


def _load(pattern, exclude=None):
    frames = []
    for f in glob.glob(os.path.join(RESULTS_DIR, pattern)):
        if exclude and exclude in f:
            continue
        d = pd.read_csv(f)
        m = d["agent"].astype(str).str.extract(r"seed(\d+)")
        d["seed"] = m[0].astype("Int64")
        d["intersection"] = d["intersection"].astype(str).str.strip()
        frames.append(d)
    if not frames:
        raise SystemExit(f"No files for pattern {pattern} in {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def seed_variability(test):
    print("\n=== Training-seed variability (test intersections) ===")
    per_seed = test.groupby("seed")["avg_wait"].mean()
    m, sd = per_seed.mean(), per_seed.std(ddof=1)
    print(per_seed.round(2).to_string())
    print(f"Overall: mean={m:.2f}  SD={sd:.2f}  CV={100*sd/m:.1f}%  "
          f"range=[{per_seed.min():.1f}, {per_seed.max():.1f}]")
    piv = test.groupby(["intersection", "seed"])["avg_wait"].mean().unstack()
    for inter, row in piv.iterrows():
        print(f"  {inter:28s} mean={row.mean():.1f} SD={row.std(ddof=1):.1f} "
              f"CV={100*row.std(ddof=1)/row.mean():.1f}%")
    return per_seed


def validation_selection():
    print("\n=== Validation selection (Monaco only) ===")
    mon = _load("evaluation_results_Monaco*_seed*.csv").groupby("seed")["avg_wait"].mean()
    best = mon.idxmin()
    print(f"Deployed seed selected on the Monaco validation intersection = {best}")
    return best


def heldout_vs_baselines(seed):
    print(f"\n=== Held-out: seed {seed} vs baselines (paired Wilcoxon) ===")
    ppo = _load(f"evaluation_results_*_seed{seed}.csv")
    ppo = ppo[ppo.intersection.isin(HELDOUT)][KEY + ["avg_wait"]].rename(columns={"avg_wait": "PPO"})
    bl = _load("evaluation_results_*_baseline.csv")
    for name in ("Static", "Actuated"):
        a = bl[bl.agent == name][KEY + ["avg_wait"]].rename(columns={"avg_wait": name})
        ppo = ppo.merge(a, on=KEY)
    INDEP = ["Herlianska - Zelená Stráň", "LuST - -3976 (Luxembourg)"]
    for scope in HELDOUT + ["HELD-OUT POOLED"]:
        if scope == "HELD-OUT POOLED":
            sub = ppo[ppo.intersection.isin(INDEP)]
        else:
            sub = ppo[ppo.intersection == scope]
        if len(sub) == 0:
            continue
        print(f"  {scope} (n={len(sub)})")
        for base in ("Static", "Actuated"):
            p, b = sub["PPO"].values, sub[base].values
            imp = 100 * (b.mean() - p.mean()) / b.mean()
            try:
                _, pv = wilcoxon(p, b); pvs = f"p={pv:.2e}"
            except ValueError:
                pvs = "n/a"
            print(f"    PPO {p.mean():6.1f}s vs {base:8s} {b.mean():6.1f}s -> {imp:+5.1f}%  {pvs}")


if __name__ == "__main__":
    test = _load("evaluation_results_*_seed*.csv", exclude="Monaco")
    test = test[~test["intersection"].str.contains("LuST")]
    seed_variability(test)
    deployed = validation_selection()
    heldout_vs_baselines(int(deployed))
