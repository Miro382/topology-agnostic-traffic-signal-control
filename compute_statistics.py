#!/usr/bin/env python3
"""Statistics over evaluate_strategy.py result CSVs: per-agent summaries, RL-vs-baseline tests, and LaTeX tables."""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["avg_wait", "total_wait"]
BASELINES_DEFAULT = ["Static", "Actuated"]


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def ci95_halfwidth(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def run_test(rl_vals, base_vals, paired):
    rl_vals = np.asarray(rl_vals, dtype=float)
    base_vals = np.asarray(base_vals, dtype=float)
    if len(rl_vals) < 2 or len(base_vals) < 2:
        return np.nan
    try:
        if paired:
            if len(rl_vals) != len(base_vals):
                return np.nan
            if np.allclose(rl_vals, base_vals):
                return 1.0
            return float(stats.wilcoxon(rl_vals, base_vals).pvalue)
        return float(stats.mannwhitneyu(rl_vals, base_vals, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def identify_agents(df, rl_name):
    agents = list(df["agent"].unique())
    rl = next((a for a in agents if rl_name.lower() in a.lower()), None)
    if rl is None:
        raise SystemExit(f"RL agent matching '{rl_name}' not found. Agents present: {agents}")
    baselines = [b for b in BASELINES_DEFAULT if b in agents] or [a for a in agents if a != rl]
    return rl, baselines


def paired_values(sub, rl, base, metric, keys):
    """Return run-aligned (rl_values, base_values) paired on `keys`."""
    r = sub[sub["agent"] == rl].set_index(keys)[metric].dropna()
    b = sub[sub["agent"] == base].set_index(keys)[metric].dropna()
    common = r.index.intersection(b.index)
    common = sorted(common)
    return r.loc[common].values, b.loc[common].values


def analyse_intersection(df, inter, rl, baselines, paired, outdir):
    df = df[df["intersection"] == inter].copy()
    df["scale"] = df["scale"].round(2)
    scenarios = sorted(df.groupby(["scale", "variance"]).groups.keys())

    summary_rows, test_rows = [], []

    for metric in METRICS:
        for (scale, var) in scenarios:
            sub = df[(df["scale"] == scale) & (df["variance"] == var)]
            present = sub["agent"].unique()
            for a in present:
                vals = sub[sub["agent"] == a][metric].dropna().values
                if len(vals) == 0:
                    continue
                summary_rows.append({
                    "intersection": inter, "metric": metric, "scale": scale, "variance": var,
                    "agent": a, "n": len(vals), "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95": ci95_halfwidth(vals),
                })
            if rl not in present:
                continue
            for base in baselines:
                if base not in present:
                    continue
                if paired:
                    rl_v, base_v = paired_values(sub, rl, base, metric, ["run"])
                else:
                    rl_v = sub[sub["agent"] == rl][metric].dropna().values
                    base_v = sub[sub["agent"] == base][metric].dropna().values
                p = run_test(rl_v, base_v, paired)
                bm, rm = float(np.mean(base_v)), float(np.mean(rl_v))
                impr = (bm - rm) / bm * 100 if bm else np.nan  # positive => RL lower wait
                test_rows.append({
                    "intersection": inter, "metric": metric, "scale": scale, "variance": var,
                    "comparison": f"{rl} vs {base}", "rl_mean": rm, "base_mean": bm,
                    "improvement_pct": impr, "p_value": p, "sig": stars(p), "n_pairs": len(rl_v),
                })

        # pooled across all scenarios (pair on scale/variance/run when paired)
        for base in baselines:
            if base not in df["agent"].unique():
                continue
            if paired:
                rl_v, base_v = paired_values(df, rl, base, metric, ["scale", "variance", "run"])
            else:
                rl_v = df[df["agent"] == rl][metric].dropna().values
                base_v = df[df["agent"] == base][metric].dropna().values
            if len(rl_v) == 0 or len(base_v) == 0:
                continue
            p = run_test(rl_v, base_v, paired)
            bm, rm = float(np.mean(base_v)), float(np.mean(rl_v))
            test_rows.append({
                "intersection": inter, "metric": metric, "scale": "ALL", "variance": "ALL",
                "comparison": f"{rl} vs {base}", "rl_mean": rm, "base_mean": bm,
                "improvement_pct": (bm - rm) / bm * 100 if bm else np.nan,
                "p_value": p, "sig": stars(p), "n_pairs": len(rl_v),
            })

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    safe = inter.replace(" ", "_").replace("/", "-")
    summary.to_csv(os.path.join(outdir, f"stats_summary_{safe}.csv"), index=False)
    tests.to_csv(os.path.join(outdir, f"stats_tests_{safe}.csv"), index=False)
    for metric in METRICS:
        write_latex_table(summary, tests, inter, metric, rl, baselines, outdir, safe, paired)
    return summary, tests


def write_latex_table(summary, tests, inter, metric, rl, baselines, outdir, safe, paired):
    s = summary[summary["metric"] == metric]
    t = tests[tests["metric"] == metric]
    scenarios = sorted(set(zip(s["scale"], s["variance"])))
    agents = [a for a in [*baselines, rl] if a in s["agent"].unique()]
    col_spec = "ll" + "r" * len(agents) + "r" * len(baselines)
    ncol = len(col_spec)

    L = [r"\begin{table*}",
         r"\caption{\textbf{%s: %s, mean $\pm$ SD over runs and improvement of %s.}}"
         % (inter, metric.replace("_", r"\_"), rl),
         r"\label{tab:stats_%s_%s}" % (safe, metric),
         r"\centering", r"\begin{tabular}{%s}" % col_spec, r"\hline",
         " & ".join(["Scale", "Var"] + agents + [r"$\Delta$\%% vs %s" % b for b in baselines]) + r" \\",
         r"\hline"]
    for (scale, var) in scenarios:
        cells = [f"{scale}", f"{var}"]
        for a in agents:
            row = s[(s["scale"] == scale) & (s["variance"] == var) & (s["agent"] == a)]
            cells.append(f"{row['mean'].iloc[0]:.1f} $\\pm$ {row['sd'].iloc[0]:.1f}" if len(row) else "--")
        for b in baselines:
            tr = t[(t["scale"] == scale) & (t["variance"] == var) & (t["comparison"] == f"{rl} vs {b}")]
            cells.append(f"{tr['improvement_pct'].iloc[0]:+.1f} {tr['sig'].iloc[0]}" if len(tr) else "--")
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\hline")
    tn = "Wilcoxon signed-rank (paired)" if paired else "Mann-Whitney U (unpaired)"
    L.append(r"\multicolumn{%d}{l}{\footnotesize %s: *** $p<0.001$, ** $p<0.01$, * $p<0.05$, ns. Positive $\Delta$\%% = lower wait than baseline.} \\" % (ncol, tn))
    L += [r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(outdir, f"table_{safe}_{metric}.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", help="result CSVs (default: evaluation_results_*.csv)")
    ap.add_argument("--rl", default="PPO", help="substring identifying the RL agent")
    ap.add_argument("--paired", action="store_true", help="paired Wilcoxon (requires shared routes per run)")
    ap.add_argument("--outdir", default="stats_out")
    args = ap.parse_args()

    paths = args.csvs or sorted(glob.glob("evaluation_results_*.csv"))
    if not paths:
        sys.exit("No result CSVs found. Run evaluate_strategy.py first or pass paths.")
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    rl, baselines = identify_agents(df, args.rl)
    print(f"RL agent: {rl} | baselines: {baselines} | test: {'Wilcoxon paired' if args.paired else 'Mann-Whitney unpaired'}")

    for inter in df["intersection"].unique():
        _, tests = analyse_intersection(df, inter, rl, baselines, args.paired, args.outdir)
        print(f"\n=== {inter} ===")
        for _, r in tests[tests["scale"] == "ALL"].iterrows():
            print(f"  [{r['metric']:<10}] {r['comparison']}: {r['improvement_pct']:+.1f}%  p={r['p_value']:.2e}  {r['sig']}  (n={r['n_pairs']})")
    print(f"\nOutputs written to: {args.outdir}/")


if __name__ == "__main__":
    main()
