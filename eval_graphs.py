"""Generate evaluation figures (metric-vs-scale, improvement heatmaps, violins, overview) from result CSVs."""

import argparse
import os
import sys
import warnings
import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)

# Style
# Try LaTeX rendering; fall back silently if unavailable
try:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    # Quick test
    fig_test, ax_test = plt.subplots()
    ax_test.set_xlabel(r"$\alpha$")
    plt.close(fig_test)
except Exception:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    })

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "errorbar.capsize": 3,
})

# Palette & markers (colour-blind-friendly, Wong 2011)
PALETTE = {
    "Static":                 "#000000",  # black
}
MARKERS = {
    "Static":                 "s",
    "Free training":          "o",
    "Restricted training":    "^",
    "Restricted training 2":  "D",
}
LINESTYLES = {
    "Static":                 (0, (5, 2)),   # dashed
    "Free training":          "-",
    "Restricted training":    "--",
    "Restricted training 2":  "-.",
}

EXTRA_COLORS = ["#D55E00", "#E69F00", "#0072B2", "#009E73", "#e377c2", "#7f7f7f", "#bcbd22"]
EXTRA_MARKERS = ["v", "p", "*", "X", "d", "P", "H", "x", "^"]
EXTRA_LINESTYLES = [":", "-.", "--", "-", ":", "-.", "--", "-", ":"]

def get_agent_style(agent):
    """Dynamically assign colors, markers, and linestyles to unknown agents."""
    if agent not in PALETTE:
        idx = (len(PALETTE) - 1) % len(EXTRA_COLORS)
        PALETTE[agent] = EXTRA_COLORS[idx]
        MARKERS[agent] = EXTRA_MARKERS[idx]
        LINESTYLES[agent] = EXTRA_LINESTYLES[idx]
    return PALETTE[agent], MARKERS[agent], LINESTYLES[agent]

METRIC_LABELS = {
    "avg_wait":     "Average wait time (s)",
    "total_wait":   "Total wait time (vehicle·s)",
    "max_wait":     "Maximum wait time (s)",
    "total_reward": "Total reward",
}

TIME_ORDER = ["08:00", "12:00", "20:00", "22:00"]


# Helpers
def load_data(file_paths: list) -> pd.DataFrame:
    """Load selected CSV files and return a single combined DataFrame."""
    frames = []
    for path in file_paths:
        if path and Path(path).exists():
            frames.append(pd.read_csv(path))
    if not frames:
        sys.exit("Error: no valid CSV files found.")
    df = pd.concat(frames, ignore_index=True)
    df["scale"] = df["scale"].round(2)
    return df


def agent_order(agents):
    """Sort agent list so Static always comes first."""
    priority = {"Static": 0, "Free training": 1,
                "Restricted training": 2, "Restricted training 2": 3}
    return sorted(agents, key=lambda a: priority.get(a, 99))


def summarise(df: pd.DataFrame, group_cols: list, metric: str) -> pd.DataFrame:
    g = df.groupby(group_cols)[metric]
    s = g.agg(["mean", "std"]).reset_index()
    s.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}, inplace=True)
    return s


def save(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fp = path / name
    fig.savefig(fp, bbox_inches="tight")
    print(f"  Saved → {fp}")
    plt.close(fig)


# Metric vs traffic scale per intersection
def plot_metric_vs_scale(df: pd.DataFrame, intersection: str,
                         metric: str, out: Path):
    sub = df[df["intersection"] == intersection]
    agents = agent_order(sub["agent"].unique())
    variances = sorted(sub["variance"].unique())

    ncols = len(variances)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4),
                             sharey=True, sharex=True)
    if ncols == 1:
        axes = [axes]

    for ax, v in zip(axes, variances):
        sub_t = sub[sub["variance"] == v]
        s = summarise(sub_t, ["agent", "scale"], metric)

        for agent in agents:
            d = s[s["agent"] == agent].sort_values("scale")
            if d.empty:
                continue
            color, marker, ls = get_agent_style(agent)
            ax.errorbar(
                d["scale"], d[f"{metric}_mean"],
                yerr=d[f"{metric}_std"].fillna(0),
                label=agent, color=color, marker=marker,
                linestyle=ls, capsize=3,
            )

        ax.set_title(f"Variance: {v}")
        ax.set_xlabel("Traffic scale")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(True, axis="both")
        ax.set_axisbelow(True)

    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric))

    # Single legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(agents), bbox_to_anchor=(0.5, -0.08),
               frameon=True)

    safe_inter = intersection.replace(" ", "_")
    title = (f"{METRIC_LABELS.get(metric, metric)} vs traffic scale\n"
             f"Intersection: {intersection}")
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, out, f"{metric}_vs_scale_{safe_inter}.pdf")


# Heatmap of % improvement over Static
def plot_improvement_heatmap(df: pd.DataFrame, intersection: str,
                              metric: str, out: Path, lower_is_better: bool = True):
    """% change relative to Static baseline (negative = worse for lower-is-better metrics)."""
    sub = df[df["intersection"] == intersection]
    agents = [a for a in agent_order(sub["agent"].unique()) if a != "Static"]
    variances = sorted(sub["variance"].unique())

    if "Static" not in sub["agent"].unique() or not agents:
        print(f"  Skipping heatmap for {intersection}: no Static baseline found.")
        return

    means = summarise(sub, ["agent", "scale", "variance"], metric)
    scales = sorted(sub["scale"].unique())

    fig, axes = plt.subplots(
        1, len(agents),
        figsize=(4.5 * len(agents), 0.55 * len(scales) + 1.5),
        sharey=True,
    )
    if len(agents) == 1:
        axes = [axes]

    for ax, agent in zip(axes, agents):
        matrix = np.full((len(scales), len(variances)), np.nan)

        for i, scale in enumerate(scales):
            for j, var in enumerate(variances):
                static_row = means[
                    (means["agent"] == "Static") &
                    (means["scale"] == scale) &
                    (means["variance"] == var)
                ]
                agent_row = means[
                    (means["agent"] == agent) &
                    (means["scale"] == scale) &
                    (means["variance"] == var)
                ]
                if static_row.empty or agent_row.empty:
                    continue
                s_val = static_row[f"{metric}_mean"].values[0]
                a_val = agent_row[f"{metric}_mean"].values[0]
                if s_val == 0:
                    continue
                pct = (s_val - a_val) / abs(s_val) * 100  # positive = RL better
                if not lower_is_better:
                    pct = -pct
                matrix[i, j] = pct

        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=-vmax, vmax=vmax, origin="upper")

        # Annotate cells
        for i in range(len(scales)):
            for j in range(len(variances)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt = f"{val:+.1f}%"
                    col = "black" if abs(val) < 0.6 * vmax else "white"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7.5, color=col)

        ax.set_xticks(range(len(variances)))
        ax.set_xticklabels([f"Var: {v}" for v in variances], rotation=30, ha="right")
        ax.set_yticks(range(len(scales)))
        ax.set_yticklabels([f"{s:.1f}" for s in scales])
        ax.set_xlabel("Density Variance")
        ax.set_title(agent, pad=6)
        plt.colorbar(im, ax=ax, label="Improvement over Static (%)", pad=0.02)

    axes[0].set_ylabel("Traffic scale")
    metric_lbl = METRIC_LABELS.get(metric, metric)
    fig.suptitle(
        f"% improvement in {metric_lbl} over Static baseline\n"
        f"Intersection: {intersection}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    safe_inter = intersection.replace(" ", "_")
    save(fig, out, f"heatmap_improvement_{metric}_{safe_inter}.pdf")


# Violin distributions
def plot_violin(df: pd.DataFrame, intersection: str,
                metric: str, out: Path):
    sub = df[df["intersection"] == intersection]
    agents = agent_order(sub["agent"].unique())
    variances = sorted(sub["variance"].unique())

    fig, axes = plt.subplots(1, len(variances),
                              figsize=(3.2 * len(variances), 4.5),
                              sharey=True)
    if len(variances) == 1:
        axes = [axes]

    for ax, v in zip(axes, variances):
        sub_t = sub[sub["variance"] == v]
        data = [sub_t[sub_t["agent"] == a][metric].dropna().values
                for a in agents]
        positions = np.arange(len(agents))

        parts = ax.violinplot(data, positions=positions,
                              showmedians=True, showextrema=True,
                              widths=0.6)

        # Colour each violin
        for i, (pc, agent) in enumerate(zip(parts["bodies"], agents)):
            color, _, _ = get_agent_style(agent)
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor(color)

        for part_name in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(0.9)

        # Overlay individual run points with jitter
        rng = np.random.default_rng(42)
        for i, (d, agent) in enumerate(zip(data, agents)):
            jitter = rng.uniform(-0.08, 0.08, size=len(d))
            color, _, _ = get_agent_style(agent)
            ax.scatter(positions[i] + jitter, d, s=12,
                       color=color,
                       alpha=0.7, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(agents, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"Variance: {v}")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.suptitle(
        f"Distribution of {METRIC_LABELS.get(metric, metric)}\n"
        f"Intersection: {intersection}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    safe_inter = intersection.replace(" ", "_")
    save(fig, out, f"violin_{metric}_{safe_inter}.pdf")


# Combined 2x3 overview panel
def plot_combined_overview(df: pd.DataFrame, out: Path):
    """One figure per intersection: 2 rows × 3 cols covering key metrics."""
    for intersection in df["intersection"].unique():
        sub = df[df["intersection"] == intersection]
        agents = agent_order(sub["agent"].unique())
        variances = sorted(sub["variance"].unique())
        metrics = ["avg_wait", "max_wait", "total_reward"]

        fig, axes = plt.subplots(
            len(variances), len(metrics),
            figsize=(4.5 * len(metrics), 3.2 * len(variances)),
            sharex=True,
        )
        # Ensure 2-D indexing even for single rows/cols
        if len(variances) == 1:
            axes = axes[np.newaxis, :]
        if len(metrics) == 1:
            axes = axes[:, np.newaxis]

        for row, v in enumerate(variances):
            sub_t = sub[sub["variance"] == v]
            for col, metric in enumerate(metrics):
                ax = axes[row, col]
                s = summarise(sub_t, ["agent", "scale"], metric)

                for agent in agents:
                    d = s[s["agent"] == agent].sort_values("scale")
                    if d.empty:
                        continue
                    color, marker, ls = get_agent_style(agent)
                    ax.errorbar(
                        d["scale"], d[f"{metric}_mean"],
                        yerr=d[f"{metric}_std"].fillna(0),
                        label=agent,
                        color=color,
                        marker=marker,
                        linestyle=ls,
                        capsize=3,
                    )

                if col == 0:
                    ax.set_ylabel(f"Var: {v}\n{METRIC_LABELS.get(metric, metric)}",
                                  fontsize=9)
                else:
                    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)

                if row == 0:
                    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)

                if row == len(variances) - 1:
                    ax.set_xlabel("Traffic scale")

                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
                ax.grid(True, axis="both")
                ax.set_axisbelow(True)

        # Shared legend
        def get_line2d(a):
            color, marker, ls = get_agent_style(a)
            return Line2D([0], [0], color=color, marker=marker, linestyle=ls, label=a)
            
        custom_handles = [get_line2d(a) for a in agents]
        fig.legend(handles=custom_handles,
                   loc="lower center", ncol=len(agents),
                   bbox_to_anchor=(0.5, -0.03), frameon=True)

        fig.suptitle(f"Evaluation overview — {intersection}", fontsize=13, y=1.01)
        fig.tight_layout()
        safe = intersection.replace(" ", "_")
        save(fig, out, f"combined_overview_{safe}.pdf")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality RL vs Static traffic evaluation plots."
    )
    parser.add_argument(
        "--out",
        default="figures",
        help="Output directory for figures (default: ./figures/)",
    )
    parser.add_argument(
        "--fmt",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output file format (default: pdf)",
    )
    args = parser.parse_args()

    # Find local CSV files
    csv_files = sorted(glob.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        selection = input("Enter comma-separated paths to CSV files:\n> ")
    else:
        print("Available CSV files:")
        for i, f in enumerate(csv_files, 1):
            print(f"  [{i}] {f}")
        selection = input("\nEnter the numbers of the files to process (comma-separated), or type paths directly:\n> ")

    file_paths = []
    for s in selection.split(","):
        s = s.strip()
        if not s: 
            continue
        if s.isdigit() and csv_files:
            idx = int(s) - 1
            if 0 <= idx < len(csv_files):
                file_paths.append(csv_files[idx])
            else:
                print(f"Warning: Index {s} is out of range.")
        else:
            file_paths.append(s)
            
    if not file_paths:
        print("No files selected. Exiting.")
        sys.exit(0)

    # Allow format override
    if args.fmt != "pdf":
        # Monkey-patch save to use the requested extension
        orig_save = save

        def patched_save(fig, path, name):
            orig_save(fig, path, name.replace(".pdf", f".{args.fmt}"))

        globals()["save"] = patched_save

    out = Path(args.out)
    print(f"\nLoading data …")
    df = load_data(file_paths)
    print(f"  {len(df)} rows | intersections: {list(df['intersection'].unique())}\n")

    intersections = df["intersection"].unique()

    for intersection in intersections:
        print(f"─── {intersection} ───────────────────────────────────────")

        # Per-metric line plots
        for metric in ["avg_wait", "total_wait", "max_wait", "total_reward"]:
            plot_metric_vs_scale(df, intersection, metric, out)

        # Improvement heatmaps (skip if no Static in this intersection)
        for metric, lib in [("avg_wait", True), ("total_wait", True),
                             ("total_reward", False)]:
            plot_improvement_heatmap(df, intersection, metric, out,
                                     lower_is_better=lib)

        # Violin distributions
        plot_violin(df, intersection, "avg_wait", out)
        plot_violin(df, intersection, "total_reward", out)

    # Combined overview
    print("─── Combined overview ────────────────────────────────────")
    plot_combined_overview(df, out)

    print(f"\nAll figures saved to: {out.resolve()}\n")


if __name__ == "__main__":
    main()