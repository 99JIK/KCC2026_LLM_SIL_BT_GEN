"""Generate paper figures from a sitl_env_bt_gen log.

Usage:
    python scripts/figures.py <log.jsonl>
                              [--config experiments/configs/default.yaml]
                              [--rescore]
                              [--out experiments/results/figures/]
                              [--format pdf]            # or png
                              [--font "Times New Roman"]
                              [--title-prefix "GPT-4 — "]

Generates 6 essential figures for the KCC paper:
    fig_main_coverage        — bar chart: coverage by strategy (raw/repaired)
    fig_main_validity        — bar chart: BT.CPP validity by strategy
    fig_h1_scatter           — H1 evidence: Δvalidity vs Δcoverage from repair
    fig_pipeline_cost        — stacked bar: per-step tokens for proposed variants
    fig_significance         — pairwise p-values + Cohen's d heatmap
    fig_best_of_n            — best-of-N coverage curve (value of sampling)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.stats import (  # noqa: E402
    best_of_n_curve, pairwise_comparison, proposed_pipeline_costs,
)
from src.bt_validator.coverage import coverage_score  # noqa: E402


STRATEGIES_ORDER = [
    "zero_shot",
    "few_shot_generic",
    "proposed",
    "proposed_with_few_shot",
]
STRATEGY_LABELS = {
    "zero_shot": "Zero-shot",
    "few_shot_generic": "Few-shot",
    "proposed": "Proposed",
    "proposed_with_few_shot": "Proposed +\nFew-shot",
}
STRATEGY_COLORS = {
    "zero_shot": "#cccccc",
    "few_shot_generic": "#9ec1e6",
    "proposed": "#3d7bb6",
    "proposed_with_few_shot": "#1f4e7d",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_log(path: str):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r.get("type") == "result"]


def load_expected(cfg_path: str) -> dict:
    """Load expected_behaviors keyed by (domain, object).

    Honors the `domains_include` directive used by the runner.
    """
    cfg = yaml.safe_load(open(cfg_path))
    if "domains" not in cfg and "domains_include" in cfg:
        include_path = os.path.join(
            os.path.dirname(os.path.abspath(cfg_path)), cfg["domains_include"],
        )
        with open(include_path) as f:
            sub = yaml.safe_load(f)
        cfg["domains"] = sub.get("domains", [])
    return {
        (d["name"], o["name"]): o.get("expected_behaviors", [])
        for d in cfg.get("domains", [])
        for o in d["objects"]
    }


def maybe_rescore(results: list[dict], expected_by_obj: dict) -> None:
    for r in results:
        eb = expected_by_obj.get((r.get("domain"), r.get("object")), [])
        if r.get("bt_xml") and eb:
            r["coverage"] = coverage_score(eb, r["bt_xml"])


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def fig_main_coverage(results, out_path):
    """Coverage (%) by strategy, raw vs repaired, with 95% CI."""
    g = defaultdict(list)
    for r in results:
        g[(r["strategy"], r["condition"])].append(r["coverage"]["ratio"])
    present = [s for s in STRATEGIES_ORDER if g.get((s, "raw"))]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.6), 6))
    x = np.arange(len(present))
    width = 0.36

    raw_m, raw_e, rep_m, rep_e = [], [], [], []
    for s in present:
        rv = g.get((s, "raw"), [])
        pv = g.get((s, "repaired"), [])
        raw_m.append(mean(rv) * 100 if rv else 0)
        raw_e.append(_ci95(rv) * 100)
        rep_m.append(mean(pv) * 100 if pv else 0)
        rep_e.append(_ci95(pv) * 100)

    bars1 = ax.bar(x - width / 2, raw_m, width, yerr=raw_e, capsize=3,
                   label="Raw", color=[STRATEGY_COLORS[s] for s in present],
                   edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, rep_m, width, yerr=rep_e, capsize=3,
           label="Repaired", color=[STRATEGY_COLORS[s] for s in present],
           edgecolor="black", linewidth=0.6, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in present])
    ax.set_ylabel("Behavior coverage (%)")
    ax.set_title("Behavior coverage by strategy (95% CI)")
    ax.set_ylim(0, max(rep_m + raw_m) * 1.3 + 5)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for b, m in zip(bars1, raw_m):
        ax.annotate(f"{m:.1f}", xy=(b.get_x() + b.get_width() / 2, m),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_main_validity(results, out_path):
    """BT.CPP validity (%) by strategy, raw vs repaired."""
    g = defaultdict(list)
    for r in results:
        g[(r["strategy"], r["condition"])].append(
            1 if r["validation"].get("is_valid") else 0
        )
    present = [s for s in STRATEGIES_ORDER if g.get((s, "raw"))]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.6), 6))
    x = np.arange(len(present))
    width = 0.36

    raw_m = [mean(g.get((s, "raw"), [0])) * 100 for s in present]
    rep_m = [mean(g.get((s, "repaired"), [0])) * 100 for s in present]

    bars1 = ax.bar(x - width / 2, raw_m, width,
                   color=[STRATEGY_COLORS[s] for s in present],
                   edgecolor="black", linewidth=0.6, label="Raw")
    bars2 = ax.bar(x + width / 2, rep_m, width,
                   color=[STRATEGY_COLORS[s] for s in present],
                   edgecolor="black", linewidth=0.6, hatch="//", label="Repaired")

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in present])
    ax.set_ylabel("Structural validity (%)")
    ax.set_title("BT.CPP v4 runtime validity by strategy")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.0f}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_h1_scatter(results, out_path):
    """For each (strategy, object), plot Δvalidity vs Δcoverage from repair."""
    by_so = defaultdict(lambda: {"raw": [], "rep": []})
    for r in results:
        key = (r["strategy"], r["domain"], r["object"])
        if r["condition"] == "raw":
            by_so[key]["raw"].append(r)
        else:
            by_so[key]["rep"].append(r)

    fig, ax = plt.subplots(figsize=(8, 6))
    for s in STRATEGIES_ORDER:
        xs, ys = [], []
        for key, ent in by_so.items():
            if key[0] != s or not ent["raw"] or not ent["rep"]:
                continue
            v_raw = mean(1 if r["validation"].get("is_valid") else 0 for r in ent["raw"])
            v_rep = mean(1 if r["validation"].get("is_valid") else 0 for r in ent["rep"])
            c_raw = mean(r["coverage"]["ratio"] for r in ent["raw"])
            c_rep = mean(r["coverage"]["ratio"] for r in ent["rep"])
            xs.append((v_rep - v_raw) * 100)
            ys.append((c_rep - c_raw) * 100)
        if xs:
            ax.scatter(xs, ys, label=STRATEGY_LABELS[s].replace("\n", " "),
                       color=STRATEGY_COLORS[s], s=80, edgecolor="black",
                       linewidth=0.5, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Δ Structural validity (repaired − raw, %)")
    ax.set_ylabel("Δ Behavior coverage (repaired − raw, %)")
    ax.set_title("H1: Repair improves validity but not coverage")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_pipeline_cost(results, out_path):
    """Stacked bar: per-step token cost for proposed variants."""
    variants = [s for s in ("proposed", "proposed_with_few_shot")
                if any(r["strategy"] == s for r in results)]
    if not variants:
        return

    steps = ["decompose", "elicit", "synthesize"]
    step_colors = ["#a8d5ba", "#6caf8d", "#2e7d56"]
    data = {v: proposed_pipeline_costs(results, v, "raw") for v in variants}

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(variants))
    bottom = np.zeros(len(variants))
    for step, color in zip(steps, step_colors):
        vals = [data[v][step] for v in variants]
        bars = ax.bar(x, vals, 0.45, bottom=bottom, label=step,
                      color=color, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            if v > 100:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f"{v:.0f}", ha="center", va="center", fontsize=10)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[v].replace("\n", " ") for v in variants])
    ax.set_ylabel("Avg tokens per generation")
    ax.set_title("Proposed pipeline: token cost breakdown by step")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    max_total = max(sum(data[v].values()) for v in variants)
    for i, v in enumerate(variants):
        total = sum(data[v].values())
        ax.text(i, total + max_total * 0.03, f"total: {total:.0f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max_total * 1.18)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_significance(results, out_path):
    """Pairwise p-values + Cohen's d heatmap."""
    pw = pairwise_comparison(results, STRATEGIES_ORDER, "raw", "coverage")
    if not pw:
        return

    present = sorted(
        {r["strategy_a"] for r in pw} | {r["strategy_b"] for r in pw},
        key=lambda s: STRATEGIES_ORDER.index(s) if s in STRATEGIES_ORDER else 999,
    )
    n = len(present)
    idx = {s: i for i, s in enumerate(present)}
    p_mat = np.full((n, n), np.nan)
    d_mat = np.full((n, n), np.nan)
    for r in pw:
        if r["strategy_a"] not in idx or r["strategy_b"] not in idx:
            continue
        i, j = idx[r["strategy_a"]], idx[r["strategy_b"]]
        p_mat[i, j] = p_mat[j, i] = r["p_value"]
        d_mat[i, j] = r["cohens_d"]
        d_mat[j, i] = -r["cohens_d"]

    cell_size = max(0.9, 5.0 / n)
    fig, axes = plt.subplots(2, 1, figsize=(cell_size * n + 3, cell_size * n * 2 + 3))
    labels = [STRATEGY_LABELS.get(s, s).replace("\n", " ") for s in present]

    # Top: -log10(p)
    log_p = -np.log10(np.where(p_mat > 0, p_mat, 1.0))
    im0 = axes[0].imshow(log_p, cmap="Reds", vmin=0, vmax=4)
    axes[0].set_xticks(range(n)); axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].set_title("Pairwise −log₁₀(p)  (Wilcoxon paired)")
    for i in range(n):
        for j in range(n):
            if i == j or math.isnan(p_mat[i, j]):
                continue
            p = p_mat[i, j]
            txt = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            axes[0].text(j, i, txt, ha="center", va="center",
                         color="white" if log_p[i, j] > 2 else "black", fontsize=11)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Bottom: Cohen's d
    im1 = axes[1].imshow(d_mat, cmap="RdBu_r", vmin=-2, vmax=2)
    axes[1].set_xticks(range(n)); axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    axes[1].set_yticklabels(labels, fontsize=10)
    axes[1].set_title("Cohen's d (row vs col, positive = row better)")
    for i in range(n):
        for j in range(n):
            if i == j or math.isnan(d_mat[i, j]):
                continue
            axes[1].text(j, i, f"{d_mat[i, j]:.2f}", ha="center", va="center",
                         color="white" if abs(d_mat[i, j]) > 1 else "black",
                         fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_best_of_n(results, out_path):
    """Best-of-N coverage curve."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in STRATEGIES_ORDER:
        curve = best_of_n_curve(results, s, "raw")
        if not curve:
            continue
        ns = list(range(1, len(curve) + 1))
        ax.plot(ns, [c * 100 for c in curve], marker="o",
                label=STRATEGY_LABELS[s].replace("\n", " "),
                color=STRATEGY_COLORS[s], linewidth=2.5, markersize=7,
                markeredgecolor="black", markeredgewidth=0.6)
    ax.set_xlabel("N (number of reps sampled)")
    ax.set_ylabel("Coverage of best-of-N (%, avg over objects)")
    ax.set_title("Best-of-N coverage: value of repeated sampling")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def _apply_title_prefix(ax, prefix: str | None):
    if prefix:
        title = ax.get_title()
        if title:
            ax.set_title(f"{prefix}{title}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log")
    p.add_argument("--config", default="experiments/configs/default.yaml")
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--out", default="experiments/results/figures")
    p.add_argument("--format", default="pdf", choices=["pdf", "png"])
    p.add_argument("--font", default=None,
                   help="Override font family (e.g. 'Times New Roman', 'Arial')")
    p.add_argument("--title-prefix", default=None,
                   help="Prefix added to every figure title (e.g. 'GPT-4 — ')")
    args = p.parse_args()

    # Global rcParams for paper quality
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.15

    if args.font:
        plt.rcParams["font.family"] = "serif" if "times" in args.font.lower() else "sans-serif"
        if "times" in args.font.lower():
            plt.rcParams["font.serif"] = [args.font, "Times New Roman", "DejaVu Serif"]
        else:
            plt.rcParams["font.sans-serif"] = [args.font, "DejaVu Sans"]
        plt.rcParams["mathtext.fontset"] = "stix"
        print(f"Font override: {args.font}")

    os.makedirs(args.out, exist_ok=True)
    results = load_log(args.log)
    print(f"Loaded {len(results)} result rows")

    if args.rescore:
        expected = load_expected(args.config)
        maybe_rescore(results, expected)
        print(f"Re-scored coverage with {args.config}")

    # Patch savefig to inject title prefix right before save.
    prefix = args.title_prefix
    if prefix:
        _orig_savefig = plt.savefig

        def _patched_savefig(*a, **kw):
            fig = plt.gcf()
            for ax in fig.get_axes():
                _apply_title_prefix(ax, prefix)
            return _orig_savefig(*a, **kw)

        plt.savefig = _patched_savefig

    ext = args.format
    figs = [
        ("fig_main_coverage", fig_main_coverage),
        ("fig_main_validity", fig_main_validity),
        ("fig_h1_scatter", fig_h1_scatter),
        ("fig_pipeline_cost", fig_pipeline_cost),
        ("fig_significance", fig_significance),
        ("fig_best_of_n", fig_best_of_n),
    ]
    for name, fn in figs:
        out_path = os.path.join(args.out, f"{name}.{ext}")
        try:
            fn(results, out_path)
            print(f"  wrote {out_path}")
        except Exception as e:
            print(f"  FAILED {name}: {e}")


if __name__ == "__main__":
    main()
