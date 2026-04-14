"""Shared statistical helpers for analyze.py and figures.py.

Functions are deliberately stdlib + scipy only — no pandas dependency.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, pstdev, stdev

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Pairing helpers
# ---------------------------------------------------------------------------
def paired_per_object(results: list[dict], strategy_a: str, strategy_b: str,
                      condition: str = "raw", metric: str = "coverage") -> tuple[list[float], list[float]]:
    """Pair runs by (domain, object, rep_index).

    For each rep_index, take the metric value from strategy_a and strategy_b
    on the same (domain, object). If reps don't align, drop the unmatched.

    Returns two equal-length lists ready for paired tests.
    """
    # Index by (strategy, domain, object) -> list of metric values in encounter order
    g = defaultdict(list)
    for r in results:
        if r.get("condition") != condition:
            continue
        if metric == "coverage":
            v = (r.get("coverage") or {}).get("ratio", 0.0)
        elif metric == "valid":
            v = 1.0 if r["validation"].get("is_valid") else 0.0
        elif metric == "tokens":
            u = r.get("generation_usage") or {}
            v = u.get("total_tokens", 0)
        elif metric == "nodes":
            v = r["validation"].get("metrics", {}).get("total_nodes", 0)
        elif metric == "leaves":
            import re
            xml = r.get("bt_xml") or ""
            v = len(re.findall(r"<(Action|Condition)\b", xml))
        else:
            v = 0
        g[(r["strategy"], r["domain"], r["object"])].append(v)

    xs, ys = [], []
    for (s, d, o), vals_a in list(g.items()):
        if s != strategy_a:
            continue
        vals_b = g.get((strategy_b, d, o), [])
        n = min(len(vals_a), len(vals_b))
        xs.extend(vals_a[:n])
        ys.extend(vals_b[:n])
    return xs, ys


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------
def cohens_d(x: list[float], y: list[float]) -> float:
    """Cohen's d for two paired samples (using pooled SD)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    sx, sy = stdev(x), stdev(y)
    pooled = math.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return (mx - my) / pooled


def cohens_d_paired(diffs: list[float]) -> float:
    """Cohen's d for paired differences (d_z)."""
    if len(diffs) < 2:
        return 0.0
    s = stdev(diffs)
    if s == 0:
        return 0.0
    return mean(diffs) / s


def effect_size_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(values: list[float], n_boot: int = 2000,
                      ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for the mean. Returns (mean, lo, hi)."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return float(arr.mean()), lo, hi


# ---------------------------------------------------------------------------
# Pairwise comparison table
# ---------------------------------------------------------------------------
def pairwise_comparison(results: list[dict], strategies: list[str],
                        condition: str = "raw", metric: str = "coverage",
                        bootstrap: bool = True) -> list[dict]:
    """Run paired Wilcoxon, Cohen's d, and (Bonferroni + Holm)-corrected
    p-values for every ordered pair (a, b).

    Each row also includes a bootstrap 95% CI on the paired mean difference,
    so reviewers can see the magnitude of the effect with uncertainty.
    """
    out = []
    raw_pvals: list[float] = []
    for i, a in enumerate(strategies):
        for j, b in enumerate(strategies):
            if i >= j:
                continue
            xs, ys = paired_per_object(results, a, b, condition, metric)
            if len(xs) < 2 or len(xs) != len(ys):
                continue
            diffs = [xs[k] - ys[k] for k in range(len(xs))]
            try:
                w_stat, p_val = stats.wilcoxon(xs, ys, zero_method="wilcox",
                                               alternative="two-sided")
            except ValueError:
                w_stat, p_val = float("nan"), float("nan")
            d = cohens_d_paired(diffs)
            mean_diff, lo, hi = bootstrap_mean_ci(diffs) if bootstrap else (
                mean(diffs) if diffs else 0.0, 0.0, 0.0
            )
            row = {
                "strategy_a": a,
                "strategy_b": b,
                "n_pairs": len(xs),
                "mean_a": mean(xs) if xs else 0,
                "mean_b": mean(ys) if ys else 0,
                "mean_diff": mean_diff,
                "diff_ci_lo": lo,
                "diff_ci_hi": hi,
                "wilcoxon_W": float(w_stat),
                "p_value": float(p_val),
                "cohens_d": d,
                "effect_size": effect_size_label(d),
            }
            raw_pvals.append(float(p_val))
            out.append(row)

    # Multiple-comparison correction. We do BOTH Bonferroni and Holm so the
    # paper can report whichever convention the venue prefers. Holm is
    # uniformly more powerful at the same family-wise error rate.
    if raw_pvals:
        bonf = _bonferroni(raw_pvals)
        holm = _holm(raw_pvals)
        for row, b_corr, h_corr in zip(out, bonf, holm):
            row["p_bonferroni"] = b_corr
            row["p_holm"] = h_corr
            # Sig markers based on Holm-corrected p
            row["sig_holm"] = _sig_marker(h_corr)
    return out


def _bonferroni(pvals: list[float]) -> list[float]:
    """Bonferroni correction: cap at 1.0."""
    n = len(pvals)
    return [min(p * n, 1.0) for p in pvals]


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down. Returns adjusted p-values in original order."""
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        # Monotonicity: adjusted p-values must be non-decreasing
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def _sig_marker(p: float) -> str:
    if p is None or (isinstance(p, float) and (p != p)):  # NaN
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Win-rate / head-to-head
# ---------------------------------------------------------------------------
def win_rate_per_object(results: list[dict], strategies: list[str],
                        condition: str = "raw") -> dict:
    """For each (domain, object), find the strategy with the highest mean
    coverage across reps. Return {strategy: win_count} and per-object table."""
    by_so = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["condition"] != condition:
            continue
        cov = (r.get("coverage") or {}).get("ratio", 0.0)
        by_so[(r["domain"], r["object"])][r["strategy"]].append(cov)

    wins = defaultdict(int)
    rows = []
    for so, by_strat in sorted(by_so.items()):
        means = {s: mean(v) for s, v in by_strat.items() if s in strategies and v}
        if not means:
            continue
        winner = max(means, key=means.get)
        wins[winner] += 1
        rows.append({
            "domain": so[0],
            "object": so[1],
            "winner": winner,
            **{f"{s}_mean": means.get(s, float("nan")) for s in strategies},
        })
    return {"wins": dict(wins), "per_object": rows}


# ---------------------------------------------------------------------------
# Best-of-N curves
# ---------------------------------------------------------------------------
def best_of_n_curve(results: list[dict], strategy: str, condition: str = "raw",
                    metric: str = "coverage") -> list[float]:
    """For each N from 1..max_reps, compute the mean (over objects) of the
    best-of-N coverage. Useful for showing how diverse sampling helps.

    The 'best of N' is the maximum coverage across the first N reps for each
    object, averaged over objects.
    """
    by_so = defaultdict(list)
    for r in results:
        if r["condition"] != condition or r["strategy"] != strategy:
            continue
        v = (r.get("coverage") or {}).get("ratio", 0.0)
        by_so[(r["domain"], r["object"])].append(v)

    if not by_so:
        return []
    max_n = max(len(v) for v in by_so.values())
    curve = []
    for n in range(1, max_n + 1):
        per_obj_best = []
        for vals in by_so.values():
            if len(vals) >= n:
                per_obj_best.append(max(vals[:n]))
        if per_obj_best:
            curve.append(mean(per_obj_best))
    return curve


# ---------------------------------------------------------------------------
# Per-category coverage
# ---------------------------------------------------------------------------
def per_category_coverage(results: list[dict], strategy: str,
                          condition: str = "raw") -> dict[str, float]:
    """For each behavior category, compute (matched / total) across all
    runs of the given strategy."""
    counts = defaultdict(lambda: [0, 0])
    for r in results:
        if r["condition"] != condition or r["strategy"] != strategy:
            continue
        cov = r.get("coverage") or {}
        for pb in cov.get("per_behavior", []):
            cat = pb.get("category") or "_uncategorized"
            counts[cat][1] += 1
            if pb.get("matched"):
                counts[cat][0] += 1
    return {c: (counts[c][0] / counts[c][1]) if counts[c][1] else 0.0
            for c in sorted(counts.keys())}


# ---------------------------------------------------------------------------
# Intra-strategy diversity (Jaccard between BT haystacks)
# ---------------------------------------------------------------------------
def intra_strategy_diversity(results: list[dict], strategy: str,
                             condition: str = "raw") -> dict:
    """For each (domain, object), compute the average pairwise Jaccard
    distance between the haystacks of all reps. Higher = more diverse."""
    from src.bt_validator.coverage import _bt_haystack
    by_so = defaultdict(list)
    for r in results:
        if r["condition"] != condition or r["strategy"] != strategy:
            continue
        if not r.get("bt_xml"):
            continue
        by_so[(r["domain"], r["object"])].append(_bt_haystack(r["bt_xml"]))

    out = {}
    for so, hays in by_so.items():
        if len(hays) < 2:
            continue
        dists = []
        for i in range(len(hays)):
            for j in range(i + 1, len(hays)):
                a, b = hays[i], hays[j]
                if not a and not b:
                    continue
                inter = len(a & b)
                union = len(a | b)
                jac = (inter / union) if union else 0.0
                dists.append(1 - jac)  # distance
        if dists:
            out[so] = mean(dists)
    return out


# ---------------------------------------------------------------------------
# Pipeline cost breakdown for proposed
# ---------------------------------------------------------------------------
def proposed_pipeline_costs(results: list[dict], strategy: str = "proposed",
                            condition: str = "raw") -> dict:
    """Sum tokens per pipeline step (decompose / elicit / synthesize)."""
    sums = {"decompose": 0, "elicit": 0, "synthesize": 0}
    counts = {"decompose": 0, "elicit": 0, "synthesize": 0}
    for r in results:
        if r["condition"] != condition or r["strategy"] != strategy:
            continue
        pipe = r.get("pipeline") or {}
        for step in ("decompose", "elicit", "synthesize"):
            usage = (pipe.get(step) or {}).get("usage") or {}
            tot = usage.get("total_tokens", 0)
            if tot:
                sums[step] += tot
                counts[step] += 1
    return {step: (sums[step] / counts[step]) if counts[step] else 0
            for step in sums}
