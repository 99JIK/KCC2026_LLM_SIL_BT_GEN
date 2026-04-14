"""Analyze a sitl_env_bt_gen experiment log.

Usage:
    python scripts/analyze.py <log.jsonl> [--config experiments/configs/default.yaml]
                              [--rescore]   # re-run coverage_score with current config
                              [--out experiments/results/]
                              [--csv]       # also dump CSV tables

Produces:
    - Strategy x condition main table (valid%, coverage%, ci, tokens)
    - By-domain breakdown (raw, repaired)
    - By-object breakdown
    - Repair-pass statistics
    - Validity gain table (raw -> repaired)
    - Per-behavior coverage rates (worst-N, best-N)
    - Top error messages

The script accepts logs produced by run_experiment.py with the
{condition: raw|repaired} schema.
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

# Make src importable when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from scripts.stats import (  # noqa: E402
    best_of_n_curve, bootstrap_mean_ci, intra_strategy_diversity,
    pairwise_comparison, per_category_coverage, proposed_pipeline_costs,
    win_rate_per_object,
)
from src.bt_validator.coverage import coverage_score  # noqa: E402

STRATEGIES_ORDER = [
    "zero_shot",
    "few_shot_generic",
    "proposed",
    "proposed_with_few_shot",
]
CONDITIONS_ORDER = ["raw", "repaired"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_log(path: str) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    results = [r for r in rows if r.get("type") == "result"]
    errors = [r for r in rows if r.get("type") == "error"]
    return results, errors


def load_expected(config_path: str) -> dict:
    """Load expected_behaviors keyed by (domain, object).

    Honors the `domains_include` directive used by the runner so that
    config files which delegate domain definitions to a shared file
    (e.g. _domains.yaml) still work.
    """
    cfg = yaml.safe_load(open(config_path))
    if "domains" not in cfg and "domains_include" in cfg:
        include_path = os.path.join(
            os.path.dirname(os.path.abspath(config_path)),
            cfg["domains_include"],
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
    """Replace each row's coverage with a freshly computed score."""
    for r in results:
        eb = expected_by_obj.get((r.get("domain"), r.get("object")), [])
        if r.get("bt_xml") and eb:
            r["coverage"] = coverage_score(eb, r["bt_xml"])
        elif eb:
            r["coverage"] = {
                "total": len(eb), "covered": 0, "ratio": 0.0,
                "per_behavior": [], "per_category": {},
            }


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def _ci95(values: list[float]) -> tuple[float, float]:
    """Approx 95% CI half-width using t≈1.96; returns (mean, half_width)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = mean(values)
    if n < 2:
        return m, 0.0
    sd = stdev(values)
    return m, 1.96 * sd / math.sqrt(n)


def _safe_div(a, b):
    return a / b if b else 0.0


def main_table(results: list[dict]) -> list[dict]:
    """Aggregate by (strategy, condition).

    Reports:
      - n, valid%, btcpp%
      - coverage mean with both parametric (±95) and bootstrap (lo/hi) CIs
      - min/max coverage, avg nodes, avg tokens
    """
    g = defaultdict(list)
    for r in results:
        g[(r["strategy"], r["condition"])].append(r)

    rows = []
    for s in STRATEGIES_ORDER:
        for c in CONDITIONS_ORDER:
            rs = g.get((s, c), [])
            if not rs:
                continue
            n = len(rs)
            valid_n = sum(1 for r in rs if r["validation"].get("is_valid"))
            btcpp_n = sum(
                1 for r in rs
                if r["validation"].get("metrics", {}).get("btcpp_load")
            )
            covs = [r["coverage"]["ratio"] for r in rs if r.get("coverage")]
            cov_mean, cov_ci = _ci95(covs)
            cov_boot_mean, cov_boot_lo, cov_boot_hi = bootstrap_mean_ci(covs)
            nodes = [
                r["validation"].get("metrics", {}).get("total_nodes", 0)
                for r in rs
            ]
            tokens = [
                r.get("generation_usage", {}).get("total_tokens", 0)
                if isinstance(r.get("generation_usage"), dict) else 0
                for r in rs
            ]
            rows.append({
                "strategy": s,
                "condition": c,
                "n": n,
                "valid_pct": _safe_div(valid_n, n),
                "btcpp_pct": _safe_div(btcpp_n, n),
                "coverage_mean": cov_mean,
                "coverage_ci95_halfwidth": cov_ci,
                "coverage_boot_lo": cov_boot_lo,
                "coverage_boot_hi": cov_boot_hi,
                "coverage_min": min(covs) if covs else 0.0,
                "coverage_max": max(covs) if covs else 0.0,
                "nodes_avg": mean(nodes) if nodes else 0.0,
                "tokens_avg": mean(tokens) if tokens else 0.0,
            })
    return rows


def by_domain_table(results: list[dict], condition: str = "raw") -> list[dict]:
    g = defaultdict(list)
    for r in results:
        if r["condition"] != condition:
            continue
        g[(r["domain"], r["strategy"])].append(r)

    out = []
    for (dom, strat), rs in sorted(g.items()):
        n = len(rs)
        valid_n = sum(1 for r in rs if r["validation"].get("is_valid"))
        covs = [r["coverage"]["ratio"] for r in rs]
        cov_mean, cov_ci = _ci95(covs)
        out.append({
            "domain": dom,
            "strategy": strat,
            "n": n,
            "valid_pct": _safe_div(valid_n, n),
            "coverage_mean": cov_mean,
            "coverage_ci95": cov_ci,
        })
    return out


def by_object_table(results: list[dict], condition: str = "raw") -> list[dict]:
    g = defaultdict(list)
    for r in results:
        if r["condition"] != condition:
            continue
        g[(r["domain"], r["object"], r["strategy"])].append(r)

    out = []
    for (dom, obj, strat), rs in sorted(g.items()):
        covs = [r["coverage"]["ratio"] for r in rs]
        cov_mean, cov_ci = _ci95(covs)
        out.append({
            "domain": dom,
            "object": obj,
            "strategy": strat,
            "n": len(rs),
            "coverage_mean": cov_mean,
            "coverage_ci95": cov_ci,
            "coverage_min": min(covs) if covs else 0.0,
            "coverage_max": max(covs) if covs else 0.0,
        })
    return out


def repair_stats(results: list[dict]) -> list[dict]:
    out = []
    for s in STRATEGIES_ORDER:
        rs = [
            r for r in results
            if r["strategy"] == s and r["condition"] == "repaired"
        ]
        if not rs:
            continue
        calls = [r.get("repair", {}).get("calls", 0) for r in rs]
        needed = sum(1 for c in calls if c > 0)
        out.append({
            "strategy": s,
            "n": len(rs),
            "needed_repair": needed,
            "needed_pct": _safe_div(needed, len(rs)),
            "avg_calls": mean(calls) if calls else 0.0,
            "max_calls": max(calls) if calls else 0,
        })
    return out


def validity_gain(results: list[dict]) -> list[dict]:
    """Show how raw -> repaired changes valid% and coverage%."""
    out = []
    for s in STRATEGIES_ORDER:
        raw = [r for r in results if r["strategy"] == s and r["condition"] == "raw"]
        rep = [r for r in results if r["strategy"] == s and r["condition"] == "repaired"]
        if not raw or not rep:
            continue
        rv = _safe_div(sum(1 for r in raw if r["validation"].get("is_valid")), len(raw))
        pv = _safe_div(sum(1 for r in rep if r["validation"].get("is_valid")), len(rep))
        rc = mean([r["coverage"]["ratio"] for r in raw])
        pc = mean([r["coverage"]["ratio"] for r in rep])
        out.append({
            "strategy": s,
            "n": len(raw),
            "valid_raw": rv,
            "valid_repaired": pv,
            "valid_delta": pv - rv,
            "coverage_raw": rc,
            "coverage_repaired": pc,
            "coverage_delta": pc - rc,
        })
    return out


def per_behavior_rates(results: list[dict], condition: str = "raw") -> list[dict]:
    """Per (domain, object, behavior_index) match rate across strategies/reps."""
    counts = defaultdict(lambda: [0, 0])  # key -> [matched, total]
    texts = {}
    cats = {}
    for r in results:
        if r["condition"] != condition:
            continue
        cov = r.get("coverage") or {}
        for i, pb in enumerate(cov.get("per_behavior", [])):
            key = (r["domain"], r["object"], i)
            counts[key][0] += 1 if pb.get("matched") else 0
            counts[key][1] += 1
            texts[key] = pb.get("behavior", "")
            cats[key] = pb.get("category", "")
    out = []
    for key, (m, t) in counts.items():
        out.append({
            "domain": key[0],
            "object": key[1],
            "idx": key[2],
            "behavior": texts[key],
            "category": cats.get(key, ""),
            "matched": m,
            "total": t,
            "rate": _safe_div(m, t),
        })
    out.sort(key=lambda x: x["rate"])
    return out


def error_top(results: list[dict], condition: str, k: int = 10) -> list[tuple[str, int]]:
    counts = defaultdict(int)
    for r in results:
        if r["condition"] != condition:
            continue
        for e in r["validation"].get("errors", []):
            counts[e[:120]] += 1
    return sorted(counts.items(), key=lambda x: -x[1])[:k]


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def fmt_pct(x): return f"{x*100:5.1f}%"
def fmt_int(x): return f"{int(x):>5}"


def print_main_table(rows):
    print("\n=== Main table: strategy × condition ===")
    print("CI columns: ±95 = parametric half-width; [boot_lo, boot_hi] = bootstrap 95% CI")
    print(f"{'strategy':<26}{'cond':<10}{'n':>4}  {'valid':>6} {'btcpp':>6} "
          f"{'cov_mean':>9} {'±95':>6} {'boot_CI':<18} "
          f"{'min':>6} {'max':>6} {'nodes':>6} {'tokens':>8}")
    for r in rows:
        boot_ci = (
            f"[{r['coverage_boot_lo']*100:.1f},{r['coverage_boot_hi']*100:.1f}]"
        )
        print(f"{r['strategy']:<26}{r['condition']:<10}{r['n']:>4}  "
              f"{fmt_pct(r['valid_pct']):>6} {fmt_pct(r['btcpp_pct']):>6} "
              f"{fmt_pct(r['coverage_mean']):>9} {r['coverage_ci95_halfwidth']*100:>5.1f} "
              f"{boot_ci:<18} "
              f"{fmt_pct(r['coverage_min']):>6} {fmt_pct(r['coverage_max']):>6} "
              f"{r['nodes_avg']:>6.1f} {r['tokens_avg']:>8.0f}")


def print_by_domain(rows, label):
    print(f"\n=== Coverage by domain ({label}) ===")
    print(f"{'domain':<15}{'strategy':<26}{'n':>4} {'valid':>7} {'cov_mean':>9} {'±95':>6}")
    for r in rows:
        print(f"{r['domain']:<15}{r['strategy']:<26}{r['n']:>4} "
              f"{fmt_pct(r['valid_pct']):>7} {fmt_pct(r['coverage_mean']):>9} "
              f"{r['coverage_ci95']*100:>5.1f}")


def print_by_object(rows, label):
    print(f"\n=== Coverage by object ({label}) ===")
    # Pivot: object x strategy
    by_obj = defaultdict(dict)
    for r in rows:
        by_obj[(r["domain"], r["object"])][r["strategy"]] = r
    print(f"{'object':<35}", end="")
    for s in STRATEGIES_ORDER:
        print(f"{s:<26}", end="")
    print()
    for (dom, obj), strats in sorted(by_obj.items()):
        line = f"{(dom[:8] + '/' + obj):<35}"
        for s in STRATEGIES_ORDER:
            if s in strats:
                m = strats[s]["coverage_mean"]
                ci = strats[s]["coverage_ci95"]
                line += f"{fmt_pct(m)+'±'+f'{ci*100:.0f}':<26}"
            else:
                line += " " * 26
        print(line)


def print_repair(rows):
    print("\n=== Repair pass statistics ===")
    print(f"{'strategy':<26}{'n':>4} {'needed':>8} {'pct':>7} {'avg_calls':>11} {'max':>5}")
    for r in rows:
        print(f"{r['strategy']:<26}{r['n']:>4} {r['needed_repair']:>8} "
              f"{fmt_pct(r['needed_pct']):>7} {r['avg_calls']:>11.2f} {r['max_calls']:>5}")


def print_validity_gain(rows):
    print("\n=== Validity gain from repair (H1 evidence) ===")
    print(f"{'strategy':<26}{'n':>4}  {'valid_raw':>10} {'valid_rep':>10} {'Δvalid':>8}  "
          f"{'cov_raw':>8} {'cov_rep':>8} {'Δcov':>8}")
    for r in rows:
        print(f"{r['strategy']:<26}{r['n']:>4}  "
              f"{fmt_pct(r['valid_raw']):>10} {fmt_pct(r['valid_repaired']):>10} "
              f"{r['valid_delta']*100:>+7.1f} "
              f"{fmt_pct(r['coverage_raw']):>8} {fmt_pct(r['coverage_repaired']):>8} "
              f"{r['coverage_delta']*100:>+7.1f}")


def print_per_behavior(rows, k=10):
    print(f"\n=== Worst {k} behaviors (lowest match rate, raw) ===")
    for r in rows[:k]:
        print(f"  [{fmt_pct(r['rate'])}] {r['matched']}/{r['total']}  "
              f"{r['domain']}/{r['object']}#{r['idx']}: {r['behavior'][:60]}")
    print(f"\n=== Best {k} behaviors (highest match rate, raw) ===")
    for r in list(reversed(rows))[:k]:
        print(f"  [{fmt_pct(r['rate'])}] {r['matched']}/{r['total']}  "
              f"{r['domain']}/{r['object']}#{r['idx']}: {r['behavior'][:60]}")


def print_pairwise(rows):
    print("\n=== Pairwise comparison (Wilcoxon paired, raw coverage) ===")
    print("p-values: *** p<0.001  ** p<0.01  * p<0.05  ns p>=0.05")
    print("Multiple-comparison correction: Holm (and Bonferroni in CSV)")
    print(f"{'A':<24}{'B':<26}{'n':>4} {'mean_A':>8} {'mean_B':>8} "
          f"{'Δ':>8} {'95%CI':<16} "
          f"{'p_raw':>9} {'p_holm':>9} {'sig':>5} {'cohens_d':>10} {'effect':>12}")
    for r in rows:
        p_raw = r["p_value"]
        p_holm = r.get("p_holm", p_raw)
        sig = r.get("sig_holm", "ns")
        ci_str = f"[{r['diff_ci_lo']*100:+.1f},{r['diff_ci_hi']*100:+.1f}]"
        print(f"{r['strategy_a']:<24}{r['strategy_b']:<26}{r['n_pairs']:>4} "
              f"{r['mean_a']*100:>7.1f}% {r['mean_b']*100:>7.1f}% "
              f"{r['mean_diff']*100:>+7.1f} {ci_str:<16} "
              f"{p_raw:>9.4f} {p_holm:>9.4f} {sig:>5} "
              f"{r['cohens_d']:>10.3f} {r['effect_size']:>12}")


def print_winrate(wr):
    print("\n=== Win rate (best mean coverage per object, raw) ===")
    total = sum(wr["wins"].values())
    for s in STRATEGIES_ORDER:
        n = wr["wins"].get(s, 0)
        bar = "█" * n
        print(f"  {s:<26} {n:>3}/{total} {bar}")
    print("\nPer-object winners:")
    for r in wr["per_object"]:
        scores = "  ".join(
            f"{s[:8]}={r.get(f'{s}_mean', 0)*100:5.1f}%"
            for s in STRATEGIES_ORDER if not math.isnan(r.get(f'{s}_mean', float('nan')))
        )
        print(f"  {r['domain']:<14}/{r['object']:<22} → {r['winner']:<26} {scores}")


def print_best_of_n(curves: dict):
    if not any(curves.values()):
        return
    print("\n=== Best-of-N coverage (raw) ===")
    print("(For each object, take max coverage over first N reps; average over objects)")
    max_n = max(len(c) for c in curves.values() if c)
    header = " ".join(f"N={n+1:>2}" for n in range(max_n))
    print(f"  {'strategy':<26} {header}")
    for s, curve in curves.items():
        if not curve:
            continue
        line = " ".join(f"{v*100:5.1f}" for v in curve)
        print(f"  {s:<26} {line}")


def print_per_category(cat_data: dict):
    print("\n=== Coverage by behavior category (raw, %) ===")
    all_cats = sorted({c for d in cat_data.values() for c in d.keys()})
    print(f"  {'category':<22}", end="")
    for s in STRATEGIES_ORDER:
        if s in cat_data:
            print(f"{s[:14]:>14}", end="")
    print()
    for cat in all_cats:
        line = f"  {cat[:22]:<22}"
        for s in STRATEGIES_ORDER:
            if s in cat_data:
                v = cat_data[s].get(cat, 0)
                line += f"{v*100:>13.1f}%"
        print(line)


def print_diversity(div_data: dict):
    if not any(div_data.values()):
        return
    print("\n=== Intra-strategy diversity (raw, avg pairwise Jaccard distance) ===")
    print("(Higher = more diverse output across reps. 0 = identical, 1 = no overlap)")
    for s in STRATEGIES_ORDER:
        if s not in div_data or not div_data[s]:
            continue
        vals = list(div_data[s].values())
        if vals:
            print(f"  {s:<26} mean={mean(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}  n_objects={len(vals)}")


def print_pipeline_cost(pc: dict):
    if not pc or not any(pc.values()):
        return
    print("\n=== Proposed pipeline token cost (avg per call) ===")
    for step in ("decompose", "elicit", "synthesize"):
        if step in pc and pc[step] > 0:
            print(f"  {step:<14} {pc[step]:>8.0f} tokens")
    total = sum(pc.values())
    print(f"  {'total':<14} {total:>8.0f} tokens")


def print_errors(raw_errs, rep_errs):
    print("\n=== Top errors (raw) ===")
    for k, c in raw_errs:
        print(f"  [{c:>3}] {k}")
    print("\n=== Top errors (repaired) ===")
    for k, c in rep_errs:
        print(f"  [{c:>3}] {k}")


# ---------------------------------------------------------------------------
# CSV dump
# ---------------------------------------------------------------------------
def dump_csv(out_dir: str, name: str, rows: list[dict]):
    if not rows:
        return
    import csv
    path = os.path.join(out_dir, f"{name}.csv")
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", help="Path to sitl_env_bt_gen JSONL log")
    p.add_argument("--config", default="experiments/configs/default.yaml")
    p.add_argument("--rescore", action="store_true",
                   help="Recompute coverage with current config keywords")
    p.add_argument("--out", default="experiments/results")
    p.add_argument("--csv", action="store_true", help="Also write CSV tables")
    p.add_argument("--top", type=int, default=10,
                   help="N for worst/best per-behavior tables")
    args = p.parse_args()

    results, errors = load_log(args.log)
    print(f"Loaded {len(results)} result rows, {len(errors)} errors from {args.log}")

    if args.rescore:
        expected = load_expected(args.config)
        maybe_rescore(results, expected)
        print(f"Re-scored coverage with {args.config}")

    print(f"Strategies present: {sorted(set(r['strategy'] for r in results))}")
    print(f"Conditions present: {sorted(set(r['condition'] for r in results))}")
    print(f"Domains present: {sorted(set(r['domain'] for r in results))}")

    main_rows = main_table(results)
    print_main_table(main_rows)

    by_dom_raw = by_domain_table(results, "raw")
    print_by_domain(by_dom_raw, "raw")
    by_dom_rep = by_domain_table(results, "repaired")
    print_by_domain(by_dom_rep, "repaired")

    by_obj_raw = by_object_table(results, "raw")
    print_by_object(by_obj_raw, "raw")

    rep_rows = repair_stats(results)
    print_repair(rep_rows)

    val_rows = validity_gain(results)
    print_validity_gain(val_rows)

    per_b = per_behavior_rates(results, "raw")
    print_per_behavior(per_b, k=args.top)

    # === Statistical comparisons ===
    pw_rows = pairwise_comparison(results, STRATEGIES_ORDER, "raw", "coverage")
    print_pairwise(pw_rows)

    wr = win_rate_per_object(results, STRATEGIES_ORDER, "raw")
    print_winrate(wr)

    # Per-category breakdown
    cat_data = {s: per_category_coverage(results, s, "raw") for s in STRATEGIES_ORDER}
    cat_data = {s: d for s, d in cat_data.items() if d}
    print_per_category(cat_data)

    # Best-of-N curves
    bn_curves = {s: best_of_n_curve(results, s, "raw") for s in STRATEGIES_ORDER}
    print_best_of_n(bn_curves)

    # Intra-strategy diversity
    div_data = {s: intra_strategy_diversity(results, s, "raw") for s in STRATEGIES_ORDER}
    print_diversity(div_data)

    # Proposed pipeline cost breakdown
    for s in ("proposed", "proposed_with_few_shot"):
        pc = proposed_pipeline_costs(results, s, "raw")
        if any(pc.values()):
            print(f"\n--- Pipeline cost for {s} ---")
            print_pipeline_cost(pc)

    raw_errs = error_top(results, "raw")
    rep_errs = error_top(results, "repaired")
    print_errors(raw_errs, rep_errs)

    if args.csv:
        os.makedirs(args.out, exist_ok=True)
        print(f"\nWriting CSVs to {args.out}/")
        dump_csv(args.out, "main_table", main_rows)
        dump_csv(args.out, "by_domain_raw", by_dom_raw)
        dump_csv(args.out, "by_domain_repaired", by_dom_rep)
        dump_csv(args.out, "by_object_raw", by_obj_raw)
        dump_csv(args.out, "repair_stats", rep_rows)
        dump_csv(args.out, "validity_gain", val_rows)
        dump_csv(args.out, "per_behavior", per_b)
        dump_csv(args.out, "pairwise_tests", pw_rows)
        dump_csv(args.out, "winrate_per_object", wr["per_object"])
        # category coverage as flat rows
        cat_rows = []
        for s, d in cat_data.items():
            for cat, v in d.items():
                cat_rows.append({"strategy": s, "category": cat, "coverage": v})
        dump_csv(args.out, "category_coverage", cat_rows)


if __name__ == "__main__":
    main()
