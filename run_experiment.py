"""Main experiment runner.

For each (object, strategy, repetition):
  1. Generate the initial BT (zero_shot / few_shot_generic / proposed /
     proposed_with_few_shot).
  2. Score the RAW result (structural validity + coverage).
  3. Apply the structural_repair post-pass.
  4. Score the REPAIRED result.

Supports --resume <log.jsonl> to continue an interrupted run.
"""

import argparse
import json
import os
import time

import yaml

from src.bt_validator.coverage import coverage_score
from src.bt_validator.validator import validate_bt_xml
from src.generators.bt_generator import BTGenerator
from src.generators.llm_client import LLMClient
from src.utils.logging import ExperimentLogger


# ---------------------------------------------------------------------------
# Config + examples
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Load a YAML config. Supports `domains_include` for shared domain files."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    include = cfg.pop("domains_include", None)
    if include and "domains" not in cfg:
        include_path = os.path.join(os.path.dirname(os.path.abspath(path)), include)
        with open(include_path) as f:
            sub = yaml.safe_load(f)
        if not isinstance(sub, dict) or "domains" not in sub:
            raise ValueError(f"{include_path} must define a top-level 'domains:' key")
        cfg["domains"] = sub["domains"]
    return cfg


def load_generic_examples(example_dir: str = "data/few_shot_examples") -> str:
    folder = os.path.join(example_dir, "generic")
    if not os.path.isdir(folder):
        return ""
    out = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".xml"):
            continue
        with open(os.path.join(folder, fname)) as f:
            out.append(f"### Example: {fname}\n```xml\n{f.read().strip()}\n```")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_for_log(bt_xml: str, expected_behaviors: list) -> tuple[dict, dict]:
    if not bt_xml:
        return (
            {"is_valid": False, "errors": ["no XML produced"],
             "warnings": [], "metrics": {}},
            {"total": len(expected_behaviors), "covered": 0, "ratio": 0.0,
             "per_behavior": [], "per_category": {}},
        )
    vr = validate_bt_xml(bt_xml)
    validation = {
        "is_valid": vr.is_valid,
        "errors": vr.errors,
        "warnings": vr.warnings,
        "metrics": vr.metrics,
    }
    coverage = coverage_score(expected_behaviors, bt_xml)
    return validation, coverage


def _structural_validator_fn(bt_xml: str):
    vr = validate_bt_xml(bt_xml)
    if vr.is_valid:
        return None
    return "\n".join(vr.errors)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_resume_state(log_path: str) -> set[tuple]:
    """Return the set of (domain, object, strategy, rep) cells with BOTH
    raw AND repaired rows already in the log."""
    if not os.path.isfile(log_path):
        return set()
    raw_seen, rep_seen = set(), set()
    for line in open(log_path):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "result":
            continue
        key = (row.get("domain"), row.get("object"),
               row.get("strategy"), row.get("rep_index"))
        if row.get("condition") == "raw":
            raw_seen.add(key)
        elif row.get("condition") == "repaired":
            rep_seen.add(key)
    return {k for k in (raw_seen & rep_seen) if k[3] is not None}


# ---------------------------------------------------------------------------
# Run one (object, strategy, rep), with retry
# ---------------------------------------------------------------------------
def run_one(generator, strategy, domain_cfg, obj, generic_examples, repair_iterations):
    domain = domain_cfg["name"]
    sut = domain_cfg["sut_description"]
    obj_name = obj["name"]
    obj_desc = obj["description"]
    expected = obj.get("expected_behaviors", [])

    if strategy == "zero_shot":
        gen = generator.zero_shot(obj_name, obj_desc, domain, sut)
    elif strategy == "few_shot_generic":
        gen = generator.few_shot_generic(
            obj_name, obj_desc, generic_examples, domain, sut,
        )
    elif strategy == "proposed":
        gen = generator.proposed(obj_name, obj_desc, domain, sut)
    elif strategy == "proposed_with_few_shot":
        gen = generator.proposed_with_few_shot(
            obj_name, obj_desc, domain, sut, generic_examples,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    raw_xml = gen.get("bt_xml")
    raw_val, raw_cov = _validate_for_log(raw_xml, expected)

    if raw_xml:
        repair = generator.structural_repair(
            raw_xml, validator_fn=_structural_validator_fn,
            max_iterations=repair_iterations,
        )
    else:
        repair = {
            "bt_xml": None, "repair_calls": 0, "history": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "elapsed_seconds": 0.0,
        }
    repaired_xml = repair["bt_xml"]
    rep_val, rep_cov = _validate_for_log(repaired_xml, expected)

    return {
        "domain": domain, "object": obj_name, "strategy": strategy,
        "expected_behaviors": expected, "generation": gen,
        "raw": {"bt_xml": raw_xml, "validation": raw_val, "coverage": raw_cov},
        "repair": repair,
        "repaired": {"bt_xml": repaired_xml, "validation": rep_val, "coverage": rep_cov},
    }


def run_one_with_retry(*args, max_attempts=3, backoff=5.0, **kwargs):
    """Wrap run_one with an outer retry for rare cases where all
    LLMClient-level retries are exhausted."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return run_one(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                print(f"       run_one attempt {attempt + 1}/{max_attempts} "
                      f"failed: {e}. Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
            else:
                raise
    raise last_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run SITL environment object BT generation experiments",
    )
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--object", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default=None, metavar="LOG_PATH",
                        help="Append to an existing JSONL log; skip completed cells.")
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config["experiment"]

    if args.dry_run:
        print(json.dumps({k: v for k, v in config.items() if k != "domains"},
                         indent=2, ensure_ascii=False))
        total = sum(len(d["objects"]) for d in config["domains"])
        print(f"\nObjects: {total}")
        print(f"Strategies: {len(config['strategies'])}")
        print(f"Repetitions: {exp['repetitions']}")
        print(f"Generation calls (approx): "
              f"{total * len(config['strategies']) * exp['repetitions']}")
        return

    client = LLMClient(
        model=config["model"]["name"],
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"],
        seed=config["model"].get("seed"),
        provider=config["model"].get("provider"),
        base_url=config["model"].get("base_url"),
    )
    generator = BTGenerator(client)

    completed: set[tuple] = set()
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume target missing: {args.resume}")
        completed = load_resume_state(args.resume)
        print(f"Resuming from {args.resume} "
              f"({len(completed)} cells already complete)")
        logger = ExperimentLogger(exp["name"], log_path=args.resume)
    else:
        logger = ExperimentLogger(exp["name"])

    generic_examples = load_generic_examples()

    strategies = [args.strategy] if args.strategy else config["strategies"]
    domains = config["domains"]
    if args.domain:
        domains = [d for d in domains if d["name"] == args.domain]
    repetitions = exp.get("repetitions", 1)
    repair_iters = config.get("repair", {}).get("max_iterations", 3)

    total_objects = sum(
        len([o for o in d["objects"] if not args.object or o["name"] == args.object])
        for d in domains
    )
    total = len(strategies) * total_objects * repetitions
    count = 0

    for domain_cfg in domains:
        domain_name = domain_cfg["name"]
        objects = domain_cfg["objects"]
        if args.object:
            objects = [o for o in objects if o["name"] == args.object]

        for strategy in strategies:
            for obj in objects:
                for rep in range(repetitions):
                    count += 1
                    cell = (domain_name, obj["name"], strategy, rep)

                    if cell in completed:
                        print(f"[{count}/{total}] {domain_name} | {strategy} | "
                              f"{obj['name']} | rep {rep + 1}/{repetitions}  "
                              f"[SKIP — resumed]")
                        continue

                    print(f"[{count}/{total}] {domain_name} | {strategy} | "
                          f"{obj['name']} | rep {rep + 1}/{repetitions}")

                    try:
                        out = run_one_with_retry(
                            generator, strategy, domain_cfg, obj,
                            generic_examples, repair_iters,
                        )
                        logger.log_result(
                            condition="raw",
                            domain=out["domain"], object_name=out["object"],
                            strategy=out["strategy"], generation=out["generation"],
                            validation=out["raw"]["validation"],
                            coverage=out["raw"]["coverage"],
                            bt_xml=out["raw"]["bt_xml"],
                            repair_info=None,
                            expected_behaviors=out["expected_behaviors"],
                            rep_index=rep,
                        )
                        logger.log_result(
                            condition="repaired",
                            domain=out["domain"], object_name=out["object"],
                            strategy=out["strategy"], generation=out["generation"],
                            validation=out["repaired"]["validation"],
                            coverage=out["repaired"]["coverage"],
                            bt_xml=out["repaired"]["bt_xml"],
                            repair_info=out["repair"],
                            expected_behaviors=out["expected_behaviors"],
                            rep_index=rep,
                        )

                        rv = out["raw"]["validation"]["is_valid"]
                        rc = out["raw"]["coverage"]["covered"]
                        rt = out["raw"]["coverage"]["total"]
                        pv = out["repaired"]["validation"]["is_valid"]
                        pc = out["repaired"]["coverage"]["covered"]
                        rep_calls = out["repair"]["repair_calls"]
                        print(f"       raw: {'V' if rv else 'X'} cov={rc}/{rt}  "
                              f"repaired: {'V' if pv else 'X'} cov={pc}/{rt}  "
                              f"(repair_calls={rep_calls})")
                    except Exception as e:
                        print(f"       -> ERROR (after retries): {e}")
                        logger.log({
                            "type": "error",
                            "domain": domain_name, "strategy": strategy,
                            "object": obj["name"], "rep_index": rep,
                            "error": str(e),
                        })

    print(f"\nDone! Log: {logger.log_path}")


if __name__ == "__main__":
    main()
