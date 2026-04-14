"""Experiment logging utilities."""

import json
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "experiments/logs",
        log_path: str | None = None,
    ):
        """Open a log file for the experiment.

        If ``log_path`` is given, append to that exact file (used by --resume).
        Otherwise create a new timestamped file under ``log_dir``.
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        if log_path is not None:
            self.log_path = log_path
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self.timestamp = "resumed"
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}.jsonl")
            os.makedirs(log_dir, exist_ok=True)

    def log(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        entry["experiment"] = self.experiment_name
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_result(
        self,
        condition: str,
        domain: str,
        object_name: str,
        strategy: str,
        generation: dict,
        validation: dict,
        coverage: dict,
        bt_xml: str | None,
        repair_info: dict | None = None,
        expected_behaviors: list | None = None,
        rep_index: int | None = None,
    ):
        """Log one row of experiment results.

        condition: "raw" (just-generated BT) or "repaired" (after structural_repair).
        rep_index: 0-based repetition number — used by --resume to skip
                   already-completed (object, strategy, rep) cells.
        """
        entry = {
            "type": "result",
            "condition": condition,
            "domain": domain,
            "object": object_name,
            "strategy": strategy,
            "rep_index": rep_index,
            "bt_xml": bt_xml,
            "generation_usage": generation.get("usage"),
            "generation_elapsed": generation.get("elapsed_seconds"),
            "generation_call_count": generation.get("call_count"),
            "generation_model": generation.get("model"),
            "generation_provider": generation.get("provider"),
            "generation_seed": generation.get("seed"),
            "generation_system_fingerprint": generation.get("system_fingerprint"),
            "validation": {
                "is_valid": validation.get("is_valid"),
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
                "metrics": validation.get("metrics", {}),
            },
            "coverage": coverage or {},
            "expected_behaviors": expected_behaviors or [],
        }
        if repair_info is not None:
            entry["repair"] = {
                "calls": repair_info.get("repair_calls", 0),
                "history": repair_info.get("history", []),
                "usage": repair_info.get("usage"),
                "elapsed_seconds": repair_info.get("elapsed_seconds"),
            }
        # Pipeline trace for proposed strategy (decompose/elicit intermediate texts)
        if "pipeline" in generation and condition == "raw":
            entry["pipeline"] = generation["pipeline"]
        self.log(entry)
