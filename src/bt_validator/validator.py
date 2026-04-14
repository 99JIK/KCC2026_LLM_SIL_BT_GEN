"""Behavior Tree validation using py_trees and BT.CPP v4 runtime."""

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import py_trees

BTCPP_LOADER = os.path.join(os.path.dirname(__file__), "btcpp_loader")


VALID_CONTROL_NODES = {"Sequence", "Fallback", "Parallel", "Selector"}
VALID_DECORATOR_NODES = {"Repeat", "Inverter", "ForceSuccess", "ForceFailure", "RetryUntilSuccessful"}
VALID_LEAF_NODES = {"Action", "Condition"}
ALL_VALID_NODES = VALID_CONTROL_NODES | VALID_DECORATOR_NODES | VALID_LEAF_NODES


class BTValidationResult:
    def __init__(self):
        self.is_valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.metrics: dict = {}

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def feedback_string(self) -> str | None:
        if self.is_valid and not self.warnings:
            return None
        lines = []
        for e in self.errors:
            lines.append(f"ERROR: {e}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def validate_bt_xml(xml_string: str) -> BTValidationResult:
    """Validate a Behavior Tree XML string through the full pipeline.

    Stages:
      1. XML well-formedness (Python xml parser)
      2. Schema check (custom: known control/decorator/leaf node types)
      3. Structural check (children counts, leaf attributes)
      4. Runtime load + tick via BT.CPP v4 (external btcpp_loader)
    """
    result = BTValidationResult()

    # Stage 1: XML well-formedness
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        result.add_error(f"XML parse error: {e}")
        return result

    # Stages 2 & 3: Schema + structure validation
    _validate_node(root, result, depth=0)

    # Metrics
    result.metrics = _compute_metrics(root)

    # Stage 4: BT.CPP v4 runtime load + tick (writes into result.metrics)
    _validate_btcpp_runtime(xml_string, result)

    return result


def _validate_btcpp_runtime(xml_string: str, result: BTValidationResult) -> None:
    """Stage 4: load the XML through BT.CPP v4 and tick it once.

    Records pass/fail in result.metrics['btcpp_load'] and ['btcpp_tick'].
    Errors from BT.CPP are added to result.errors.
    """
    result.metrics["btcpp_load"] = False
    result.metrics["btcpp_tick"] = False

    if not os.path.isfile(BTCPP_LOADER) or not os.access(BTCPP_LOADER, os.X_OK):
        result.add_warning(
            "BT.CPP loader not built — skipping stage 4 runtime validation"
        )
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, encoding="utf-8",
    ) as tmp:
        tmp.write(xml_string)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [BTCPP_LOADER, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        result.add_error("BT.CPP runtime: tick timed out (>10s)")
        os.unlink(tmp_path)
        return
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    stderr = proc.stderr.strip()
    if proc.returncode == 0:
        result.metrics["btcpp_load"] = True
        result.metrics["btcpp_tick"] = True
    elif proc.returncode == 3:
        # Loaded fine; tick was infinite (Repeat -1 in a repertoire BT).
        # This is expected for repertoire BTs and is NOT an error.
        result.metrics["btcpp_load"] = True
        result.metrics["btcpp_tick"] = "deferred"
    elif proc.returncode == 1:
        result.add_error(f"BT.CPP load failed: {stderr or 'unknown error'}")
    elif proc.returncode == 2:
        result.metrics["btcpp_load"] = True
        result.add_error(f"BT.CPP tick failed: {stderr or 'unknown error'}")
    else:
        result.add_error(f"BT.CPP loader returned code {proc.returncode}: {stderr}")


def _validate_node(node: ET.Element, result: BTValidationResult, depth: int):
    """Recursively validate BT node structure."""
    tag = node.tag

    # Skip root/BehaviorTree wrapper
    if tag in ("root", "BehaviorTree", "TreeNodesModel"):
        for child in node:
            _validate_node(child, result, depth)
        return

    if tag not in ALL_VALID_NODES:
        result.add_warning(f"Unknown node type: '{tag}' at depth {depth}")

    # Control nodes must have children
    if tag in VALID_CONTROL_NODES:
        if len(node) == 0:
            result.add_error(f"{tag} node at depth {depth} has no children")
        if tag != "Parallel" and len(node) < 2:
            result.add_warning(f"{tag} node at depth {depth} has only 1 child")

    # Decorator nodes must have exactly one child
    if tag in VALID_DECORATOR_NODES:
        if len(node) != 1:
            result.add_error(
                f"{tag} decorator at depth {depth} must have exactly 1 child, has {len(node)}"
            )

    # Leaf nodes should have a name attribute
    if tag in VALID_LEAF_NODES:
        if "name" not in node.attrib and "ID" not in node.attrib:
            result.add_warning(f"{tag} at depth {depth} missing 'name' attribute")
        if len(node) > 0:
            result.add_error(f"{tag} leaf at depth {depth} should not have children")

    for child in node:
        _validate_node(child, result, depth + 1)


def _compute_metrics(root: ET.Element) -> dict:
    """Compute structural metrics of the BT."""
    metrics = {
        "total_nodes": 0,
        "max_depth": 0,
        "action_count": 0,
        "condition_count": 0,
        "control_count": 0,
        "decorator_count": 0,
    }
    _count_nodes(root, metrics, depth=0)
    return metrics


def _count_nodes(node: ET.Element, metrics: dict, depth: int):
    tag = node.tag
    if tag in ("root", "BehaviorTree", "TreeNodesModel"):
        for child in node:
            _count_nodes(child, metrics, depth)
        return

    metrics["total_nodes"] += 1
    metrics["max_depth"] = max(metrics["max_depth"], depth)

    if tag in VALID_CONTROL_NODES:
        metrics["control_count"] += 1
    elif tag in VALID_DECORATOR_NODES:
        metrics["decorator_count"] += 1
    elif tag == "Action":
        metrics["action_count"] += 1
    elif tag == "Condition":
        metrics["condition_count"] += 1

    for child in node:
        _count_nodes(child, metrics, depth + 1)


def build_py_tree(xml_string: str) -> py_trees.behaviour.Behaviour | None:
    """Convert BT XML to a py_trees tree for execution testing."""
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return None

    # Find the actual BT root (skip wrappers)
    bt_root = root
    while bt_root.tag in ("root", "BehaviorTree") and len(bt_root) > 0:
        bt_root = bt_root[0]

    return _xml_to_pytree(bt_root)


def _xml_to_pytree(node: ET.Element) -> py_trees.behaviour.Behaviour:
    """Recursively convert XML node to py_trees behaviour."""
    tag = node.tag
    name = node.attrib.get("name", node.attrib.get("ID", tag))

    if tag == "Sequence":
        composite = py_trees.composites.Sequence(name=name, memory=True)
        for child in node:
            composite.add_child(_xml_to_pytree(child))
        return composite

    elif tag in ("Fallback", "Selector"):
        composite = py_trees.composites.Selector(name=name, memory=True)
        for child in node:
            composite.add_child(_xml_to_pytree(child))
        return composite

    elif tag == "Parallel":
        composite = py_trees.composites.Parallel(
            name=name,
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
        )
        for child in node:
            composite.add_child(_xml_to_pytree(child))
        return composite

    elif tag == "Inverter" and len(node) == 1:
        child = _xml_to_pytree(node[0])
        return py_trees.decorators.Inverter(name=name, child=child)

    elif tag in VALID_LEAF_NODES:
        return py_trees.behaviours.Success(name=name)

    else:
        return py_trees.behaviours.Success(name=f"{tag}:{name}")
