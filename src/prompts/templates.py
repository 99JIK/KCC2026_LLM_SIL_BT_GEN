"""Prompt templates for SITL environment object behavior repertoire generation.

Four strategies supported:
  1. zero_shot               — single call, no examples
  2. few_shot_generic        — single call with SITL-unrelated examples
  3. proposed                — decompose → elicit → synthesize (3 calls)
  4. proposed_with_few_shot  — same pipeline, examples injected at synthesis

Plus a structural-repair template for the post-pass. NOTHING in this file
refers to the expected_behaviors list — the LLM never sees the evaluation
reference.
"""

# =============================================================================
# Shared system prompt (all generation strategies)
# =============================================================================
SYSTEM_PROMPT = """\
You are an expert in simulation-based testing and behavior design for \
Software-In-The-Loop (SITL) environments.

You generate Behavior Trees (BTs) for ENVIRONMENT OBJECTS in SITL simulation \
— NOT for the System Under Test, but for the surrounding agents and entities \
that interact with the SUT.

A behavior repertoire BT is NOT a script for one test scenario. It contains \
ALL possible behaviors the object could exhibit in reality, organized as \
distinct subtrees. Simulation conditions activate the appropriate subtrees \
at runtime.

BehaviorTree.CPP v4 XML format:
- Root wrapper: <root BTCPP_format="4"> with <BehaviorTree ID="...">
- Control nodes: <Sequence>, <Fallback>, <Parallel>
- Decorator nodes: <Repeat>, <Inverter>, <ForceSuccess>, <ForceFailure>, <RetryUntilSuccessful>
- Leaf nodes: <Action ID="..."/>, <Condition ID="..."/>

Output ONLY valid BehaviorTree.CPP v4 XML wrapped in ```xml ... ``` blocks."""


# =============================================================================
# Strategy 1: Zero-shot
# =============================================================================
ZERO_SHOT_TEMPLATE = """\
Generate a complete behavior repertoire as a Behavior Tree for the following \
SITL environment object.

Domain: {domain}
SUT: {sut_description}
Environment Object: {object_name} — {object_description}

Include ALL possible behaviors this object could exhibit in a realistic \
simulation, organized as distinct subtrees. Do NOT generate a script for a \
single scenario."""


# =============================================================================
# Strategy 2: Few-shot generic (SITL-unrelated examples)
# =============================================================================
FEW_SHOT_TEMPLATE = """\
Generate a complete behavior repertoire as a Behavior Tree for a SITL \
environment object.

Here are examples of well-structured behavior repertoire BTs from UNRELATED \
domains (a video game NPC and a smart thermostat). Use them only as a guide \
to the XML format and to the general "repertoire" pattern (multiple distinct \
behavior subtrees under one root). The actual behaviors of your target object \
will be entirely different from these examples.

{examples}

---

Now generate a behavior repertoire for:

Domain: {domain}
SUT: {sut_description}
Environment Object: {object_name} — {object_description}

Include ALL possible behaviors this object could exhibit in a realistic \
simulation, organized as distinct subtrees."""


# =============================================================================
# Strategy 3: Proposed method (decompose → elicit → synthesize)
# =============================================================================

# Neutral system prompt for the two intermediate (non-XML) steps.
PROPOSED_INTERMEDIATE_SYSTEM = """\
You are an expert in simulation-based testing and behavior design for SITL \
environments. You are helping enumerate behaviors for an environment object \
in a simulation. Output plain text only — no XML, no code blocks."""


PROPOSED_DECOMPOSE_TEMPLATE = """\
You are analyzing a SITL environment object to enumerate the BEHAVIOR \
DIMENSIONS it could exhibit in a realistic simulation.

A behavior dimension is a high-level category of related behaviors — e.g.,
"emergency response", "normal operation", "fault handling", "user interaction".

Domain: {domain}
SUT (which the object interacts with): {sut_description}
Environment Object: {object_name} — {object_description}

List 6 to 10 distinct behavior dimensions for this object. Cover the full \
range: normal operation, fault/error handling, safety responses, environmental \
disturbances, interactions with other agents, and edge cases. Be exhaustive \
— prefer over-listing to under-listing.

Output ONLY a numbered list, one dimension per line, with a short name and \
a one-sentence description. No XML, no preamble."""


PROPOSED_ELICIT_TEMPLATE = """\
You previously identified the following behavior dimensions for the SITL \
environment object "{object_name}":

{dimensions}

For EACH dimension above, list 2 to 4 concrete, specific behaviors the object \
could exhibit. A concrete behavior is something that could be implemented as \
a Behavior Tree subtree — it has a clear trigger condition and a clear sequence \
of actions.

Be exhaustive within each dimension. Think about variations, edge cases, and \
realistic situations the simulation might need to reproduce.

Output as a structured list, grouped by dimension:

Dimension 1: <name>
  - <concrete behavior>
  - <concrete behavior>
  ...
Dimension 2: <name>
  - <concrete behavior>
  ...

No XML, no preamble. Just the structured list."""


PROPOSED_SYNTHESIZE_TEMPLATE = """\
You have enumerated the following behaviors for the SITL environment object \
"{object_name}" (a {object_description}) in the {domain} domain:

{enumeration}

Now convert this enumeration into a single behavior repertoire Behavior Tree \
in BehaviorTree.CPP v4 XML format. Each dimension from the enumeration should \
become a distinct subtree under the root. Each concrete behavior within a \
dimension should be implementable as a sequence/fallback within that subtree.

Use a Fallback at the root with priority order: emergency/safety responses \
first, then fault handling, then normal operation. Or use Parallel if the \
behaviors run as independent subsystems.

Output ONLY the BehaviorTree.CPP v4 XML wrapped in ```xml ... ``` blocks. \
Make sure every behavior from the enumeration above is represented as a \
subtree or sequence in the resulting BT."""


# =============================================================================
# Strategy 4: Proposed + few-shot (examples injected at synthesis step only)
# =============================================================================
PROPOSED_SYNTHESIZE_WITH_EXAMPLES_TEMPLATE = """\
You have enumerated the following behaviors for the SITL environment object \
"{object_name}" (a {object_description}) in the {domain} domain:

{enumeration}

Here are examples of well-structured behavior repertoire BTs from UNRELATED \
domains. Use them only as a guide to the XML format and Action ID naming \
convention. The actual behaviors of your target object are entirely different \
from these examples — do NOT copy their structure or behaviors, only their \
style of fine-grained, specific node naming.

{examples}

---

Now convert your enumeration above into a single behavior repertoire Behavior \
Tree in BehaviorTree.CPP v4 XML format. Each dimension from the enumeration \
should become a distinct subtree under the root. Each concrete behavior \
within a dimension should be implementable as a sequence/fallback within \
that subtree.

Use a Fallback at the root with priority order: emergency/safety responses \
first, then fault handling, then normal operation. Or use Parallel if the \
behaviors run as independent subsystems.

Output ONLY the BehaviorTree.CPP v4 XML wrapped in ```xml ... ``` blocks. \
Make sure every behavior from your enumeration above is represented as a \
subtree or sequence in the resulting BT."""


# =============================================================================
# Repair pass: structural feedback only (NO coverage signal — fully fair)
# =============================================================================
STRUCTURAL_REPAIR_TEMPLATE = """\
Your previous Behavior Tree has structural errors that prevent it from \
loading or executing in the BehaviorTree.CPP v4 runtime.

ERRORS:
{feedback}

Previous BT:
```xml
{previous_bt}
```

Please produce a corrected version that resolves these structural errors. \
PRESERVE the original tree's intent and structure as much as possible — only \
fix the specific errors above. Do NOT remove or simplify any subtrees.

Output ONLY the corrected BehaviorTree.CPP v4 XML wrapped in ```xml ... ``` blocks."""
