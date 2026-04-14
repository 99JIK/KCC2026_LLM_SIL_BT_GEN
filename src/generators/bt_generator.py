"""BT generation strategies and a structural repair post-pass.

Four strategies:
  - zero_shot               : single LLM call, no examples
  - few_shot_generic        : single LLM call with SITL-unrelated examples
  - proposed                : 3-call pipeline (decompose → elicit → synthesize)
  - proposed_with_few_shot  : same pipeline, examples injected at synthesis step

Repair post-pass:
  - structural_repair       : iteratively fixes BT.CPP load/tick errors using
                              only structural feedback (no coverage signal).
"""

from __future__ import annotations

from src.generators.llm_client import LLMClient
from src.prompts.templates import (
    FEW_SHOT_TEMPLATE,
    PROPOSED_DECOMPOSE_TEMPLATE,
    PROPOSED_ELICIT_TEMPLATE,
    PROPOSED_INTERMEDIATE_SYSTEM,
    PROPOSED_SYNTHESIZE_TEMPLATE,
    PROPOSED_SYNTHESIZE_WITH_EXAMPLES_TEMPLATE,
    STRUCTURAL_REPAIR_TEMPLATE,
    SYSTEM_PROMPT,
    ZERO_SHOT_TEMPLATE,
)


class BTGenerator:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

    # -------------------------------------------------------------------------
    # Strategy 1: zero-shot
    # -------------------------------------------------------------------------
    def zero_shot(
        self,
        object_name: str,
        object_description: str,
        domain: str,
        sut_description: str,
    ) -> dict:
        prompt = ZERO_SHOT_TEMPLATE.format(
            object_name=object_name,
            object_description=object_description,
            domain=domain,
            sut_description=sut_description,
        )
        result = self.client.generate(SYSTEM_PROMPT, prompt)
        result["strategy"] = "zero_shot"
        result["call_count"] = 1
        return result

    # -------------------------------------------------------------------------
    # Strategy 2: few-shot generic (SITL-unrelated examples)
    # -------------------------------------------------------------------------
    def few_shot_generic(
        self,
        object_name: str,
        object_description: str,
        examples: str,
        domain: str,
        sut_description: str,
    ) -> dict:
        prompt = FEW_SHOT_TEMPLATE.format(
            object_name=object_name,
            object_description=object_description,
            examples=examples,
            domain=domain,
            sut_description=sut_description,
        )
        result = self.client.generate(SYSTEM_PROMPT, prompt)
        result["strategy"] = "few_shot_generic"
        result["call_count"] = 1
        return result

    # -------------------------------------------------------------------------
    # Strategy 3: proposed (decompose → elicit → synthesize)
    # -------------------------------------------------------------------------
    def proposed(
        self,
        object_name: str,
        object_description: str,
        domain: str,
        sut_description: str,
    ) -> dict:
        return self._proposed_pipeline(
            object_name, object_description, domain, sut_description,
            examples=None, strategy_name="proposed",
        )

    # -------------------------------------------------------------------------
    # Strategy 4: proposed + few-shot examples at synthesis
    # -------------------------------------------------------------------------
    def proposed_with_few_shot(
        self,
        object_name: str,
        object_description: str,
        domain: str,
        sut_description: str,
        examples: str,
    ) -> dict:
        return self._proposed_pipeline(
            object_name, object_description, domain, sut_description,
            examples=examples, strategy_name="proposed_with_few_shot",
        )

    # -------------------------------------------------------------------------
    # Pipeline steps (public so they can be reused)
    # -------------------------------------------------------------------------
    def step_decompose(
        self, object_name, object_description, domain, sut_description,
    ) -> dict:
        prompt = PROPOSED_DECOMPOSE_TEMPLATE.format(
            object_name=object_name,
            object_description=object_description,
            domain=domain,
            sut_description=sut_description,
        )
        return self.client.generate(PROPOSED_INTERMEDIATE_SYSTEM, prompt)

    def step_elicit(self, object_name, dimensions_text) -> dict:
        prompt = PROPOSED_ELICIT_TEMPLATE.format(
            object_name=object_name,
            dimensions=dimensions_text,
        )
        return self.client.generate(PROPOSED_INTERMEDIATE_SYSTEM, prompt)

    def step_synthesize(
        self, object_name, object_description, domain, enumeration_text,
        examples: str | None = None,
    ) -> dict:
        if examples:
            prompt = PROPOSED_SYNTHESIZE_WITH_EXAMPLES_TEMPLATE.format(
                object_name=object_name,
                object_description=object_description,
                domain=domain,
                enumeration=enumeration_text,
                examples=examples,
            )
        else:
            prompt = PROPOSED_SYNTHESIZE_TEMPLATE.format(
                object_name=object_name,
                object_description=object_description,
                domain=domain,
                enumeration=enumeration_text,
            )
        return self.client.generate(SYSTEM_PROMPT, prompt)

    # -------------------------------------------------------------------------
    # Shared pipeline composer
    # -------------------------------------------------------------------------
    def _proposed_pipeline(
        self, object_name, object_description, domain, sut_description,
        examples: str | None, strategy_name: str,
    ) -> dict:
        decompose = self.step_decompose(
            object_name, object_description, domain, sut_description,
        )
        elicit = self.step_elicit(object_name, decompose["content"])
        synth = self.step_synthesize(
            object_name, object_description, domain,
            elicit["content"], examples=examples,
        )
        steps = [("decompose", decompose), ("elicit", elicit), ("synthesize", synth)]

        total_prompt = sum(s[1]["usage"]["prompt_tokens"] for s in steps)
        total_completion = sum(s[1]["usage"]["completion_tokens"] for s in steps)
        total_elapsed = round(sum(s[1]["elapsed_seconds"] for s in steps), 2)

        pipeline_log = {}
        for name, step in steps:
            entry = {"usage": step["usage"]}
            if name != "synthesize":
                entry["content"] = step["content"]
            else:
                entry["with_examples"] = examples is not None
            pipeline_log[name] = entry

        return {
            "content": synth["content"],
            "bt_xml": synth["bt_xml"],
            "usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
            "elapsed_seconds": total_elapsed,
            "model": synth["model"],
            "strategy": strategy_name,
            "call_count": 3,
            "pipeline": pipeline_log,
        }

    # -------------------------------------------------------------------------
    # Repair post-pass: structural feedback only
    # -------------------------------------------------------------------------
    def structural_repair(
        self,
        bt_xml: str,
        validator_fn,
        max_iterations: int = 3,
    ) -> dict:
        """Iteratively repair structural errors.

        validator_fn(xml) returns None if OK, else an error-message string.
        Coverage is never passed in — only structural errors.
        """
        history: list[dict] = []
        current_xml = bt_xml
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_elapsed = 0.0
        repair_calls = 0

        for i in range(max_iterations):
            feedback = validator_fn(current_xml)
            if feedback is None:
                history.append({"iteration": i, "status": "ok"})
                break

            history.append({
                "iteration": i, "status": "repair_attempt", "errors": feedback,
            })
            repair_prompt = STRUCTURAL_REPAIR_TEMPLATE.format(
                feedback=feedback, previous_bt=current_xml,
            )
            repair = self.client.generate(SYSTEM_PROMPT, repair_prompt)
            repair_calls += 1
            total_prompt_tokens += repair["usage"]["prompt_tokens"]
            total_completion_tokens += repair["usage"]["completion_tokens"]
            total_elapsed += repair["elapsed_seconds"]

            if repair["bt_xml"]:
                current_xml = repair["bt_xml"]
            else:
                history.append({"iteration": i, "status": "repair_failed_no_xml"})
                break
        else:
            feedback = validator_fn(current_xml)
            history.append({
                "iteration": max_iterations,
                "status": "ok" if feedback is None else "max_iterations_reached",
                "errors": feedback,
            })

        return {
            "bt_xml": current_xml,
            "repair_calls": repair_calls,
            "history": history,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "elapsed_seconds": round(total_elapsed, 2),
        }
