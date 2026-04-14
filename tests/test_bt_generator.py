"""Unit tests for src.generators.bt_generator.

The tests use a fake LLM client that records calls and returns canned
responses, so they run instantly with no API access.
"""

from __future__ import annotations

from src.generators.bt_generator import BTGenerator


class FakeLLMClient:
    """Records every call and returns canned XML."""

    def __init__(self, responses: list[str] | None = None):
        # Each call to .generate() returns the next item in `responses`,
        # cycling at the end. If responses is None, returns a tiny valid BT.
        self.responses = responses or [self._default_xml()]
        self.calls: list[dict] = []
        self._idx = 0

    @staticmethod
    def _default_xml() -> str:
        return (
            '```xml\n'
            '<root BTCPP_format="4">\n'
            '  <BehaviorTree ID="Test">\n'
            '    <Sequence>\n'
            '      <Action ID="One"/>\n'
            '      <Action ID="Two"/>\n'
            '    </Sequence>\n'
            '  </BehaviorTree>\n'
            '</root>\n'
            '```'
        )

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        content = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        from src.generators.llm_client import LLMClient
        return {
            "content": content,
            "bt_xml": LLMClient._extract_xml(content),
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "elapsed_seconds": 0.1,
            "model": "fake",
        }


def test_zero_shot_one_call():
    fake = FakeLLMClient()
    g = BTGenerator(client=fake)
    result = g.zero_shot(
        object_name="elev_car",
        object_description="elevator car hardware",
        domain="elevator",
        sut_description="elevator control software",
    )
    assert len(fake.calls) == 1
    assert result["strategy"] == "zero_shot"
    assert result["call_count"] == 1
    assert result["bt_xml"] is not None
    assert "<root" in result["bt_xml"]


def test_few_shot_generic_includes_examples():
    fake = FakeLLMClient()
    g = BTGenerator(client=fake)
    g.few_shot_generic(
        object_name="elev_car",
        object_description="elevator car hardware",
        examples="### Example\n```xml\n<root/>\n```",
        domain="elevator",
        sut_description="elevator control software",
    )
    assert len(fake.calls) == 1
    assert "Example" in fake.calls[0]["user"]


def test_proposed_makes_three_calls():
    fake = FakeLLMClient(responses=[
        "1. dimension A\n2. dimension B",
        "Dimension A:\n  - behavior 1\n  - behavior 2",
        FakeLLMClient._default_xml(),
    ])
    g = BTGenerator(client=fake)
    result = g.proposed(
        object_name="elev_car",
        object_description="elevator car hardware",
        domain="elevator",
        sut_description="elevator control software",
    )
    assert len(fake.calls) == 3
    assert result["strategy"] == "proposed"
    assert result["call_count"] == 3
    assert "decompose" in result["pipeline"]
    assert "elicit" in result["pipeline"]
    assert "synthesize" in result["pipeline"]
    assert result["pipeline"]["synthesize"]["with_examples"] is False
    assert result["usage"]["total_tokens"] == 150 * 3


def test_proposed_pipeline_threads_outputs():
    fake = FakeLLMClient(responses=[
        "DIMENSIONS_TEXT",
        "ENUMERATION_TEXT",
        FakeLLMClient._default_xml(),
    ])
    g = BTGenerator(client=fake)
    g.proposed(
        object_name="x", object_description="y",
        domain="z", sut_description="w",
    )
    assert "DIMENSIONS_TEXT" in fake.calls[1]["user"]
    assert "ENUMERATION_TEXT" in fake.calls[2]["user"]


def test_proposed_with_few_shot_injects_examples_only_at_synthesis():
    fake = FakeLLMClient(responses=[
        "DIMENSIONS",
        "ENUMERATION",
        FakeLLMClient._default_xml(),
    ])
    g = BTGenerator(client=fake)
    g.proposed_with_few_shot(
        object_name="x", object_description="y",
        domain="z", sut_description="w",
        examples="### Example\n```xml\n<root/>\n```",
    )
    assert "Example" not in fake.calls[0]["user"]
    assert "Example" not in fake.calls[1]["user"]
    assert "Example" in fake.calls[2]["user"]


def test_structural_repair_no_op_when_valid():
    fake = FakeLLMClient()
    g = BTGenerator(client=fake)
    bt = "<root><Action/></root>"
    result = g.structural_repair(bt, validator_fn=lambda _: None, max_iterations=3)
    assert result["repair_calls"] == 0
    assert result["bt_xml"] == bt
    assert len(fake.calls) == 0


def test_structural_repair_fixes_then_succeeds():
    fake = FakeLLMClient(responses=[
        '```xml\n<root><Action ID="Fixed"/></root>\n```',
    ])
    g = BTGenerator(client=fake)
    state = {"calls": 0}

    def validator(_xml):
        state["calls"] += 1
        if state["calls"] == 1:
            return "FAKE_ERROR"
        return None

    result = g.structural_repair(
        "<root><BadNode/></root>", validator_fn=validator, max_iterations=3,
    )
    assert result["repair_calls"] == 1
    assert "Fixed" in result["bt_xml"]


def test_structural_repair_max_iterations():
    fake = FakeLLMClient(responses=['```xml\n<root><Action/></root>\n```'])
    g = BTGenerator(client=fake)
    result = g.structural_repair(
        "<root/>",
        validator_fn=lambda _: "always broken",
        max_iterations=2,
    )
    assert result["repair_calls"] == 2


def test_step_decompose_returns_text():
    fake = FakeLLMClient(responses=["DIMENSIONS"])
    g = BTGenerator(client=fake)
    out = g.step_decompose("x", "y", "z", "w")
    assert out["content"] == "DIMENSIONS"
    assert len(fake.calls) == 1


def test_step_elicit_takes_dimensions_text():
    fake = FakeLLMClient(responses=["ENUMERATION"])
    g = BTGenerator(client=fake)
    out = g.step_elicit("obj", "DIMS")
    assert "DIMS" in fake.calls[0]["user"]
    assert out["content"] == "ENUMERATION"


def test_step_synthesize_with_examples_uses_different_template():
    fake = FakeLLMClient()
    g = BTGenerator(client=fake)
    g.step_synthesize("obj", "desc", "dom", "ENUM", examples="EX")
    assert "EX" in fake.calls[0]["user"]
