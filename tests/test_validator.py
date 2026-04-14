"""Tests for BT validator."""

from src.bt_validator.validator import validate_bt_xml, build_py_tree


VALID_BT = """<root BTCPP_format="4">
  <BehaviorTree ID="TestTree">
    <Sequence>
      <Condition ID="IsReady"/>
      <Action ID="DoTask"/>
      <Action ID="Report"/>
    </Sequence>
  </BehaviorTree>
</root>"""

INVALID_XML = """<root><Sequence><Action name="test"></root>"""

EMPTY_SEQUENCE = """<root>
  <BehaviorTree ID="TestTree">
    <Sequence name="empty"/>
  </BehaviorTree>
</root>"""


def test_valid_bt():
    result = validate_bt_xml(VALID_BT)
    assert result.is_valid
    assert len(result.errors) == 0
    assert result.metrics["total_nodes"] == 4  # Sequence + Condition + 2 Actions
    assert result.metrics["action_count"] == 2
    assert result.metrics["condition_count"] == 1


def test_invalid_xml():
    result = validate_bt_xml(INVALID_XML)
    assert not result.is_valid
    assert any("parse error" in e.lower() for e in result.errors)


def test_empty_sequence():
    result = validate_bt_xml(EMPTY_SEQUENCE)
    assert not result.is_valid
    assert any("no children" in e.lower() for e in result.errors)


def test_build_py_tree():
    tree = build_py_tree(VALID_BT)
    assert tree is not None
    # name defaults to the tag when no name/ID attribute is present
    assert tree.name in ("Sequence", "main")


def test_feedback_string_valid():
    result = validate_bt_xml(VALID_BT)
    assert result.feedback_string() is None


def test_feedback_string_invalid():
    result = validate_bt_xml(INVALID_XML)
    feedback = result.feedback_string()
    assert feedback is not None
    assert "ERROR" in feedback
