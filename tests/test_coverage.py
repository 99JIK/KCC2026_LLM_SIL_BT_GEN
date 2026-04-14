"""Unit tests for src.bt_validator.coverage."""

from src.bt_validator.coverage import (
    _bt_haystack,
    _check_keywords,
    _tokenize,
    coverage_score,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def test_tokenize_camel_case():
    assert _tokenize("OpenDoor") == {"open", "door"}


def test_tokenize_snake_case():
    assert _tokenize("press_call_button") == {"press", "call", "button"}


def test_tokenize_strips_short_and_stopwords():
    out = _tokenize("the quick fox of doom")
    # 'the' and 'of' are stopwords, removed
    assert "the" not in out
    assert "of" not in out
    assert "quick" in out
    assert "fox" in out
    assert "doom" in out


def test_tokenize_drops_3char_threshold():
    # 'rc' length 2 -> dropped
    assert "rc" not in _tokenize("rc loss")


def test_tokenize_handles_empty():
    assert _tokenize("") == set()
    assert _tokenize(None) == set()


# ---------------------------------------------------------------------------
# Haystack from BT XML
# ---------------------------------------------------------------------------
def test_haystack_from_node_tags():
    xml = '<root><Sequence><Action ID="OpenDoor"/></Sequence></root>'
    h = _bt_haystack(xml)
    # Should contain tokens from tag names AND from ID attribute value
    assert "sequence" in h
    assert "action" in h
    assert "open" in h
    assert "door" in h


def test_haystack_from_comments():
    xml = '<root><!-- this handles emergency response --></root>'
    h = _bt_haystack(xml)
    assert "emergency" in h
    assert "response" in h


def test_haystack_handles_malformed_xml():
    h = _bt_haystack("<root><not closing")
    # Returns empty set instead of raising
    assert h == set()


# ---------------------------------------------------------------------------
# Keyword matching (required + any_of)
# ---------------------------------------------------------------------------
def test_required_token_present_passes():
    haystack = {"door", "open", "close"}
    keywords = {"required": ["door"], "any_of": [["open", "shut"]]}
    matched, _ = _check_keywords(haystack, keywords)
    assert matched is True


def test_required_token_missing_fails():
    haystack = {"window", "open"}
    keywords = {"required": ["door"], "any_of": [["open"]]}
    matched, detail = _check_keywords(haystack, keywords)
    assert matched is False
    assert detail["required_hit"] is None


def test_any_of_group_must_pass():
    haystack = {"door"}
    keywords = {"required": ["door"], "any_of": [["open", "close"], ["sensor", "beam"]]}
    matched, detail = _check_keywords(haystack, keywords)
    # 'door' satisfies required, but neither any_of group has a hit
    assert matched is False


def test_any_of_all_groups_pass():
    haystack = {"door", "open", "sensor"}
    keywords = {
        "required": ["door"],
        "any_of": [["open", "shut"], ["sensor", "beam"]],
    }
    matched, _ = _check_keywords(haystack, keywords)
    assert matched is True


def test_no_required_no_any_of():
    haystack = {"anything"}
    keywords = {"required": [], "any_of": []}
    matched, _ = _check_keywords(haystack, keywords)
    # No constraints -> trivially passes
    assert matched is True


# ---------------------------------------------------------------------------
# coverage_score: behavior with curated keywords
# ---------------------------------------------------------------------------
def test_coverage_full_match():
    behaviors = [
        {
            "text": "Door open and close with sensor",
            "category": "door_safety",
            "keywords": {
                "required": ["door"],
                "any_of": [["open"], ["close"], ["sensor"]],
            },
        }
    ]
    bt = '<root><Action ID="OpenDoor"/><Action ID="CloseDoor"/><Action ID="DoorSensor"/></root>'
    cov = coverage_score(behaviors, bt)
    assert cov["covered"] == 1
    assert cov["total"] == 1
    assert cov["ratio"] == 1.0


def test_coverage_partial_miss():
    behaviors = [
        {
            "text": "Door open and close with sensor",
            "keywords": {
                "required": ["door"],
                "any_of": [["open"], ["close"], ["sensor"]],
            },
        }
    ]
    # Missing the 'sensor' group
    bt = '<root><Action ID="OpenDoor"/><Action ID="CloseDoor"/></root>'
    cov = coverage_score(behaviors, bt)
    assert cov["covered"] == 0
    assert cov["ratio"] == 0.0


def test_coverage_multiple_behaviors_partial():
    behaviors = [
        {
            "text": "A",
            "keywords": {"required": ["alpha"], "any_of": [["one"]]},
        },
        {
            "text": "B",
            "keywords": {"required": ["beta"], "any_of": [["two"]]},
        },
        {
            "text": "C",
            "keywords": {"required": ["gamma"], "any_of": [["three"]]},
        },
    ]
    bt = '<root><Action ID="AlphaOne"/><Action ID="BetaTwo"/></root>'
    cov = coverage_score(behaviors, bt)
    assert cov["covered"] == 2
    assert cov["total"] == 3


def test_coverage_per_category_aggregation():
    behaviors = [
        {
            "text": "X",
            "category": "alpha",
            "keywords": {"required": ["x"], "any_of": [["foo"]]},
        },
        {
            "text": "Y",
            "category": "alpha",
            "keywords": {"required": ["y"], "any_of": [["foo"]]},
        },
        {
            "text": "Z",
            "category": "beta",
            "keywords": {"required": ["z"], "any_of": [["foo"]]},
        },
    ]
    # Note: 'x' is 1 char, dropped by tokenizer (<3). Use longer tokens.
    behaviors = [
        {
            "text": "X",
            "category": "alpha",
            "keywords": {"required": ["xxx"], "any_of": [["foo"]]},
        },
        {
            "text": "Y",
            "category": "alpha",
            "keywords": {"required": ["yyy"], "any_of": [["foo"]]},
        },
        {
            "text": "Z",
            "category": "beta",
            "keywords": {"required": ["zzz"], "any_of": [["foo"]]},
        },
    ]
    bt = '<root><Action ID="XxxFoo"/><Action ID="ZzzFoo"/></root>'
    cov = coverage_score(behaviors, bt)
    cats = cov["per_category"]
    assert cats["alpha"]["total"] == 2
    assert cats["alpha"]["covered"] == 1  # only xxx, not yyy
    assert cats["beta"]["total"] == 1
    assert cats["beta"]["covered"] == 1


def test_coverage_handles_empty_inputs():
    assert coverage_score([], "")["ratio"] == 0.0
    assert coverage_score([], "<root/>")["ratio"] == 0.0
    assert coverage_score([{"text": "x", "keywords": {"required": ["a"]}}], "")["ratio"] == 0.0


# ---------------------------------------------------------------------------
# coverage_score: backward-compat (no keywords field, falls back to overlap)
# ---------------------------------------------------------------------------
def test_coverage_fallback_text_overlap():
    behaviors = [{"text": "open the door with a sensor"}]  # no keywords
    bt = '<root><Action ID="OpenDoorWithSensor"/></root>'
    cov = coverage_score(behaviors, bt, min_overlap=0.4)
    assert cov["per_behavior"][0]["match_method"] == "text_overlap"
    assert cov["covered"] == 1


def test_coverage_accepts_plain_string():
    behaviors = ["open the door with a sensor"]
    bt = '<root><Action ID="OpenDoorWithSensor"/></root>'
    cov = coverage_score(behaviors, bt, min_overlap=0.4)
    assert cov["covered"] == 1
