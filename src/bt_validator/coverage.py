"""Behavior coverage scoring via curated keyword matching.

Each expected behavior is associated with a manually-curated keyword set
(`required` + `any_of` groups) derived from its source documentation. A
behavior is "covered" iff:
  - At least one of the `required` tokens appears in the BT haystack, AND
  - For each `any_of` group, at least one token in the group appears.

This is intentionally conservative and reproducible — no LLM-as-judge, no
embeddings — but vocabulary-tolerant via the `any_of` groups (which encode
synonyms and alternative phrasings).

Backward compatibility: if an expected behavior has no `keywords` field,
we fall back to a token-overlap heuristic on the `text` field.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "if", "then", "else",
    "when", "while", "into", "onto", "via", "per",
    "behavior", "tree", "system", "object", "state", "mode",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase tokens, splitting on non-alnum and CamelCase boundaries."""
    if not text:
        return set()
    out: set[str] = set()
    for piece in _TOKEN_SPLIT.split(text):
        if not piece:
            continue
        for sub in _CAMEL_SPLIT.split(piece):
            sub = sub.lower().strip()
            if len(sub) < 3 or sub in _STOPWORDS:
                continue
            out.add(sub)
    return out


def _bt_haystack(bt_xml: str) -> set[str]:
    """Collect all identifier tokens from the BT XML (tags, attrs, comments)."""
    haystack: set[str] = set()
    try:
        root = ET.fromstring(bt_xml)
    except ET.ParseError:
        return haystack

    for elem in root.iter():
        haystack.update(_tokenize(elem.tag))
        for attr_name, attr_value in elem.attrib.items():
            haystack.update(_tokenize(attr_name))
            haystack.update(_tokenize(attr_value))

    for match in re.finditer(r"<!--(.*?)-->", bt_xml, re.DOTALL):
        haystack.update(_tokenize(match.group(1)))

    return haystack


def _normalize_keyword_list(items) -> list[str]:
    """Lowercase + strip a list of keyword strings."""
    if not items:
        return []
    return [str(x).lower().strip() for x in items if str(x).strip()]


def _check_keywords(haystack: set[str], keywords: dict) -> tuple[bool, dict]:
    """Apply (required + any_of) matching against the haystack.

    A behavior is matched iff:
      (1) at least one `required` token appears in the haystack, AND
      (2) every `any_of` group has at least one hit.

    Returns (matched, detail).
    """
    required = _normalize_keyword_list(keywords.get("required", []))
    any_of_groups = keywords.get("any_of", []) or []

    required_hit = None
    if required:
        for tok in required:
            if tok in haystack:
                required_hit = tok
                break
        if required_hit is None:
            return False, {
                "required_hit": None,
                "groups": [],
                "reason": "no required token matched",
            }

    group_results = []
    all_groups_pass = True
    for group in any_of_groups:
        tokens = _normalize_keyword_list(group)
        hits = [t for t in tokens if t in haystack]
        passed = bool(hits)
        if not passed:
            all_groups_pass = False
        group_results.append({
            "tokens": tokens,
            "hits": hits,
            "passed": passed,
        })

    return all_groups_pass and (required_hit is not None or not required), {
        "required_hit": required_hit,
        "groups": group_results,
        "reason": "ok" if all_groups_pass else "missing any_of group",
    }


def _fallback_text_overlap(
    text: str, haystack: set[str], min_overlap: float,
) -> tuple[bool, float, list[str]]:
    """For behaviors without curated keywords: simple token overlap."""
    expected = _tokenize(text)
    if not expected:
        return False, 0.0, []
    matched = expected & haystack
    overlap = len(matched) / len(expected)
    return overlap >= min_overlap, overlap, sorted(matched)


def _entry_text(b) -> str:
    return b.get("text", "") if isinstance(b, dict) else str(b)


def _entry_source(b) -> str:
    return b.get("source", "") if isinstance(b, dict) else ""


def _entry_category(b) -> str:
    return b.get("category", "") if isinstance(b, dict) else ""


def _entry_keywords(b) -> dict | None:
    if isinstance(b, dict) and isinstance(b.get("keywords"), dict):
        return b["keywords"]
    return None


def coverage_score(
    expected_behaviors: list,
    bt_xml: str,
    *,
    min_overlap: float = 0.4,  # only used for legacy entries without keywords
) -> dict:
    """Score how many expected behaviors are covered by a BT.

    Each entry may be:
      - {"text": ..., "source": ..., "category": ..., "keywords": {...}}  (preferred)
      - {"text": ..., "source": ...}  (fallback to token overlap)
      - "plain string"  (fallback to token overlap)

    Returns a dict with totals, per-behavior detail, and per-category coverage.
    """
    if not expected_behaviors or not bt_xml:
        return {
            "total": len(expected_behaviors or []),
            "covered": 0,
            "ratio": 0.0,
            "per_behavior": [],
            "per_category": {},
        }

    haystack = _bt_haystack(bt_xml)
    per: list[dict] = []
    covered = 0

    for entry in expected_behaviors:
        text = _entry_text(entry)
        source = _entry_source(entry)
        category = _entry_category(entry)
        keywords = _entry_keywords(entry)

        if keywords is not None:
            matched, detail = _check_keywords(haystack, keywords)
            per.append({
                "behavior": text,
                "source": source,
                "category": category,
                "matched": matched,
                "match_method": "keywords",
                "detail": detail,
            })
        else:
            matched, overlap, hits = _fallback_text_overlap(
                text, haystack, min_overlap,
            )
            per.append({
                "behavior": text,
                "source": source,
                "category": category,
                "matched": matched,
                "match_method": "text_overlap",
                "detail": {"overlap": round(overlap, 3), "hits": hits},
            })

        if matched:
            covered += 1

    # Per-category aggregation
    cat_total: dict[str, int] = {}
    cat_covered: dict[str, int] = {}
    for pb in per:
        cat = pb["category"] or "_uncategorized"
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if pb["matched"]:
            cat_covered[cat] = cat_covered.get(cat, 0) + 1
    per_category = {
        cat: {
            "total": cat_total[cat],
            "covered": cat_covered.get(cat, 0),
            "ratio": round(cat_covered.get(cat, 0) / cat_total[cat], 3),
        }
        for cat in cat_total
    }

    return {
        "total": len(expected_behaviors),
        "covered": covered,
        "ratio": round(covered / len(expected_behaviors), 3),
        "per_behavior": per,
        "per_category": per_category,
    }
