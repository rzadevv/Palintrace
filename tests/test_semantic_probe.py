import hashlib
import json
from collections import Counter
from pathlib import Path


def test_semantic_probe_v0_1_is_the_fixed_independent_18_case_suite() -> None:
    path = Path("tests/fixtures/semantic_probe_v0.1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "e277c04b9b18d5717f94b524e65467b0240ec515961abed49398132dc8777fb4"
    )
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert len(cases) == 18
    assert [case["id"] for case in cases] == [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
    ]
    assert Counter(case["expected_relation"] for case in cases) == {
        "entailment": 6,
        "contradiction": 6,
        "neutral": 6,
    }
    assert all(
        set(case) == {"id", "premise", "hypothesis", "expected_relation"}
        for case in cases
    )


def test_evidence_composition_probe_v0_1_has_frozen_content_and_shape() -> None:
    path = Path("tests/fixtures/evidence_composition_probe_v0.1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "84f824548b1ae2ee2d75fc04e5069bb1d8e45580092515a6c1aaa5d656675237"
    )
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert len(cases) == 18
    assert [case["id"] for case in cases] == [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
    ]
    assert Counter(case["expected_relation"] for case in cases) == {
        "entailment": 6,
        "contradiction": 6,
        "neutral": 6,
    }
    assert all(
        set(case) == {"id", "segments", "hypothesis", "expected_relation"}
        for case in cases
    )
    assert all(
        set(segment)
        == {"source_ref_index", "transcript_id", "turn_idx", "role", "span", "text"}
        for case in cases
        for segment in case["segments"]
    )
