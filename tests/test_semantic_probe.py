import json
from collections import Counter
from pathlib import Path


def test_semantic_probe_v0_1_is_the_fixed_independent_18_case_suite() -> None:
    path = Path("tests/fixtures/semantic_probe_v0.1.json")
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
