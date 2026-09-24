from pathlib import Path

from palintrace.taxonomy import TAXONOMY_VERSION, DefectClass

EXPECTED_LABELS = (
    "unsupported_claim",
    "internal_contradiction",
    "stale_active",
    "orphaned_provenance",
    "retrieval_shadowing",
    "injected_instruction",
    "privacy_scope_violation",
    "redundancy_bloat",
)


def test_taxonomy_version_and_labels() -> None:
    assert TAXONOMY_VERSION == "1.1"
    assert tuple(defect.value for defect in DefectClass) == EXPECTED_LABELS


def test_taxonomy_documentation_covers_every_class() -> None:
    documentation = Path("docs/taxonomy.md").read_text(encoding="utf-8")

    for label in EXPECTED_LABELS:
        assert f"## `{label}`" in documentation
