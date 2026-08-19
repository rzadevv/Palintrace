from pathlib import Path

from memlint.taxonomy import TAXONOMY_VERSION, DefectClass

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


def test_taxonomy_version_and_labels_are_frozen() -> None:
    assert TAXONOMY_VERSION == "1.0"
    assert len(DefectClass) == 8
    assert tuple(defect.value for defect in DefectClass) == EXPECTED_LABELS


def test_taxonomy_documentation_covers_every_frozen_class() -> None:
    documentation = Path("docs/taxonomy.md").read_text(encoding="utf-8")

    assert "Taxonomy version: `1.0`" in documentation
    for label in EXPECTED_LABELS:
        assert f"`{label}`" in documentation
    for field in (
        "Definition",
        "Inclusion criteria",
        "Exclusion criteria",
        "Example",
        "Non-example",
        "Required evidence",
        "Gold label target",
        "Establishment",
        "Mutation strategy",
    ):
        assert documentation.count(f"**{field}.**") == 8
