import copy

from scripts.build_lopo_manifests import STYLE_GROUP, build_folds


def test_build_folds_rotates_holdout_and_validation():
    payload = {
        "manifest_id": "base",
        "partitions": {
            "train": [{"id": "a", "manuscript_id": "c2av01", "xml_path": "1.xml"}],
            "validation": [{"id": "b", "manuscript_id": "c2av02", "xml_path": "2.xml"}],
            "holdout": [{"id": "c", "manuscript_id": "c2av03", "xml_path": "3.xml"}],
        },
        "style_groups": {STYLE_GROUP: {"manuscript_ids": [], "base_model_override": "base.safetensors"}},
        "metadata": {},
    }
    original = copy.deepcopy(payload)

    folds = build_folds(payload)

    assert payload == original
    assert [fold["metadata"]["lopo_holdout"] for fold in folds] == ["c2av01", "c2av02", "c2av03"]
    assert [fold["metadata"]["lopo_validation"] for fold in folds] == ["c2av02", "c2av03", "c2av01"]
    assert [len(fold["partitions"]["train"]) for fold in folds] == [1, 1, 1]
