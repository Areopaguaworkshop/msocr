#!/usr/bin/env python3
"""Build leave-one-plate-out manifests from the c2av fine-tune manifest."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data/manifests/c2av-finetune.json"
DEFAULT_OUT = ROOT / "data/manifests/lopo"
STYLE_GROUP = "c2av-syriac-finetune"


def _plate_key(case: dict) -> str:
    return str(case["manuscript_id"])


def build_folds(payload: dict) -> list[dict]:
    cases = sorted(
        [case for part in ("train", "validation", "holdout") for case in payload["partitions"].get(part, [])],
        key=_plate_key,
    )
    folds = []
    for idx, holdout in enumerate(cases):
        val = cases[(idx + 1) % len(cases)]
        train = [case for case in cases if case not in (holdout, val)]

        fold = copy.deepcopy(payload)
        holdout_id = holdout["manuscript_id"]
        val_id = val["manuscript_id"]
        fold["manifest_id"] = f"{payload['manifest_id']}-lopo-{holdout_id}"
        fold["partitions"] = {
            "train": [_with_id(case, "train") for case in train],
            "validation": [_with_id(val, "val")],
            "holdout": [_with_id(holdout, "holdout")],
        }
        fold["style_groups"][STYLE_GROUP]["manuscript_ids"] = [case["manuscript_id"] for case in cases]
        fold["metadata"] = {
            **payload.get("metadata", {}),
            "lopo_holdout": holdout_id,
            "lopo_validation": val_id,
            "lopo_train_count": len(train),
            "purpose": "leave-one-plate-out CV fold",
        }
        folds.append(fold)
    return folds


def _with_id(case: dict, suffix: str) -> dict:
    out = copy.deepcopy(case)
    out["id"] = f"{case['manuscript_id']}-{suffix}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold in build_folds(payload):
        path = args.out_dir / f"{fold['manifest_id']}.json"
        path.write_text(json.dumps(fold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
