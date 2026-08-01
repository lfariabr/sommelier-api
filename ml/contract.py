"""MLN601 assessment contracts and validation helpers."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from ml import DATA_DIR

CLASSIFICATION_CONTRACT_PATH = Path(__file__).with_name("assessment_contract.json")
REGRESSION_CONTRACT_PATH = Path(__file__).with_name("regression_contract.json")


@lru_cache(maxsize=1)
def load_assessment_contract() -> dict:
    """Return the checked-in contract for the submitted A2 classifier."""
    return json.loads(CLASSIFICATION_CONTRACT_PATH.read_text())


@lru_cache(maxsize=1)
def load_regression_contract() -> dict:
    """Return the A1 source protocol and its production adaptation."""
    return json.loads(REGRESSION_CONTRACT_PATH.read_text())


def validate_contract_compatibility() -> None:
    """Ensure both model contracts describe the same input data and features."""
    classification = load_assessment_contract()
    regression = load_regression_contract()
    if regression["relationship"] != "assessment_derived":
        raise ValueError("Regression contract must declare assessment_derived lineage")
    if regression["dataset"]["files"] != classification["dataset"]["files"]:
        raise ValueError("Regression and classification dataset hashes differ")
    if regression["feature_order"] != classification["feature_order"]:
        raise ValueError("Regression and classification feature order differs")

    raw_rows = regression["dataset"]["raw_rows"]
    serving = regression["serving_adaptation"]
    if raw_rows - serving["duplicates_removed"] != serving["model_rows"]:
        raise ValueError("Regression serving row counts are inconsistent")


def sha256_file(path: Path) -> str:
    """Calculate a file digest without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_files(data_dir: Path = DATA_DIR) -> None:
    """Fail if either local UCI source file differs from the contract dataset."""
    contract = load_assessment_contract()
    for filename, expected_hash in contract["dataset"]["files"].items():
        actual_hash = sha256_file(data_dir / filename)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Dataset parity failure for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
