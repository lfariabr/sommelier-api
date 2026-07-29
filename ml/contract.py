"""MLN601 Assessment 2 parity contract and validation helpers."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from ml import DATA_DIR

CONTRACT_PATH = Path(__file__).with_name("assessment_contract.json")


@lru_cache(maxsize=1)
def load_assessment_contract() -> dict:
    """Return the checked-in contract for the submitted A2 classifier."""
    return json.loads(CONTRACT_PATH.read_text())


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
