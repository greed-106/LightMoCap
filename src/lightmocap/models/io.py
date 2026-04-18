from __future__ import annotations

from pathlib import Path

import numpy as np


GENDER_TO_FILE = {
    "neutral": "SMPLX_NEUTRAL.npz",
    "male": "SMPLX_MALE.npz",
    "female": "SMPLX_FEMALE.npz",
}


def resolve_smplx_model_path(model_root: str | Path, gender: str = "neutral") -> Path:
    gender = gender.lower()
    if gender not in GENDER_TO_FILE:
        raise ValueError(f"Unsupported gender: {gender}")
    path = Path(model_root) / GENDER_TO_FILE[gender]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_smplx_npz(model_root: str | Path, gender: str = "neutral") -> dict[str, np.ndarray]:
    path = resolve_smplx_model_path(model_root, gender)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
