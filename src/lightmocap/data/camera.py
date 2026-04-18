from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import yaml


def natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


class OpenCVYamlLoader(yaml.SafeLoader):
    """YAML loader that understands OpenCV's !!opencv-matrix tag."""


def _opencv_matrix_constructor(loader: OpenCVYamlLoader, node: yaml.Node) -> list[list[float]]:
    mapping = loader.construct_mapping(node, deep=True)
    rows = int(mapping["rows"])
    cols = int(mapping["cols"])
    data = np.asarray(mapping["data"], dtype=np.float64)
    return data.reshape(rows, cols).tolist()


OpenCVYamlLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", _opencv_matrix_constructor)
OpenCVYamlLoader.add_constructor("!opencv-matrix", _opencv_matrix_constructor)


def _load_yaml_document(path: str | Path) -> dict[str, Any]:
    camera_path = Path(path)
    if not camera_path.exists():
        raise FileNotFoundError(camera_path)
    text = camera_path.read_text()
    if text.startswith("%YAML:1.0"):
        text = "\n".join(text.splitlines()[1:])
    payload = yaml.load(text, Loader=OpenCVYamlLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported camera file format: {camera_path}")
    return payload


def _as_matrix(value: Any, *, shape: tuple[int, int] | None = None, key: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing matrix value for {key}")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1 and shape is not None:
        matrix = matrix.reshape(shape)
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"Expected {key} to have shape {shape}, got {matrix.shape}")
    return matrix


def _as_translation(value: Any, *, key: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"Expected {key} to contain 3 translation values, got {vector.shape}")
    return vector.reshape(3, 1)


def _as_distortion(value: Any, *, key: str) -> np.ndarray:
    dist = np.asarray(value, dtype=np.float64).reshape(-1)
    if dist.size not in {4, 5}:
        raise ValueError(f"Expected {key} to contain 4 or 5 distortion values, got {dist.size}")
    return dist.reshape(1, -1)


def _parse_image_size(entry: dict[str, Any], name: str) -> tuple[int | None, int | None]:
    if "image_size" in entry:
        image_size = entry["image_size"]
        if isinstance(image_size, dict):
            width = image_size.get("width")
            height = image_size.get("height")
            return (None if width is None else int(width), None if height is None else int(height))
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            width, height = image_size
            return int(width), int(height)
        raise ValueError(f"Unsupported image_size format for camera {name}")
    width = entry.get("W", entry.get("width"))
    height = entry.get("H", entry.get("height"))
    return (None if width is None else int(width), None if height is None else int(height))


def _camera_to_dict(camera: "PinholeCamera") -> dict[str, Any]:
    image_size: dict[str, int] = {}
    if camera.W is not None:
        image_size["width"] = int(camera.W)
    if camera.H is not None:
        image_size["height"] = int(camera.H)
    payload: dict[str, Any] = {
        "name": camera.name,
        "K": camera.K.tolist(),
        "dist": camera.dist.reshape(-1).tolist(),
        "R": camera.R.tolist(),
        "T": camera.T.tolist(),
    }
    if image_size:
        payload["image_size"] = image_size
    return payload


@dataclass(frozen=True)
class PinholeCamera:
    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    T: np.ndarray
    H: int | None = None
    W: int | None = None

    @property
    def invK(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    @property
    def RT(self) -> np.ndarray:
        return np.hstack((self.R, self.T))

    @property
    def P(self) -> np.ndarray:
        return self.K @ self.RT

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.T


class CameraSet:
    def __init__(self, cameras: dict[str, PinholeCamera]) -> None:
        self._cameras = cameras

    def __getitem__(self, key: str) -> PinholeCamera:
        return self._cameras[key]

    def __len__(self) -> int:
        return len(self._cameras)

    def __iter__(self):
        return iter(self._cameras)

    @property
    def names(self) -> list[str]:
        return list(self._cameras.keys())

    @classmethod
    def _from_legacy_yaml(cls, intri_path: str | Path, extri_path: str | Path) -> "CameraSet":
        intri = _load_yaml_document(intri_path)
        extri = _load_yaml_document(extri_path)
        cameras: dict[str, PinholeCamera] = {}
        names = sorted([str(item) for item in intri.get("names", []) if str(item) != "none"], key=natural_sort_key)
        for name in names:
            K = _as_matrix(intri.get(f"K_{name}"), shape=(3, 3), key=f"K_{name}")
            dist_value = intri.get(f"dist_{name}", intri.get(f"D_{name}"))
            dist = _as_distortion(dist_value, key=f"dist_{name}")
            rotation_matrix = extri.get(f"Rot_{name}")
            if rotation_matrix is not None:
                R = _as_matrix(rotation_matrix, shape=(3, 3), key=f"Rot_{name}")
            else:
                rotation_vector = _as_matrix(extri.get(f"R_{name}"), key=f"R_{name}")
                R = cv2.Rodrigues(rotation_vector.reshape(3, 1))[0]
            T = _as_translation(extri.get(f"T_{name}"), key=f"T_{name}")
            cameras[name] = PinholeCamera(
                name=name,
                K=K.astype(np.float64),
                dist=dist.astype(np.float64),
                R=R.astype(np.float64),
                T=T.astype(np.float64),
                H=None if intri.get(f"H_{name}") is None else int(intri[f"H_{name}"]),
                W=None if intri.get(f"W_{name}") is None else int(intri[f"W_{name}"]),
            )
        return cls(cameras)

    @classmethod
    def _from_compact_yaml(cls, camera_path: str | Path) -> "CameraSet":
        payload = _load_yaml_document(camera_path)
        entries = payload.get("cameras")
        if not isinstance(entries, list):
            raise ValueError(f"Expected cameras list in {camera_path}")
        cameras: dict[str, PinholeCamera] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid camera entry in {camera_path}: {entry!r}")
            name = str(entry["name"])
            width, height = _parse_image_size(entry, name)
            cameras[name] = PinholeCamera(
                name=name,
                K=_as_matrix(entry.get("K"), shape=(3, 3), key=f"{name}.K"),
                dist=_as_distortion(entry.get("dist"), key=f"{name}.dist"),
                R=_as_matrix(entry.get("R"), shape=(3, 3), key=f"{name}.R"),
                T=_as_translation(entry.get("T"), key=f"{name}.T"),
                H=height,
                W=width,
            )
        ordered = {name: cameras[name] for name in sorted(cameras, key=natural_sort_key)}
        return cls(ordered)

    @classmethod
    def from_yaml(cls, intri_path: str | Path, extri_path: str | Path | None = None) -> "CameraSet":
        if extri_path is not None:
            return cls._from_legacy_yaml(intri_path, extri_path)
        payload = _load_yaml_document(intri_path)
        if "cameras" in payload:
            return cls._from_compact_yaml(intri_path)
        raise ValueError(
            "Single camera YAML must contain a top-level `cameras` list. "
            "Pass both intri/extri paths to read legacy EasyMocap files."
        )

    def projection_matrices(self, names: list[str] | None = None) -> np.ndarray:
        names = self.names if names is None else names
        return np.stack([self[name].P for name in names], axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "camera_model": "pinhole_opencv",
            "cameras": [_camera_to_dict(self[name]) for name in self.names],
        }

    def to_yaml(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=False)
        return output_path

    def summary(self) -> dict[str, object]:
        return {
            "num_cameras": len(self),
            "camera_names": self.names,
            "image_sizes": {
                name: {"width": self[name].W, "height": self[name].H}
                for name in self.names
            },
        }
