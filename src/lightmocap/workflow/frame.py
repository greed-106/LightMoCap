from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from lightmocap.data.camera import CameraSet
from lightmocap.data.io import read_json, write_json
from lightmocap.viz.render import RenderOptions, render_mesh_overlay

if TYPE_CHECKING:
    from lightmocap.models.smplx import SMPLXLayer


@dataclass(frozen=True)
class FrameOutputLayout:
    root: Path
    annots: Path
    keypoints3d: Path
    smplx: Path
    renders: Path


def output_layout(root: str | Path) -> FrameOutputLayout:
    root_path = Path(root)
    return FrameOutputLayout(
        root=root_path,
        annots=root_path / "annots",
        keypoints3d=root_path / "keypoints3d",
        smplx=root_path / "smplx",
        renders=root_path / "renders",
    )


def normalize_frame_name(frame: str | int) -> str:
    if isinstance(frame, int):
        return f"{frame:06d}"
    frame = str(frame)
    return frame.zfill(6) if frame.isdigit() and len(frame) < 6 else frame


def build_frame_image_paths(image_root: str | Path, camera_names: list[str], frame: str | int, image_ext: str = ".jpg") -> dict[str, Path]:
    root = Path(image_root)
    frame_name = normalize_frame_name(frame)
    image_paths: dict[str, Path] = {}
    for camera_name in camera_names:
        direct = root / camera_name / f"{frame_name}{image_ext}"
        if direct.exists():
            image_paths[camera_name] = direct
            continue
        candidates = sorted((root / camera_name).glob(f"*{image_ext}"))
        if len(candidates) == 1:
            image_paths[camera_name] = candidates[0]
            continue
        image_paths[camera_name] = direct
    return image_paths


def save_multiview_annotations(annotations: dict[str, dict], output_root: str | Path, frame: str | int) -> None:
    output_root = Path(output_root)
    frame_name = normalize_frame_name(frame)
    for camera_name, annotation in annotations.items():
        write_json(output_root / camera_name / f"{frame_name}.json", annotation)


def load_multiview_annotations(annotation_root: str | Path, camera_names: list[str], frame: str | int) -> dict[str, dict]:
    annotation_root = Path(annotation_root)
    frame_name = normalize_frame_name(frame)
    return {camera_name: read_json(annotation_root / camera_name / f"{frame_name}.json") for camera_name in camera_names}


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_frame_result(
    output_root: str | Path,
    frame: str | int,
    result: dict[str, object],
) -> None:
    layout = output_layout(output_root)
    frame_name = normalize_frame_name(frame)
    write_json(
        layout.keypoints3d / f"{frame_name}.json",
        {"frame": frame_name, "keypoints3d": _as_numpy(result["keypoints3d"]).tolist()},
    )
    write_json(
        layout.smplx / f"{frame_name}.json",
        {"frame": frame_name, "smplx_params": {key: _as_numpy(value).tolist() for key, value in result["smplx_params"].items()}},
    )


def load_frame_smplx_params(output_root: str | Path, frame: str | int) -> dict[str, np.ndarray]:
    layout = output_layout(output_root)
    frame_name = normalize_frame_name(frame)
    payload = read_json(layout.smplx / f"{frame_name}.json")
    params = payload.get("smplx_params")
    if not isinstance(params, dict):
        raise ValueError(f"Missing smplx_params in {layout.smplx / f'{frame_name}.json'}")
    return {key: np.asarray(value, dtype=np.float32) for key, value in params.items()}


def infer_single_saved_smplx_frame(output_root: str | Path) -> str:
    layout = output_layout(output_root)
    candidates = sorted(layout.smplx.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No saved SMPL-X params found under {layout.smplx}")
    if len(candidates) > 1:
        names = ", ".join(path.stem for path in candidates[:5])
        suffix = "" if len(candidates) <= 5 else ", ..."
        raise ValueError(
            f"Multiple SMPL-X result files found under {layout.smplx}: {names}{suffix}. "
            "Please specify --frame explicitly."
        )
    return candidates[0].stem


def render_result_views(
    output_root: str | Path,
    frame: str | int,
    image_paths: dict[str, Path],
    cameras: CameraSet,
    vertices: np.ndarray,
    faces: np.ndarray,
    images: dict[str, np.ndarray] | None = None,
    max_views: int | None = None,
    render_options: RenderOptions | None = None,
) -> list[str]:
    layout = output_layout(output_root)
    frame_name = normalize_frame_name(frame)
    layout.renders.mkdir(parents=True, exist_ok=True)
    rendered = []
    camera_items = list(image_paths.items())
    if max_views is not None:
        camera_items = camera_items[:max_views]
    for camera_name, image_path in camera_items:
        if images is not None and camera_name in images:
            image = images[camera_name]
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
        overlay = render_mesh_overlay(image, vertices, faces, cameras[camera_name], options=render_options)
        out_path = layout.renders / f"{camera_name}_{frame_name}.jpg"
        cv2.imwrite(str(out_path), overlay)
        rendered.append(str(out_path))
    return rendered


def render_smplx_result_views(
    output_root: str | Path,
    frame: str | int,
    image_paths: dict[str, Path],
    cameras: CameraSet,
    body_model: "SMPLXLayer",
    smplx_params: dict[str, np.ndarray | torch.Tensor],
    images: dict[str, np.ndarray] | None = None,
    max_views: int | None = None,
    render_options: RenderOptions | None = None,
) -> list[str]:
    vertices = body_model(return_verts=True, return_tensor=False, **smplx_params)
    vertices = np.asarray(vertices)
    if vertices.ndim == 3:
        vertices = vertices[0]
    return render_result_views(
        output_root,
        frame,
        image_paths,
        cameras,
        vertices,
        body_model.faces,
        images=images,
        max_views=max_views,
        render_options=render_options,
    )
