from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from lightmocap.data.camera import PinholeCamera


@dataclass(frozen=True)
class RenderOptions:
    color: tuple[int, int, int] = (60, 200, 80)
    alpha: float = 0.55
    edge_color: tuple[int, int, int] | None = (30, 30, 30)
    render_device: str = "auto"
    metallic_factor: float = 0.05
    roughness_factor: float = 0.65


def _world_to_camera(vertices: np.ndarray, camera: PinholeCamera) -> np.ndarray:
    return vertices @ camera.R.T + camera.T.reshape(1, 3)


def _project_camera_points(camera_points: np.ndarray, K: np.ndarray) -> np.ndarray:
    projected = camera_points @ K.T
    projected[:, :2] /= projected[:, 2:3]
    return projected[:, :2]


def _normalize_color(color: tuple[int, int, int], alpha: float = 1.0) -> tuple[float, float, float, float]:
    return tuple(channel / 255.0 for channel in color) + (float(alpha),)


def _set_pyopengl_platform(render_device: str) -> None:
    if "PYOPENGL_PLATFORM" in os.environ:
        return
    if render_device == "gpu":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif render_device == "cpu":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def _pyrender_camera_pose(camera: PinholeCamera) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = camera.R.T
    pose[:3, 3] = (-camera.R.T @ camera.T).reshape(3)
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return pose @ opencv_to_opengl


def _render_with_pyrender(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: PinholeCamera,
    options: RenderOptions,
) -> np.ndarray:
    _set_pyopengl_platform(options.render_device)
    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "pyrender backend requires the `viz` extras. Run `uv sync --extra viz`."
        ) from exc

    height, width = image.shape[:2]
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ambient_light=np.array([0.12, 0.12, 0.12, 1.0], dtype=np.float32),
    )
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=_normalize_color(options.color, options.alpha),
        metallicFactor=float(options.metallic_factor),
        roughnessFactor=float(options.roughness_factor),
        alphaMode="OPAQUE" if options.alpha >= 0.999 else "BLEND",
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
    scene.add(
        pyrender.IntrinsicsCamera(
            fx=float(camera.K[0, 0]),
            fy=float(camera.K[1, 1]),
            cx=float(camera.K[0, 2]),
            cy=float(camera.K[1, 2]),
        ),
        pose=_pyrender_camera_pose(camera),
    )

    camera_pose = _pyrender_camera_pose(camera)
    light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=4.5)
    scene.add(light, pose=camera_pose)
    fill_pose = camera_pose.copy()
    fill_pose[:3, 3] += np.array([0.6, 0.4, 0.2], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0), pose=fill_pose)
    rim_pose = camera_pose.copy()
    rim_pose[:3, 3] += np.array([-0.6, 0.5, -0.2], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=1.5), pose=rim_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    render_bgr = color_rgba[..., :3][:, :, ::-1].astype(np.float32)
    alpha = (color_rgba[..., 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
    output = image.astype(np.float32) * (1.0 - alpha) + render_bgr * alpha
    return np.clip(np.round(output), 0, 255).astype(np.uint8)


def render_mesh_overlay_from_camera(
    image: np.ndarray,
    camera_points: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int] = (60, 200, 80),
    alpha: float = 0.55,
    edge_color: tuple[int, int, int] | None = (30, 30, 30),
) -> np.ndarray:
    height, width = image.shape[:2]
    projected = _project_camera_points(camera_points, K)
    face_points = projected[faces]
    face_depth = camera_points[faces, 2].mean(axis=1)
    order = np.argsort(face_depth)[::-1]
    color_layer = image.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    for face_index in order:
        pts3d = camera_points[faces[face_index]]
        if np.any(pts3d[:, 2] <= 1e-4):
            continue
        pts2d = face_points[face_index]
        if np.any(~np.isfinite(pts2d)):
            continue
        if np.max(pts2d[:, 0]) < 0 or np.max(pts2d[:, 1]) < 0 or np.min(pts2d[:, 0]) >= width or np.min(pts2d[:, 1]) >= height:
            continue
        polygon = np.round(pts2d).astype(np.int32)
        cv2.fillConvexPoly(color_layer, polygon, color)
        cv2.fillConvexPoly(mask, polygon, 255)
        if edge_color is not None:
            cv2.polylines(color_layer, [polygon], True, edge_color, 1, lineType=cv2.LINE_AA)
    output = image.copy()
    valid = mask > 0
    output[valid] = np.round(image[valid] * (1.0 - alpha) + color_layer[valid] * alpha).astype(np.uint8)
    return output


def render_mesh_overlay(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: PinholeCamera,
    options: RenderOptions | None = None,
    color: tuple[int, int, int] = (60, 200, 80),
    alpha: float = 0.55,
    edge_color: tuple[int, int, int] | None = (30, 30, 30),
) -> np.ndarray:
    options = options or RenderOptions(color=color, alpha=alpha, edge_color=edge_color)
    return _render_with_pyrender(image, vertices, faces, camera, options)


def make_panel(images: list[np.ndarray], labels: list[str], bg_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image, label in zip(images, labels):
        canvas = np.full((max_height + 32, image.shape[1], 3), bg_color, dtype=np.uint8)
        canvas[32 : 32 + image.shape[0], : image.shape[1]] = image
        cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        padded.append(canvas)
    return cv2.hconcat(padded)
