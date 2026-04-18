from __future__ import annotations

import cv2
import numpy as np

from lightmocap.data.camera import PinholeCamera


def _world_to_camera(vertices: np.ndarray, camera: PinholeCamera) -> np.ndarray:
    return vertices @ camera.R.T + camera.T.reshape(1, 3)


def _project_camera_points(camera_points: np.ndarray, K: np.ndarray) -> np.ndarray:
    projected = camera_points @ K.T
    projected[:, :2] /= projected[:, 2:3]
    return projected[:, :2]


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
    color: tuple[int, int, int] = (60, 200, 80),
    alpha: float = 0.55,
    edge_color: tuple[int, int, int] | None = (30, 30, 30),
) -> np.ndarray:
    camera_points = _world_to_camera(vertices, camera)
    return render_mesh_overlay_from_camera(image, camera_points, faces, camera.K, color=color, alpha=alpha, edge_color=edge_color)


def make_panel(images: list[np.ndarray], labels: list[str], bg_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image, label in zip(images, labels):
        canvas = np.full((max_height + 32, image.shape[1], 3), bg_color, dtype=np.uint8)
        canvas[32 : 32 + image.shape[0], : image.shape[1]] = image
        cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        padded.append(canvas)
    return cv2.hconcat(padded)
