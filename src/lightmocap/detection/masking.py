from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lightmocap.detection.adapter import bbox_from_keypoints


def load_mask(path: str | Path) -> np.ndarray:
    mask_path = Path(path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    return mask


def mask_to_foreground(mask: np.ndarray, threshold: int | float = 0) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array.max(axis=2)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected mask with shape (H, W) or (H, W, C), got {mask_array.shape}")
    return mask_array > threshold


def mask_foreground_ratio(mask: np.ndarray, threshold: int | float = 0) -> float:
    foreground = mask_to_foreground(mask, threshold=threshold)
    return float(foreground.mean())


def is_mask_usable(mask: np.ndarray, threshold: int | float = 0, min_area_ratio: float = 0.0) -> bool:
    if min_area_ratio <= 0:
        return True
    return mask_foreground_ratio(mask, threshold=threshold) >= min_area_ratio


def _validate_mask_shape(mask: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    foreground = mask_to_foreground(mask)
    if foreground.shape != image_shape[:2]:
        raise ValueError(f"Mask shape {foreground.shape} does not match image shape {image_shape[:2]}")
    return foreground


def apply_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
    threshold: int | float = 0,
    background_value: int = 0,
) -> np.ndarray:
    foreground = mask_to_foreground(mask, threshold=threshold)
    if foreground.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {foreground.shape} does not match image shape {image.shape[:2]}")
    masked = image.copy()
    masked[~foreground] = background_value
    return masked


def _filter_keypoints_by_foreground(keypoints: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    filtered = np.asarray(keypoints, dtype=np.float32).copy()
    if filtered.size == 0:
        return filtered.reshape(0, 3)
    if filtered.ndim != 2 or filtered.shape[1] != 3:
        raise ValueError(f"Expected keypoints with shape (N, 3), got {filtered.shape}")
    height, width = foreground.shape
    xy = filtered[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    valid = finite & (filtered[:, 2] > 0)
    xi = np.zeros(filtered.shape[0], dtype=np.int64)
    yi = np.zeros(filtered.shape[0], dtype=np.int64)
    xi[finite] = np.rint(xy[finite, 0]).astype(np.int64)
    yi[finite] = np.rint(xy[finite, 1]).astype(np.int64)
    inside_image = valid & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    inside_mask = np.zeros(filtered.shape[0], dtype=bool)
    inside_mask[inside_image] = foreground[yi[inside_image], xi[inside_image]]
    filtered[~inside_mask, 2] = 0.0
    return filtered


def filter_annotation_by_mask(
    annotation: dict[str, Any],
    mask: np.ndarray,
    threshold: int | float = 0,
    min_body_keypoints: int = 3,
    min_area_ratio: float = 0.0,
) -> dict[str, Any]:
    foreground = mask_to_foreground(mask, threshold=threshold)
    if min_area_ratio > 0 and float(foreground.mean()) < min_area_ratio:
        return deepcopy(annotation)
    if annotation.get("height") is not None and int(annotation["height"]) != foreground.shape[0]:
        raise ValueError(f"Mask height {foreground.shape[0]} does not match annotation height {annotation['height']}")
    if annotation.get("width") is not None and int(annotation["width"]) != foreground.shape[1]:
        raise ValueError(f"Mask width {foreground.shape[1]} does not match annotation width {annotation['width']}")

    masked_annotation = deepcopy(annotation)
    masked_annots: list[dict[str, Any]] = []
    for person in masked_annotation.get("annots", []):
        body = _filter_keypoints_by_foreground(np.asarray(person.get("keypoints", []), dtype=np.float32), foreground)
        hand_left = _filter_keypoints_by_foreground(np.asarray(person.get("handl2d", []), dtype=np.float32), foreground)
        hand_right = _filter_keypoints_by_foreground(np.asarray(person.get("handr2d", []), dtype=np.float32), foreground)
        face = _filter_keypoints_by_foreground(np.asarray(person.get("face2d", []), dtype=np.float32), foreground)

        if int((body[:, 2] > 0).sum()) < min_body_keypoints:
            continue

        person["keypoints"] = body.tolist()
        person["bbox"] = bbox_from_keypoints(body)
        person["handl2d"] = hand_left.tolist()
        person["bbox_handl2d"] = bbox_from_keypoints(hand_left)
        person["handr2d"] = hand_right.tolist()
        person["bbox_handr2d"] = bbox_from_keypoints(hand_right)
        person["face2d"] = face.tolist()
        person["bbox_face2d"] = bbox_from_keypoints(face)
        masked_annots.append(person)

    for person_id, person in enumerate(masked_annots):
        person["personID"] = person_id
    masked_annotation["annots"] = masked_annots
    return masked_annotation
