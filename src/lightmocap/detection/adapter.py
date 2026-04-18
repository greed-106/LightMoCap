from __future__ import annotations

from typing import Any

import numpy as np

from lightmocap.data.skeleton import BODY25_SIZE, FACE_SIZE, HAND21_SIZE


def bbox_from_keypoints(
    keypoints: np.ndarray,
    rescale: float = 1.2,
    detection_thresh: float = 0.05,
    min_pixel: float = 5.0,
) -> list[float]:
    valid = keypoints[:, -1] > detection_thresh
    if valid.sum() < 3:
        return [0.0, 0.0, 100.0, 100.0, 0.0]
    valid_keypoints = keypoints[valid, :2]
    center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0)) / 2.0
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    if bbox_size[0] < min_pixel or bbox_size[1] < min_pixel:
        return [0.0, 0.0, 100.0, 100.0, 0.0]
    bbox_size = bbox_size * rescale
    return [
        float(center[0] - bbox_size[0] / 2.0),
        float(center[1] - bbox_size[1] / 2.0),
        float(center[0] + bbox_size[0] / 2.0),
        float(center[1] + bbox_size[1] / 2.0),
        float(keypoints[valid, 2].mean()),
    ]


def _reshape_keypoints(data: list[float], num_points: int) -> np.ndarray:
    if not data:
        return np.zeros((num_points, 3), dtype=np.float32)
    array = np.asarray(data, dtype=np.float32).reshape(-1, 3)
    if array.shape[0] != num_points:
        if array.shape[0] > num_points:
            array = array[:num_points]
        else:
            padded = np.zeros((num_points, 3), dtype=np.float32)
            padded[: array.shape[0]] = array
            array = padded
    return array


def openpose_people_to_annotation(
    payload: dict[str, Any],
    filename: str,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    annots: list[dict[str, Any]] = []
    for person_id, person in enumerate(payload.get("people", [])):
        body = _reshape_keypoints(person.get("pose_keypoints_2d", []), BODY25_SIZE)
        hand_left = _reshape_keypoints(person.get("hand_left_keypoints_2d", []), HAND21_SIZE)
        hand_right = _reshape_keypoints(person.get("hand_right_keypoints_2d", []), HAND21_SIZE)
        face = _reshape_keypoints(person.get("face_keypoints_2d", []), 70)
        face = face[17 : 17 + FACE_SIZE]
        annots.append(
            {
                "personID": person_id,
                "bbox": bbox_from_keypoints(body),
                "keypoints": body.tolist(),
                "bbox_handl2d": bbox_from_keypoints(hand_left),
                "handl2d": hand_left.tolist(),
                "bbox_handr2d": bbox_from_keypoints(hand_right),
                "handr2d": hand_right.tolist(),
                "bbox_face2d": bbox_from_keypoints(face),
                "face2d": face.tolist(),
                "isKeyframe": False,
            }
        )
    return {
        "filename": filename,
        "height": height,
        "width": width,
        "annots": annots,
        "isKeyframe": False,
    }


BODY18_TO_BODY25 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
}

def body18_to_body25(keypoints18: np.ndarray) -> np.ndarray:
    if keypoints18.shape != (18, 3):
        raise ValueError(f"Expected body18 keypoints with shape (18, 3), got {keypoints18.shape}")
    body25 = np.zeros((BODY25_SIZE, 3), dtype=np.float32)
    for src, dst in BODY18_TO_BODY25.items():
        body25[dst] = keypoints18[src]
    right_hip = keypoints18[8]
    left_hip = keypoints18[11]
    if right_hip[2] > 0 and left_hip[2] > 0:
        body25[8, :2] = (right_hip[:2] + left_hip[:2]) / 2.0
        body25[8, 2] = min(right_hip[2], left_hip[2])
    elif right_hip[2] > 0:
        body25[8] = right_hip
    elif left_hip[2] > 0:
        body25[8] = left_hip
    return body25


def body24_to_body25(keypoints24: np.ndarray) -> np.ndarray:
    """Convert rtmlib openpose-style 24 body+foot points to body25.

    Expected input layout with `Wholebody(to_openpose=True)` is:
    0..17  -> body18
    18..23 -> [LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel]
    """
    if keypoints24.shape != (24, 3):
        raise ValueError(f"Expected body24 keypoints with shape (24, 3), got {keypoints24.shape}")
    body25 = body18_to_body25(keypoints24[:18])
    # body25 feet order matches rtmlib openpose134 order when to_openpose=True.
    body25[19:25] = keypoints24[18:24]
    return body25


def rtmlib_body_to_annotation(
    keypoints: np.ndarray,
    scores: np.ndarray,
    filename: str,
    width: int | None = None,
    height: int | None = None,
    bboxes: np.ndarray | None = None,
) -> dict[str, Any]:
    annots: list[dict[str, Any]] = []
    if keypoints.ndim != 3:
        raise ValueError(f"Expected keypoints with shape (N, K, 2), got {keypoints.shape}")
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape (N, K), got {scores.shape}")
    for person_id in range(keypoints.shape[0]):
        body18 = np.concatenate([keypoints[person_id], scores[person_id][..., None]], axis=-1).astype(np.float32)
        body25 = body18_to_body25(body18)
        bbox = bbox_from_keypoints(body25)
        if bboxes is not None and person_id < bboxes.shape[0]:
            bbox = [
                float(bboxes[person_id][0]),
                float(bboxes[person_id][1]),
                float(bboxes[person_id][2]),
                float(bboxes[person_id][3]),
                float(body25[body25[:, 2] > 0, 2].mean()) if (body25[:, 2] > 0).any() else 0.0,
            ]
        annots.append(
            {
                "personID": person_id,
                "bbox": bbox,
                "keypoints": body25.tolist(),
                "bbox_handl2d": [0.0, 0.0, 100.0, 100.0, 0.0],
                "handl2d": np.zeros((HAND21_SIZE, 3), dtype=np.float32).tolist(),
                "bbox_handr2d": [0.0, 0.0, 100.0, 100.0, 0.0],
                "handr2d": np.zeros((HAND21_SIZE, 3), dtype=np.float32).tolist(),
                "bbox_face2d": [0.0, 0.0, 100.0, 100.0, 0.0],
                "face2d": np.zeros((FACE_SIZE, 3), dtype=np.float32).tolist(),
                "isKeyframe": False,
            }
        )
    return {
        "filename": filename,
        "height": height,
        "width": width,
        "annots": annots,
        "isKeyframe": False,
    }


def rtmlib_wholebody_to_annotation(
    keypoints: np.ndarray,
    scores: np.ndarray,
    filename: str,
    width: int | None = None,
    height: int | None = None,
    bboxes: np.ndarray | None = None,
) -> dict[str, Any]:
    annots: list[dict[str, Any]] = []
    if keypoints.ndim != 3:
        raise ValueError(f"Expected keypoints with shape (N, K, 2), got {keypoints.shape}")
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape (N, K), got {scores.shape}")
    for person_id in range(keypoints.shape[0]):
        wholebody = np.concatenate([keypoints[person_id], scores[person_id][..., None]], axis=-1).astype(np.float32)
        if wholebody.shape[0] < 134:
            raise ValueError(f"Expected at least 134 wholebody keypoints, got {wholebody.shape[0]}")
        body25 = body24_to_body25(wholebody[:24])
        face51 = wholebody[24:92][17 : 17 + FACE_SIZE]
        hand_left = wholebody[92:113]
        hand_right = wholebody[113:134]
        bbox = bbox_from_keypoints(body25)
        if bboxes is not None and person_id < bboxes.shape[0]:
            bbox = [
                float(bboxes[person_id][0]),
                float(bboxes[person_id][1]),
                float(bboxes[person_id][2]),
                float(bboxes[person_id][3]),
                float(body25[body25[:, 2] > 0, 2].mean()) if (body25[:, 2] > 0).any() else 0.0,
            ]
        annots.append(
            {
                "personID": person_id,
                "bbox": bbox,
                "keypoints": body25.tolist(),
                "bbox_handl2d": bbox_from_keypoints(hand_left),
                "handl2d": hand_left.tolist(),
                "bbox_handr2d": bbox_from_keypoints(hand_right),
                "handr2d": hand_right.tolist(),
                "bbox_face2d": bbox_from_keypoints(face51),
                "face2d": face51.tolist(),
                "isKeyframe": False,
            }
        )
    return {
        "filename": filename,
        "height": height,
        "width": width,
        "annots": annots,
        "isKeyframe": False,
    }


def canonical_keypoints_from_annotation(annotation: dict[str, Any], mode: str = "body25") -> np.ndarray:
    if not annotation["annots"]:
        if mode == "body25":
            return np.zeros((BODY25_SIZE, 3), dtype=np.float32)
        if mode == "bodyhandface":
            return np.zeros((BODY25_SIZE + HAND21_SIZE * 2 + FACE_SIZE, 3), dtype=np.float32)
        raise ValueError(f"Unsupported mode: {mode}")
    person = annotation["annots"][0]
    body = np.asarray(person["keypoints"], dtype=np.float32)
    if mode == "body25":
        return body
    if mode == "bodyhandface":
        handl = np.asarray(person["handl2d"], dtype=np.float32)
        handr = np.asarray(person["handr2d"], dtype=np.float32)
        face = np.asarray(person["face2d"], dtype=np.float32)
        return np.vstack([body, handl, handr, face])
    raise ValueError(f"Unsupported mode: {mode}")
