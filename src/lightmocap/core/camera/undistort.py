from __future__ import annotations

import cv2
import numpy as np


class Undistort:
    @staticmethod
    def image(frame: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        return cv2.undistort(frame, K, dist, None)

    @staticmethod
    def points(keypoints: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        if keypoints.ndim != 2:
            raise ValueError(f"Expected (N, C) keypoints, got {keypoints.shape}")
        kpts = np.ascontiguousarray(keypoints[:, None, :2])
        undistorted = cv2.undistortPoints(kpts, K, dist, P=K)
        return np.hstack([undistorted[:, 0], keypoints[:, 2:]])

    @staticmethod
    def bbox(bbox: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        points = np.array([[bbox[0], bbox[1], 1.0], [bbox[2], bbox[3], 1.0]], dtype=np.float64)
        points = Undistort.points(points, K, dist)
        return np.array([points[0, 0], points[0, 1], points[1, 0], points[1, 1], bbox[4]], dtype=np.float64)
