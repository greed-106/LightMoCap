from __future__ import annotations

import numpy as np

from lightmocap.core.camera.undistort import Undistort
from lightmocap.data.camera import CameraSet


def batch_triangulate(keypoints_: np.ndarray, Pall: np.ndarray, min_view: int = 2) -> np.ndarray:
    v = (keypoints_[:, :, -1] > 0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0) / v[valid_joint]
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    A = np.hstack([conf * (uP2 - P0), conf * (vP2 - P1)])
    _, _, vh = np.linalg.svd(A)
    X = vh[:, -1, :]
    X = X / X[:, 3:]
    result = np.zeros((keypoints_.shape[1], 4), dtype=np.float64)
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result


def project_points(keypoints3d: np.ndarray, Pall: np.ndarray) -> np.ndarray:
    homo = np.concatenate([keypoints3d[..., :3], np.ones_like(keypoints3d[..., :1])], axis=-1)
    projected = np.einsum("vab,kb->vka", Pall, homo)
    projected[..., :2] /= projected[..., 2:]
    return projected


def triangulate_multiview_points(
    keypoints2d: np.ndarray,
    cameras: CameraSet,
    camera_names: list[str] | None = None,
    min_view: int = 2,
    undistort: bool = True,
) -> np.ndarray:
    camera_names = cameras.names if camera_names is None else camera_names
    if keypoints2d.shape[0] != len(camera_names):
        raise ValueError("Number of views does not match number of camera names")
    if undistort:
        undistorted = []
        for index, camera_name in enumerate(camera_names):
            camera = cameras[camera_name]
            undistorted.append(Undistort.points(keypoints2d[index], camera.K, camera.dist))
        keypoints2d = np.stack(undistorted, axis=0)
    Pall = cameras.projection_matrices(camera_names)
    return batch_triangulate(keypoints2d, Pall, min_view=min_view)
