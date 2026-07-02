from __future__ import annotations

import numpy as np

from lightmocap.core.camera.undistort import Undistort
from lightmocap.data.camera import CameraSet


def _linear_batch_triangulate(keypoints_: np.ndarray, Pall: np.ndarray, min_view: int = 2) -> np.ndarray:
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


def _ransac_triangulate(
    keypoints_: np.ndarray,
    Pall: np.ndarray,
    min_view: int,
    max_reprojection_error: float,
) -> np.ndarray:
    n_joints = keypoints_.shape[1]
    result = _linear_batch_triangulate(keypoints_, Pall, min_view=min_view)
    for joint_index in range(n_joints):
        views = np.flatnonzero(keypoints_[:, joint_index, 2] > 0)
        if views.size <= min_view:
            continue

        best_inliers: np.ndarray | None = None
        best_score: tuple[int, float, float] | None = None
        for left in range(views.size - 1):
            for right in range(left + 1, views.size):
                pair = views[[left, right]]
                candidate = _linear_batch_triangulate(keypoints_[pair, joint_index : joint_index + 1], Pall[pair], min_view=2)
                if candidate[0, 3] <= 0 or not np.isfinite(candidate[0, :3]).all():
                    continue
                projected = project_points(candidate[:, :3], Pall[views])[:, 0, :2]
                errors = np.linalg.norm(projected - keypoints_[views, joint_index, :2], axis=1)
                finite = np.isfinite(errors)
                inliers = views[finite & (errors <= max_reprojection_error)]
                if inliers.size < min_view:
                    continue
                inlier_errors = errors[finite & (errors <= max_reprojection_error)]
                score = (int(inliers.size), -float(np.median(inlier_errors)), -float(np.mean(inlier_errors)))
                if best_score is None or score > best_score:
                    best_score = score
                    best_inliers = inliers

        if best_inliers is None:
            continue
        refined = _linear_batch_triangulate(keypoints_[best_inliers, joint_index : joint_index + 1], Pall[best_inliers], min_view=min_view)
        result[joint_index] = refined[0]
    return result


def batch_triangulate(
    keypoints_: np.ndarray,
    Pall: np.ndarray,
    min_view: int = 2,
    max_reprojection_error: float | None = None,
) -> np.ndarray:
    if max_reprojection_error is None or max_reprojection_error <= 0:
        return _linear_batch_triangulate(keypoints_, Pall, min_view=min_view)
    return _ransac_triangulate(
        keypoints_,
        Pall,
        min_view=min_view,
        max_reprojection_error=float(max_reprojection_error),
    )


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
    max_reprojection_error: float | None = None,
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
    return batch_triangulate(keypoints2d, Pall, min_view=min_view, max_reprojection_error=max_reprojection_error)
