import numpy as np

from lightmocap.core.triangulation.triangulate import batch_triangulate, triangulate_multiview_points
from lightmocap.data.camera import CameraSet, PinholeCamera


def test_batch_triangulate_recovers_simple_point():
    point = np.array([1.0, 2.0, 5.0, 1.0])
    P0 = np.array([[1000.0, 0.0, 0.0, 0.0], [0.0, 1000.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    P1 = np.array([[1000.0, 0.0, 0.0, -1000.0], [0.0, 1000.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    x0 = (P0 @ point)[:2] / (P0 @ point)[2]
    x1 = (P1 @ point)[:2] / (P1 @ point)[2]
    keypoints = np.array([
        [[x0[0], x0[1], 1.0]],
        [[x1[0], x1[1], 1.0]],
    ])
    result = batch_triangulate(keypoints, np.stack([P0, P1], axis=0), min_view=2)
    assert np.allclose(result[0, :3], point[:3], atol=1e-6)


def test_batch_triangulate_rejects_reprojection_outlier():
    point = np.array([1.0, 2.0, 5.0, 1.0])
    P0 = np.array([[1000.0, 0.0, 0.0, 0.0], [0.0, 1000.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    P1 = np.array([[1000.0, 0.0, 0.0, -1000.0], [0.0, 1000.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    P2 = np.array([[1000.0, 0.0, 0.0, 1000.0], [0.0, 1000.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    Pall = np.stack([P0, P1, P2], axis=0)
    x0 = (P0 @ point)[:2] / (P0 @ point)[2]
    x1 = (P1 @ point)[:2] / (P1 @ point)[2]
    x2 = (P2 @ point)[:2] / (P2 @ point)[2]
    keypoints = np.array([
        [[x0[0], x0[1], 1.0]],
        [[x1[0], x1[1], 1.0]],
        [[x2[0] + 400.0, x2[1] + 50.0, 1.0]],
    ])

    result = batch_triangulate(keypoints, Pall, min_view=2, max_reprojection_error=5.0)

    assert np.allclose(result[0, :3], point[:3], atol=1e-6)


def test_triangulate_multiview_points_passes_reprojection_threshold():
    point = np.array([1.0, 2.0, 5.0, 1.0])
    K = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]])
    cameras = CameraSet({
        "cam0": PinholeCamera("cam0", K, np.zeros((1, 5)), np.eye(3), np.array([[0.0], [0.0], [0.0]])),
        "cam1": PinholeCamera("cam1", K, np.zeros((1, 5)), np.eye(3), np.array([[-1.0], [0.0], [0.0]])),
        "cam2": PinholeCamera("cam2", K, np.zeros((1, 5)), np.eye(3), np.array([[1.0], [0.0], [0.0]])),
    })
    projections = cameras.projection_matrices()
    points2d = []
    for projection in projections:
        projected = (projection @ point)[:2] / (projection @ point)[2]
        points2d.append(projected.tolist() + [1.0])
    points2d[2][0] += 400.0
    points2d[2][1] += 50.0

    result = triangulate_multiview_points(
        np.array(points2d, dtype=np.float64)[:, None, :],
        cameras,
        undistort=False,
        max_reprojection_error=5.0,
    )

    assert np.allclose(result[0, :3], point[:3], atol=1e-6)
