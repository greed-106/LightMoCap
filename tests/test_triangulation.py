import numpy as np

from lightmocap.core.triangulation.triangulate import batch_triangulate


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
