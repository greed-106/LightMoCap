import json
import sys
from pathlib import Path

import numpy as np

from lightmocap.data.camera import CameraSet
from lightmocap.core.triangulation.triangulate import batch_triangulate

sys.path.insert(0, "/home/ymj/code/python/EasyMocap")

from easymocap.mytools.camera_utils import read_camera  # noqa: E402
from easymocap.mytools.triangulator import batch_triangulate as v1_batch_triangulate  # noqa: E402


def test_camera_projection_matches_v1_on_coreview():
    root = Path("/home/ymj/code/python/EasyMocap/CoreView_377")
    our_cameras = CameraSet.from_yaml(root / "intri.yml", root / "extri.yml")
    v1_cameras = read_camera(root / "intri.yml", root / "extri.yml")
    for name in our_cameras.names:
        assert np.allclose(our_cameras[name].P, v1_cameras[name]["P"], atol=1e-12)


def test_batch_triangulate_matches_v1_on_coreview_frame():
    root = Path("/home/ymj/code/python/EasyMocap/CoreView_377")
    cameras = CameraSet.from_yaml(root / "intri.yml", root / "extri.yml")
    frame_name = "000003"
    keypoints = []
    for camera_name in cameras.names:
        with (root / "keypoints2d" / camera_name / f"{frame_name}_keypoints.json").open() as handle:
            payload = json.load(handle)
        if payload["people"]:
            body25 = np.array(payload["people"][0]["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
        else:
            body25 = np.zeros((25, 3), dtype=np.float64)
        keypoints.append(body25)
    keypoints = np.stack(keypoints, axis=0)
    Pall = cameras.projection_matrices(cameras.names)
    ours = batch_triangulate(keypoints, Pall, min_view=2)
    v1 = v1_batch_triangulate(keypoints, Pall, min_view=2)
    assert np.allclose(ours, v1, atol=1e-12)
