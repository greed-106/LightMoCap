from pathlib import Path

import numpy as np

from lightmocap.data.camera import CameraSet


def test_coreview_camera_yaml_can_be_read():
    cameras = CameraSet.from_yaml(
        "/home/ymj/code/python/EasyMocap/CoreView_377/intri.yml",
        "/home/ymj/code/python/EasyMocap/CoreView_377/extri.yml",
    )
    assert len(cameras) == 23
    assert cameras.names[0].startswith("Camera_B")
    assert cameras[cameras.names[0]].K.shape == (3, 3)


def test_compact_camera_yaml_can_be_read():
    cameras = CameraSet.from_yaml("/home/ymj/code/python/EasyMocap/CoreView_377/cameras.yaml")
    assert len(cameras) == 23
    assert cameras.names[:3] == ["Camera_B1", "Camera_B2", "Camera_B3"]
    assert cameras[cameras.names[0]].R.shape == (3, 3)
    assert cameras[cameras.names[0]].dist.shape == (1, 5)


def test_camera_yaml_roundtrip_matches_legacy_projection(tmp_path: Path):
    legacy = CameraSet.from_yaml(
        "/home/ymj/code/python/EasyMocap/CoreView_377/intri.yml",
        "/home/ymj/code/python/EasyMocap/CoreView_377/extri.yml",
    )
    roundtrip_path = legacy.to_yaml(tmp_path / "cameras.yaml")
    compact = CameraSet.from_yaml(roundtrip_path)
    for name in legacy.names:
        assert np.allclose(compact[name].P, legacy[name].P, atol=1e-12)
