import numpy as np

from lightmocap.core.fitting.pipeline import _sanitize_keypoints_for_smplx
from lightmocap.data.camera import CameraSet, natural_sort_key
from lightmocap.data.dataset import MultiViewImageSequence


def test_natural_sort_key_orders_camera_names_human_readably():
    values = ["Camera_B1", "Camera_B10", "Camera_B2", "Camera_B11", "Camera_B3"]
    assert sorted(values, key=natural_sort_key) == ["Camera_B1", "Camera_B2", "Camera_B3", "Camera_B10", "Camera_B11"]


def test_coreview_camera_names_are_naturally_sorted():
    cameras = CameraSet.from_yaml(
        "/home/ymj/code/python/EasyMocap/CoreView_377/intri.yml",
        "/home/ymj/code/python/EasyMocap/CoreView_377/extri.yml",
    )
    assert cameras.names[:5] == ["Camera_B1", "Camera_B2", "Camera_B3", "Camera_B4", "Camera_B5"]
    assert cameras.names[-3:] == ["Camera_B21", "Camera_B22", "Camera_B23"]


def test_coreview_dataset_uses_natural_camera_order():
    sequence = MultiViewImageSequence("/home/ymj/code/python/EasyMocap/CoreView_377")
    assert sequence.camera_names[:5] == ["Camera_B1", "Camera_B2", "Camera_B3", "Camera_B4", "Camera_B5"]


def test_multiview_sequence_discovers_png_frames(tmp_path):
    for camera_name in ["Camera_B1", "Camera_B2"]:
        camera_dir = tmp_path / camera_name
        camera_dir.mkdir()
        (camera_dir / "000001.png").touch()
        (camera_dir / "000002.png").touch()

    sequence = MultiViewImageSequence(tmp_path)

    assert sequence.frame_names == ["000001", "000002"]
    assert sequence.frame_paths(0)["Camera_B1"] == tmp_path / "Camera_B1" / "000001.png"


def test_sanitize_keypoints_aligns_hand_roots_to_body_wrists():
    keypoints = np.zeros((1, 25 + 21 + 21 + 51, 4), dtype=np.float32)
    keypoints[0, 7] = [1.0, 2.0, 3.0, 0.9]
    keypoints[0, 4] = [4.0, 5.0, 6.0, 0.8]
    keypoints[0, 25] = [10.0, 20.0, 30.0, 0.1]
    keypoints[0, 46] = [40.0, 50.0, 60.0, 0.2]
    sanitized = _sanitize_keypoints_for_smplx(keypoints, unsupported_indices=[])
    assert np.allclose(sanitized[0, 25], keypoints[0, 7])
    assert np.allclose(sanitized[0, 46], keypoints[0, 4])
