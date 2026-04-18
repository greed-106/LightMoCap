import json
from pathlib import Path

import cv2
import numpy as np

from lightmocap.detection.rtmlib import RTMLibDetector


def test_rtmlib_wholebody_feet_grouping_is_compatible_with_openpose_body25():
    image_path = Path("/home/ymj/code/python/EasyMocap/CoreView_377/Camera_B11/000003.jpg")
    annot_path = Path("/home/ymj/code/python/EasyMocap/CoreView_377/keypoints2d/Camera_B11/000003_keypoints.json")
    image = cv2.imread(str(image_path))
    detector = RTMLibDetector(mode="lightweight", backend="onnxruntime", device="cpu", solution="wholebody")
    annotation = detector.detect_annotation(image, image_path.name)
    with annot_path.open() as handle:
        openpose = json.load(handle)
    openpose_body25 = np.array(openpose["people"][0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    rtmlib_body25 = np.array(annotation["annots"][0]["keypoints"], dtype=np.float32)
    # Check grouped feet order: left foot occupies body25[19:22], right foot occupies body25[22:25].
    assert openpose_body25[19:22, 2].sum() > 0
    assert openpose_body25[22:25, 2].sum() > 0
    assert rtmlib_body25[19:22, 2].sum() > 0
    assert rtmlib_body25[22:25, 2].sum() > 0
    # The left-foot group from rtmlib should be closer to the left-foot group from OpenPose than to the right-foot group.
    left_direct = np.linalg.norm(rtmlib_body25[19:22, :2] - openpose_body25[19:22, :2], axis=-1).mean()
    left_swapped = np.linalg.norm(rtmlib_body25[19:22, :2] - openpose_body25[22:25, :2], axis=-1).mean()
    right_direct = np.linalg.norm(rtmlib_body25[22:25, :2] - openpose_body25[22:25, :2], axis=-1).mean()
    right_swapped = np.linalg.norm(rtmlib_body25[22:25, :2] - openpose_body25[19:22, :2], axis=-1).mean()
    assert left_direct < left_swapped
    assert right_direct < right_swapped
