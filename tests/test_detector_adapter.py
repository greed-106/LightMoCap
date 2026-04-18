import numpy as np

from lightmocap.data.io import read_json
from lightmocap.detection.adapter import (
    body18_to_body25,
    body24_to_body25,
    openpose_people_to_annotation,
    rtmlib_wholebody_to_annotation,
)


def test_openpose_json_converts_to_canonical_annotation():
    payload = read_json(
        "/home/ymj/code/python/EasyMocap/CoreView_377/keypoints2d/Camera_B11/000003_keypoints.json"
    )
    annotation = openpose_people_to_annotation(payload, "Camera_B11/000003.jpg")
    assert annotation["annots"]
    person = annotation["annots"][0]
    assert len(person["keypoints"]) == 25
    assert len(person["handl2d"]) == 21
    assert len(person["handr2d"]) == 21
    assert len(person["face2d"]) == 51


def test_body18_can_be_promoted_to_body25():
    body18 = np.zeros((18, 3), dtype=np.float32)
    body18[0] = [10.0, 20.0, 1.0]
    body18[1] = [11.0, 21.0, 1.0]
    body18[8] = [30.0, 40.0, 0.8]
    body18[11] = [50.0, 60.0, 0.9]
    body25 = body18_to_body25(body18)
    assert body25.shape == (25, 3)
    assert np.allclose(body25[0], body18[0])
    assert np.allclose(body25[1], body18[1])
    assert np.allclose(body25[8, :2], [40.0, 50.0])
    assert np.isclose(body25[8, 2], 0.8)


def test_body24_can_be_promoted_to_body25_with_feet():
    body24 = np.zeros((24, 3), dtype=np.float32)
    body24[8] = [30.0, 40.0, 0.8]   # RHip in body18
    body24[11] = [50.0, 60.0, 0.9]  # LHip in body18
    body24[18] = [1.0, 2.0, 0.9]    # LBigToe
    body24[19] = [3.0, 4.0, 0.8]    # LSmallToe
    body24[20] = [5.0, 6.0, 0.7]    # LHeel
    body24[21] = [7.0, 8.0, 0.6]    # RBigToe
    body24[22] = [9.0, 10.0, 0.5]   # RSmallToe
    body24[23] = [11.0, 12.0, 0.4]  # RHeel
    body25 = body24_to_body25(body24)
    assert np.allclose(body25[8, :2], [40.0, 50.0])
    assert np.isclose(body25[8, 2], 0.8)
    assert np.allclose(body25[19], body24[18])
    assert np.allclose(body25[20], body24[19])
    assert np.allclose(body25[21], body24[20])
    assert np.allclose(body25[22], body24[21])
    assert np.allclose(body25[23], body24[22])
    assert np.allclose(body25[24], body24[23])


def test_rtmlib_wholebody_body_slice_uses_24_points_before_face():
    keypoints = np.zeros((1, 134, 2), dtype=np.float32)
    scores = np.zeros((1, 134), dtype=np.float32)
    keypoints[0, 23] = [123.0, 456.0]
    scores[0, 23] = 0.77
    keypoints[0, 24] = [999.0, 888.0]
    scores[0, 24] = 0.66
    annotation = rtmlib_wholebody_to_annotation(keypoints, scores, "frame.jpg")
    body25 = np.asarray(annotation["annots"][0]["keypoints"], dtype=np.float32)
    face51 = np.asarray(annotation["annots"][0]["face2d"], dtype=np.float32)
    assert np.allclose(body25[24], [123.0, 456.0, 0.77])
    assert np.allclose(face51[0], [0.0, 0.0, 0.0])
def test_rtmlib_wholebody_converts_to_canonical_annotation():
    keypoints = np.zeros((1, 134, 2), dtype=np.float32)
    scores = np.zeros((1, 134), dtype=np.float32)
    keypoints[0, 0] = [10.0, 20.0]
    scores[0, 0] = 1.0
    keypoints[0, 18] = [1.0, 2.0]
    scores[0, 18] = 0.9
    keypoints[0, 92] = [50.0, 60.0]
    scores[0, 92] = 0.7
    keypoints[0, 113] = [70.0, 80.0]
    scores[0, 113] = 0.6
    keypoints[0, 24 + 17] = [90.0, 100.0]
    scores[0, 24 + 17] = 0.8
    annotation = rtmlib_wholebody_to_annotation(keypoints, scores, 'frame.jpg', width=640, height=480)
    person = annotation['annots'][0]
    assert len(person['keypoints']) == 25
    assert len(person['handl2d']) == 21
    assert len(person['handr2d']) == 21
    assert len(person['face2d']) == 51
