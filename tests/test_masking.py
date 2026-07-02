import numpy as np
import pytest

from lightmocap.detection.masking import apply_image_mask, filter_annotation_by_mask, is_mask_usable, mask_to_foreground


def test_mask_to_foreground_accepts_gray_and_rgb_masks():
    gray = np.array([[0, 1], [127, 255]], dtype=np.uint8)
    assert mask_to_foreground(gray).tolist() == [[False, True], [True, True]]
    assert mask_to_foreground(gray, threshold=127).tolist() == [[False, False], [False, True]]

    rgb = np.dstack([np.zeros_like(gray), gray, np.zeros_like(gray)])
    assert mask_to_foreground(rgb, threshold=127).tolist() == [[False, False], [False, True]]


def test_apply_image_mask_zeros_background_and_preserves_foreground():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    masked = apply_image_mask(image, mask)

    assert masked[0, 0].tolist() == [0, 0, 0]
    assert masked[0, 1].tolist() == image[0, 1].tolist()
    assert masked[1, 0].tolist() == image[1, 0].tolist()
    assert masked[1, 1].tolist() == [0, 0, 0]


def test_apply_image_mask_rejects_shape_mismatch():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((3, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not match image shape"):
        apply_image_mask(image, mask)


def test_is_mask_usable_rejects_tiny_foreground_ratio():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 255

    assert is_mask_usable(mask, min_area_ratio=0.005)
    assert not is_mask_usable(mask, min_area_ratio=0.02)


def test_filter_annotation_by_mask_zeros_outside_points_and_recomputes_bbox():
    annotation = {
        "filename": "000001.jpg",
        "height": 20,
        "width": 20,
        "annots": [
            {
                "personID": 3,
                "bbox": [0.0, 0.0, 20.0, 20.0, 1.0],
                "keypoints": [
                    [2.0, 2.0, 0.9],
                    [10.0, 2.0, 0.8],
                    [2.0, 10.0, 0.7],
                    [18.0, 18.0, 0.6],
                ],
                "bbox_handl2d": [0.0, 0.0, 20.0, 20.0, 1.0],
                "handl2d": [[2.0, 2.0, 0.5], [18.0, 18.0, 0.5], [10.0, 2.0, 0.5]],
                "bbox_handr2d": [0.0, 0.0, 20.0, 20.0, 1.0],
                "handr2d": [[2.0, 10.0, 0.5], [18.0, 18.0, 0.5], [10.0, 2.0, 0.5]],
                "bbox_face2d": [0.0, 0.0, 20.0, 20.0, 1.0],
                "face2d": [[2.0, 2.0, 0.5], [18.0, 18.0, 0.5], [10.0, 2.0, 0.5]],
                "isKeyframe": False,
            }
        ],
        "isKeyframe": False,
    }
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2, 2] = 255
    mask[2, 10] = 255
    mask[10, 2] = 255

    filtered = filter_annotation_by_mask(annotation, mask)
    person = filtered["annots"][0]
    body = np.asarray(person["keypoints"], dtype=np.float32)

    assert person["personID"] == 0
    assert np.allclose(body[:, 2], [0.9, 0.8, 0.7, 0.0])
    assert person["bbox"][4] > 0.0
    assert person["bbox"][2] < 12.0
    assert np.asarray(person["handl2d"], dtype=np.float32)[1, 2] == 0.0


def test_filter_annotation_by_mask_drops_people_without_enough_body_points():
    annotation = {
        "filename": "000001.jpg",
        "height": 5,
        "width": 5,
        "annots": [
            {
                "personID": 0,
                "bbox": [0.0, 0.0, 5.0, 5.0, 1.0],
                "keypoints": [[4.0, 4.0, 0.9], [4.0, 3.0, 0.8], [3.0, 4.0, 0.7]],
                "bbox_handl2d": [0.0, 0.0, 5.0, 5.0, 1.0],
                "handl2d": [[4.0, 4.0, 0.5]],
                "bbox_handr2d": [0.0, 0.0, 5.0, 5.0, 1.0],
                "handr2d": [[4.0, 4.0, 0.5]],
                "bbox_face2d": [0.0, 0.0, 5.0, 5.0, 1.0],
                "face2d": [[4.0, 4.0, 0.5]],
                "isKeyframe": False,
            }
        ],
        "isKeyframe": False,
    }
    mask = np.zeros((5, 5), dtype=np.uint8)

    filtered = filter_annotation_by_mask(annotation, mask)

    assert filtered["annots"] == []
