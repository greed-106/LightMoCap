import json
from pathlib import Path

import numpy as np

from lightmocap.core.camera.undistort import Undistort
from lightmocap.core.fitting import FittingConfig, SMPLXFittingPipeline
from lightmocap.core.triangulation import project_points, triangulate_multiview_points
from lightmocap.data.camera import CameraSet
from lightmocap.detection.adapter import canonical_keypoints_from_annotation, openpose_people_to_annotation
from lightmocap.detection.rtmlib import RTMLibDetector
from lightmocap.models.smplx import SMPLXLayer


ROOT = Path("/home/ymj/code/python/EasyMocap/CoreView_377")
MODEL_ROOT = "/home/ymj/code/python/EasyMocap/models/smplx"
FRAME_NAME = "000003"


def _load_openpose_annotations(frame_name: str = FRAME_NAME) -> tuple[CameraSet, dict[str, dict]]:
    cameras = CameraSet.from_yaml(ROOT / "intri.yml", ROOT / "extri.yml")
    annotations: dict[str, dict] = {}
    for camera_name in cameras.names:
        with (ROOT / "keypoints2d" / camera_name / f"{frame_name}_keypoints.json").open() as handle:
            payload = json.load(handle)
        annotations[camera_name] = openpose_people_to_annotation(payload, f"{camera_name}/{frame_name}.jpg")
    return cameras, annotations


def test_coreview_annots_npy_matches_camera_ordering_contract():
    cameras = CameraSet.from_yaml(ROOT / "intri.yml", ROOT / "extri.yml")
    annots = np.load(ROOT / "annots.npy", allow_pickle=True).item()
    assert len(annots["cams"]["K"]) == len(cameras.names) == 23
    assert len(annots["ims"]) == 617
    first_frame_paths = annots["ims"][0]["ims"]
    first_frame_cameras = [Path(path).parent.name for path in first_frame_paths]
    assert first_frame_cameras == cameras.names
    assert np.array(annots["ims"][0]["kpts2d"]).shape == (23, 25, 3)


def test_coreview_openpose_priors_are_body_only_for_this_sequence():
    cameras, annotations = _load_openpose_annotations()
    for camera_name in cameras.names[:3]:
        person = annotations[camera_name]["annots"][0]
        assert np.asarray(person["handl2d"]).shape == (21, 3)
        assert np.asarray(person["handr2d"]).shape == (21, 3)
        assert np.asarray(person["face2d"]).shape == (51, 3)
        assert float(np.asarray(person["handl2d"])[:, 2].sum()) == 0.0
        assert float(np.asarray(person["handr2d"])[:, 2].sum()) == 0.0
        assert float(np.asarray(person["face2d"])[:, 2].sum()) == 0.0


def test_real_coreview_body_fit_regression_from_v1_openpose():
    cameras, annotations = _load_openpose_annotations()
    camera_names = cameras.names
    keypoints2d = np.stack(
        [canonical_keypoints_from_annotation(annotations[camera_name], mode="body25") for camera_name in camera_names],
        axis=0,
    )
    keypoints3d = triangulate_multiview_points(keypoints2d, cameras, camera_names=camera_names)
    model = SMPLXLayer(MODEL_ROOT, gender="neutral", device="cpu")
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=15, shape_maxiters=5))
    fitted_params = fitter.fit(keypoints3d[None])
    fitted_keypoints = model(return_verts=False, return_tensor=False, **fitted_params)
    valid = np.array(model.body25_supported_indices)
    error = np.linalg.norm(fitted_keypoints[:, valid] - keypoints3d[None, valid, :3], axis=-1)
    assert float(error.mean()) < 0.05


def test_real_coreview_bodyhandface_fit_regression_from_rtmlib_wholebody():
    cameras = CameraSet.from_yaml(ROOT / "intri.yml", ROOT / "extri.yml")
    camera_names = cameras.names[:4]
    detector = RTMLibDetector(mode="lightweight", backend="onnxruntime", device="cpu", solution="wholebody")
    annotations: dict[str, dict] = {}
    for camera_name in camera_names:
        image_path = ROOT / camera_name / f"{FRAME_NAME}.jpg"
        import cv2

        image = cv2.imread(str(image_path))
        annotations[camera_name] = detector.detect_annotation(image, image_path.name)

    keypoints2d = np.stack(
        [canonical_keypoints_from_annotation(annotations[camera_name], mode="bodyhandface") for camera_name in camera_names],
        axis=0,
    )
    keypoints3d = triangulate_multiview_points(keypoints2d, cameras, camera_names=camera_names)
    model = SMPLXLayer(MODEL_ROOT, gender="neutral", device="cpu")
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=15, shape_maxiters=5))
    fitted_params = fitter.fit(keypoints3d[None])
    fitted_keypoints = model(return_verts=False, return_tensor=False, keypoint_mode="bodyhandface", **fitted_params)

    body_idx = np.array(model.body25_supported_indices)
    hand_idx = np.arange(25, 25 + 42)
    face_idx = np.arange(25 + 42, 25 + 42 + 51)

    body_err = np.linalg.norm(fitted_keypoints[:, body_idx] - keypoints3d[None, body_idx, :3], axis=-1)
    hand_err = np.linalg.norm(fitted_keypoints[:, hand_idx] - keypoints3d[None, hand_idx, :3], axis=-1)
    face_err = np.linalg.norm(fitted_keypoints[:, face_idx] - keypoints3d[None, face_idx, :3], axis=-1)

    assert float(body_err.mean()) < 0.05
    assert float(hand_err.mean()) < 0.05
    assert float(face_err.mean()) < 0.02


def test_real_coreview_bodyhandface_reprojection_regression_from_rtmlib_wholebody():
    cameras = CameraSet.from_yaml(ROOT / "intri.yml", ROOT / "extri.yml")
    camera_names = cameras.names[:4]
    detector = RTMLibDetector(mode="lightweight", backend="onnxruntime", device="cpu", solution="wholebody")
    annotations: dict[str, dict] = {}
    for camera_name in camera_names:
        image_path = ROOT / camera_name / f"{FRAME_NAME}.jpg"
        import cv2

        image = cv2.imread(str(image_path))
        annotations[camera_name] = detector.detect_annotation(image, image_path.name)

    keypoints2d = np.stack(
        [canonical_keypoints_from_annotation(annotations[camera_name], mode="bodyhandface") for camera_name in camera_names],
        axis=0,
    )
    keypoints3d = triangulate_multiview_points(keypoints2d, cameras, camera_names=camera_names)
    model = SMPLXLayer(MODEL_ROOT, gender="neutral", device="cpu")
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=15, shape_maxiters=5))
    fitted_params = fitter.fit(keypoints3d[None])
    fitted_keypoints = model(return_verts=False, return_tensor=False, keypoint_mode="bodyhandface", **fitted_params)[0]

    undistorted_2d = []
    for view_idx, camera_name in enumerate(camera_names):
        camera = cameras[camera_name]
        undistorted_2d.append(Undistort.points(keypoints2d[view_idx], camera.K, camera.dist))
    undistorted_2d = np.stack(undistorted_2d, axis=0)

    fitted_points4 = np.concatenate([fitted_keypoints, np.ones((fitted_keypoints.shape[0], 1), dtype=np.float32)], axis=-1)
    projected = project_points(fitted_points4, cameras.projection_matrices(camera_names))

    valid = undistorted_2d[..., 2] > 0.2
    body_valid = valid[:, model.body25_supported_indices]
    hand_valid = valid[:, 25 : 25 + 42]
    face_valid = valid[:, 25 + 42 :]
    body_err = np.linalg.norm(projected[:, model.body25_supported_indices, :2] - undistorted_2d[:, model.body25_supported_indices, :2], axis=-1)[body_valid]
    hand_err = np.linalg.norm(projected[:, 25 : 25 + 42, :2] - undistorted_2d[:, 25 : 25 + 42, :2], axis=-1)[hand_valid]
    face_err = np.linalg.norm(projected[:, 25 + 42 :, :2] - undistorted_2d[:, 25 + 42 :, :2], axis=-1)[face_valid]

    assert float(body_err.mean()) < 20.0
    assert float(hand_err.mean()) < 10.0
    assert float(face_err.mean()) < 5.0


def test_reprojection_refinement_does_not_worsen_real_body_reprojection():
    cameras, annotations = _load_openpose_annotations()
    camera_names = cameras.names
    keypoints2d = np.stack(
        [canonical_keypoints_from_annotation(annotations[camera_name], mode="body25") for camera_name in camera_names],
        axis=0,
    )
    keypoints3d = triangulate_multiview_points(keypoints2d, cameras, camera_names=camera_names)
    bboxes = np.stack([np.asarray(annotations[camera_name]["annots"][0]["bbox"], dtype=np.float32) for camera_name in camera_names], axis=0)
    model = SMPLXLayer(MODEL_ROOT, gender="neutral", device="cpu")
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=15, shape_maxiters=5))

    params_3d_only = fitter.fit(keypoints3d[None])
    params_with_2d = fitter.fit(
        keypoints3d[None],
        keypoints2d=keypoints2d[None],
        bboxes=bboxes[None],
        projection_matrices=cameras.projection_matrices(camera_names),
    )

    fitted_3d_only = model(return_verts=False, return_tensor=False, keypoint_mode="body25", **params_3d_only)[0]
    fitted_with_2d = model(return_verts=False, return_tensor=False, keypoint_mode="body25", **params_with_2d)[0]
    undistorted = np.stack([Undistort.points(keypoints2d[i], cameras[camera_name].K, cameras[camera_name].dist) for i, camera_name in enumerate(camera_names)], axis=0)
    proj_3d_only = project_points(
        np.concatenate([fitted_3d_only, np.ones((fitted_3d_only.shape[0], 1), dtype=np.float32)], axis=-1),
        cameras.projection_matrices(camera_names),
    )
    proj_with_2d = project_points(
        np.concatenate([fitted_with_2d, np.ones((fitted_with_2d.shape[0], 1), dtype=np.float32)], axis=-1),
        cameras.projection_matrices(camera_names),
    )
    valid = undistorted[..., 2] > 0.2
    supported = np.zeros((25,), dtype=bool)
    supported[np.array(model.body25_supported_indices)] = True
    valid[:, ~supported] = False
    err_3d_only = np.linalg.norm(proj_3d_only[:, :, :2] - undistorted[:, :, :2], axis=-1)[valid]
    err_with_2d = np.linalg.norm(proj_with_2d[:, :, :2] - undistorted[:, :, :2], axis=-1)[valid]
    assert float(err_with_2d.mean()) <= float(err_3d_only.mean()) * 1.05
