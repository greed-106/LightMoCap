import json
from pathlib import Path

import numpy as np

from lightmocap.core.fitting import FittingConfig, SMPLXFittingPipeline
from lightmocap.core.triangulation import triangulate_multiview_points
from lightmocap.data.camera import CameraSet
from lightmocap.detection.adapter import openpose_people_to_annotation
from lightmocap.models.smplx import SMPLXLayer


def test_real_coreview_frame_can_be_triangulated_and_fitted():
    root = Path("/home/ymj/code/python/EasyMocap/CoreView_377")
    cameras = CameraSet.from_yaml(root / "intri.yml", root / "extri.yml")
    frame_name = "000003"
    keypoints2d = []
    for camera_name in cameras.names:
        with (root / "keypoints2d" / camera_name / f"{frame_name}_keypoints.json").open() as handle:
            payload = json.load(handle)
        annotation = openpose_people_to_annotation(payload, f"{camera_name}/{frame_name}.jpg")
        keypoints2d.append(np.asarray(annotation["annots"][0]["keypoints"], dtype=np.float64))
    keypoints2d = np.stack(keypoints2d, axis=0)
    keypoints3d = triangulate_multiview_points(keypoints2d, cameras)
    model = SMPLXLayer("/home/ymj/code/python/EasyMocap/models/smplx", gender="neutral", device="cpu")
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=15, shape_maxiters=5))
    fitted_params = fitter.fit(keypoints3d[None])
    fitted_keypoints = model(return_verts=False, return_tensor=False, **fitted_params)
    valid = np.array(model.body25_supported_indices)
    error = np.linalg.norm(fitted_keypoints[:, valid] - keypoints3d[None, valid, :3], axis=-1)
    assert float(error.mean()) < 0.05
