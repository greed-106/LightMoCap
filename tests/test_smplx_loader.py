import numpy as np

from lightmocap.core.fitting import FittingConfig, SMPLXFittingPipeline
from lightmocap.models.io import load_smplx_npz
from lightmocap.models.smplx import SMPLXLayer


def test_smplx_npz_can_be_loaded():
    data = load_smplx_npz("/home/ymj/code/python/EasyMocap/models/smplx", "neutral")
    assert "v_template" in data
    assert "weights" in data
    assert "kintree_table" in data


def test_smplx_layer_forward_returns_body25_joints():
    model = SMPLXLayer("/home/ymj/code/python/EasyMocap/models/smplx", gender="neutral", device="cpu")
    params = model.init_params(n_frames=1, n_shapes=1, ret_tensor=False)
    joints = model(return_verts=False, return_tensor=False, **params)
    assert joints.shape == (1, 25, 3)


def test_smplx_layer_can_return_bodyhandface_keypoints():
    model = SMPLXLayer("/home/ymj/code/python/EasyMocap/models/smplx", gender="neutral", device="cpu")
    params = model.init_params(n_frames=1, n_shapes=1, ret_tensor=False)
    keypoints = model(return_verts=False, return_tensor=False, keypoint_mode="bodyhandface", **params)
    assert keypoints.shape == (1, 25 + 21 + 21 + 51, 3)
    # fingertip should come from mesh vertices instead of duplicating distal joints.
    assert not np.allclose(keypoints[0, 25 + 4], keypoints[0, 25 + 3])
    assert not np.allclose(keypoints[0, 25 + 8], keypoints[0, 25 + 7])


def test_smplx_fitting_pipeline_recovers_simple_pose_target():
    model = SMPLXLayer("/home/ymj/code/python/EasyMocap/models/smplx", gender="neutral", device="cpu")
    target = model.init_params(n_frames=1, n_shapes=1, ret_tensor=False)
    target["Th"][0] = np.array([0.05, -0.03, 0.2], dtype=np.float32)
    target["Rh"][0] = np.array([0.02, -0.01, 0.03], dtype=np.float32)
    keypoints3d = model(return_verts=False, return_tensor=False, **target)
    keypoints3d = np.concatenate([keypoints3d, np.ones((1, 25, 1), dtype=np.float32)], axis=-1)
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=15, shape_maxiters=5))
    fitted = fitter.fit(keypoints3d)
    fitted_kpts = model(return_verts=False, return_tensor=False, **fitted)
    assert np.allclose(fitted_kpts[:, model.body25_supported_indices], keypoints3d[:, model.body25_supported_indices, :3], atol=1e-2)


def test_smplx_fitting_pipeline_supports_bodyhandface_targets():
    model = SMPLXLayer("/home/ymj/code/python/EasyMocap/models/smplx", gender="neutral", device="cpu")
    target = model.init_params(n_frames=1, n_shapes=1, ret_tensor=False)
    target["poses"][0, 66:78] = np.array([0.1, -0.05, 0.03, 0.08, -0.02, 0.01, -0.1, 0.07, 0.04, -0.03, 0.02, 0.05], dtype=np.float32)
    target["poses"][0, 78:] = np.array([0.02, -0.01, 0.03, 0.01, 0.0, -0.01, -0.01, 0.02, 0.01], dtype=np.float32)
    target["expression"][0] = np.linspace(-0.2, 0.2, 10, dtype=np.float32)
    keypoints = model(return_verts=False, return_tensor=False, keypoint_mode="bodyhandface", **target)
    keypoints = np.concatenate([keypoints, np.ones((1, keypoints.shape[1], 1), dtype=np.float32)], axis=-1)
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=15, shape_maxiters=5))
    fitted = fitter.fit(keypoints)
    fitted_keypoints = model(return_verts=False, return_tensor=False, keypoint_mode="bodyhandface", **fitted)
    assert np.allclose(fitted_keypoints, keypoints[..., :3], atol=5e-2)
