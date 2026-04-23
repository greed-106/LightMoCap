import numpy as np

from lightmocap.core.fitting import FittingConfig, SMPLXFittingPipeline
from lightmocap.core.fitting.losses import LossJointPrior
from lightmocap.models.smplx import SMPLXLayer


MODEL_ROOT = "/home/ymj/code/python/EasyMocap/models/smplx"


def _with_confidence(keypoints3d: np.ndarray) -> np.ndarray:
    conf = np.ones((*keypoints3d.shape[:-1], 1), dtype=np.float32)
    return np.concatenate([keypoints3d.astype(np.float32), conf], axis=-1)


def test_torso_stage_recovers_global_alignment_from_torso_initialization():
    model = SMPLXLayer(MODEL_ROOT, gender="neutral", device="cpu")
    fitter = SMPLXFittingPipeline(model, FittingConfig(device="cpu", maxiters=10, shape_maxiters=2))

    gt_params = model.init_params(n_frames=1, n_shapes=1, ret_tensor=False)
    gt_params["shapes"][0, :3] = np.array([0.3, -0.1, 0.2], dtype=np.float32)
    gt_params["Rh"][0] = np.array([0.25, -0.15, 0.35], dtype=np.float32)
    gt_params["Th"][0] = np.array([0.3, 1.0, 2.0], dtype=np.float32)

    keypoints3d = model(return_verts=False, return_tensor=False, keypoint_mode="body25", **gt_params)
    keypoints3d = _with_confidence(keypoints3d)

    init_params = model.init_params(n_frames=1, n_shapes=1, ret_tensor=False)
    init_params["shapes"][:] = gt_params["shapes"]
    fitted = fitter.fit_global_torso(init_params, keypoints3d, maxiters=10, initialize=True)

    assert np.linalg.norm(fitted["Rh"][0] - gt_params["Rh"][0]) < 0.1
    assert np.linalg.norm(fitted["Th"][0] - gt_params["Th"][0]) < 0.05


def test_joint_prior_penalizes_anatomically_implausible_pose():
    loss_joint_prior = LossJointPrior()
    reasonable = np.zeros((1, 87), dtype=np.float32)
    impossible = np.zeros((1, 87), dtype=np.float32)

    # Reasonable flexion.
    reasonable[0, 3 + 3 * 3 + 0] = 1.0
    reasonable[0, 3 + 3 * 4 + 0] = 1.0
    reasonable[0, 3 + 3 * 17 + 1] = -1.0
    reasonable[0, 3 + 3 * 18 + 1] = 1.0

    # Hyperextension / wrong bending directions.
    impossible[0, 3 + 3 * 3 + 0] = -1.0
    impossible[0, 3 + 3 * 4 + 0] = -1.2
    impossible[0, 3 + 3 * 17 + 1] = 1.0
    impossible[0, 3 + 3 * 18 + 1] = -1.0

    reasonable_loss = float(loss_joint_prior(poses=np.asarray(reasonable)).item())
    impossible_loss = float(loss_joint_prior(poses=np.asarray(impossible)).item())
    assert impossible_loss > reasonable_loss + 0.5


def test_lbfgs_inner_iterations_are_capped_per_stage():
    model = SMPLXLayer(MODEL_ROOT, gender="neutral", device="cpu")
    fitter = SMPLXFittingPipeline(
        model,
        FittingConfig(device="cpu", maxiters=15, shape_maxiters=5, lbfgs_inner_max_iter=8),
    )

    assert fitter._lbfgs_max_iter(30) == 8
    assert fitter._lbfgs_max_iter(15) == 8
    assert fitter._lbfgs_max_iter(5) == 5
