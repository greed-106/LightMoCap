from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lightmocap.data.skeleton import BODY25_KINTREE, BODY25_SIZE, HAND21_SIZE
from lightmocap.models.smplx import SMPLXLayer

from .losses import (
    LossInit,
    LossKeypoints3D,
    LossKeypointsMV2D,
    LossRegPoses,
    LossRegPosesZero,
    LossRegShapes,
    LossSmoothBodyMean,
    LossSmoothPoses,
    LossSmoothRh,
)
from .optimizer import FittingMonitor, LBFGS, grad_require


@dataclass
class FittingConfig:
    device: str = "cpu"
    maxiters: int = 20
    shape_maxiters: int = 10
    enable_k2d_refine: bool = True
    weight_loss: dict[str, float] = field(
        default_factory=lambda: {
            "k3d": 1.0,
            "k3d_hand": 5.0,
            "k3d_face": 2.0,
            "reg_poses": 1e-3,
            "reg_poses_zero": 1e-2,
            "reg_hand": 1e-4,
            "reg_head": 1e-2,
            "reg_expr": 1e-2,
            "reg_shapes": 5e-3,
            "k2d": 1e-4,
            "smooth_body": 5e-1,
            "smooth_poses": 1e-1,
            "smooth_Rh": 1e-1,
            "smooth_hand": 1e-3,
            "smooth_head": 0.0,
            "init_poses": 1e-2,
            "init_shapes": 1e-2,
            "s3d": 1.0,
        }
    )


def _sanitize_keypoints_for_smplx(keypoints3d: np.ndarray, unsupported_indices: list[int]) -> np.ndarray:
    keypoints3d = keypoints3d.copy()
    if keypoints3d.shape[-1] == 3:
        conf = np.ones((*keypoints3d.shape[:-1], 1), dtype=keypoints3d.dtype)
        keypoints3d = np.concatenate([keypoints3d, conf], axis=-1)
    keypoints3d[..., unsupported_indices, 3] = 0.0
    if keypoints3d.shape[-2] > BODY25_SIZE:
        # Match V1 behavior: hand roots are anchored to the body wrists.
        left_root = BODY25_SIZE
        right_root = BODY25_SIZE + HAND21_SIZE
        keypoints3d[..., left_root, :] = keypoints3d[..., 7, :]
        keypoints3d[..., right_root, :] = keypoints3d[..., 4, :]
    return keypoints3d


def _sanitize_keypoints2d_for_smplx(keypoints2d: np.ndarray, unsupported_indices: list[int]) -> np.ndarray:
    keypoints2d = keypoints2d.copy()
    if keypoints2d.shape[-1] != 3:
        raise ValueError(f"Expected keypoints2d with shape (..., 3), got {keypoints2d.shape}")
    keypoints2d[..., unsupported_indices, 2] = 0.0
    if keypoints2d.shape[-2] > BODY25_SIZE:
        left_root = BODY25_SIZE
        right_root = BODY25_SIZE + HAND21_SIZE
        keypoints2d[..., left_root, :] = keypoints2d[..., 7, :]
        keypoints2d[..., right_root, :] = keypoints2d[..., 4, :]
    return keypoints2d


class SMPLXFittingPipeline:
    def __init__(self, body_model: SMPLXLayer, config: FittingConfig | None = None) -> None:
        self.body_model = body_model
        self.config = config or FittingConfig(device=str(body_model.device))

    def fit_shape(self, body_params: dict[str, np.ndarray], keypoints3d: np.ndarray) -> dict[str, np.ndarray]:
        """Estimate betas from coarse limb-length consistency.

        This stage intentionally mirrors the V1 shape-first strategy before any
        pose-heavy optimization is attempted.
        """
        device = self.body_model.device
        kintree = np.array(BODY25_KINTREE, dtype=int)
        limb_length = np.linalg.norm(keypoints3d[:, kintree[:, 1], :3] - keypoints3d[:, kintree[:, 0], :3], axis=2, keepdims=True)
        limb_conf = np.minimum(keypoints3d[:, kintree[:, 1], 3:], keypoints3d[:, kintree[:, 0], 3:])
        limb_length = torch.tensor(limb_length, dtype=torch.float32, device=device)
        limb_conf = torch.tensor(limb_conf, dtype=torch.float32, device=device)
        params = {key: torch.tensor(val, dtype=torch.float32, device=device) for key, val in body_params.items()}
        params_init = {key: val.clone() for key, val in params.items()}
        opt_params = [params["shapes"]]
        grad_require(opt_params, True)
        optimizer = LBFGS(opt_params, line_search_fn="strong_wolfe", max_iter=self.config.shape_maxiters)

        def closure(debug: bool = False):
            optimizer.zero_grad()
            kpts_est = self.body_model(return_verts=False, return_tensor=True, only_shape=True, keypoint_mode="body25", **params)
            src = kpts_est[:, kintree[:, 0], :3]
            dst = kpts_est[:, kintree[:, 1], :3]
            direct_est = (dst - src).detach()
            direct_norm = torch.norm(direct_est, dim=2, keepdim=True)
            direct_normalized = direct_est / (direct_norm + 1e-4)
            err = dst - src - direct_normalized * limb_length
            loss_dict = {
                "s3d": torch.sum(err**2 * limb_conf) / keypoints3d.shape[0],
                "reg_shapes": torch.sum(params["shapes"]**2),
                "init_shapes": torch.sum((params["shapes"] - params_init["shapes"]) ** 2),
            }
            loss = sum(loss_dict[name] * self.config.weight_loss.get(name, 0.0) for name in loss_dict)
            if debug:
                return loss_dict
            loss.backward()
            return loss

        FittingMonitor(ftol=1e-4, maxiters=self.config.shape_maxiters).run_fitting(optimizer, closure, opt_params)
        return {key: val.detach().cpu().numpy() for key, val in params.items()}

    def fit_pose3d(self, body_params: dict[str, np.ndarray], keypoints3d: np.ndarray) -> dict[str, np.ndarray]:
        """Fit SMPL-X to triangulated 3D keypoints."""
        device = self.body_model.device
        params = {key: torch.tensor(val, dtype=torch.float32, device=device) for key, val in body_params.items()}
        keypoints3d = _sanitize_keypoints_for_smplx(keypoints3d, self.body_model.unsupported_body25_indices())
        keypoint_mode = "bodyhandface" if keypoints3d.shape[1] > 25 else "body25"
        loss_k3d = LossKeypoints3D(keypoints3d, device)
        loss_reg_pose = LossRegPoses()
        loss_reg_shapes = LossRegShapes()
        loss_init = LossInit(body_params, device)
        loss_smooth_body = LossSmoothBodyMean()
        loss_smooth_pose = LossSmoothPoses()
        loss_smooth_rh = LossSmoothRh()
        opt_params = [params["Rh"], params["Th"], params["poses"]]
        if keypoints3d.shape[1] > 25:
            opt_params.append(params["expression"])
        grad_require(opt_params, True)
        optimizer = LBFGS(opt_params, line_search_fn="strong_wolfe", max_iter=self.config.maxiters)

        def closure(debug: bool = False):
            optimizer.zero_grad()
            kpts_est = self.body_model(return_verts=False, return_tensor=True, keypoint_mode=keypoint_mode, **params)
            loss_dict = {
                "k3d": loss_k3d.body(kpts_est=kpts_est, **params),
                "smooth_body": loss_smooth_body.body(kpts_est=kpts_est, **params),
                "smooth_poses": loss_smooth_pose.poses(poses=params["poses"]),
                "smooth_Rh": loss_smooth_rh(Rh=params["Rh"]),
                "reg_poses": loss_reg_pose.reg_body(poses=params["poses"]),
                "reg_shapes": loss_reg_shapes(shapes=params["shapes"]),
                "init_poses": loss_init.init_poses(poses=params["poses"]),
                "init_shapes": loss_init.init_shapes(shapes=params["shapes"]),
            }
            if keypoint_mode == "bodyhandface":
                loss_dict["k3d_hand"] = loss_k3d.hand(kpts_est=kpts_est, **params)
                loss_dict["k3d_face"] = loss_k3d.face(kpts_est=kpts_est, **params)
                loss_dict["smooth_hand"] = loss_smooth_body.hand(kpts_est=kpts_est, **params)
                loss_dict["smooth_head"] = loss_smooth_pose.head(poses=params["poses"])
                loss_dict["reg_hand"] = loss_reg_pose.reg_hand(poses=params["poses"])
                loss_dict["reg_head"] = loss_reg_pose.reg_head(poses=params["poses"])
                loss_dict["reg_expr"] = loss_reg_pose.reg_expr(expression=params["expression"])
            loss = sum(loss_dict[name] * self.config.weight_loss.get(name, 0.0) for name in loss_dict)
            if debug:
                return loss_dict
            loss.backward()
            return loss

        FittingMonitor(ftol=1e-4, maxiters=self.config.maxiters).run_fitting(optimizer, closure, opt_params)
        return {key: val.detach().cpu().numpy() for key, val in params.items()}

    def fit_pose2d(
        self,
        body_params: dict[str, np.ndarray],
        keypoints2d: np.ndarray,
        bboxes: np.ndarray,
        projection_matrices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Refine a 3D fit with V1-style multiview reprojection constraints."""
        device = self.body_model.device
        params = {key: torch.tensor(val, dtype=torch.float32, device=device) for key, val in body_params.items()}
        keypoints2d = _sanitize_keypoints2d_for_smplx(keypoints2d, self.body_model.unsupported_body25_indices())
        keypoint_mode = "bodyhandface" if keypoints2d.shape[2] > 25 else "body25"
        loss_k2d = LossKeypointsMV2D(keypoints2d, bboxes, projection_matrices, device)
        loss_reg_pose = LossRegPoses()
        loss_reg_zero = LossRegPosesZero(keypoints2d, model_type="smplx")
        loss_reg_shapes = LossRegShapes()
        loss_init = LossInit(body_params, device)
        loss_smooth_body = LossSmoothBodyMean()
        loss_smooth_pose = LossSmoothPoses()
        loss_smooth_rh = LossSmoothRh()
        opt_params = [params["Rh"], params["Th"], params["poses"]]
        if keypoint_mode == "bodyhandface":
            opt_params.append(params["expression"])
        grad_require(opt_params, True)
        optimizer = LBFGS(opt_params, line_search_fn="strong_wolfe", max_iter=self.config.maxiters)

        def closure(debug: bool = False):
            optimizer.zero_grad()
            kpts_est = self.body_model(return_verts=False, return_tensor=True, keypoint_mode=keypoint_mode, **params)
            loss_dict = {
                "k2d": loss_k2d(kpts_est=kpts_est, **params),
                "smooth_body": loss_smooth_body.body(kpts_est=kpts_est, **params),
                "smooth_poses": loss_smooth_pose.poses(poses=params["poses"]),
                "smooth_Rh": loss_smooth_rh(Rh=params["Rh"]),
                "reg_poses": loss_reg_pose.reg_body(poses=params["poses"]),
                "reg_poses_zero": loss_reg_zero(poses=params["poses"]),
                "reg_shapes": loss_reg_shapes(shapes=params["shapes"]),
                "init_poses": loss_init.init_poses(poses=params["poses"]),
                "init_shapes": loss_init.init_shapes(shapes=params["shapes"]),
            }
            if keypoint_mode == "bodyhandface":
                loss_dict["smooth_hand"] = loss_smooth_body.hand(kpts_est=kpts_est, **params)
                loss_dict["smooth_head"] = loss_smooth_pose.head(poses=params["poses"])
                loss_dict["reg_hand"] = loss_reg_pose.reg_hand(poses=params["poses"])
                loss_dict["reg_head"] = loss_reg_pose.reg_head(poses=params["poses"])
                loss_dict["reg_expr"] = loss_reg_pose.reg_expr(expression=params["expression"])
            loss = sum(loss_dict[name] * self.config.weight_loss.get(name, 0.0) for name in loss_dict)
            if debug:
                return loss_dict
            loss.backward()
            return loss

        FittingMonitor(ftol=1e-4, maxiters=self.config.maxiters).run_fitting(optimizer, closure, opt_params)
        return {key: val.detach().cpu().numpy() for key, val in params.items()}

    def fit(
        self,
        keypoints3d: np.ndarray,
        body_params: dict[str, np.ndarray] | None = None,
        keypoints2d: np.ndarray | None = None,
        bboxes: np.ndarray | None = None,
        projection_matrices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the full fitting schedule.

        The sequence is shape -> pose3d -> optional pose2d refinement.
        """
        if body_params is None:
            body_params = self.body_model.init_params(n_frames=keypoints3d.shape[0], n_shapes=1, ret_tensor=False)
        keypoints3d = _sanitize_keypoints_for_smplx(keypoints3d, self.body_model.unsupported_body25_indices())
        body_params = self.fit_shape(body_params, keypoints3d[:, :25])
        body_params = self.fit_pose3d(body_params, keypoints3d)
        if self.config.enable_k2d_refine and keypoints2d is not None and bboxes is not None and projection_matrices is not None:
            body_params = self.fit_pose2d(body_params, keypoints2d, bboxes, projection_matrices)
        return body_params
