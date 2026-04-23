from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from lightmocap.data.skeleton import BODY25_KINTREE, BODY25_SIZE, HAND21_SIZE
from lightmocap.models.smplx import SMPLXLayer

from .losses import (
    LossInit,
    LossJointPrior,
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


TORSO_BODY25_INDICES = np.array([1, 2, 5, 8, 9, 12], dtype=int)


def _default_stages() -> list[dict[str, object]]:
    return [
        {"name": "shape"},
        {"name": "torso", "maxiters": 30, "shoulder_weight": 5.0, "hip_weight": 5.0, "initialize": True},
        {"name": "pose", "maxiters": 20},
    ]


@dataclass
class FittingConfig:
    device: str = "cpu"
    maxiters: int = 20
    shape_maxiters: int = 10
    lbfgs_inner_max_iter: int = 8
    enable_k2d_refine: bool = True
    k3d_robust_sigma: float = 0.05
    stages: list[dict[str, object]] = field(default_factory=_default_stages)
    weight_loss: dict[str, float] = field(
        default_factory=lambda: {
            "k3d": 1.0,
            "k3d_hand": 5.0,
            "k3d_face": 2.0,
            "joint_prior": 0.1,
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


def _to_numpy_params(body_params: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    for key, value in body_params.items():
        if isinstance(value, torch.Tensor):
            params[key] = value.detach().cpu().numpy()
        else:
            params[key] = np.asarray(value, dtype=np.float32)
    return params


def _safe_normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def _torso_basis(points: np.ndarray, conf: np.ndarray) -> np.ndarray | None:
    pelvis = points[8]
    neck = points[1]
    if conf[8] <= 0 or conf[1] <= 0:
        return None
    x_candidates = []
    if conf[2] > 0 and conf[5] > 0:
        x_candidates.append(points[2] - points[5])
    if conf[9] > 0 and conf[12] > 0:
        x_candidates.append(points[9] - points[12])
    if not x_candidates:
        return None
    x_axis = _safe_normalize(np.mean(np.stack(x_candidates, axis=0), axis=0))
    y_axis = _safe_normalize(neck - pelvis)
    if x_axis is None or y_axis is None:
        return None
    z_axis = _safe_normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _weighted_center(points: np.ndarray, conf: np.ndarray) -> np.ndarray | None:
    valid = conf > 0
    if valid.sum() < 3:
        return None
    weights = conf[valid]
    weights = weights / max(float(weights.sum()), 1e-6)
    return np.sum(points[valid] * weights[:, None], axis=0)


def _estimate_torso_transform(source: np.ndarray, target: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    source_basis = _torso_basis(source, conf)
    target_basis = _torso_basis(target, conf)
    source_center = _weighted_center(source[TORSO_BODY25_INDICES], conf[TORSO_BODY25_INDICES])
    target_center = _weighted_center(target[TORSO_BODY25_INDICES], conf[TORSO_BODY25_INDICES])
    if source_basis is None or target_basis is None or source_center is None or target_center is None:
        return None
    rotation = target_basis @ source_basis.T
    if np.linalg.det(rotation) < 0:
        target_basis[:, 2] *= -1.0
        rotation = target_basis @ source_basis.T
    translation = target_center - source_center @ rotation.T
    return rotation.astype(np.float32), translation.astype(np.float32)


def _torso_joint_weights(device: torch.device, shoulder_weight: float, hip_weight: float) -> torch.Tensor:
    weights = torch.zeros((BODY25_SIZE,), dtype=torch.float32, device=device)
    weights[1] = 1.0
    weights[8] = 1.0
    weights[2] = shoulder_weight
    weights[5] = shoulder_weight
    weights[9] = hip_weight
    weights[12] = hip_weight
    return weights


class SMPLXFittingPipeline:
    def __init__(self, body_model: SMPLXLayer, config: FittingConfig | None = None) -> None:
        self.body_model = body_model
        self.config = config or FittingConfig(device=str(body_model.device))

    def _lbfgs_max_iter(self, maxiters: int) -> int:
        # The outer FittingMonitor already drives convergence checks. Cap the
        # inner LBFGS iterations so stage maxiters are not multiplied twice.
        return max(1, min(int(maxiters), int(self.config.lbfgs_inner_max_iter)))

    def _initialize_global_from_torso(
        self,
        body_params: dict[str, np.ndarray | torch.Tensor],
        keypoints3d: np.ndarray,
    ) -> dict[str, np.ndarray]:
        params = _to_numpy_params(body_params)
        zero_Rh = np.zeros_like(params["Rh"], dtype=np.float32)
        zero_Th = np.zeros_like(params["Th"], dtype=np.float32)
        model_body = self.body_model(
            poses=params["poses"],
            shapes=params["shapes"],
            Rh=zero_Rh,
            Th=zero_Th,
            expression=params["expression"],
            return_verts=False,
            return_tensor=False,
            keypoint_mode="body25",
        )
        for frame_index in range(keypoints3d.shape[0]):
            conf = keypoints3d[frame_index, :, 3]
            transform = _estimate_torso_transform(
                model_body[frame_index],
                keypoints3d[frame_index, :, :3],
                conf,
            )
            if transform is None:
                continue
            rotation, translation = transform
            params["Rh"][frame_index] = cv2.Rodrigues(rotation)[0][:, 0]
            params["Th"][frame_index] = translation
        return params

    def _fit_shape_tensor(self, body_params: dict[str, np.ndarray | torch.Tensor], keypoints3d: np.ndarray) -> dict[str, torch.Tensor]:
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
        params = self.body_model.check_params(body_params)
        params_init = {key: val.clone() for key, val in params.items()}
        opt_params = [params["shapes"]]
        grad_require(opt_params, True)
        optimizer = LBFGS(
            opt_params,
            line_search_fn="strong_wolfe",
            max_iter=self._lbfgs_max_iter(self.config.shape_maxiters),
        )

        def closure(debug: bool = False):
            optimizer.zero_grad()
            kpts_est = self.body_model(
                return_verts=False,
                return_tensor=True,
                only_shape=True,
                keypoint_mode="body25",
                validated=True,
                **params,
            )
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
        return params

    def fit_shape(self, body_params: dict[str, np.ndarray | torch.Tensor], keypoints3d: np.ndarray) -> dict[str, np.ndarray]:
        return _to_numpy_params(self._fit_shape_tensor(body_params, keypoints3d))

    def _fit_global_torso_tensor(
        self,
        body_params: dict[str, np.ndarray | torch.Tensor],
        keypoints3d: np.ndarray,
        maxiters: int | None = None,
        shoulder_weight: float = 5.0,
        hip_weight: float = 5.0,
        initialize: bool = True,
    ) -> dict[str, torch.Tensor]:
        device = self.body_model.device
        keypoints3d = _sanitize_keypoints_for_smplx(keypoints3d, self.body_model.unsupported_body25_indices())
        if initialize:
            body_params = self._initialize_global_from_torso(body_params, keypoints3d[:, :25])
        params = self.body_model.check_params(body_params)
        loss_k3d = LossKeypoints3D(
            keypoints3d[:, :25],
            device,
            robust=True,
            sigma_squared=self.config.k3d_robust_sigma**2,
        )
        loss_smooth_rh = LossSmoothRh()
        joint_weights = _torso_joint_weights(device, shoulder_weight=shoulder_weight, hip_weight=hip_weight)
        opt_params = [params["Rh"], params["Th"]]
        grad_require(opt_params, True)
        stage_maxiters = maxiters or self.config.maxiters
        optimizer = LBFGS(
            opt_params,
            line_search_fn="strong_wolfe",
            max_iter=self._lbfgs_max_iter(stage_maxiters),
        )

        def closure(debug: bool = False):
            optimizer.zero_grad()
            kpts_est = self.body_model(
                return_verts=False,
                return_tensor=True,
                keypoint_mode="body25",
                validated=True,
                **params,
            )
            loss_dict = {
                "k3d": loss_k3d.body(kpts_est=kpts_est, joint_weights=joint_weights),
                "smooth_Rh": loss_smooth_rh(Rh=params["Rh"]),
            }
            loss = sum(loss_dict[name] * self.config.weight_loss.get(name, 0.0) for name in loss_dict)
            if debug:
                return loss_dict
            loss.backward()
            return loss

        FittingMonitor(ftol=1e-4, maxiters=stage_maxiters).run_fitting(optimizer, closure, opt_params)
        return params

    def fit_global_torso(
        self,
        body_params: dict[str, np.ndarray | torch.Tensor],
        keypoints3d: np.ndarray,
        maxiters: int | None = None,
        shoulder_weight: float = 5.0,
        hip_weight: float = 5.0,
        initialize: bool = True,
    ) -> dict[str, np.ndarray]:
        return _to_numpy_params(
            self._fit_global_torso_tensor(
                body_params,
                keypoints3d,
                maxiters=maxiters,
                shoulder_weight=shoulder_weight,
                hip_weight=hip_weight,
                initialize=initialize,
            )
        )

    def _fit_pose3d_tensor(
        self,
        body_params: dict[str, np.ndarray | torch.Tensor],
        keypoints3d: np.ndarray,
        maxiters: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fit SMPL-X to triangulated 3D keypoints."""
        device = self.body_model.device
        params = self.body_model.check_params(body_params)
        keypoints3d = _sanitize_keypoints_for_smplx(keypoints3d, self.body_model.unsupported_body25_indices())
        keypoint_mode = "bodyhandface" if keypoints3d.shape[1] > 25 else "body25"
        loss_k3d = LossKeypoints3D(
            keypoints3d,
            device,
            robust=False,
            sigma_squared=self.config.k3d_robust_sigma**2,
        )
        loss_joint_prior = LossJointPrior()
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
        stage_maxiters = maxiters or self.config.maxiters
        optimizer = LBFGS(
            opt_params,
            line_search_fn="strong_wolfe",
            max_iter=self._lbfgs_max_iter(stage_maxiters),
        )

        def closure(debug: bool = False):
            optimizer.zero_grad()
            kpts_est = self.body_model(
                return_verts=False,
                return_tensor=True,
                keypoint_mode=keypoint_mode,
                validated=True,
                **params,
            )
            loss_dict = {
                "k3d": loss_k3d.body(kpts_est=kpts_est, **params),
                "joint_prior": loss_joint_prior(poses=params["poses"]),
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

        FittingMonitor(ftol=1e-4, maxiters=stage_maxiters).run_fitting(optimizer, closure, opt_params)
        return params

    def fit_pose3d(
        self,
        body_params: dict[str, np.ndarray | torch.Tensor],
        keypoints3d: np.ndarray,
        maxiters: int | None = None,
    ) -> dict[str, np.ndarray]:
        return _to_numpy_params(self._fit_pose3d_tensor(body_params, keypoints3d, maxiters=maxiters))

    def _fit_pose2d_tensor(
        self,
        body_params: dict[str, np.ndarray | torch.Tensor],
        keypoints2d: np.ndarray,
        bboxes: np.ndarray,
        projection_matrices: np.ndarray,
        maxiters: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Refine a 3D fit with V1-style multiview reprojection constraints."""
        device = self.body_model.device
        params = self.body_model.check_params(body_params)
        keypoints2d = _sanitize_keypoints2d_for_smplx(keypoints2d, self.body_model.unsupported_body25_indices())
        keypoint_mode = "bodyhandface" if keypoints2d.shape[2] > 25 else "body25"
        loss_k2d = LossKeypointsMV2D(keypoints2d, bboxes, projection_matrices, device)
        loss_joint_prior = LossJointPrior()
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
        stage_maxiters = maxiters or self.config.maxiters
        optimizer = LBFGS(
            opt_params,
            line_search_fn="strong_wolfe",
            max_iter=self._lbfgs_max_iter(stage_maxiters),
        )

        def closure(debug: bool = False):
            optimizer.zero_grad()
            kpts_est = self.body_model(
                return_verts=False,
                return_tensor=True,
                keypoint_mode=keypoint_mode,
                validated=True,
                **params,
            )
            loss_dict = {
                "k2d": loss_k2d(kpts_est=kpts_est, **params),
                "joint_prior": loss_joint_prior(poses=params["poses"]),
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

        FittingMonitor(ftol=1e-4, maxiters=stage_maxiters).run_fitting(optimizer, closure, opt_params)
        return params

    def fit_pose2d(
        self,
        body_params: dict[str, np.ndarray | torch.Tensor],
        keypoints2d: np.ndarray,
        bboxes: np.ndarray,
        projection_matrices: np.ndarray,
        maxiters: int | None = None,
    ) -> dict[str, np.ndarray]:
        return _to_numpy_params(
            self._fit_pose2d_tensor(
                body_params,
                keypoints2d,
                bboxes,
                projection_matrices,
                maxiters=maxiters,
            )
        )

    def fit_tensor(
        self,
        keypoints3d: np.ndarray,
        body_params: dict[str, np.ndarray | torch.Tensor] | None = None,
        keypoints2d: np.ndarray | None = None,
        bboxes: np.ndarray | None = None,
        projection_matrices: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Internal fast path for end-to-end single-frame inference.

        This keeps optimization variables on-device across fitting stages and
        only materializes numpy arrays at explicit I/O boundaries.
        """
        if body_params is None:
            body_params = self.body_model.init_params(n_frames=keypoints3d.shape[0], n_shapes=1, ret_tensor=False)
        keypoints3d = _sanitize_keypoints_for_smplx(keypoints3d, self.body_model.unsupported_body25_indices())
        body_params_t = self.body_model.check_params(body_params)
        stages = self.config.stages or _default_stages()
        for stage in stages:
            stage_name = str(stage.get("name", "")).lower()
            if stage_name == "shape":
                body_params_t = self._fit_shape_tensor(body_params_t, keypoints3d[:, :25])
            elif stage_name == "torso":
                body_params_t = self._fit_global_torso_tensor(
                    body_params_t,
                    keypoints3d,
                    maxiters=int(stage.get("maxiters", self.config.maxiters)),
                    shoulder_weight=float(stage.get("shoulder_weight", 5.0)),
                    hip_weight=float(stage.get("hip_weight", 5.0)),
                    initialize=bool(stage.get("initialize", True)),
                )
            elif stage_name in {"pose", "pose3d"}:
                body_params_t = self._fit_pose3d_tensor(
                    body_params_t,
                    keypoints3d,
                    maxiters=int(stage.get("maxiters", self.config.maxiters)),
                )
            elif stage_name in {"pose2d", "k2d", "refine2d"}:
                # Pose2d is still gated by the presence of multiview 2D inputs below.
                continue
            else:
                raise ValueError(f"Unsupported fitting stage: {stage_name}")
        if self.config.enable_k2d_refine and keypoints2d is not None and bboxes is not None and projection_matrices is not None:
            pose2d_stage = next((stage for stage in stages if str(stage.get("name", "")).lower() in {"pose2d", "k2d", "refine2d"}), None)
            body_params_t = self._fit_pose2d_tensor(
                body_params_t,
                keypoints2d,
                bboxes,
                projection_matrices,
                maxiters=int(pose2d_stage.get("maxiters", self.config.maxiters)) if pose2d_stage is not None else None,
            )
        return body_params_t

    def fit(
        self,
        keypoints3d: np.ndarray,
        body_params: dict[str, np.ndarray] | None = None,
        keypoints2d: np.ndarray | None = None,
        bboxes: np.ndarray | None = None,
        projection_matrices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the full fitting schedule.

        The public API returns numpy arrays for CLI and file I/O boundaries.
        """
        return _to_numpy_params(
            self.fit_tensor(
                keypoints3d,
                body_params=body_params,
                keypoints2d=keypoints2d,
                bboxes=bboxes,
                projection_matrices=projection_matrices,
            )
        )
