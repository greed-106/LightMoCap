from __future__ import annotations

import numpy as np
import torch

from lightmocap.data.skeleton import BODY25_SIZE, FACE_SIZE, HAND21_SIZE
from lightmocap.models.lbs import batch_rodrigues


funcl2 = lambda x: torch.sum(x**2)


def gmof(squared_residual: torch.Tensor, sigma_squared: float) -> torch.Tensor:
    return (sigma_squared * squared_residual) / (sigma_squared + squared_residual)


class LossKeypoints3D:
    def __init__(self, keypoints3d: np.ndarray, device: torch.device) -> None:
        keypoints3d = torch.tensor(keypoints3d, dtype=torch.float32, device=device)
        self.keypoints3d = keypoints3d[..., :3]
        self.conf = keypoints3d[..., 3:]
        self.n_frames = keypoints3d.shape[0]

    def body(self, kpts_est: torch.Tensor, **kwargs) -> torch.Tensor:
        n_joints = min(kpts_est.shape[1], self.keypoints3d.shape[1], BODY25_SIZE)
        diff = (kpts_est[:, :n_joints, :3] - self.keypoints3d[:, :n_joints, :3]) * self.conf[:, :n_joints]
        return funcl2(diff) / self.n_frames

    def hand(self, kpts_est: torch.Tensor, **kwargs) -> torch.Tensor:
        start = BODY25_SIZE
        end = min(kpts_est.shape[1], self.keypoints3d.shape[1], BODY25_SIZE + HAND21_SIZE * 2)
        if end <= start:
            return torch.zeros((), dtype=kpts_est.dtype, device=kpts_est.device)
        diff = (kpts_est[:, start:end, :3] - self.keypoints3d[:, start:end, :3]) * self.conf[:, start:end]
        return funcl2(diff) / self.n_frames

    def face(self, kpts_est: torch.Tensor, **kwargs) -> torch.Tensor:
        start = BODY25_SIZE + HAND21_SIZE * 2
        end = min(kpts_est.shape[1], self.keypoints3d.shape[1], start + FACE_SIZE)
        if end <= start:
            return torch.zeros((), dtype=kpts_est.dtype, device=kpts_est.device)
        diff = (kpts_est[:, start:end, :3] - self.keypoints3d[:, start:end, :3]) * self.conf[:, start:end]
        return funcl2(diff) / self.n_frames


class LossKeypointsMV2D:
    def __init__(
        self,
        keypoints2d: np.ndarray,
        bboxes: np.ndarray,
        projection_matrices: np.ndarray,
        device: torch.device,
    ) -> None:
        # keypoints2d: (n_frames, n_views, n_joints, 3)
        # bboxes: (n_frames, n_views, 5)
        self.device = device
        keypoints2d = np.asarray(keypoints2d, dtype=np.float32)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        projection_matrices = np.asarray(projection_matrices, dtype=np.float32)
        if keypoints2d.ndim != 4:
            raise ValueError(f"Expected keypoints2d with shape (F, V, J, 3), got {keypoints2d.shape}")
        if bboxes.ndim != 3:
            raise ValueError(f"Expected bboxes with shape (F, V, 5), got {bboxes.shape}")
        if projection_matrices.ndim != 3:
            raise ValueError(f"Expected projection matrices with shape (V, 3, 4), got {projection_matrices.shape}")
        keypoints2d = keypoints2d.transpose(1, 0, 2, 3)
        bboxes = bboxes.transpose(1, 0, 2)
        bbox_sizes = np.maximum(bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1])
        bbox_conf = bboxes[..., 4]
        bbox_sizes = (bbox_sizes * bbox_conf).sum(axis=-1) / (1e-3 + bbox_conf.sum(axis=-1))
        bbox_sizes = bbox_sizes[..., None, None, None]
        bbox_sizes[bbox_sizes < 10] = 1e6
        inv_bbox_sizes = torch.tensor(1.0 / bbox_sizes, dtype=torch.float32, device=device)
        keypoints2d_t = torch.tensor(keypoints2d, dtype=torch.float32, device=device)
        self.keypoints2d = keypoints2d_t[..., :2]
        self.conf = keypoints2d_t[..., 2:] * inv_bbox_sizes * 100.0
        self.Pall = torch.tensor(projection_matrices, dtype=torch.float32, device=device)
        self.n_views, self.n_frames, self.n_joints = keypoints2d.shape[:3]
        self.kpt_homo = torch.ones((self.n_frames, self.n_joints, 1), dtype=torch.float32, device=device)

    def __call__(self, kpts_est: torch.Tensor, **kwargs) -> torch.Tensor:
        kpts_homo = torch.cat([kpts_est[..., : self.n_joints, :], self.kpt_homo], dim=2)
        points_cam = torch.einsum("vab,fnb->vfna", self.Pall, kpts_homo)
        img_points = points_cam[..., :2] / points_cam[..., 2:].clamp(min=1e-6)
        residual = (img_points - self.keypoints2d) * self.conf
        squared = gmof(residual**2, 200.0)
        return torch.sum(squared) / self.n_views / self.n_frames


class LossSmoothBodyMean:
    @staticmethod
    def _smooth(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] <= 2:
            return torch.zeros((), dtype=values.dtype, device=values.device)
        interp = values.clone().detach()
        interp[1:-1] = (interp[:-2] + interp[2:]) / 2.0
        return funcl2(values[1:-1] - interp[1:-1]) / (values.shape[0] - 2)

    def body(self, kpts_est: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._smooth(kpts_est[:, :25])

    def hand(self, kpts_est: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._smooth(kpts_est[:, 25 : 25 + 42])


class LossSmoothPoses:
    @staticmethod
    def _smooth(poses: torch.Tensor) -> torch.Tensor:
        if poses.shape[0] <= 2:
            return torch.zeros((), dtype=poses.dtype, device=poses.device)
        interp = poses.clone().detach()
        interp[1:-1] = (interp[1:-1] + interp[:-2] + interp[2:]) / 3.0
        return funcl2(poses[1:-1] - interp[1:-1]) / (poses.shape[0] - 2)

    def poses(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._smooth(poses[:, :66])

    def hands(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._smooth(poses[:, 66 : 66 + 12])

    def head(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._smooth(poses[:, 78:])


class LossSmoothRh:
    def __call__(self, Rh: torch.Tensor, **kwargs) -> torch.Tensor:
        if Rh.shape[0] <= 2:
            return torch.zeros((), dtype=Rh.dtype, device=Rh.device)
        rot = batch_rodrigues(Rh)
        interp = rot.clone().detach()
        interp[1:-1] = (interp[1:-1] + interp[:-2] + interp[2:]) / 3.0
        return funcl2(rot[1:-1] - interp[1:-1]) / (Rh.shape[0] - 2)


class LossRegPoses:
    def reg_body(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return funcl2(poses[:, :66]) / poses.shape[0]

    def reg_hand(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return funcl2(poses[:, 66:78]) / poses.shape[0]

    def reg_head(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return funcl2(poses[:, 78:]) / poses.shape[0]

    def reg_expr(self, expression: torch.Tensor, **kwargs) -> torch.Tensor:
        return funcl2(expression) / expression.shape[0]


class LossRegPosesZero:
    def __init__(self, keypoints2d: np.ndarray, model_type: str = "smplx") -> None:
        if keypoints2d.shape[-2] <= 15:
            use_feet = False
            use_head = False
        else:
            use_feet = keypoints2d[..., [19, 20, 21, 22, 23, 24], 2].sum() > 0.1
            use_head = keypoints2d[..., [15, 16, 17, 18], 2].sum() > 0.1
        if model_type == "smpl":
            joint_zero_idx = [3, 6, 9, 10, 11, 13, 14, 20, 21, 22, 23]
        elif model_type in ["smplh", "smplx"]:
            joint_zero_idx = [3, 6, 9, 10, 11, 13, 14]
        else:
            raise NotImplementedError(model_type)
        if not use_feet:
            joint_zero_idx.extend([7, 8])
        if not use_head:
            joint_zero_idx.extend([12, 15])
        pose_zero_idx = [[idx for idx in range(3 * joint, 3 * joint + 3)] for joint in joint_zero_idx]
        self.idx = sum(pose_zero_idx, [])

    def __call__(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.sum(torch.abs(poses[:, self.idx])) / poses.shape[0]


class LossRegShapes:
    def __call__(self, shapes: torch.Tensor, **kwargs) -> torch.Tensor:
        return funcl2(shapes) / shapes.shape[0]


class LossInit:
    def __init__(self, params: dict[str, np.ndarray], device: torch.device) -> None:
        self.poses = torch.tensor(params["poses"], dtype=torch.float32, device=device)
        self.shapes = torch.tensor(params["shapes"], dtype=torch.float32, device=device)

    def init_poses(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        return funcl2(poses - self.poses) / poses.shape[0]

    def init_shapes(self, shapes: torch.Tensor, **kwargs) -> torch.Tensor:
        return funcl2(shapes - self.shapes) / shapes.shape[0]
