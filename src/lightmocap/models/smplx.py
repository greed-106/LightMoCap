from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from lightmocap.data.skeleton import BODY25_SIZE, SMPLX_BODY25_SUPPORTED

from .io import load_smplx_npz
from .lbs import batch_rodrigues, lbs, vertices2landmarks


def to_tensor(array: Any, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    tensor = array if isinstance(array, torch.Tensor) else torch.tensor(array, dtype=dtype)
    return tensor if device is None else tensor.to(device)


def to_np(array: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


BODY25_NAME_TO_SMPLX = {
    1: "Neck",
    2: "R_Shoulder",
    3: "R_Elbow",
    4: "R_Wrist",
    5: "L_Shoulder",
    6: "L_Elbow",
    7: "L_Wrist",
    8: "Pelvis",
    9: "R_Hip",
    10: "R_Knee",
    11: "R_Ankle",
    12: "L_Hip",
    13: "L_Knee",
    14: "L_Ankle",
    15: "R_Eye",
    16: "L_Eye",
    19: "L_Foot",
    20: "L_Foot",
    22: "R_Foot",
    23: "R_Foot",
}

LEFT_HAND_ORDER = [
    "L_Wrist",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
]

RIGHT_HAND_ORDER = [
    "R_Wrist",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
]

SMPLX_FINGERTIP_VERTICES = {
    "left": {
        "thumb": 5361,
        "index": 4933,
        "middle": 5058,
        "ring": 5169,
        "pinky": 5286,
    },
    "right": {
        "thumb": 8079,
        "index": 7669,
        "middle": 7794,
        "ring": 7905,
        "pinky": 8022,
    },
}

HAND21_LAYOUT = {
    "left": [
        ("joint", "L_Wrist"),
        ("joint", "L_Thumb1"),
        ("joint", "L_Thumb2"),
        ("joint", "L_Thumb3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["left"]["thumb"]),
        ("joint", "L_Index1"),
        ("joint", "L_Index2"),
        ("joint", "L_Index3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["left"]["index"]),
        ("joint", "L_Middle1"),
        ("joint", "L_Middle2"),
        ("joint", "L_Middle3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["left"]["middle"]),
        ("joint", "L_Ring1"),
        ("joint", "L_Ring2"),
        ("joint", "L_Ring3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["left"]["ring"]),
        ("joint", "L_Pinky1"),
        ("joint", "L_Pinky2"),
        ("joint", "L_Pinky3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["left"]["pinky"]),
    ],
    "right": [
        ("joint", "R_Wrist"),
        ("joint", "R_Thumb1"),
        ("joint", "R_Thumb2"),
        ("joint", "R_Thumb3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["right"]["thumb"]),
        ("joint", "R_Index1"),
        ("joint", "R_Index2"),
        ("joint", "R_Index3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["right"]["index"]),
        ("joint", "R_Middle1"),
        ("joint", "R_Middle2"),
        ("joint", "R_Middle3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["right"]["middle"]),
        ("joint", "R_Ring1"),
        ("joint", "R_Ring2"),
        ("joint", "R_Ring3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["right"]["ring"]),
        ("joint", "R_Pinky1"),
        ("joint", "R_Pinky2"),
        ("joint", "R_Pinky3"),
        ("vertex", SMPLX_FINGERTIP_VERTICES["right"]["pinky"]),
    ],
}


@dataclass(frozen=True)
class SMPLXModelSpec:
    gender: str
    model_root: str


class SMPLXLayer(nn.Module):
    NUM_BODY_POSES = 66
    NUM_HAND_PCA = 6
    NUM_FACE_POSES = 9

    def __init__(
        self,
        model_path: str,
        gender: str = "neutral",
        device: str | torch.device = "cpu",
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
        use_pose_blending: bool = True,
        use_shape_blending: bool = True,
    ) -> None:
        super().__init__()
        self.model_type = "smplx"
        self.dtype = torch.float32
        self.use_pose_blending = use_pose_blending
        self.use_shape_blending = use_shape_blending
        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs
        self.num_pca_comps = self.NUM_HAND_PCA
        self.use_pca = True
        self.use_flat_mean = True
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        self.device = device
        self.data = load_smplx_npz(model_path, gender)
        self._register_buffers_from_data(self.data)
        self.body25_supported_indices = SMPLX_BODY25_SUPPORTED
        self.to(self.device)

    @property
    def compact_pose_dim(self) -> int:
        return self.NUM_BODY_POSES + self.num_pca_comps * 2 + self.NUM_FACE_POSES

    @property
    def full_pose_dim(self) -> int:
        return 165

    def _register_buffers_from_data(self, data: dict[str, np.ndarray]) -> None:
        self.faces = to_np(data["f"], dtype=np.int64)
        self.register_buffer("faces_tensor", to_tensor(self.faces, dtype=torch.long, device=self.device))
        for key in ["J_regressor", "v_template", "weights"]:
            self.register_buffer(key, to_tensor(to_np(data[key]), dtype=self.dtype, device=self.device))
        num_pose_basis = data["posedirs"].shape[-1]
        posedirs = np.reshape(data["posedirs"], [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype, device=self.device))
        shapedirs = to_tensor(to_np(data["shapedirs"]), dtype=self.dtype, device=self.device)
        shapedirs = shapedirs[:, :, : self.num_betas + self.num_expression_coeffs]
        self.register_buffer("shapedirs", shapedirs)
        parents = to_tensor(to_np(data["kintree_table"][0]), dtype=torch.long, device=self.device)
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("mHandsMeanL", to_tensor(to_np(data["hands_meanl"]).reshape(1, -1), dtype=self.dtype, device=self.device))
        self.register_buffer("mHandsMeanR", to_tensor(to_np(data["hands_meanr"]).reshape(1, -1), dtype=self.dtype, device=self.device))
        self.register_buffer(
            "mHandsComponentsL",
            to_tensor(to_np(data["hands_componentsl"][: self.num_pca_comps]), dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "mHandsComponentsR",
            to_tensor(to_np(data["hands_componentsr"][: self.num_pca_comps]), dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "lmk_faces_idx",
            to_tensor(to_np(data["lmk_faces_idx"], dtype=np.int64), dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "lmk_bary_coords",
            to_tensor(to_np(data["lmk_bary_coords"]), dtype=self.dtype, device=self.device),
        )
        joint2num = data.get("joint2num")
        if isinstance(joint2num, np.ndarray) and joint2num.shape == ():
            joint2num = joint2num.item()
        self.joint2num = joint2num or {}

    def init_params(self, n_frames: int = 1, n_shapes: int = 1, ret_tensor: bool = False) -> dict[str, np.ndarray | torch.Tensor]:
        params: dict[str, np.ndarray | torch.Tensor] = {
            "poses": np.zeros((n_frames, self.compact_pose_dim), dtype=np.float32),
            "shapes": np.zeros((n_shapes, self.num_betas), dtype=np.float32),
            "Rh": np.zeros((n_frames, 3), dtype=np.float32),
            "Th": np.zeros((n_frames, 3), dtype=np.float32),
            "expression": np.zeros((n_frames, self.num_expression_coeffs), dtype=np.float32),
        }
        if ret_tensor:
            params = {key: to_tensor(val, dtype=self.dtype, device=self.device) for key, val in params.items()}
        return params

    def check_params(self, body_params: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor]:
        body_params = body_params.copy()
        n_frames = body_params["poses"].shape[0]
        for key in ["poses", "shapes", "Rh", "Th", "expression"]:
            if key not in body_params:
                if key == "expression":
                    body_params[key] = np.zeros((n_frames, self.num_expression_coeffs), dtype=np.float32)
                elif key == "shapes":
                    body_params[key] = np.zeros((1, self.num_betas), dtype=np.float32)
                else:
                    body_params[key] = np.zeros((n_frames, 3 if key in ["Rh", "Th"] else self.compact_pose_dim), dtype=np.float32)
        for key in ["poses", "shapes", "Rh", "Th", "expression"]:
            body_params[key] = to_tensor(body_params[key], dtype=self.dtype, device=self.device)
        if body_params["shapes"].shape[0] < n_frames:
            body_params["shapes"] = body_params["shapes"].expand(n_frames, -1)
        return body_params

    def extend_pose(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.shape[-1] == self.full_pose_dim:
            return poses
        if poses.shape[-1] != self.compact_pose_dim:
            raise ValueError(f"Expected compact pose dim {self.compact_pose_dim} or full dim {self.full_pose_dim}, got {poses.shape[-1]}")
        poses_lh = poses[:, self.NUM_BODY_POSES : self.NUM_BODY_POSES + self.num_pca_comps]
        poses_rh = poses[:, self.NUM_BODY_POSES + self.num_pca_comps : self.NUM_BODY_POSES + 2 * self.num_pca_comps]
        poses_head = poses[:, self.NUM_BODY_POSES + 2 * self.num_pca_comps :]
        poses_lh = poses_lh @ self.mHandsComponentsL
        poses_rh = poses_rh @ self.mHandsComponentsR
        if not self.use_flat_mean:
            poses_lh = poses_lh + self.mHandsMeanL
            poses_rh = poses_rh + self.mHandsMeanR
        return torch.cat([poses[:, : self.NUM_BODY_POSES], poses_head, poses_lh, poses_rh], dim=1)

    def _native_joints_to_body25(self, joints: torch.Tensor) -> torch.Tensor:
        batch = joints.shape[0]
        body25 = torch.zeros((batch, BODY25_SIZE, 3), dtype=joints.dtype, device=joints.device)
        for body25_idx, smplx_name in BODY25_NAME_TO_SMPLX.items():
            if smplx_name in self.joint2num:
                body25[:, body25_idx] = joints[:, self.joint2num[smplx_name]]
        return body25

    def _named_joint_stack(self, joints: torch.Tensor, names: list[str]) -> torch.Tensor:
        return torch.stack([joints[:, self.joint2num[name]] for name in names], dim=1)

    def _hand21_single(self, joints: torch.Tensor, vertices: torch.Tensor, side: str) -> torch.Tensor:
        output = []
        for kind, value in HAND21_LAYOUT[side]:
            if kind == "joint":
                output.append(joints[:, self.joint2num[value]])
            else:
                output.append(vertices[:, value])
        return torch.stack(output, dim=1)

    def _hand21(self, joints: torch.Tensor, vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._hand21_single(joints, vertices, "left"), self._hand21_single(joints, vertices, "right")

    def _face51(self, vertices: torch.Tensor) -> torch.Tensor:
        batch = vertices.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(0).expand(batch, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(0).expand(batch, -1, -1)
        return vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

    def bodyhandface_keypoints(self, joints: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
        body25 = self._native_joints_to_body25(joints)
        handl, handr = self._hand21(joints, vertices)
        face = self._face51(vertices)
        return torch.cat([body25, handl, handr, face], dim=1)

    def unsupported_body25_indices(self) -> list[int]:
        return [index for index in range(BODY25_SIZE) if index not in self.body25_supported_indices]

    def forward(
        self,
        poses: np.ndarray | torch.Tensor,
        shapes: np.ndarray | torch.Tensor,
        Rh: np.ndarray | torch.Tensor | None = None,
        Th: np.ndarray | torch.Tensor | None = None,
        expression: np.ndarray | torch.Tensor | None = None,
        return_verts: bool = True,
        return_tensor: bool = True,
        return_smpl_joints: bool = False,
        keypoint_mode: str = "body25",
        only_shape: bool = False,
        pose2rot: bool = True,
        validated: bool = False,
        **kwargs,
    ) -> torch.Tensor | np.ndarray:
        if validated:
            params = {
                "poses": poses,
                "shapes": shapes,
                "Rh": Rh,
                "Th": Th,
                "expression": expression,
            }
        else:
            params = self.check_params({"poses": poses, "shapes": shapes, "Rh": Rh, "Th": Th, "expression": expression})
        poses_t = params["poses"]
        shapes_t = params["shapes"]
        Rh_t = params["Rh"]
        Th_t = params["Th"]
        expr_t = params["expression"]
        if expr_t is not None:
            shapes_t = torch.cat([shapes_t, expr_t], dim=1)
        if pose2rot:
            poses_t = self.extend_pose(poses_t)
        rot = batch_rodrigues(Rh_t) if Rh_t.ndim == 2 else Rh_t
        transl = Th_t.unsqueeze(1)
        need_vertices = return_verts or keypoint_mode == "bodyhandface"
        vertices, joints = lbs(
            shapes_t,
            poses_t,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.weights,
            pose2rot=pose2rot,
            dtype=self.dtype,
            only_shape=only_shape,
            use_pose_blending=self.use_pose_blending,
            use_shape_blending=self.use_shape_blending,
            compute_verts=need_vertices,
        )
        if vertices is not None:
            vertices = torch.matmul(vertices, rot.transpose(1, 2)) + transl
        joints = torch.matmul(joints, rot.transpose(1, 2)) + transl
        output: torch.Tensor
        if return_verts:
            if vertices is None:
                raise RuntimeError("Vertices were not computed")
            output = vertices
        elif return_smpl_joints:
            output = joints
        else:
            if keypoint_mode == "body25":
                output = self._native_joints_to_body25(joints)
            elif keypoint_mode == "bodyhandface":
                if vertices is None:
                    raise RuntimeError("Vertices are required for bodyhandface keypoints")
                output = self.bodyhandface_keypoints(joints, vertices)
            else:
                raise ValueError(f"Unsupported keypoint mode: {keypoint_mode}")
        if return_tensor:
            return output
        return output.detach().cpu().numpy()
