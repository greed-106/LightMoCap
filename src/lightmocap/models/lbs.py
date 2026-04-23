from __future__ import annotations

import torch
import torch.nn.functional as F


def vertices2landmarks(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    lmk_faces_idx: torch.Tensor,
    lmk_bary_coords: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.reshape(-1)).reshape(batch_size, -1, 3)
    lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts
    lmk_vertices = vertices.reshape(-1, 3)[lmk_faces].reshape(batch_size, -1, 3, 3)
    return torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bl,mkl->bmk", [betas, shape_disps])


def batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    rot_vec_ori = rot_vecs if rot_vecs.ndim > 2 else None
    if rot_vec_ori is not None:
        rot_vecs = rot_vecs.reshape(-1, 3)
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(batch_size, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    rot_mat = ident + sin * K + (1.0 - cos) * torch.bmm(K, K)
    if rot_vec_ori is not None:
        rot_mat = rot_mat.reshape(*rot_vec_ori.shape[:-1], 3, 3)
    return rot_mat


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.contiguous().reshape(-1, 3, 1))
    transforms_mat = transforms_mat.view(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for index in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[index]], transforms_mat[:, index])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms


def lbs(
    betas: torch.Tensor,
    pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
    pose2rot: bool = True,
    dtype: torch.dtype = torch.float32,
    only_shape: bool = False,
    use_shape_blending: bool = True,
    use_pose_blending: bool = True,
    J_shaped: torch.Tensor | None = None,
    compute_verts: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device
    if use_shape_blending:
        v_shaped = v_template + blend_shapes(betas, shapedirs)
        J = vertices2joints(J_regressor, v_shaped)
    else:
        v_shaped = v_template.unsqueeze(0).expand(batch_size, -1, -1)
        if J_shaped is None:
            raise ValueError("J_shaped is required when shape blending is disabled")
        J = J_shaped[None].expand(batch_size, -1, -1)
    if only_shape:
        return v_shaped, J
    if pose2rot:
        rot_mats = batch_rodrigues(pose.reshape(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)
    if use_pose_blending:
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch_size, -1)
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    if not compute_verts:
        return None, J_transformed
    W = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed
