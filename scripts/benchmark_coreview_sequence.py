from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from lightmocap.core.camera.undistort import Undistort
from lightmocap.core.fitting import FittingConfig, SMPLXFittingPipeline
from lightmocap.core.triangulation import project_points, triangulate_multiview_points
from lightmocap.data.camera import CameraSet
from lightmocap.data.io import write_json
from lightmocap.detection.adapter import canonical_keypoints_from_annotation, openpose_people_to_annotation
from lightmocap.pipeline.mocap import MoCapPipeline
from lightmocap.viz.render import make_panel, render_mesh_overlay

ROOT = Path("/home/ymj/code/python/EasyMocap/CoreView_377")
CONFIG_PATH = Path("/home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml")


def _load_config(detector_mode: str | None, detector_device: str | None, backend: str | None, enable_k2d_refine: bool) -> object:
    config = OmegaConf.load(CONFIG_PATH)
    if detector_mode is not None:
        config.detector.mode = detector_mode
    if detector_device is not None:
        config.detector.device = detector_device
        config.model.device = detector_device
    if backend is not None:
        config.detector.backend = backend
    config.fitting.enable_k2d_refine = enable_k2d_refine
    return config


def _load_v1_annotation(camera_name: str, frame_id: int) -> dict:
    with (ROOT / "keypoints2d" / camera_name / f"{frame_id:06d}_keypoints.json").open() as handle:
        payload = json.load(handle)
    return openpose_people_to_annotation(payload, f"{camera_name}/{frame_id:06d}.jpg")


def _body2d_error(pred_annotation: dict, gt_annotation: dict) -> np.ndarray:
    pred = np.asarray(pred_annotation["annots"][0]["keypoints"], dtype=np.float32)
    gt = np.asarray(gt_annotation["annots"][0]["keypoints"], dtype=np.float32)
    valid = gt[:, 2] > 0.2
    return np.linalg.norm(pred[valid, :2] - gt[valid, :2], axis=-1)


def _triangulation_error(pred_keypoints3d: np.ndarray, gt_keypoints3d: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
    valid = gt_keypoints3d[:, 3] > 0.2
    mask = np.zeros_like(valid, dtype=bool)
    mask[valid_indices] = True
    valid &= mask
    return np.linalg.norm(pred_keypoints3d[valid, :3] - gt_keypoints3d[valid, :3], axis=-1)


def _load_v1_gt(frame_id: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    params = np.load(ROOT / f"new_params/{frame_id}.npy", allow_pickle=True).item()
    vertices = np.load(ROOT / f"new_vertices/{frame_id}.npy")
    return params, vertices


def _belly_depth(vertices: np.ndarray, Rh: np.ndarray, Th: np.ndarray) -> float:
    R, _ = cv2.Rodrigues(np.asarray(Rh, dtype=np.float32).reshape(3, 1))
    local = (vertices - Th.reshape(1, 3)) @ R
    y = local[:, 1]
    y0 = y.min() + 0.42 * (y.max() - y.min())
    y1 = y.min() + 0.52 * (y.max() - y.min())
    select = (y >= y0) & (y <= y1)
    zs = local[select, 2]
    return float(zs.max() - zs.min())


def _second_diff_mean(values: np.ndarray) -> float:
    if values.shape[0] <= 2:
        return 0.0
    accel = values[2:] - 2 * values[1:-1] + values[:-2]
    return float(np.linalg.norm(accel.reshape(accel.shape[0], -1), axis=-1).mean())


def _root_relative_body_joints(body_model, params: dict[str, np.ndarray]) -> np.ndarray:
    joints = body_model(return_verts=False, return_tensor=False, keypoint_mode="body25", **params)
    pelvis = joints[:, 8:9]
    return joints - pelvis


def _reprojection_error(body_model, params: dict[str, np.ndarray], annotations_by_frame: list[dict[str, dict]], cameras: CameraSet, camera_names: list[str], mode: str) -> dict[str, float]:
    keypoints = body_model(return_verts=False, return_tensor=False, keypoint_mode=mode, **params)
    errors = {"body": [], "hand": [], "face": []}
    body_supported = np.array(body_model.body25_supported_indices)
    Pall = cameras.projection_matrices(camera_names)
    for frame_index, frame_annotations in enumerate(annotations_by_frame):
        keypoints2d = np.stack([canonical_keypoints_from_annotation(frame_annotations[c], mode=mode) for c in camera_names], axis=0)
        undistorted = np.stack([Undistort.points(keypoints2d[i], cameras[c].K, cameras[c].dist) for i, c in enumerate(camera_names)], axis=0)
        fitted_points4 = np.concatenate([keypoints[frame_index], np.ones((keypoints.shape[1], 1), dtype=np.float32)], axis=-1)
        projected = project_points(fitted_points4, Pall)
        valid = undistorted[..., 2] > 0.2
        body_valid = valid[:, body_supported]
        errors["body"].append(np.linalg.norm(projected[:, body_supported, :2] - undistorted[:, body_supported, :2], axis=-1)[body_valid])
        if mode == "bodyhandface":
            hand_valid = valid[:, 25:67]
            face_valid = valid[:, 67:]
            errors["hand"].append(np.linalg.norm(projected[:, 25:67, :2] - undistorted[:, 25:67, :2], axis=-1)[hand_valid])
            errors["face"].append(np.linalg.norm(projected[:, 67:, :2] - undistorted[:, 67:, :2], axis=-1)[face_valid])
    out = {}
    for key, parts in errors.items():
        if not parts or all(part.size == 0 for part in parts):
            continue
        merged = np.concatenate([part for part in parts if part.size > 0])
        out[key] = float(merged.mean())
    return out


def _param_diff(pred_params: dict[str, np.ndarray], gt_params_seq: list[dict[str, np.ndarray]]) -> dict[str, float]:
    Rh = []
    Th = []
    pose = []
    for frame_index, gt in enumerate(gt_params_seq):
        Rh.append(np.linalg.norm(pred_params["Rh"][frame_index] - gt["Rh"][0]))
        Th.append(np.linalg.norm(pred_params["Th"][frame_index] - gt["Th"][0]))
        pose.append(np.linalg.norm(pred_params["poses"][frame_index, 3:66] - gt["poses"][0, 3:66]))
    return {
        "Rh_l2_mean": float(np.mean(Rh)),
        "Th_l2_mean": float(np.mean(Th)),
        "body_pose63_l2_mean": float(np.mean(pose)),
    }


def _strategy_summary(
    name: str,
    body_model,
    params: dict[str, np.ndarray],
    annotations_by_frame: list[dict[str, dict]],
    cameras: CameraSet,
    camera_names: list[str],
    gt_params_seq: list[dict[str, np.ndarray]],
    gt_vertices_seq: list[np.ndarray],
    mode: str,
) -> dict[str, object]:
    vertices = body_model(return_verts=True, return_tensor=False, **params)
    body_joints_rel = _root_relative_body_joints(body_model, params)
    belly = [_belly_depth(vertices[i], params["Rh"][i], params["Th"][i]) for i in range(vertices.shape[0])]
    gt_belly = [_belly_depth(gt_vertices_seq[i], gt_params_seq[i]["Rh"][0], gt_params_seq[i]["Th"][0]) for i in range(len(gt_vertices_seq))]
    return {
        "name": name,
        "param_diff_vs_v1": _param_diff(params, gt_params_seq),
        "reprojection_mean_px": _reprojection_error(body_model, params, annotations_by_frame, cameras, camera_names, mode),
        "jitter": {
            "poses_accel_mean": _second_diff_mean(params["poses"][:, :66]),
            "Rh_accel_mean": _second_diff_mean(params["Rh"]),
            "Th_accel_mean": _second_diff_mean(params["Th"]),
            "body_joint_rel_accel_mean": _second_diff_mean(body_joints_rel),
            "vertices_accel_mean": _second_diff_mean(vertices[:, ::20]),
            "belly_depth_std": float(np.std(belly)),
        },
        "belly_depth_mean": float(np.mean(belly)),
        "belly_depth_vs_v1_mean_abs": float(np.mean(np.abs(np.array(belly) - np.array(gt_belly)))),
        "vertices": vertices,
    }


def _render_sequence_comparison(
    output_dir: Path,
    frame_ids: list[int],
    camera_name: str,
    cameras: CameraSet,
    independent_vertices: np.ndarray,
    seq_vertices: np.ndarray,
    smooth_vertices: np.ndarray,
    v1_vertices_seq: list[np.ndarray],
    faces: np.ndarray,
) -> list[str]:
    render_frames = [frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]]
    files = []
    for frame_id in render_frames:
        idx = frame_ids.index(frame_id)
        image = cv2.imread(str(ROOT / camera_name / f"{frame_id:06d}.jpg"))
        camera = cameras[camera_name]
        image_und = Undistort.image(image, camera.K, camera.dist)
        panel = make_panel(
            [
                image_und,
                render_mesh_overlay(image_und, independent_vertices[idx], faces, camera, color=(80, 180, 240)),
                render_mesh_overlay(image_und, seq_vertices[idx], faces, camera, color=(240, 180, 80)),
                render_mesh_overlay(image_und, smooth_vertices[idx], faces, camera, color=(60, 220, 90)),
                render_mesh_overlay(image_und, v1_vertices_seq[idx], np.load(ROOT / "lbs/faces.npy"), camera, color=(240, 120, 60)),
            ],
            ["input", "independent", "sequence_no_temporal", "sequence_temporal", "v1_smpl"],
        )
        out = output_dir / f"sequence_compare_{camera_name}_{frame_id:06d}.jpg"
        cv2.imwrite(str(out), panel)
        files.append(str(out))
    return files


def evaluate_coreview_sequence_benchmark(
    start_frame: int = 0,
    num_frames: int = 12,
    detector_mode: str = "balanced",
    detector_device: str = "cuda",
    backend: str = "onnxruntime",
    output: str | Path = "/home/ymj/code/python/EasyMocap/lightmocap/outputs/coreview_sequence_benchmark",
) -> dict[str, object]:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = list(range(start_frame, start_frame + num_frames))
    base_config = _load_config(detector_mode, detector_device, backend, enable_k2d_refine=True)
    pipeline = MoCapPipeline.from_config(base_config)
    cameras = pipeline.camera
    camera_names = cameras.names
    mode = pipeline.detector.annotation_mode

    annotations_by_frame: list[dict[str, dict]] = []
    keypoints3d_seq = []
    body2d_errors = []
    body3d_errors = []
    gt_params_seq = []
    gt_vertices_seq = []
    for frame_id in frame_ids:
        frame_annotations = {}
        for camera_name in camera_names:
            image = cv2.imread(str(ROOT / camera_name / f"{frame_id:06d}.jpg"))
            frame_annotations[camera_name] = pipeline.detector.detect_annotation(image, f"{camera_name}/{frame_id:06d}.jpg")
            gt_annotation = _load_v1_annotation(camera_name, frame_id)
            body2d_errors.append(_body2d_error(frame_annotations[camera_name], gt_annotation))
        annotations_by_frame.append(frame_annotations)
        pred_k3d = pipeline.triangulate_annotations(frame_annotations, camera_names=camera_names, mode=mode)
        gt_body2d = np.stack([canonical_keypoints_from_annotation(_load_v1_annotation(camera_name, frame_id), mode="body25") for camera_name in camera_names], axis=0)
        gt_k3d = triangulate_multiview_points(gt_body2d, cameras, camera_names=camera_names)
        body3d_errors.append(_triangulation_error(pred_k3d[:25], gt_k3d, np.array(pipeline.body_model.body25_supported_indices)))
        keypoints3d_seq.append(pred_k3d)
        gt_params, gt_vertices = _load_v1_gt(frame_id)
        gt_params_seq.append(gt_params)
        gt_vertices_seq.append(gt_vertices)

    keypoints3d_seq = np.stack(keypoints3d_seq, axis=0)
    keypoints2d_seq = np.stack([
        np.stack([canonical_keypoints_from_annotation(frame_annotations[c], mode=mode) for c in camera_names], axis=0)
        for frame_annotations in annotations_by_frame
    ], axis=0)
    bboxes_seq = np.stack([
        np.stack([np.asarray(frame_annotations[c]["annots"][0]["bbox"], dtype=np.float32) for c in camera_names], axis=0)
        for frame_annotations in annotations_by_frame
    ], axis=0)
    Pall = cameras.projection_matrices(camera_names)

    independent_params = {
        "poses": [],
        "shapes": [],
        "Rh": [],
        "Th": [],
        "expression": [],
    }
    for frame_index, frame_id in enumerate(frame_ids):
        params = pipeline.fit_keypoints3d(
            keypoints3d_seq[frame_index],
            annotations=annotations_by_frame[frame_index],
            camera_names=camera_names,
            mode=mode,
        )
        for key in independent_params:
            independent_params[key].append(params[key][0])
    independent_params = {key: np.asarray(value, dtype=np.float32) for key, value in independent_params.items()}

    sequence_init = {
        "poses": independent_params["poses"].copy(),
        "Rh": independent_params["Rh"].copy(),
        "Th": independent_params["Th"].copy(),
        "expression": independent_params["expression"].copy(),
        "shapes": independent_params["shapes"].mean(axis=0, keepdims=True),
    }

    no_temporal_cfg = FittingConfig(
        device=str(pipeline.body_model.device),
        maxiters=pipeline.fitter.config.maxiters,
        shape_maxiters=pipeline.fitter.config.shape_maxiters,
        enable_k2d_refine=True,
        weight_loss={**pipeline.fitter.config.weight_loss, **{k: 0.0 for k in ["smooth_body", "smooth_poses", "smooth_Rh", "smooth_hand", "smooth_head"]}},
    )
    no_temporal_fitter = SMPLXFittingPipeline(pipeline.body_model, no_temporal_cfg)
    seq_no_temporal_params = no_temporal_fitter.fit(
        keypoints3d_seq,
        body_params=sequence_init,
        keypoints2d=keypoints2d_seq,
        bboxes=bboxes_seq,
        projection_matrices=Pall,
    )

    temporal_cfg = FittingConfig(
        device=str(pipeline.body_model.device),
        maxiters=pipeline.fitter.config.maxiters,
        shape_maxiters=pipeline.fitter.config.shape_maxiters,
        enable_k2d_refine=True,
        weight_loss=pipeline.fitter.config.weight_loss,
    )
    temporal_fitter = SMPLXFittingPipeline(pipeline.body_model, temporal_cfg)
    seq_temporal_params = temporal_fitter.fit(
        keypoints3d_seq,
        body_params=sequence_init,
        keypoints2d=keypoints2d_seq,
        bboxes=bboxes_seq,
        projection_matrices=Pall,
    )

    independent_summary = _strategy_summary("independent", pipeline.body_model, independent_params, annotations_by_frame, cameras, camera_names, gt_params_seq, gt_vertices_seq, mode)
    no_temporal_summary = _strategy_summary("sequence_no_temporal", pipeline.body_model, seq_no_temporal_params, annotations_by_frame, cameras, camera_names, gt_params_seq, gt_vertices_seq, mode)
    temporal_summary = _strategy_summary("sequence_temporal", pipeline.body_model, seq_temporal_params, annotations_by_frame, cameras, camera_names, gt_params_seq, gt_vertices_seq, mode)

    render_files = _render_sequence_comparison(
        output_dir,
        frame_ids,
        camera_names[0],
        cameras,
        independent_summary["vertices"],
        no_temporal_summary["vertices"],
        temporal_summary["vertices"],
        gt_vertices_seq,
        pipeline.body_model.faces,
    )

    # Remove raw vertices from JSON summary to keep it compact.
    for summary in [independent_summary, no_temporal_summary, temporal_summary]:
        summary.pop("vertices")

    report = {
        "frames": frame_ids,
        "detector": {
            "type": base_config.detector.type,
            "solution": base_config.detector.solution,
            "mode": base_config.detector.mode,
            "device": base_config.detector.device,
            "backend": base_config.detector.backend,
        },
        "notes": {
            "sequence_init": "sequence-level fitting is initialized from the per-frame independent results, with the sequence shape initialized by the frame-wise mean beta",
        },
        "detection_and_triangulation": {
            "body2d_vs_v1_openpose_mean_px": float(np.concatenate(body2d_errors).mean()),
            "body3d_vs_v1_triangulation_mean_m": float(np.concatenate(body3d_errors).mean()),
        },
        "strategies": {
            "independent": independent_summary,
            "sequence_no_temporal": no_temporal_summary,
            "sequence_temporal": temporal_summary,
        },
        "temporal_effect": {
            "body_joint_rel_accel_delta": temporal_summary["jitter"]["body_joint_rel_accel_mean"] - no_temporal_summary["jitter"]["body_joint_rel_accel_mean"],
            "vertices_accel_delta": temporal_summary["jitter"]["vertices_accel_mean"] - no_temporal_summary["jitter"]["vertices_accel_mean"],
            "belly_depth_std_delta": temporal_summary["jitter"]["belly_depth_std"] - no_temporal_summary["jitter"]["belly_depth_std"],
            "body_pose63_l2_mean_delta_vs_v1": temporal_summary["param_diff_vs_v1"]["body_pose63_l2_mean"] - no_temporal_summary["param_diff_vs_v1"]["body_pose63_l2_mean"],
        },
        "render_files": render_files,
    }
    write_json(output_dir / f"summary_{frame_ids[0]:06d}_{frame_ids[-1]:06d}.json", report)
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--detector-mode", choices=["lightweight", "balanced", "performance"], default="balanced")
    parser.add_argument("--detector-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--output", default="/home/ymj/code/python/EasyMocap/lightmocap/outputs/coreview_sequence_benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    report = evaluate_coreview_sequence_benchmark(
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        detector_mode=args.detector_mode,
        detector_device=args.detector_device,
        backend=args.backend,
        output=args.output,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
