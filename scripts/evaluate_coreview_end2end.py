from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from lightmocap.core.camera.undistort import Undistort
from lightmocap.core.triangulation import project_points, triangulate_multiview_points
from lightmocap.data.camera import CameraSet
from lightmocap.data.io import write_json
from lightmocap.detection.adapter import canonical_keypoints_from_annotation, openpose_people_to_annotation
from lightmocap.pipeline.mocap import MoCapPipeline
from lightmocap.viz.render import make_panel, render_mesh_overlay

ROOT = Path("/home/ymj/code/python/EasyMocap/CoreView_377")
CONFIG_PATH = Path("/home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml")


def _draw_keypoints(image: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    for x, y, conf in keypoints:
        if conf <= 0.2:
            continue
        cv2.circle(output, (int(round(x)), int(round(y))), 3, color, -1, lineType=cv2.LINE_AA)
    return output


def _load_v1_openpose_annotations(cameras: CameraSet, frame_name: str) -> dict[str, dict]:
    annotations: dict[str, dict] = {}
    for camera_name in cameras.names:
        with (ROOT / "keypoints2d" / camera_name / f"{frame_name}_keypoints.json").open() as handle:
            payload = json.load(handle)
        annotations[camera_name] = openpose_people_to_annotation(payload, f"{camera_name}/{frame_name}.jpg")
    return annotations


def _body_detection_metrics(pred_annotations: dict[str, dict], gt_annotations: dict[str, dict], camera_names: list[str]) -> dict[str, float | int | None]:
    errors = []
    for camera_name in camera_names:
        pred = np.asarray(pred_annotations[camera_name]["annots"][0]["keypoints"], dtype=np.float32)
        gt = np.asarray(gt_annotations[camera_name]["annots"][0]["keypoints"], dtype=np.float32)
        valid = gt[:, 2] > 0.2
        if valid.any():
            errors.append(np.linalg.norm(pred[valid, :2] - gt[valid, :2], axis=-1))
    if not errors:
        return {"count": 0, "mean": None, "max": None}
    stacked = np.concatenate(errors)
    return {"count": int(stacked.size), "mean": float(stacked.mean()), "max": float(stacked.max())}


def _triangulation_metrics(pred_keypoints3d: np.ndarray, gt_keypoints3d: np.ndarray, valid_indices: np.ndarray | None = None) -> dict[str, float | int]:
    valid = gt_keypoints3d[:, 3] > 0.2
    if valid_indices is not None:
        mask = np.zeros_like(valid, dtype=bool)
        mask[valid_indices] = True
        valid &= mask
    errors = np.linalg.norm(pred_keypoints3d[valid, :3] - gt_keypoints3d[valid, :3], axis=-1)
    return {"count": int(errors.size), "mean": float(errors.mean()), "max": float(errors.max())}


def _parameter_metrics(pred_params: dict[str, np.ndarray], gt_params: dict[str, np.ndarray]) -> dict[str, float]:
    pred_pose = pred_params["poses"][0]
    gt_pose = gt_params["poses"][0]
    return {
        "Rh_l2": float(np.linalg.norm(pred_params["Rh"][0] - gt_params["Rh"][0])),
        "Th_l2": float(np.linalg.norm(pred_params["Th"][0] - gt_params["Th"][0])),
        "body_pose63_l2": float(np.linalg.norm(pred_pose[3:66] - gt_pose[3:66])),
        "shapes_l2_note_model_mismatch": float(np.linalg.norm(pred_params["shapes"][0] - gt_params["shapes"][0])),
    }


def _belly_metrics(vertices: np.ndarray, Rh: np.ndarray, Th: np.ndarray) -> dict[str, float]:
    R, _ = cv2.Rodrigues(np.asarray(Rh, dtype=np.float32).reshape(3, 1))
    local = (vertices - Th.reshape(1, 3)) @ R
    y = local[:, 1]
    y0 = y.min() + 0.42 * (y.max() - y.min())
    y1 = y.min() + 0.52 * (y.max() - y.min())
    select = (y >= y0) & (y <= y1)
    xs = local[select, 0]
    zs = local[select, 2]
    return {
        "slice_vertex_count": int(select.sum()),
        "width_x_range": float(xs.max() - xs.min()),
        "depth_z_range": float(zs.max() - zs.min()),
        "front_z_max": float(zs.max()),
        "back_z_min": float(zs.min()),
    }


def _reprojection_metrics(
    fitted_keypoints: np.ndarray,
    annotations: dict[str, dict],
    cameras: CameraSet,
    camera_names: list[str],
    body_supported_indices: np.ndarray,
    mode: str,
) -> dict[str, dict[str, float | int | None]]:
    keypoints2d = np.stack([canonical_keypoints_from_annotation(annotations[c], mode=mode) for c in camera_names], axis=0)
    undistorted = np.stack([Undistort.points(keypoints2d[i], cameras[c].K, cameras[c].dist) for i, c in enumerate(camera_names)], axis=0)
    fitted_points4 = np.concatenate([fitted_keypoints, np.ones((fitted_keypoints.shape[0], 1), dtype=np.float32)], axis=-1)
    projected = project_points(fitted_points4, cameras.projection_matrices(camera_names))
    valid = undistorted[..., 2] > 0.2
    body_valid = valid[:, body_supported_indices]
    body_err = np.linalg.norm(projected[:, body_supported_indices, :2] - undistorted[:, body_supported_indices, :2], axis=-1)[body_valid]
    metrics = {
        "body": {
            "count": int(body_err.size),
            "mean": float(body_err.mean()) if body_err.size else None,
            "max": float(body_err.max()) if body_err.size else None,
        }
    }
    if mode == "bodyhandface":
        hand_valid = valid[:, 25:67]
        face_valid = valid[:, 67:]
        hand_err = np.linalg.norm(projected[:, 25:67, :2] - undistorted[:, 25:67, :2], axis=-1)[hand_valid]
        face_err = np.linalg.norm(projected[:, 67:, :2] - undistorted[:, 67:, :2], axis=-1)[face_valid]
        metrics["hand"] = {
            "count": int(hand_err.size),
            "mean": float(hand_err.mean()) if hand_err.size else None,
            "max": float(hand_err.max()) if hand_err.size else None,
        }
        metrics["face"] = {
            "count": int(face_err.size),
            "mean": float(face_err.mean()) if face_err.size else None,
            "max": float(face_err.max()) if face_err.size else None,
        }
    return metrics


def _staged_compare(pipeline: MoCapPipeline, images: dict[str, str], frame_name: str) -> tuple[dict[str, object], dict[str, dict], dict[str, float]]:
    staged_annotations = {}
    for camera_name, image_path in images.items():
        image = cv2.imread(str(image_path))
        staged_annotations[camera_name] = pipeline.detector.detect_annotation(image, f"{camera_name}/{frame_name}.jpg")
    camera_names = list(images.keys())
    staged_k3d = pipeline.triangulate_annotations(staged_annotations, camera_names=camera_names, mode=pipeline.detector.annotation_mode)
    staged_params = pipeline.fit_keypoints3d(staged_k3d, annotations=staged_annotations, camera_names=camera_names, mode=pipeline.detector.annotation_mode)
    staged_vertices = pipeline.body_model(return_verts=True, return_tensor=False, **staged_params)
    end2end = pipeline.process_frame(images, frame_id=int(frame_name))
    diffs = {
        "annotation_body_max_abs_diff": 0.0,
        "triangulation_max_abs_diff": float(np.max(np.abs(end2end["keypoints3d"] - staged_k3d))),
        "poses_max_abs_diff": float(np.max(np.abs(end2end["smplx_params"]["poses"] - staged_params["poses"]))),
        "Rh_max_abs_diff": float(np.max(np.abs(end2end["smplx_params"]["Rh"] - staged_params["Rh"]))),
        "Th_max_abs_diff": float(np.max(np.abs(end2end["smplx_params"]["Th"] - staged_params["Th"]))),
        "vertices_max_abs_diff": float(np.max(np.abs(end2end["vertices"] - staged_vertices))),
    }
    for camera_name in images:
        end_body = np.asarray(end2end["annotations"][camera_name]["annots"][0]["keypoints"], dtype=np.float32)
        staged_body = np.asarray(staged_annotations[camera_name]["annots"][0]["keypoints"], dtype=np.float32)
        diffs["annotation_body_max_abs_diff"] = max(diffs["annotation_body_max_abs_diff"], float(np.max(np.abs(end_body - staged_body))))
    return end2end, staged_annotations, diffs


def _render_views(
    output_dir: Path,
    frame_name: str,
    camera_names: list[str],
    cameras: CameraSet,
    our_vertices: np.ndarray,
    v1_vertices: np.ndarray,
    v1_faces: np.ndarray,
    pred_annotations: dict[str, dict],
    gt_annotations: dict[str, dict],
    our_faces: np.ndarray,
) -> list[str]:
    rendered_files: list[str] = []
    for camera_name in camera_names:
        image = cv2.imread(str(ROOT / camera_name / f"{frame_name}.jpg"))
        camera = cameras[camera_name]
        image_und = Undistort.image(image, camera.K, camera.dist)
        pred_body = Undistort.points(np.asarray(pred_annotations[camera_name]["annots"][0]["keypoints"], dtype=np.float32), camera.K, camera.dist)
        gt_body = Undistort.points(np.asarray(gt_annotations[camera_name]["annots"][0]["keypoints"], dtype=np.float32), camera.K, camera.dist)
        kp_panel = _draw_keypoints(_draw_keypoints(image_und, gt_body, (50, 220, 50)), pred_body, (50, 80, 240))
        our_overlay = render_mesh_overlay(image_und, our_vertices, our_faces, camera, color=(60, 220, 90))
        v1_overlay = render_mesh_overlay(image_und, v1_vertices, v1_faces, camera, color=(240, 120, 60))
        panel = make_panel(
            [image_und, kp_panel, our_overlay, v1_overlay],
            ["input_undistorted", "body2d gt(green) vs pred(red)", "our_smplx_overlay", "v1_smpl_overlay"],
        )
        out_path = output_dir / f"render_compare_{camera_name}_{frame_name}.jpg"
        cv2.imwrite(str(out_path), panel)
        rendered_files.append(str(out_path))
    return rendered_files


def evaluate_coreview_end2end(
    frame: str = "000003",
    render_views: int = 4,
    output: str | Path = "/home/ymj/code/python/EasyMocap/lightmocap/outputs/coreview_e2e",
    detector_mode: str | None = None,
    detector_device: str | None = None,
    backend: str | None = None,
    disable_k2d_refine: bool = False,
) -> dict[str, object]:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(CONFIG_PATH)
    if detector_mode is not None:
        config.detector.mode = detector_mode
    if detector_device is not None:
        config.detector.device = detector_device
        config.model.device = detector_device
    if backend is not None:
        config.detector.backend = backend
    if disable_k2d_refine:
        config.fitting.enable_k2d_refine = False
    elif not hasattr(config.fitting, "enable_k2d_refine"):
        config.fitting.enable_k2d_refine = True

    pipeline = MoCapPipeline.from_config(config)
    cameras = pipeline.camera
    images = {camera_name: str(ROOT / camera_name / f"{frame}.jpg") for camera_name in cameras.names}

    end2end, pred_annotations, staged_diffs = _staged_compare(pipeline, images, frame)
    gt_annotations = _load_v1_openpose_annotations(cameras, frame)
    gt_body2d = np.stack([canonical_keypoints_from_annotation(gt_annotations[camera_name], mode="body25") for camera_name in cameras.names], axis=0)
    gt_body3d = triangulate_multiview_points(gt_body2d, cameras, camera_names=cameras.names)
    gt_params = np.load(ROOT / f"new_params/{int(frame)}.npy", allow_pickle=True).item()
    v1_vertices = np.load(ROOT / f"new_vertices/{int(frame)}.npy")
    v1_faces = np.load(ROOT / "lbs/faces.npy")
    fitted_keypoints = pipeline.body_model(return_verts=False, return_tensor=False, keypoint_mode=pipeline.detector.annotation_mode, **end2end["smplx_params"])[0]

    summary = {
        "frame": frame,
        "detector": {
            "type": config.detector.type,
            "solution": config.detector.solution,
            "mode": config.detector.mode,
            "device": config.detector.device,
            "backend": config.detector.backend,
        },
        "model": {
            "gender": config.model.gender,
            "device": getattr(config.model, "device", str(pipeline.body_model.device)),
        },
        "fitting": {
            "enable_k2d_refine": bool(getattr(config.fitting, "enable_k2d_refine", True)),
        },
        "notes": {
            "dataset_body_prior": "CoreView_377/keypoints2d is V1 OpenPose body-only output",
            "bodyhandface_truth": "No V1 hand/face truth is stored in this sequence; bodyhandface is checked against real rtmlib wholebody detections and reprojection consistency",
            "smpl_param_compare": "new_params/new_vertices are V1 SMPL results; Rh/Th/body pose overlap are comparable, shapes are reported with model mismatch note",
        },
        "end_to_end_vs_staged": staged_diffs,
        "body2d_compare_vs_dataset_openpose": _body_detection_metrics(end2end["annotations"], gt_annotations, cameras.names),
        "body3d_compare_vs_dataset_openpose_triangulation": _triangulation_metrics(end2end["keypoints3d"][:25], gt_body3d, valid_indices=np.array(pipeline.body_model.body25_supported_indices)),
        "body_param_compare_vs_v1_new_params": _parameter_metrics(end2end["smplx_params"], gt_params),
        "our_belly_metrics": _belly_metrics(end2end["vertices"][0], end2end["smplx_params"]["Rh"][0], end2end["smplx_params"]["Th"][0]),
        "v1_belly_metrics": _belly_metrics(v1_vertices, gt_params["Rh"][0], gt_params["Th"][0]),
        "reprojection_to_input_2d": _reprojection_metrics(
            fitted_keypoints,
            end2end["annotations"],
            cameras,
            cameras.names,
            np.array(pipeline.body_model.body25_supported_indices),
            pipeline.detector.annotation_mode,
        ),
    }

    render_camera_names = cameras.names[:render_views]
    render_files = _render_views(
        output_dir=output_dir,
        frame_name=frame,
        camera_names=render_camera_names,
        cameras=cameras,
        our_vertices=end2end["vertices"][0],
        v1_vertices=v1_vertices,
        v1_faces=v1_faces,
        pred_annotations=end2end["annotations"],
        gt_annotations=gt_annotations,
        our_faces=pipeline.body_model.faces,
    )
    summary["render_files"] = render_files

    result_payload = {
        "frame": frame,
        "camera_names": cameras.names,
        "keypoints3d": end2end["keypoints3d"].tolist(),
        "smplx_params": {key: value.tolist() for key, value in end2end["smplx_params"].items()},
    }

    summary_path = output_dir / f"summary_{frame}.json"
    result_path = output_dir / f"result_{frame}.json"
    vertices_path = output_dir / f"vertices_{frame}.npy"
    write_json(summary_path, summary)
    write_json(result_path, result_payload)
    np.save(vertices_path, end2end["vertices"])
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="000003")
    parser.add_argument("--render-views", type=int, default=4)
    parser.add_argument("--output", default="/home/ymj/code/python/EasyMocap/lightmocap/outputs/coreview_e2e")
    parser.add_argument("--detector-mode", choices=["lightweight", "balanced", "performance"], default=None)
    parser.add_argument("--detector-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--disable-k2d-refine", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    summary = evaluate_coreview_end2end(
        frame=args.frame,
        render_views=args.render_views,
        output=args.output,
        detector_mode=args.detector_mode,
        detector_device=args.detector_device,
        backend=args.backend,
        disable_k2d_refine=args.disable_k2d_refine,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
