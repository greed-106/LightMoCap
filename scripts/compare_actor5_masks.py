from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from lightmocap.data.io import write_json
from lightmocap.detection.masking import load_mask, mask_foreground_ratio
from lightmocap.pipeline.mocap import MoCapPipeline
from lightmocap.workflow.frame import build_frame_image_paths, build_frame_mask_paths, save_frame_result, save_multiview_annotations


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _frame_name(frame: int) -> str:
    return f"{frame:06d}"


def _frame_dir(frame: int) -> str:
    return f"frame{frame:03d}"


def _save_pipeline_result(output_root: Path, frame_name: str, result: dict[str, Any]) -> None:
    save_multiview_annotations(result["annotations"], output_root / "annots", frame_name)
    save_frame_result(output_root, frame_name, result)


def _compare_keypoints3d(nomask: np.ndarray, masked: np.ndarray) -> dict[str, Any]:
    valid0 = nomask[..., 3] > 0
    valid1 = masked[..., 3] > 0
    valid_both = valid0 & valid1
    dist = np.linalg.norm(nomask[..., :3] - masked[..., :3], axis=-1)
    return {
        "exact_equal": bool(np.array_equal(nomask, masked)),
        "shape": list(nomask.shape),
        "valid_nomask": int(valid0.sum()),
        "valid_masked": int(valid1.sum()),
        "valid_both": int(valid_both.sum()),
        "mean_valid_both": float(dist[valid_both].mean()) if valid_both.any() else None,
        "median_valid_both": float(np.median(dist[valid_both])) if valid_both.any() else None,
        "p95_valid_both": float(np.percentile(dist[valid_both], 95)) if valid_both.any() else None,
        "max_valid_both": float(dist[valid_both].max()) if valid_both.any() else None,
        "conf_mean_abs": float(np.abs(nomask[..., 3] - masked[..., 3]).mean()),
    }


def _compare_smplx(nomask: dict[str, Any], masked: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in sorted(set(nomask) & set(masked)):
        value0 = _as_numpy(nomask[key]).astype(np.float64)
        value1 = _as_numpy(masked[key]).astype(np.float64)
        diff = np.abs(value0 - value1)
        params[key] = {
            "shape": list(value0.shape),
            "exact_equal": bool(np.array_equal(value0, value1)),
            "mean_abs": float(diff.mean()),
            "max_abs": float(diff.max()),
            "rms": float(np.sqrt((diff**2).mean())),
        }
    return {
        "keys_equal": set(nomask) == set(masked),
        "params": params,
    }


def _body_stats(annotation0: dict[str, Any], annotation1: dict[str, Any]) -> dict[str, Any]:
    n0 = len(annotation0.get("annots", []))
    n1 = len(annotation1.get("annots", []))
    stats: dict[str, Any] = {
        "people_nomask": n0,
        "people_masked": n1,
        "exact_equal": annotation0 == annotation1,
    }
    if n0 == 0 or n1 == 0:
        return stats
    body0 = np.asarray(annotation0["annots"][0]["keypoints"], dtype=np.float64)
    body1 = np.asarray(annotation1["annots"][0]["keypoints"], dtype=np.float64)
    valid = (body0[:, 2] > 0) & (body1[:, 2] > 0)
    xy = np.linalg.norm(body0[:, :2] - body1[:, :2], axis=1)
    conf = np.abs(body0[:, 2] - body1[:, 2])
    stats.update(
        {
            "valid_body_nomask": int((body0[:, 2] > 0).sum()),
            "valid_body_masked": int((body1[:, 2] > 0).sum()),
            "valid_body_both": int(valid.sum()),
            "body_xy_mean_px": float(xy[valid].mean()) if valid.any() else None,
            "body_xy_p95_px": float(np.percentile(xy[valid], 95)) if valid.any() else None,
            "body_xy_max_px": float(xy[valid].max()) if valid.any() else None,
            "body_conf_mean_abs": float(conf.mean()),
        }
    )
    return stats


def _compare_annotations(nomask: dict[str, dict], masked: dict[str, dict]) -> dict[str, Any]:
    per_camera = {camera_name: _body_stats(nomask[camera_name], masked[camera_name]) for camera_name in sorted(nomask)}
    xy_means = [value["body_xy_mean_px"] for value in per_camera.values() if value.get("body_xy_mean_px") is not None]
    xy_p95 = [value["body_xy_p95_px"] for value in per_camera.values() if value.get("body_xy_p95_px") is not None]
    xy_max = [value["body_xy_max_px"] for value in per_camera.values() if value.get("body_xy_max_px") is not None]
    conf = [value["body_conf_mean_abs"] for value in per_camera.values() if value.get("body_conf_mean_abs") is not None]
    changed_people = [camera for camera, value in per_camera.items() if value["people_nomask"] != value["people_masked"]]
    return {
        "all_exact_equal": all(value["exact_equal"] for value in per_camera.values()),
        "people_count_all_equal": not changed_people,
        "people_count_changed_cameras": changed_people,
        "body_xy_mean_px_mean_camera": float(np.mean(xy_means)) if xy_means else None,
        "body_xy_p95_px_mean_camera": float(np.mean(xy_p95)) if xy_p95 else None,
        "body_xy_max_px_max_camera": float(np.max(xy_max)) if xy_max else None,
        "body_conf_mean_abs_mean_camera": float(np.mean(conf)) if conf else None,
        "per_camera": per_camera,
    }


def _compare_results(nomask: dict[str, Any], masked: dict[str, Any]) -> dict[str, Any]:
    return {
        "keypoints3d": _compare_keypoints3d(_as_numpy(nomask["keypoints3d"]), _as_numpy(masked["keypoints3d"])),
        "smplx": _compare_smplx(nomask["smplx_params"], masked["smplx_params"]),
        "annotations": _compare_annotations(nomask["annotations"], masked["annotations"]),
    }


def _mask_quality(mask_paths: dict[str, Path], threshold: float, min_area_ratio: float) -> dict[str, Any]:
    per_camera: dict[str, Any] = {}
    skipped = []
    for camera_name, mask_path in mask_paths.items():
        ratio = mask_foreground_ratio(load_mask(mask_path), threshold=threshold)
        usable = ratio >= min_area_ratio if min_area_ratio > 0 else True
        per_camera[camera_name] = {"foreground_ratio": ratio, "usable": usable}
        if not usable:
            skipped.append(camera_name)
    return {"skipped_cameras": skipped, "per_camera": per_camera}


def _aggregate(per_frame: list[dict[str, Any]]) -> dict[str, Any]:
    k3d_mean = [item["metrics"]["keypoints3d"]["mean_valid_both"] for item in per_frame if item["metrics"]["keypoints3d"]["mean_valid_both"] is not None]
    k3d_p95 = [item["metrics"]["keypoints3d"]["p95_valid_both"] for item in per_frame if item["metrics"]["keypoints3d"]["p95_valid_both"] is not None]
    k3d_max = [item["metrics"]["keypoints3d"]["max_valid_both"] for item in per_frame if item["metrics"]["keypoints3d"]["max_valid_both"] is not None]
    body_xy = [item["metrics"]["annotations"]["body_xy_mean_px_mean_camera"] for item in per_frame if item["metrics"]["annotations"]["body_xy_mean_px_mean_camera"] is not None]
    changed_people = [item["frame"] for item in per_frame if not item["metrics"]["annotations"]["people_count_all_equal"]]
    skipped_masks = {
        item["frame"]: item["mask_quality"]["skipped_cameras"]
        for item in per_frame
        if item["mask_quality"]["skipped_cameras"]
    }
    smplx_max: dict[str, float] = {}
    for item in per_frame:
        for key, value in item["metrics"]["smplx"]["params"].items():
            smplx_max[key] = max(smplx_max.get(key, 0.0), float(value["max_abs"]))
    return {
        "num_frames": len(per_frame),
        "keypoints3d_mean_diff_mean": float(np.mean(k3d_mean)) if k3d_mean else None,
        "keypoints3d_p95_diff_mean": float(np.mean(k3d_p95)) if k3d_p95 else None,
        "keypoints3d_max_diff_max": float(np.max(k3d_max)) if k3d_max else None,
        "body_xy_mean_px_mean": float(np.mean(body_xy)) if body_xy else None,
        "frames_with_people_count_changes": changed_people,
        "frames_with_skipped_masks": skipped_masks,
        "smplx_max_abs_by_param": smplx_max,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = OmegaConf.load(args.config)
    pipeline = MoCapPipeline.from_config(config)
    output_root = Path(args.output)
    mask_root = Path(args.masks)
    image_ext = OmegaConf.select(config, "data.image_ext")
    per_frame = []

    for frame in range(args.start_frame, args.end_frame + 1):
        frame_name = _frame_name(frame)
        image_paths = build_frame_image_paths(config.data.images, pipeline.camera.names, frame_name, image_ext=image_ext)
        mask_paths = build_frame_mask_paths(mask_root, config.data.images, image_paths)
        mask_quality = _mask_quality(mask_paths, args.mask_threshold, args.min_mask_area_ratio)

        nomask_result = pipeline.process_frame(image_paths, frame_id=frame)
        masked_result = pipeline.process_frame(
            image_paths,
            frame_id=frame,
            masks=mask_paths,
            mask_threshold=args.mask_threshold,
            min_mask_area_ratio=args.min_mask_area_ratio,
        )

        nomask_output = output_root / "nomask" / _frame_dir(frame)
        masked_output = output_root / "masked" / _frame_dir(frame)
        _save_pipeline_result(nomask_output, frame_name, nomask_result)
        _save_pipeline_result(masked_output, frame_name, masked_result)

        metrics = _compare_results(nomask_result, masked_result)
        per_frame.append({"frame": frame_name, "mask_quality": mask_quality, "metrics": metrics})
        print(json.dumps({"frame": frame_name, "metrics": metrics["keypoints3d"]}, indent=2), flush=True)

    summary = {
        "config": str(Path(args.config).resolve()),
        "masks": str(mask_root.resolve()),
        "output": str(output_root.resolve()),
        "start_frame": _frame_name(args.start_frame),
        "end_frame": _frame_name(args.end_frame),
        "mask_threshold": args.mask_threshold,
        "min_mask_area_ratio": args.min_mask_area_ratio,
        "aggregate": _aggregate(per_frame),
        "per_frame": per_frame,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare actor5 MVSP outputs with and without mirrored masks.")
    parser.add_argument("--config", default="configs/actor5_scaled.yaml")
    parser.add_argument("--masks", default="/mnt/g/code/datasets/zju4dv/actor5/other-formats/wu-4dgs/masks")
    parser.add_argument("--output", default="/tmp/lightmocap_actor5_mask_experiment/first100")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=100)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    summary = run_experiment(parse_args())
    print(json.dumps({key: summary[key] for key in ["output", "start_frame", "end_frame", "aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
