from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import cv2
from omegaconf import DictConfig, OmegaConf

from lightmocap.data.camera import CameraSet
from lightmocap.data.dataset import MultiViewImageSequence
from lightmocap.data.io import read_json, write_json
from lightmocap.detection.adapter import openpose_people_to_annotation
from lightmocap.detection.rtmlib import RTMLibDetector
from lightmocap.pipeline.mocap import MoCapPipeline
from lightmocap.workflow.frame import (
    build_frame_image_paths,
    load_multiview_annotations,
    normalize_frame_name,
    output_layout,
    render_result_views,
    save_frame_result,
    save_multiview_annotations,
)


def _dump_json(data: dict, output: str | Path | None) -> int:
    if output is None:
        print(json.dumps(data, indent=2))
        return 0
    write_json(output, data)
    return 0


def _read_image(path: str | Path) -> cv2.typing.MatLike:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)


def _output_root(config: DictConfig, output: str | None) -> Path:
    if output is not None:
        return Path(output)
    configured = OmegaConf.select(config, "data.output")
    if configured is not None:
        return Path(str(configured))
    return Path.cwd() / "outputs"


def _annotation_root(args: argparse.Namespace, config: DictConfig) -> Path:
    if getattr(args, "annots", None):
        return Path(args.annots)
    return output_layout(_output_root(config, getattr(args, "output", None))).annots


def _layout_summary(root: str | Path) -> dict[str, str]:
    layout = output_layout(root)
    return {
        "root": str(layout.root.resolve()),
        "annots": str(layout.annots.resolve()),
        "keypoints3d": str(layout.keypoints3d.resolve()),
        "smplx": str(layout.smplx.resolve()),
        "vertices": str(layout.vertices.resolve()),
        "renders": str(layout.renders.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a LightMocap config and summarize the current multiview inputs")
    validate.add_argument("--config", required=True, help="LightMocap 配置文件路径")

    detect_image = subparsers.add_parser("detect-image", help="Run rtmlib on a single image and emit canonical annotation JSON")
    detect_image.add_argument("--input", required=True, help="单张输入图像路径")
    detect_image.add_argument("--output", help="输出 annotation JSON 路径；不填则打印到终端")
    detect_image.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"], help="rtmlib 检测模式")
    detect_image.add_argument("--device", default="cpu", help="推理设备，如 cpu 或 cuda")
    detect_image.add_argument("--backend", default="onnxruntime", help="rtmlib 后端，一般使用 onnxruntime")
    detect_image.add_argument("--solution", default="body", choices=["body", "wholebody"], help="body 表示仅身体，wholebody 表示身体+手+脸")

    convert = subparsers.add_parser("convert-openpose", help="Convert an OpenPose JSON file to canonical annotation JSON")
    convert.add_argument("--input", required=True, help="OpenPose 原始 JSON 文件路径")
    convert.add_argument("--output", help="输出 canonical annotation JSON 路径；不填则打印到终端")
    convert.add_argument("--filename", default="frame.jpg", help="写入 annotation 的 filename 字段")
    convert.add_argument("--width", type=int, help="可选图像宽度")
    convert.add_argument("--height", type=int, help="可选图像高度")

    convert_cameras = subparsers.add_parser("convert-cameras", help="Convert legacy intri/extri YAML files into the compact single-camera YAML")
    convert_cameras.add_argument("--intri", required=True, help="旧版内参 YAML 路径")
    convert_cameras.add_argument("--extri", required=True, help="旧版外参 YAML 路径")
    convert_cameras.add_argument("--output", required=True, help="新的单文件相机 YAML 输出路径")

    detect = subparsers.add_parser("detect", help="Detect one synchronized multiview frame and save canonical annotations")
    detect.add_argument("--config", required=True, help="LightMocap 配置文件路径")
    detect.add_argument("--frame", required=True, help="要处理的帧编号；3 和 000003 等价")
    detect.add_argument("--output", help="输出根目录；默认读取 config.data.output")

    triangulate = subparsers.add_parser("triangulate", help="Triangulate one frame from saved multiview annotations")
    triangulate.add_argument("--config", required=True, help="LightMocap 配置文件路径")
    triangulate.add_argument("--frame", required=True, help="要处理的帧编号；3 和 000003 等价")
    triangulate.add_argument("--output", help="输出根目录；默认读取 config.data.output")
    triangulate.add_argument("--annots", help="annotation 根目录；默认使用 <output>/annots")

    fit = subparsers.add_parser("fit", help="Fit SMPL-X for one frame from saved annotations")
    fit.add_argument("--config", required=True, help="LightMocap 配置文件路径")
    fit.add_argument("--frame", required=True, help="要处理的帧编号；3 和 000003 等价")
    fit.add_argument("--output", help="输出根目录；默认读取 config.data.output")
    fit.add_argument("--annots", help="annotation 根目录；默认使用 <output>/annots")
    fit.add_argument("--render-views", type=int, default=0, help="导出前多少个相机视角的渲染图；0 表示不导出")

    run = subparsers.add_parser("run", help="Run detect -> triangulate -> fit for one synchronized multiview frame")
    run.add_argument("--config", required=True, help="LightMocap 配置文件路径")
    run.add_argument("--frame", required=True, help="要处理的帧编号；3 和 000003 等价")
    run.add_argument("--output", help="输出根目录；默认读取 config.data.output")
    run.add_argument("--render-views", type=int, default=0, help="导出前多少个相机视角的渲染图；0 表示不导出")

    return parser


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if hasattr(config.cameras, "path"):
        cameras = CameraSet.from_yaml(config.cameras.path)
    else:
        cameras = CameraSet.from_yaml(config.cameras.intri, config.cameras.extri)
    dataset = MultiViewImageSequence(config.data.images, cameras)
    return _dump_json(
        {
            "mode": config.mode.type,
            "camera_summary": cameras.summary(),
            "dataset_summary": dataset.summary(),
        },
        output=None,
    )


def _cmd_detect_image(args: argparse.Namespace) -> int:
    detector = RTMLibDetector(mode=args.mode, device=args.device, backend=args.backend, solution=args.solution)
    annotation = detector.detect_annotation(_read_image(args.input), Path(args.input).name)
    return _dump_json(annotation, output=args.output)


def _cmd_convert_openpose(args: argparse.Namespace) -> int:
    payload = read_json(args.input)
    annotation = openpose_people_to_annotation(payload, args.filename, width=args.width, height=args.height)
    return _dump_json(annotation, output=args.output)


def _cmd_convert_cameras(args: argparse.Namespace) -> int:
    cameras = CameraSet.from_yaml(args.intri, args.extri)
    output_path = cameras.to_yaml(args.output)
    return _dump_json(
        {
            "camera_yaml": str(output_path.resolve()),
            "num_cameras": len(cameras),
            "camera_names": cameras.names,
        },
        output=None,
    )


def _cmd_detect(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    pipeline = MoCapPipeline.from_config(config)
    frame_name = normalize_frame_name(args.frame)
    output_root = _output_root(config, args.output)
    layout = output_layout(output_root)
    image_paths = build_frame_image_paths(pipeline.config.data.images, pipeline.camera.names, frame_name)
    annotations = {}
    for camera_name, image_path in image_paths.items():
        annotations[camera_name] = pipeline.detector.detect_annotation(_read_image(image_path), image_path.name)
    save_multiview_annotations(annotations, layout.annots, frame_name)
    return _dump_json(
        {
            "frame": frame_name,
            "output": str(output_root.resolve()),
            "annotation_root": str(layout.annots.resolve()),
            "output_layout": _layout_summary(output_root),
            "camera_names": pipeline.camera.names,
            "num_cameras": len(pipeline.camera.names),
        },
        output=None,
    )


def _cmd_triangulate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    pipeline = MoCapPipeline.from_config(config)
    frame_name = normalize_frame_name(args.frame)
    output_root = _output_root(config, args.output)
    layout = output_layout(output_root)
    annotation_root = _annotation_root(args, config)
    annotations = load_multiview_annotations(annotation_root, pipeline.camera.names, frame_name)
    keypoints3d = pipeline.triangulate_annotations(annotations, camera_names=pipeline.camera.names, mode=pipeline.detector.annotation_mode)
    output_path = layout.keypoints3d / f"{frame_name}.json"
    write_json(output_path, {"frame": frame_name, "keypoints3d": keypoints3d.tolist()})
    return _dump_json(
        {
            "frame": frame_name,
            "output": str(output_root.resolve()),
            "annotation_root": str(annotation_root.resolve()),
            "keypoints3d_path": str(output_path.resolve()),
            "output_layout": _layout_summary(output_root),
            "shape": list(keypoints3d.shape),
        },
        output=None,
    )


def _cmd_fit(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    pipeline = MoCapPipeline.from_config(config)
    frame_name = normalize_frame_name(args.frame)
    output_root = _output_root(config, args.output)
    annotation_root = _annotation_root(args, config)
    annotations = load_multiview_annotations(annotation_root, pipeline.camera.names, frame_name)
    keypoints3d = pipeline.triangulate_annotations(annotations, camera_names=pipeline.camera.names, mode=pipeline.detector.annotation_mode)
    params = pipeline.fit_keypoints3d(keypoints3d, annotations=annotations, camera_names=pipeline.camera.names, mode=pipeline.detector.annotation_mode)
    vertices = pipeline.body_model(return_verts=True, return_tensor=False, **params)
    result = {
        "frame_id": int(frame_name),
        "annotations": annotations,
        "keypoints3d": keypoints3d,
        "smplx_params": params,
        "vertices": vertices,
    }
    save_frame_result(output_root, frame_name, result)
    rendered: list[str] = []
    if args.render_views > 0:
        image_paths = build_frame_image_paths(pipeline.config.data.images, pipeline.camera.names, frame_name)
        rendered = render_result_views(
            output_root,
            frame_name,
            image_paths,
            pipeline.camera,
            vertices[0],
            pipeline.body_model.faces,
            max_views=args.render_views,
        )
    return _dump_json(
        {
            "frame": frame_name,
            "output": str(output_root.resolve()),
            "annotation_root": str(annotation_root.resolve()),
            "output_layout": _layout_summary(output_root),
            "render_files": rendered,
        },
        output=None,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    pipeline = MoCapPipeline.from_config(config)
    frame_name = normalize_frame_name(args.frame)
    output_root = _output_root(config, args.output)
    layout = output_layout(output_root)
    image_paths = build_frame_image_paths(pipeline.config.data.images, pipeline.camera.names, frame_name)
    result = pipeline.process_frame(image_paths, frame_id=int(frame_name))
    save_multiview_annotations(result["annotations"], layout.annots, frame_name)
    save_frame_result(output_root, frame_name, result)
    rendered: list[str] = []
    if args.render_views > 0:
        rendered = render_result_views(
            output_root,
            frame_name,
            image_paths,
            pipeline.camera,
            result["vertices"][0],
            pipeline.body_model.faces,
            max_views=args.render_views,
        )
    return _dump_json(
        {
            "frame": frame_name,
            "output": str(output_root.resolve()),
            "output_layout": _layout_summary(output_root),
            "render_files": rendered,
        },
        output=None,
    )


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "validate": _cmd_validate,
    "detect-image": _cmd_detect_image,
    "convert-openpose": _cmd_convert_openpose,
    "convert-cameras": _cmd_convert_cameras,
    "detect": _cmd_detect,
    "triangulate": _cmd_triangulate,
    "fit": _cmd_fit,
    "run": _cmd_run,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
