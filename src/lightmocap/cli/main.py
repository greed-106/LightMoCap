from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import cv2
from omegaconf import DictConfig, OmegaConf

from lightmocap.data.camera import CameraSet
from lightmocap.data.dataset import MultiViewImageSequence
from lightmocap.data.io import read_json, write_json
from lightmocap.detection.adapter import openpose_people_to_annotation
from lightmocap.detection.rtmlib import RTMLibDetector
from lightmocap.models import SMPLXLayer
from lightmocap.pipeline.mocap import MoCapPipeline
from lightmocap.viz import RenderOptions
from lightmocap.workflow.frame import (
    build_frame_image_paths,
    infer_single_saved_smplx_frame,
    load_frame_smplx_params,
    load_multiview_annotations,
    normalize_frame_name,
    output_layout,
    render_smplx_result_views,
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
        "renders": str(layout.renders.resolve()),
    }


def _load_cameras(config: DictConfig) -> CameraSet:
    if hasattr(config.cameras, "path"):
        return CameraSet.from_yaml(config.cameras.path)
    return CameraSet.from_yaml(config.cameras.intri, config.cameras.extri)


def _load_body_model(config: DictConfig) -> SMPLXLayer:
    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        raise RuntimeError("Rendering requires a `model` section in the config.")
    detector_cfg = getattr(config, "detector", None)
    return SMPLXLayer(
        model_path=model_cfg.model_path,
        gender=getattr(model_cfg, "gender", "neutral"),
        device=getattr(model_cfg, "device", getattr(detector_cfg, "device", "cpu") if detector_cfg is not None else "cpu"),
    )


def _resolve_saved_render_frame(args: argparse.Namespace, output_root: Path) -> str:
    if getattr(args, "frame", None) is not None:
        return normalize_frame_name(args.frame)
    return infer_single_saved_smplx_frame(output_root)


def _parse_color(value: str | list[int] | tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, str) and isinstance(value, Sequence):
        if len(value) != 3:
            raise ValueError(f"Expected 3 color channels, got {value}")
        return tuple(int(channel) for channel in value)
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    if text.startswith("#"):
        text = text[1:]
        if len(text) != 6:
            raise ValueError(f"Expected 6-digit hex color, got {value}")
        return tuple(int(text[index : index + 2], 16) for index in range(0, 6, 2))
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected color like `255,255,255` or `#ffffff`, got {value}")
    return tuple(int(part) for part in parts)


def _cfg_select(config: DictConfig | None, key: str):
    return None if config is None else OmegaConf.select(config, key)


def _resolve_render_options(
    config: DictConfig,
    args: argparse.Namespace,
    *,
    default_color: tuple[int, int, int],
    default_alpha: float,
    default_edge_color: tuple[int, int, int] | None,
    default_render_device: str,
) -> RenderOptions:
    render_cfg = getattr(config, "render", None)

    mesh_color_arg = getattr(args, "mesh_color", None)
    if mesh_color_arg is None:
        mesh_color_cfg = _cfg_select(render_cfg, "mesh_color")
        color = _parse_color(mesh_color_cfg) if mesh_color_cfg is not None else default_color
    else:
        parsed_color = _parse_color(mesh_color_arg)
        color = default_color if parsed_color is None else parsed_color

    alpha = getattr(args, "alpha", None)
    if alpha is None:
        alpha = _cfg_select(render_cfg, "alpha")
    if alpha is None:
        alpha = default_alpha

    edge_color_arg = getattr(args, "edge_color", None)
    if edge_color_arg is None:
        edge_color_cfg = _cfg_select(render_cfg, "edge_color")
        edge_color = _parse_color(edge_color_cfg) if edge_color_cfg is not None else default_edge_color
    else:
        edge_color = _parse_color(edge_color_arg)

    render_device = getattr(args, "render_device", None) or _cfg_select(render_cfg, "render_device") or default_render_device
    metallic = getattr(args, "metallic", None)
    if metallic is None:
        metallic = _cfg_select(render_cfg, "metallic_factor")
    if metallic is None:
        metallic = 0.05
    roughness = getattr(args, "roughness", None)
    if roughness is None:
        roughness = _cfg_select(render_cfg, "roughness_factor")
    if roughness is None:
        roughness = 0.65

    return RenderOptions(
        color=color,
        alpha=float(alpha),
        edge_color=edge_color,
        render_device=str(render_device),
        metallic_factor=float(metallic),
        roughness_factor=float(roughness),
    )


def _add_render_style_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--render-device", choices=["auto", "gpu", "cpu"], help="pyrender 设备偏好；gpu 会优先设置 EGL，cpu 会优先设置 OSMesa")
    parser.add_argument("--mesh-color", help="网格颜色，支持 `255,255,255` 或 `#ffffff`")
    parser.add_argument("--alpha", type=float, help="网格透明度；1.0 表示完全不透明")
    parser.add_argument("--edge-color", help="边线颜色，支持 `255,255,255`、`#ffffff` 或 `none`")
    parser.add_argument("--metallic", type=float, help="pyrender 材质 metallic 因子")
    parser.add_argument("--roughness", type=float, help="pyrender 材质 roughness 因子")


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
    _add_render_style_args(fit)

    run = subparsers.add_parser("run", help="Run detect -> triangulate -> fit for one synchronized multiview frame")
    run.add_argument("--config", required=True, help="LightMocap 配置文件路径")
    run.add_argument("--frame", required=True, help="要处理的帧编号；3 和 000003 等价")
    run.add_argument("--output", help="输出根目录；默认读取 config.data.output")
    run.add_argument("--render-views", type=int, default=0, help="导出前多少个相机视角的渲染图；0 表示不导出")
    _add_render_style_args(run)

    render = subparsers.add_parser("render", help="Render saved SMPL-X parameters onto the source images")
    render.add_argument("--config", required=True, help="LightMocap 配置文件路径")
    render.add_argument("--frame", help="要渲染的帧编号；不填时会在 `<output>/smplx` 中自动推断单个结果文件")
    render.add_argument("--output", help="结果根目录；默认读取 config.data.output")
    render.add_argument("--render-views", type=int, help="导出前多少个相机视角；不填则渲染全部")
    _add_render_style_args(render)

    return parser


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    cameras = _load_cameras(config)
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
    result = {
        "frame_id": int(frame_name),
        "annotations": annotations,
        "keypoints3d": keypoints3d,
        "smplx_params": params,
    }
    save_frame_result(output_root, frame_name, result)
    rendered: list[str] = []
    if args.render_views > 0:
        image_paths = build_frame_image_paths(pipeline.config.data.images, pipeline.camera.names, frame_name)
        render_options = _resolve_render_options(
            config,
            args,
            default_color=(245, 245, 245),
            default_alpha=1.0,
            default_edge_color=None,
            default_render_device="gpu",
        )
        rendered = render_smplx_result_views(
            output_root,
            frame_name,
            image_paths,
            pipeline.camera,
            pipeline.body_model,
            params,
            max_views=args.render_views,
            render_options=render_options,
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
    rendered: list[str] = []
    if args.render_views > 0:
        render_options = _resolve_render_options(
            config,
            args,
            default_color=(245, 245, 245),
            default_alpha=1.0,
            default_edge_color=None,
            default_render_device="gpu",
        )
        rendered = render_smplx_result_views(
            output_root,
            frame_name,
            image_paths,
            pipeline.camera,
            pipeline.body_model,
            result["smplx_params"],
            images=result.get("images"),
            max_views=args.render_views,
            render_options=render_options,
        )
    save_frame_result(output_root, frame_name, result)
    return _dump_json(
        {
            "frame": frame_name,
            "output": str(output_root.resolve()),
            "output_layout": _layout_summary(output_root),
            "render_files": rendered,
        },
        output=None,
    )


def _cmd_render(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    cameras = _load_cameras(config)
    body_model = _load_body_model(config)
    output_root = _output_root(config, args.output)
    frame_name = _resolve_saved_render_frame(args, output_root)
    image_paths = build_frame_image_paths(config.data.images, cameras.names, frame_name)
    smplx_params = load_frame_smplx_params(output_root, frame_name)
    render_options = _resolve_render_options(
        config,
        args,
        default_color=(245, 245, 245),
        default_alpha=1.0,
        default_edge_color=None,
        default_render_device="gpu",
    )
    rendered = render_smplx_result_views(
        output_root,
        frame_name,
        image_paths,
        cameras,
        body_model,
        smplx_params,
        max_views=args.render_views,
        render_options=render_options,
    )
    return _dump_json(
        {
            "frame": frame_name,
            "output": str(output_root.resolve()),
            "output_layout": _layout_summary(output_root),
            "render_device": render_options.render_device,
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
    "render": _cmd_render,
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
