from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lightmocap.core.fitting import FittingConfig, SMPLXFittingPipeline
from lightmocap.core.triangulation import triangulate_multiview_points
from lightmocap.data.camera import CameraSet
from lightmocap.data.dataset import MultiViewImageSequence
from lightmocap.detection.adapter import canonical_keypoints_from_annotation
from lightmocap.detection.rtmlib import RTMLibDetector
from lightmocap.modes.single import SingleMode
from lightmocap.models.smplx import SMPLXLayer


class MoCapPipeline:
    """High-level multiview mocap pipeline.

    The current Phase 1 path is intentionally narrow and only enables MVSP:

    1. detect 2D keypoints per view
    2. triangulate canonical keypoints into 3D
    3. fit SMPL-X with 3D and optional multiview 2D refinement
    """

    def __init__(
        self,
        mode: str,
        camera: CameraSet,
        dataset: MultiViewImageSequence,
        detector=None,
        body_model: SMPLXLayer | None = None,
        fitter: SMPLXFittingPipeline | None = None,
        config: DictConfig | None = None,
    ) -> None:
        if mode == "mvsp":
            self.runner = SingleMode(camera, dataset)
        elif mode == "mvmp":
            raise NotImplementedError("MVMP mode is reserved in config but is not implemented yet.")
        else:
            raise ValueError(f"Unsupported mode.type: {mode}")
        self.mode = mode
        self.camera = camera
        self.dataset = dataset
        self.detector = detector
        self.body_model = body_model
        self.fitter = fitter
        self.config = config

    @classmethod
    def from_config(cls, config: DictConfig) -> "MoCapPipeline":
        camera_cfg = config.cameras
        if hasattr(camera_cfg, "path"):
            camera = CameraSet.from_yaml(camera_cfg.path)
        else:
            camera = CameraSet.from_yaml(camera_cfg.intri, camera_cfg.extri)
        dataset = MultiViewImageSequence(config.data.images, camera)
        detector = None
        detector_cfg = getattr(config, "detector", None)
        if detector_cfg is not None and detector_cfg.type == "rtmlib":
            detector = RTMLibDetector(
                mode=getattr(detector_cfg, "mode", "balanced"),
                backend=getattr(detector_cfg, "backend", "onnxruntime"),
                device=getattr(detector_cfg, "device", "cpu"),
                solution=getattr(detector_cfg, "solution", "body"),
            )
        body_model = None
        fitter = None
        model_cfg = getattr(config, "model", None)
        if model_cfg is not None:
            body_model = SMPLXLayer(
                model_path=model_cfg.model_path,
                gender=getattr(model_cfg, "gender", "neutral"),
                device=getattr(model_cfg, "device", getattr(detector_cfg, "device", "cpu") if detector_cfg is not None else "cpu"),
            )
            fit_cfg = getattr(config, "fitting", None)
            weight_loss = None
            if fit_cfg is not None and hasattr(fit_cfg, "loss_weights"):
                weight_loss = OmegaConf.to_container(fit_cfg.loss_weights, resolve=True)
            stages = None
            if fit_cfg is not None and hasattr(fit_cfg, "stages"):
                stages = OmegaConf.to_container(fit_cfg.stages, resolve=True)
            fitter = SMPLXFittingPipeline(
                body_model,
                FittingConfig(
                    device=str(body_model.device),
                    maxiters=getattr(fit_cfg, "maxiters", 20) if fit_cfg is not None else 20,
                    shape_maxiters=getattr(fit_cfg, "shape_maxiters", 10) if fit_cfg is not None else 10,
                    lbfgs_inner_max_iter=getattr(fit_cfg, "lbfgs_inner_max_iter", 8) if fit_cfg is not None else 8,
                    enable_k2d_refine=getattr(fit_cfg, "enable_k2d_refine", True) if fit_cfg is not None else True,
                    k3d_robust_sigma=getattr(fit_cfg, "k3d_robust_sigma", 0.05) if fit_cfg is not None else 0.05,
                    stages=stages if stages is not None else FittingConfig().stages,
                    weight_loss=weight_loss if weight_loss is not None else FittingConfig().weight_loss,
                ),
            )
        return cls(
            mode=config.mode.type,
            camera=camera,
            dataset=dataset,
            detector=detector,
            body_model=body_model,
            fitter=fitter,
            config=config,
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "MoCapPipeline":
        config = OmegaConf.load(config_path)
        return cls.from_config(config)

    def validate_inputs(self) -> dict[str, object]:
        return self.runner.validate()

    def triangulate_annotations(
        self,
        annotations: dict[str, dict],
        camera_names: list[str] | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        camera_names = self.camera.names if camera_names is None else camera_names
        if mode is None:
            mode = getattr(self.detector, "annotation_mode", "body25") if self.detector is not None else "body25"
        keypoints2d = []
        for camera_name in camera_names:
            annot = annotations[camera_name]
            keypoints2d.append(canonical_keypoints_from_annotation(annot, mode=mode).astype(np.float64))
        keypoints2d = np.stack(keypoints2d, axis=0)
        return triangulate_multiview_points(keypoints2d, self.camera, camera_names=camera_names)

    def _bboxes_from_annotations(self, annotations: dict[str, dict], camera_names: list[str]) -> np.ndarray:
        bboxes = []
        for camera_name in camera_names:
            annot = annotations[camera_name]
            if not annot["annots"]:
                bboxes.append(np.array([0.0, 0.0, 100.0, 100.0, 0.0], dtype=np.float32))
                continue
            bboxes.append(np.asarray(annot["annots"][0]["bbox"], dtype=np.float32))
        return np.stack(bboxes, axis=0)

    def fit_keypoints3d(
        self,
        keypoints3d: np.ndarray,
        annotations: dict[str, dict] | None = None,
        camera_names: list[str] | None = None,
        mode: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Fit SMPL-X from triangulated 3D keypoints.

        When multiview annotations are provided, the fitter runs an additional
        V1-style reprojection refinement stage using 2D observations and view
        projection matrices.
        """
        if self.fitter is None:
            raise RuntimeError("Fitting pipeline is not initialized")
        if keypoints3d.ndim == 2:
            keypoints3d = keypoints3d[None]
        if annotations is None:
            return self.fitter.fit(keypoints3d)
        camera_names = self.camera.names if camera_names is None else camera_names
        mode = getattr(self.detector, "annotation_mode", "body25") if mode is None else mode
        keypoints2d = np.stack(
            [canonical_keypoints_from_annotation(annotations[camera_name], mode=mode).astype(np.float32) for camera_name in camera_names],
            axis=0,
        )
        bboxes = self._bboxes_from_annotations(annotations, camera_names)
        return self.fitter.fit(
            keypoints3d,
            keypoints2d=keypoints2d[None],
            bboxes=bboxes[None],
            projection_matrices=self.camera.projection_matrices(camera_names),
        )

    def _fit_keypoints3d_tensor(
        self,
        keypoints3d: np.ndarray,
        annotations: dict[str, dict] | None = None,
        camera_names: list[str] | None = None,
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.fitter is None:
            raise RuntimeError("Fitting pipeline is not initialized")
        if keypoints3d.ndim == 2:
            keypoints3d = keypoints3d[None]
        if annotations is None:
            return self.fitter.fit_tensor(keypoints3d)
        camera_names = self.camera.names if camera_names is None else camera_names
        mode = getattr(self.detector, "annotation_mode", "body25") if mode is None else mode
        keypoints2d = np.stack(
            [canonical_keypoints_from_annotation(annotations[camera_name], mode=mode).astype(np.float32) for camera_name in camera_names],
            axis=0,
        )
        bboxes = self._bboxes_from_annotations(annotations, camera_names)
        return self.fitter.fit_tensor(
            keypoints3d,
            keypoints2d=keypoints2d[None],
            bboxes=bboxes[None],
            projection_matrices=self.camera.projection_matrices(camera_names),
        )

    def process_frame(self, images: dict[str, str | Path | np.ndarray], frame_id: int | None = None) -> dict[str, object]:
        """Run detect -> triangulate -> fit for a single synchronized frame."""
        if self.detector is None:
            raise RuntimeError("Detector is not initialized")
        loaded_images: dict[str, np.ndarray] = {}
        annotations: dict[str, dict] = {}
        for camera_name, image_value in images.items():
            if isinstance(image_value, np.ndarray):
                image = image_value
                filename = f"{camera_name}/{frame_id if frame_id is not None else 'frame'}.jpg"
            else:
                image_path = Path(image_value)
                image = cv2.imread(str(image_path))
                if image is None:
                    raise FileNotFoundError(image_path)
                filename = str(image_path.name)
            loaded_images[camera_name] = image
            annotations[camera_name] = self.detector.detect_annotation(image, filename)
        camera_names = list(images.keys())
        annotation_mode = getattr(self.detector, "annotation_mode", "body25")
        keypoints3d = self.triangulate_annotations(annotations, camera_names=camera_names, mode=annotation_mode)
        smplx_params = self._fit_keypoints3d_tensor(keypoints3d, annotations=annotations, camera_names=camera_names, mode=annotation_mode)
        return {
            "frame_id": frame_id,
            "images": loaded_images,
            "annotations": annotations,
            "keypoints3d": keypoints3d,
            "smplx_params": smplx_params,
        }
