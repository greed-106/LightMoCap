from __future__ import annotations

from typing import Any

import numpy as np

from .adapter import rtmlib_body_to_annotation, rtmlib_wholebody_to_annotation
from .base import BaseDetector


class RTMLibDetector(BaseDetector):
    name = "rtmlib"
    """Thin adapter around rtmlib.

    `solution="body"` returns body-only canonical annotations.
    `solution="wholebody"` returns body + hands + face annotations.
    """

    def __init__(
        self,
        mode: str = "balanced",
        backend: str = "onnxruntime",
        device: str = "cpu",
        solution: str = "body",
        to_openpose: bool = True,
    ) -> None:
        self.mode = mode
        self.backend = backend
        self.device = device
        self.solution = solution
        self.to_openpose = to_openpose
        self.annotation_mode = "body25" if solution == "body" else "bodyhandface"
        if solution == "body":
            from rtmlib import Body

            self.detector = Body(
                mode=mode,
                to_openpose=to_openpose,
                backend=backend,
                device=device,
            )
        elif solution == "wholebody":
            from rtmlib import Wholebody

            self.detector = Wholebody(
                mode=mode,
                to_openpose=to_openpose,
                backend=backend,
                device=device,
            )
        else:
            raise NotImplementedError(f"Unsupported rtmlib solution: {solution}")

    def detect_raw(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = self.detector.det_model(image)
        keypoints, scores = self.detector.pose_model(image, bboxes=bboxes)
        return np.asarray(bboxes), np.asarray(keypoints), np.asarray(scores)

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        bboxes, keypoints, scores = self.detect_raw(image)
        if self.solution == "body":
            annots = rtmlib_body_to_annotation(
                keypoints=keypoints,
                scores=scores,
                filename="runtime.jpg",
                width=image.shape[1],
                height=image.shape[0],
                bboxes=bboxes,
            )
        else:
            annots = rtmlib_wholebody_to_annotation(
                keypoints=keypoints,
                scores=scores,
                filename="runtime.jpg",
                width=image.shape[1],
                height=image.shape[0],
                bboxes=bboxes,
            )
        return annots["annots"]

    def detect_annotation(
        self,
        image: np.ndarray,
        filename: str,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        bboxes, keypoints, scores = self.detect_raw(image)
        if self.solution == "body":
            return rtmlib_body_to_annotation(
                keypoints=keypoints,
                scores=scores,
                filename=filename,
                width=image.shape[1] if width is None else width,
                height=image.shape[0] if height is None else height,
                bboxes=bboxes,
            )
        return rtmlib_wholebody_to_annotation(
            keypoints=keypoints,
            scores=scores,
            filename=filename,
            width=image.shape[1] if width is None else width,
            height=image.shape[0] if height is None else height,
            bboxes=bboxes,
        )
