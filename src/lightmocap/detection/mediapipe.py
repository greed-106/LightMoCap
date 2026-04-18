from __future__ import annotations

from .base import BaseDetector


class MediaPipeDetector(BaseDetector):
    name = "mediapipe"

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def detect(self, image):  # pragma: no cover - runtime integration later
        raise NotImplementedError("MediaPipe integration is queued after YOLOv8/rtmlib MVP.")
