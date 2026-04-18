from __future__ import annotations

from lightmocap.data.camera import CameraSet
from lightmocap.data.dataset import MultiViewImageSequence


class SingleMode:
    def __init__(self, cameras: CameraSet, dataset: MultiViewImageSequence) -> None:
        self.cameras = cameras
        self.dataset = dataset

    def validate(self) -> dict[str, object]:
        return {
            "mode": "mvsp",
            "camera_summary": self.cameras.summary(),
            "dataset_summary": self.dataset.summary(),
        }
