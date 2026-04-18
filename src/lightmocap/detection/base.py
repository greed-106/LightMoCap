from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        raise NotImplementedError

    def detect_annotation(
        self,
        image: np.ndarray,
        filename: str,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
