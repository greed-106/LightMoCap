from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .camera import CameraSet, natural_sort_key


SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _normalize_image_ext(image_ext: str | None) -> str | None:
    if image_ext is None:
        return None
    image_ext = str(image_ext).strip()
    if not image_ext:
        return None
    return image_ext if image_ext.startswith(".") else f".{image_ext}"


@dataclass(frozen=True)
class FrameRecord:
    frame_name: str
    images: dict[str, Path]


class MultiViewImageSequence:
    def __init__(self, root: str | Path, cameras: CameraSet | None = None, image_ext: str | None = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.cameras = cameras
        self.image_ext = _normalize_image_ext(image_ext)
        self.single_image_mode = False
        self._single_image_paths: dict[str, Path] = {}
        self._frame_image_paths: dict[str, dict[str, Path]] = {}
        self.camera_names = cameras.names if cameras is not None else self._discover_camera_dirs()
        self._frame_names = self._collect_shared_frames()

    def _discover_camera_dirs(self) -> list[str]:
        return sorted(
            (path.name for path in self.root.iterdir() if path.is_dir() and path.name.startswith("Camera_")),
            key=natural_sort_key,
        )

    def _collect_camera_files(self, camera_dir: Path) -> list[Path]:
        if self.image_ext is not None:
            return sorted(camera_dir.glob(f"*{self.image_ext}"), key=lambda path: natural_sort_key(path.name))
        files: dict[str, Path] = {}
        for ext in SUPPORTED_IMAGE_EXTS:
            for path in camera_dir.glob(f"*{ext}"):
                files.setdefault(path.name, path)
            upper_ext = ext.upper()
            for path in camera_dir.glob(f"*{upper_ext}"):
                files.setdefault(path.name, path)
        return sorted(files.values(), key=lambda path: natural_sort_key(path.name))

    def _collect_shared_frames(self) -> list[str]:
        frame_sets: list[set[str]] = []
        per_camera_files: dict[str, list[Path]] = {}
        per_camera_frame_paths: dict[str, dict[str, Path]] = {}
        for camera_name in self.camera_names:
            camera_dir = self.root / camera_name
            if not camera_dir.exists():
                raise FileNotFoundError(camera_dir)
            files = self._collect_camera_files(camera_dir)
            per_camera_files[camera_name] = files
            per_camera_frame_paths[camera_name] = {path.stem: path for path in files}
            frames = {path.stem for path in files}
            if not frames:
                raise ValueError(f"No frames found in {camera_dir}")
            frame_sets.append(frames)
        shared = sorted(set.intersection(*frame_sets))
        if shared:
            self._frame_image_paths = {
                frame_name: {camera_name: per_camera_frame_paths[camera_name][frame_name] for camera_name in self.camera_names}
                for frame_name in shared
            }
            return shared
        if all(len(files) == 1 for files in per_camera_files.values()):
            self.single_image_mode = True
            self._single_image_paths = {camera_name: files[0] for camera_name, files in per_camera_files.items()}
            return ["000000"]
        raise ValueError(
            "No shared frames found across cameras. Expected synchronized filenames like 000003.jpg across all views, "
            "or exactly one image per camera directory for single-frame multiview mode."
        )

    @property
    def frame_names(self) -> list[str]:
        return self._frame_names

    def __len__(self) -> int:
        return len(self._frame_names)

    def frame_paths(self, index: int) -> dict[str, Path]:
        if self.single_image_mode:
            return self._single_image_paths.copy()
        frame_name = self._frame_names[index]
        return self._frame_image_paths[frame_name].copy()

    def frame(self, index: int) -> FrameRecord:
        return FrameRecord(frame_name=self._frame_names[index], images=self.frame_paths(index))

    def summary(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "layout": "single_image_per_camera" if self.single_image_mode else "shared_frame_sequence",
            "num_cameras": len(self.camera_names),
            "camera_names": self.camera_names,
            "image_ext": self.image_ext,
            "num_frames": len(self),
            "first_frame": self._frame_names[0],
            "last_frame": self._frame_names[-1],
        }
