from pathlib import Path

import numpy as np
import torch

from lightmocap.workflow.frame import build_frame_image_paths, build_frame_mask_paths, save_frame_result


def test_save_frame_result_accepts_torch_params_and_skips_vertices_dir(tmp_path: Path):
    output_root = tmp_path / "frame_output"
    result = {
        "keypoints3d": np.zeros((1, 25, 4), dtype=np.float32),
        "smplx_params": {
            "poses": torch.zeros((1, 87), dtype=torch.float32),
            "shapes": torch.zeros((1, 10), dtype=torch.float32),
            "Rh": torch.zeros((1, 3), dtype=torch.float32),
            "Th": torch.zeros((1, 3), dtype=torch.float32),
            "expression": torch.zeros((1, 10), dtype=torch.float32),
        },
    }

    save_frame_result(output_root, 1, result)

    assert (output_root / "keypoints3d" / "000001.json").exists()
    assert (output_root / "smplx" / "000001.json").exists()
    assert not (output_root / "vertices").exists()


def test_build_frame_mask_paths_mirrors_image_layout(tmp_path: Path):
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_path = image_root / "cam01" / "000001.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.touch()
    image_paths = {"cam01": image_path}

    mask_paths = build_frame_mask_paths(mask_root, image_root, image_paths)

    assert mask_paths == {"cam01": mask_root / "cam01" / "000001.jpg"}


def test_build_frame_image_paths_detects_png_frames(tmp_path: Path):
    image_root = tmp_path / "images"
    image_path = image_root / "cam01" / "000001.png"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    image_paths = build_frame_image_paths(image_root, ["cam01"], "000001")

    assert image_paths == {"cam01": image_path}


def test_build_frame_image_paths_accepts_three_digit_png_frames(tmp_path: Path):
    image_root = tmp_path / "images"
    image_path = image_root / "cam01" / "001.png"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    assert build_frame_image_paths(image_root, ["cam01"], "1") == {"cam01": image_path}
    assert build_frame_image_paths(image_root, ["cam01"], "000001") == {"cam01": image_path}
