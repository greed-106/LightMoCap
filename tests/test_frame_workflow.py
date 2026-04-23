from pathlib import Path

import numpy as np
import torch

from lightmocap.workflow.frame import save_frame_result


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
