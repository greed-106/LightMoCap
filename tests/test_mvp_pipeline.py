from omegaconf import OmegaConf

from lightmocap.core.fitting import FittingConfig
from lightmocap.pipeline.mocap import MoCapPipeline


def test_mvp_pipeline_can_validate_coreview_inputs():
    pipeline = MoCapPipeline.from_yaml(
        "/home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml"
    )
    summary = pipeline.validate_inputs()
    assert summary["mode"] == "mvsp"
    assert summary["camera_summary"]["num_cameras"] == 23
    assert summary["dataset_summary"]["num_frames"] > 0


def test_actor5_loss_weights_match_fitting_defaults():
    config = OmegaConf.load("/home/ymj/code/python/EasyMocap/lightmocap/configs/actor5.yaml")
    assert OmegaConf.to_container(config.fitting.loss_weights, resolve=True) == FittingConfig().weight_loss
