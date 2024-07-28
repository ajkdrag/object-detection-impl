from pathlib import Path

import hydra
import lightning as L
import structlog
import torch
from omegaconf import DictConfig, OmegaConf

from object_detection_impl.utils.misc import log_useful_info, set_seed
from object_detection_impl.utils.registry import load_obj

log = structlog.get_logger()


def _predict(
    model: L.LightningModule,
    dm: L.LightningDataModule,
    cfg: DictConfig,
):
    dm.prepare_data()
    dm.setup(stage="predict")
    dl = dm.predict_dataloader()

    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(dl):
            images, preds = model.predict_step(inputs, idx)
            yield dm.visualize_batch(images, preds)


def _run(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)
    log.info("**** Running predict func ****")
    root_dir = Path(cfg.general.root_dir).joinpath(
        cfg.general.exp_name,
    )

    model = load_obj(cfg.training.lit_model.class_name).load_from_checkpoint(
        root_dir.joinpath(cfg.predict.checkpoint),
        cfg=cfg,
    )

    dm = load_obj(cfg.datamodule.class_name)(cfg=cfg)
    return _predict(model, dm, cfg)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def run(cfg: DictConfig) -> None:
    Path("logs").mkdir(exist_ok=True)
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        log_useful_info()
    _run(cfg)


if __name__ == "__main__":
    run()
