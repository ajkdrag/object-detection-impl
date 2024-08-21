from pathlib import Path

import hydra
import lightning as L
import structlog
import torch
from omegaconf import DictConfig, OmegaConf

from visionlab.utils.misc import log_useful_info, set_seed
from visionlab.utils.registry import load_obj

log = structlog.get_logger()
torch.set_float32_matmul_precision("medium")


def _run(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)
    log.info("**** Running train func ****")

    exp_name = cfg.general.exp_name
    root_dir = Path(cfg.general.root_dir).joinpath(exp_name)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks = []
    for callback_name, callback in cfg.callback.items():
        if callback_name in ["model_checkpoint", "vis_dls"]:
            dirpath = root_dir.joinpath(callback.params.dirpath)
            callback.params.dirpath = dirpath.as_posix()
        callback_instance = load_obj(callback.class_name)(**callback.params)
        callbacks.append(callback_instance)

    trainer = L.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.training.trainer_params,
    )
    model = load_obj(cfg.training.lit_model.class_name)(cfg=cfg)
    dm = load_obj(cfg.datamodule.class_name)(cfg=cfg)
    trainer_kwargs = {}
    if cfg.training.get("resume") is not None:
        ckpt_dir = Path(cfg.callback.model_checkpoint.params.dirpath)
        ckpt_path = ckpt_dir.joinpath(cfg.training.resume.checkpoint).as_posix()
        log.info(f"Resuming from ckpt: {ckpt_path}")
        trainer_kwargs["ckpt_path"] = ckpt_path
    trainer.fit(model, dm, **trainer_kwargs)
    trainer.test(model, dm, ckpt_path="best")
    log.info(f"{root_dir = }")


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
