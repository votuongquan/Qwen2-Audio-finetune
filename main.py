import hydra
from omegaconf import OmegaConf
from src.train_ddp import train_ddp
from config.config import Config
import time
@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg):
    run_config = Config()
    run_config.env.save_path += f"/{time.strftime('%H-%M-%S')}"
    cfg = OmegaConf.merge(run_config, cfg)
    if cfg.train.train_strategy == "ddp":
        train_ddp(cfg)
    else:
        train_ddp(cfg)

if __name__ == "__main__":
    main_hydra()
