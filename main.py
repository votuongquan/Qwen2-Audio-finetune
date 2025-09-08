import hydra
from omegaconf import OmegaConf
from src.train_ddp import train_ddp
from src.train_deepspeed import train_deepspeed
from config.config import Config
import time
import argparse
import sys
def parse_deepspeed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args
@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg):
    run_config = Config()
    run_config.env.save_path += f"/{time.strftime('%H-%M-%S')}"
    cfg = OmegaConf.merge(run_config, cfg)
    if cfg.train.train_strategy == "ddp":
        train_ddp(cfg)
    else:
        train_deepspeed(cfg)

if __name__ == "__main__":
    deepspeed_args, remaining_args = parse_deepspeed_args()
    sys.argv = [sys.argv[0]] + remaining_args
    main_hydra()
