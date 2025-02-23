from dataclasses import dataclass, field
from typing import Optional, List
@dataclass
class PeftConfig:
    r: int = 64
    lora_alpha: int = 16
    target_modules: List = field(default_factory=lambda: [ "q_proj", "v_proj", "o_proj", "up_proj","gate_proj","down_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class TrainConfig:
    train_strategy : str = "ddp"
    deepspeed_config: str = "./config/deepspeed.json"
    seed : int = 1234
    lr: float = 1e-4
    batch_size: int = 1
    total_train_steps: int = 100000
    eval_step: int = 20
    train_epoch: int = 5
    warmup_steps: int = 1000

@dataclass
class DataConfig:
    train_data_path: str = "./data/aishell-1/asr/test"
    eval_data_path: str = "./data/aishell-1/asr/test"
    prompt_path: str = "./data/multiprompt.jsonl"
    wav_type: str = "ark"

@dataclass
class EnvConfig:
    device_type: str = "cuda" # npu gpu
    save_path: str = "./exp"
    model_path: str = ""



@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)