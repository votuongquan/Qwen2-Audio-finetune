
import torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from functools import partial
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from .dataset import AudioDatset,collate_fn_qwen2audio
import time
import torch.distributed as dist
import os 
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import functools
import random
import deepspeed
from torch.optim import lr_scheduler
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.init_process import setup_ddp,setup_deepspeed
import os
from peft import LoraConfig
from utils.functions import compute_acc
from deepspeed.utils import RepeatingLoader

def train_deepspeed(cfg):
    ## 宏参数准备
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f"{cfg.env.device_type}:{local_rank}"
    # init
    set_seed(cfg.train.seed)
    setup_deepspeed(cfg.env.device_type)
    dist.barrier()
    if local_rank == 0:
        os.mkdir(cfg.env.save_path)
        logger = set_logger(cfg.env.save_path)
    dist.barrier()
    

    # model
    processor = AutoProcessor.from_pretrained(cfg.env.model_path,trust_remote_code=True)
    peft_cfg = dict(cfg.peft)
    peft_cfg["target_modules"] = list(peft_cfg["target_modules"])
    peft_cfg = LoraConfig(**peft_cfg)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.env.model_path,trust_remote_code=True)
    model = get_peft_model(model, peft_cfg)
    model.to(device)
    parameters = filter(lambda x:x.requires_grad,model.parameters())
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
    config=cfg.train.deepspeed_config,
    model=model,
    model_parameters=parameters
)
    model.print_trainable_parameters()
    
    train_dataset = AudioDatset(cfg.data.train_data_path,cfg.data.prompt_path,cfg.data.wav_type)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader  = torch.utils.data.DataLoader(train_dataset,batch_size=cfg.train.batch_size,collate_fn=partial(collate_fn_qwen2audio,processor=processor),sampler=sampler)

    eval_dataset = AudioDatset(cfg.data.eval_data_path,cfg.data.prompt_path,cfg.data.wav_type)
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader  = torch.utils.data.DataLoader(eval_dataset,batch_size=cfg.train.batch_size,collate_fn=partial(collate_fn_qwen2audio,processor=processor),sampler=sampler)


    ## train 
    best_eval_acc = -math.inf
    best_eval_loss = math.inf
    for epoch in range(cfg.train.train_epoch):
    # 只有 rank 0 显示训练进度条
        if dist.get_rank() == 0:
            train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}")
        else:
            train_bar = train_dataloader  # 非主进程不显示进度条
        
        for train_step, batch in enumerate(train_bar):
            # 训练过程
            model_engine.train()
            batch.to(device)
            outputs = model_engine(** batch)
            loss = outputs.loss
            acc = compute_acc(outputs["logits"], batch["labels"])
            
            # 只有 rank 0 更新训练进度条描述
            if dist.get_rank() == 0:
                train_bar.set_description(f"[Train] epoch:{epoch} rank:{local_rank}, loss:{loss:0.2}, acc:{acc:0.2}")
            
            model_engine.backward(loss)
            model_engine.step()
            scheduler.step()
            
            if (train_step + 1) % cfg.train.eval_step == 0:
                # 评估过程
                eval_acc = 0.0
                eval_loss = 0.0
                
                # 只有 rank 0 显示评估进度条
                if dist.get_rank() == 0:
                    eval_bar = tqdm(eval_dataloader, desc="[Eval]")
                else:
                    eval_bar = eval_dataloader  # 非主进程不显示进度条
                
                with torch.no_grad():
                    for eval_step, batch in enumerate(eval_bar):
                        model_engine.eval()
                        batch.to(device)
                        outputs = model_engine(**batch)
                        loss = outputs.loss
                        acc = compute_acc(outputs["logits"], batch["labels"])
                        eval_loss += loss.item()
                        eval_acc += acc.item()
                        
                        # 只有 rank 0 更新评估进度条描述
                        if dist.get_rank() == 0:
                            eval_bar.set_description(f"[Eval] rank:{local_rank}, loss:{loss:0.2}, acc:{acc:0.2}")
                
                # 计算平均评估指标（避免除零错误）
                total_eval_steps = eval_step + 1  # 从0开始计数，需+1
                eval_acc = eval_acc / total_eval_steps
                eval_loss = eval_loss / total_eval_steps
                
                # 多进程同步评估结果
                eval_loss_tensor = torch.tensor(eval_loss, device=device)
                eval_acc_tensor = torch.tensor(eval_acc, device=device)
                dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_acc_tensor, op=dist.ReduceOp.SUM)
                eval_acc = eval_acc_tensor.item() / world_size
                eval_loss = eval_loss_tensor.item() / world_size
                
                # 只有 rank 0 打印评估日志和保存模型
                if dist.get_rank() == 0:
                    logger.info(f"[Epoch {epoch} ] Eval: loss {eval_loss:.4f} acc {eval_acc:.4f}")
                    
                    # 保存最优模型
                    if best_eval_loss > eval_loss:
                        save_time = time.strftime('%H-%M', time.localtime())
                        save_path = f"{cfg.env.save_path}/{save_time}"
                        logger.info(f"[Saving] Better current loss {eval_loss:.4f} : {save_path}")
                        best_eval_loss = eval_loss
                        model_engine.save_pretrained(save_path)


