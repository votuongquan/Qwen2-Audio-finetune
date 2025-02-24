
import torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from functools import partial
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from .dataset import AudioDatset,collate_fn_qwen2audio
import time
import torch.distributed as dist
import os 
import types
import math
from src.qwen2audio_fix import _merge_input_ids_with_audio_features
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import functools
import random
from torch.optim import lr_scheduler
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.init_process import setup_ddp
import os
from peft import LoraConfig
from utils.functions import compute_acc
from utils.set_amp import set_amp

def train_ddp(cfg):
    ## 宏参数准备
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f"{cfg.env.device_type}:{local_rank}"
    # init
    set_seed(cfg.train.seed)
    setup_ddp(cfg.env.device_type)
    dist.barrier()
    scaler,autocast = set_amp(cfg.env.device_type)
    if local_rank == 0:
        os.mkdir(cfg.env.save_path)
    dist.barrier()
    # logger = set_logger(cfg.env.save_path)

    # model
    processor = AutoProcessor.from_pretrained(cfg.env.model_path,trust_remote_code=True)
    peft_cfg = dict(cfg.peft)
    peft_cfg["target_modules"] = list(peft_cfg["target_modules"])
    peft_cfg = LoraConfig(**peft_cfg)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.env.model_path,trust_remote_code=True)
    model._merge_input_ids_with_audio_features = types.MethodType(_merge_input_ids_with_audio_features, model)
    model = get_peft_model(model, peft_cfg)
    model.to(device)
    model.print_trainable_parameters()

    model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, 
        lr_lambda=lambda step: (
            min(step / cfg.train.warmup_steps, 1) if step < cfg.train.warmup_steps
            else  max(0.0, 1 - (step - cfg.train.warmup_steps) / (cfg.train.total_train_steps - cfg.train.warmup_steps))
        )
    )
    train_dataset = AudioDatset(cfg.data.train_data_path,cfg.data.prompt_path,cfg.data.wav_type)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader  = torch.utils.data.DataLoader(train_dataset,batch_size=cfg.train.batch_size,num_workers=cfg.data.num_workers,collate_fn=partial(collate_fn_qwen2audio,processor=processor),sampler=sampler,prefetch_factor=cfg.data.prefetch_factor)

    eval_dataset = AudioDatset(cfg.data.eval_data_path,cfg.data.prompt_path,cfg.data.wav_type)
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader  = torch.utils.data.DataLoader(eval_dataset,batch_size=cfg.train.batch_size,num_workers=cfg.data.num_workers,collate_fn=partial(collate_fn_qwen2audio,processor=processor),sampler=sampler,prefetch_factor=cfg.data.prefetch_factor)


    ## train 
    best_eval_acc = -math.inf
    best_eval_loss = math.inf
    for epoch in range(cfg.train.train_epoch):
        train_bar = tqdm(train_dataloader)
        model.train()
        for train_step,batch in enumerate(train_bar):
            # train 
            batch.to(device)
            with autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
            loss = outputs.loss
            acc = compute_acc(outputs["logits"],batch["labels"])
            train_bar.set_description(f"[Train] epoch:{epoch} rank:{local_rank}, loss:{loss:0.2}, acc:{acc:0.2} ")
            # scaler.scale(loss).backward()
            # if (train_step + 1) % cfg.train.grad_accumulate_step == 0 or train_step == len(train_dataloader) - 1:
            #     scaler.step(optim)
            #     scaler.update()
            #     optim.zero_grad()

            scheduler.step()
            if (train_step + 1) % cfg.train.eval_step == 0:
                # eval 
                eval_acc = 0
                eval_loss = 0
                eval_bar = tqdm(eval_dataloader)
                with torch.no_grad():
                    for eval_step,batch in enumerate(eval_bar):
                        model.eval()
                        batch.to(device)
                        outputs = model(**batch)
                        loss = outputs.loss
                        acc = compute_acc(outputs["logits"],batch["labels"])
                        eval_loss += loss
                        eval_acc += acc
                        train_bar.set_description(f"[Eval] rank:{local_rank}, loss:{loss:0.2}, acc:{acc:0.2} ")
                eval_acc = eval_acc / eval_step
                eval_loss = eval_loss/ eval_step
                dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM) 
                dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)
                eval_acc = eval_acc / world_size
                eval_loss = eval_loss / world_size
                # if dist.get_rank() == 0:
                    # logger.info(f"[Epoch {epoch} ] Eval:loss {eval_loss} acc {eval_acc}")
                # saving
                if best_eval_loss > eval_loss and dist.get_rank() == 0:
                        # logger.info(f"[Saving] Better current loss {eval_loss} :{cfg.env.save_path+'/'+time.strftime('%H-%M',time.localtime())}")
                        best_eval_loss = eval_loss
                        model.module.save_pretrained(str(cfg.env.save_path+"/"+time.strftime("%H-%M",time.localtime())))



