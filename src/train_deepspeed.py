
import torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from functools import partial
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from dataset import AudioDatset,collate_fn
import time
import torch.distributed as dist
import os 
import math
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import functools
import random
from torch.optim import lr_scheduler
import logging
import json
## 宏参数准备
rank = int(os.environ['RANK'] )
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'] )
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules = [ "q_proj", "v_proj"]
)
model_path = os.environ['model_path']
train_data_path = os.environ['train_data_path']
eval_data_path = os.environ['eval_data_path']
device_type = os.environ['device_type'] 
if device_type == "npu":
    import torch_npu
prompt_path = os.environ['prompt_path']
save_path = os.environ['save_path']
wav_type = os.environ['wav_type']
lr = float(os.environ['lr'] )
seed = int(os.environ['seed'] )
batch_size = int(os.environ['batch_size'] )
eval_step = int(os.environ['eval_step'] )
train_epoch = int(os.environ['train_epoch'] )
device = f"{device_type}:{local_rank}"
total_train_steps = int(os.environ['total_train_steps'] )
warmup_steps = int(os.environ['warmup_steps'] )
with open("./config/ds.json") as f:
    deepspeed_config = json.load(f)



## 单机多卡
def setup(rank,local_rank, world_size):

    if device_type == "npu":
        torch.npu.set_device(f"npu:{local_rank}")  # 绑定当前NPU
        deepspeed.init_distributed(
        backend='hccl',    # 使用Hccl后端（NPU场景）
    )
    elif device_type =="cuda":
        torch.cuda.set_device(f"gpu:{local_rank}")  # 绑定当前GPU
        deepspeed.init_distributed(
        backend='nccl',    # 使用NCCL后端（GPU场景）
    )
setup(rank,local_rank,world_size)
## 日志


## model 
processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)


# if dist.get_rank() == 0:
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path)
model = get_peft_model(model, peft_config)
model = model.npu(local_rank)
model.print_trainable_parameters()
parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, _, _, _ = deepspeed.initialize(
    model=model, model_parameters=parameters, config=deepspeed_config
)


train_dataset = AudioDatset(train_data_path,prompt_path,wav_type)
sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader  = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,collate_fn=partial(collate_fn,processor=processor),sampler=sampler)

eval_dataset = AudioDatset(eval_data_path,prompt_path,wav_type)
sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
eval_dataloader  = torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,collate_fn=partial(collate_fn,processor=processor),sampler=sampler)

def compute_acc(logits,labels):
    _,labels_len = labels.shape
    preds = torch.argmax(logits,dim=-1)
    labels_indices = labels != -100 
    acc = torch.sum(preds[:,-labels_len-1:-1][labels_indices] == labels[labels_indices]).float() /torch.sum(labels_indices).float()
    return acc
## train 
best_eval_acc = -math.inf
best_eval_loss = math.inf
train_bar = tqdm(train_dataloader)
eval_bar = tqdm(eval_dataloader)
for epoch in range(train_epoch):
    for train_step,batch in enumerate(train_bar):
        # train 
        model_engine.train()
        batch.to(device)
        outputs = model_engine(**batch)
        loss = outputs.loss
        acc = compute_acc(outputs["logits"],batch["labels"])
        train_bar.set_description(f"[Train] rank:{local_rank}, loss:{loss:0.2}, acc:{acc:0.2} ")
        model_engine.backward(loss)
        model_engine.step()
        if train_step + 1 % eval_step == 0:
            # eval 
            eval_acc = 0
            eval_loss = 0
            with torch.no_grad():
                for eval_step,batch in enumerate(eval_bar):
                    model_engine.eval()
                    batch.to(device)
                    outputs = model_engine(**batch)
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
            if dist.get_rank() == 0:
                logger.info(f"[Epoch {epoch} ] Eval:loss {eval_loss} acc {eval_acc}")
            # saving
            if best_eval_loss > eval_loss and dist.get_rank() == 0:
                    logger.info(f"[Saving] Better current loss {eval_loss} :{save_path+'/'+time.strftime('%H-%M',time.localtime())}")
                    best_eval_loss = eval_loss
                    # model.module.save_pretrained(save_path+"/"+time.strftime("%H-%M",time.localtime()))



