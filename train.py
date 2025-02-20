
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import torch_npu
import random

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
prompt_path = os.environ['prompt_path']

lr = float(os.environ['lr'] )
seed = int(os.environ['seed'] )
batch_size = int(os.environ['batch_size'] )
eval_step = int(os.environ['eval_step'] )
train_epoch = int(os.environ['train_epoch'] )
device = f"{device_type}:{local_rank}"




## 固定随机种子
torch.npu.manual_seed(seed)
torch.manual_seed(seed)
random.seed(seed)
## 单机多卡
def setup(rank,local_rank, world_size):
    dist.init_process_group(
        backend='hccl',    # 使用NCCL后端（GPU场景）
    )
    # torch.cuda.set_device(f"gpu:{local_rank}")  # 绑定当前GPU
    torch.npu.set_device(f"npu:{local_rank}")  # 绑定当前NPU
setup(rank,local_rank,world_size)
## 模型准备 数据集准备
processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)

# if dist.get_rank() == 0:
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path)
model = get_peft_model(model, peft_config)
model = model.npu(local_rank)
model.print_trainable_parameters()
model = DDP(model, device_ids=[local_rank])
optim = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
train_dataset = AudioDatset(train_data_path,prompt_path)
sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader  = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,collate_fn=partial(collate_fn,processor=processor),sampler=sampler)

eval_dataset = AudioDatset(eval_data_path,prompt_path)
sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
eval_dataloader  = torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,collate_fn=partial(collate_fn,processor=processor),sampler=sampler)

def compute_acc(logits,labels):
    _,labels_len = labels.shape
    preds = torch.argmax(logits,dim=-1)
    labels_indices = labels != -100 
    acc = torch.sum(preds[:,-labels_len-1:-1][labels_indices] == labels[labels_indices]).float() /torch.sum(labels_indices).float()
    return acc
## 训练
best_eval_acc = -math.inf
best_eval_loss = math.inf
train_bar = tqdm(train_dataloader)
eval_bar = tqdm(eval_dataloader)
for _ in range(train_epoch):
    for step,batch in enumerate(train_bar):
        # train 
        model.train()
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        acc = compute_acc(outputs["logits"],batch["labels"])
        train_bar.set_description(f"[Train] rank:{local_rank},loss:{loss:0.2},acc:{acc:0.2}")
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % eval_step == 0:
            # eval 
            eval_acc = 0
            eval_loss = 0
            with torch.no_grad():
                for step,batch in enumerate(eval_bar):
                    model.eval()
                    batch.to(device)
                    outputs = model(**batch)
                    loss = outputs.loss
                    acc = compute_acc(outputs["logits"],batch["labels"])
                    eval_acc += loss.item()
                    eval_loss += acc.item()
                    train_bar.set_description(f"[Eval] rank:{local_rank},loss:{loss:0.2},acc:{acc:0.2}")
            eval_acc = eval_acc / step
            eval_loss = eval_loss/ step
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)
            print(f"Eval:loss {eval_loss} acc {eval_acc}")
            if best_eval_loss > eval_loss:
                best_eval_loss = eval_loss
                if dist.get_rank() == 0:
                    peft_model_id = "test"
                    model.save_pretrained(peft_model_id)



