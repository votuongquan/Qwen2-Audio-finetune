import torch
import torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from functools import partial
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from dataset import AudioDatset,collate_fn
import time



## 宏参数准备
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules = [ "q_proj", "v_proj"]
)
model_path = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio"
train_data_path = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/test"
eval_data_path = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/test"
lr = 1e-4
seed = 1314
batch_size = 1
device = "npu:0"
eval_step = 1000
train_epoch = 5
model_path = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio"
## 固定随机种子


## 模型准备 数据集准备
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path,trust_remote_code=True,device_map=device)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
optim = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
train_dataset = AudioDatset(train_data_path)
train_dataloader  = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=partial(collate_fn,processor=processor))
eval_dataset = AudioDatset(eval_data_path)
eval_dataloader  = torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,collate_fn=partial(collate_fn,processor=processor))

## 训练
model.train()
bar = tqdm(train_dataloader)
for batch in bar:
    batch.to(device)
    outputs = model(**batch)
    loss = outputs.loss
    bar.set_description(f"loss:{loss:0.2},acc:1")
    optim.zero_grad()
    loss.backward()
    optim.step()


