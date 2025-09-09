import os 
import torch
# import torch_npu
import torch.distributed as dist
import deepspeed
# 设置分布式运行环境
def setup_ddp(device_type):
    local_rank = os.environ["LOCAL_RANK"]
    if device_type == "npu":
        torch.npu.set_device(f"npu:{local_rank}")
        dist.init_process_group(
        backend='hccl',
    )
    elif device_type =="cuda":
        torch.cuda.set_device(f"cuda:{local_rank}") 
        dist.init_process_group(
        backend='nccl',
    )
def setup_deepspeed(device_type):
    local_rank = os.environ["LOCAL_RANK"]
    if device_type == "npu":
        torch.npu.set_device(f"npu:{local_rank}")
        deepspeed.init_distributed(
        dist_backend='hccl',
    )
    elif device_type =="cuda":
        torch.cuda.set_device(f"cuda:{local_rank}")
        deepspeed.init_distributed(
        dist_backend='nccl',
    )