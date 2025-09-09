import torch
import random
# import torch_npu
import numpy as np
import deepspeed
def set_seed(seed):
    torch.npu.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    deepspeed.runtime.utils.set_random_seed(seed)
