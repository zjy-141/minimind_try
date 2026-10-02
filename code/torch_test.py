# 检查 CUDA 是否可用，以及 GPU 的名称和计算能力，并进行简单的矩阵运算测试

import torch
print("CUDA 是否可用:", torch.cuda.is_available())
print("GPU 名称:", torch.cuda.get_device_name(0))
print("计算能力:", torch.cuda.get_device_capability(0))
x = torch.randn(1000, 1000, device="cuda")
y = torch.mm(x, x)
print("GPU 运算测试通过:", y.sum().item())