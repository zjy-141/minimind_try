import torch

path = r"D:\01_Project\project\2026-project17\minimind\checkpoints\pretrain_768.pth"
state_dict = torch.load(path, map_location="cpu", weights_only=True)

print(type(state_dict))
print(list(state_dict.keys())[:10])

# 正确的键名
print(state_dict["model.embed_tokens.weight"].shape)