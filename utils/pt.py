# @Time    : 2022/3/10 16:05
# @Author  : ZYF

import torch


def describe_torch_model(model: torch.nn.Module):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
