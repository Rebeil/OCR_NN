import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer():
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

    return optimizer