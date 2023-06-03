from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, Adamax


def get_optimizer(params_model, option_optimizer: dict) -> Callable[[], Adam] | \
                                                           Callable[[], AdamW] | \
                                                           Callable[[], Adamax] | \
                                                           Callable[[], str]:
    """
    Args:
        params_model (_type_): параметры модели \n
        option_optimizer (dict): опции оптимизатора \n

    Returns:
        torch.optim: оптимизатор
    """
    name = option_optimizer['name']
    lr = option_optimizer['learning_rate']
    betas1 = option_optimizer['betas']['betas1']
    betas2 = option_optimizer['betas']['betas2']
    eps = option_optimizer['eps']
    weight_decay = option_optimizer['weight_decay']
    return {
        'Adam': lambda: torch.optim.Adam(params=params_model,
                                         lr=lr,
                                         betas=(betas1, betas2),
                                         eps=float(eps),
                                         weight_decay=weight_decay),
        'AdamW': lambda: torch.optim.AdamW(params=params_model,
                                           lr=lr,
                                           betas=(betas1, betas2),
                                           eps=float(eps),
                                           weight_decay=weight_decay),
        'Adamax': lambda: torch.optim.Adamax(params=params_model,
                                             lr=lr,
                                             betas=(betas1, betas2),
                                             eps=float(eps),
                                             weight_decay=weight_decay)

        # 'RMSProp': 2,
        # 'SMORMS3': 1,
    }.get(name, lambda: f'Такого оптимизатора \'{name}\' пока что нет')
