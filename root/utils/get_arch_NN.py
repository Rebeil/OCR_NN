from typing import Callable

from architecture.my_CNN import ConvNet
from architecture.my_VGG import VGG16
from architecture.my_ResNet import PsevdoResNet
from architecture.my_ResNet_2 import PsevdoResNet_2
from architecture.my_ResNet_v3 import PsevdoResNet_with_dropout


# from architecture.my_DenseNet import


def get_arch_NN(option_network: dict = None) -> int | \
                                                Callable[[], ConvNet] | \
                                                Callable[[], VGG16] | \
                                                Callable[[], PsevdoResNet] | \
                                                Callable[[], str]:
    """
    Args:
        option_network (dict, optional): Опции нейронной сети. Defaults to None.\n
    """
    architecture = option_network['arch']

    in_channels = option_network['nn_set']['in_ch']
    num_channels = option_network['nn_set']['num_ch']
    out_channels = option_network['nn_set']['out_ch']
    block_type = option_network['nn_set']['blk']
    stride = option_network['nn_set']['srd']
    return {
        'ConvNet': lambda: ConvNet(in_channels, num_channels, out_channels),
        'VGG16': lambda: VGG16(out_channels),
        'ResNet': lambda: PsevdoResNet(in_channels, num_channels, out_channels, block_type, stride),
        'ResNet_v2': lambda: PsevdoResNet_2(in_channels, num_channels, out_channels, block_type, stride),
        'PsevdoResNet_with_dropout': lambda: PsevdoResNet_with_dropout(in_channels, num_channels, out_channels,
                                                                       block_type, stride),
        'DenseNet': 1,
    }.get(architecture, lambda: f'Такой архитектуры \'{architecture}\' пока что нет')
