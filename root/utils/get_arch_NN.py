from typing import Callable

from architecture.my_CNN import ConvNet
from architecture.my_VGG import VGG16
from architecture.my_ResNet import PsevdoResNet


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
    architecture = option_network['architecture']

    in_channels = option_network['network_settings']['in_channels']
    num_channels = option_network['network_settings']['num_channels']
    out_channels = option_network['network_settings']['out_channels']
    block_type = option_network['network_settings']['block_type']

    return {
        'ConvNet': lambda: ConvNet(in_channels, num_channels, out_channels),
        'VGG16': lambda: VGG16(out_channels),
        'ResNet': lambda: PsevdoResNet(in_channels, num_channels, out_channels, block_type),
        'DenseNet': 1,
    }.get(architecture, lambda: f'Такой архитектуры \'{architecture}\' пока что нет')


""" version 2
    dict_arch = {
        'ConvNet': ConvNet,
        'VGG16': VGG16,
        'ResNet': PsevdoResNet,
        'DenseNet': 1,
    }
    try:
        return  dict_arch[architecture](in_channels,num_channels,out_channels,block_type)

    except NotImplementedError:
        print(f'Такой архитектуры {architecture} пока что нет')
"""

""" 
    version 1
    architecture = option_network['architecture']
    if option_network is None:
        raise NotImplementedError('Опции не загружены')
    elif architecture == 'ConvNet':
        architecture = ConvNet()
    elif architecture == 'VGG16':
        out_channels = option_network['network_settings']['out_channels']
        architecture = VGG16(out_channels)
    elif architecture == 'ResNet':
        in_channels = option_network['network_settings']['in_channels']
        num_channels = option_network['network_settings']['num_channels']
        out_channels = option_network['network_settings']['out_channels']
        block_type = option_network['network_settings']['block_type']
        architecture = PsevdoResNet(in_channels,num_channels,out_channels,block_type)
    elif architecture == 'DenseNet':
        raise Exception(f'Аархитектура не добавлена(DenseNet)')
    else:
        raise NotImplementedError(f'Такой архитектуры {architecture} пока что нет')

    return architecture """
