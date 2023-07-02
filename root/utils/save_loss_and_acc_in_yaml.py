import yaml


def write_list_of_data_in_yaml(data: list, graphic_name: str, type_NN: str, add_param: str):
    with open(
            '/home/rain/vs_code/relize/saves_loss_and_acc_for_NNs/' + type_NN + '/' + add_param + '_' + graphic_name + '.yaml',
            'w') as fw:
        yaml.dump(data, fw, sort_keys=False, default_flow_style=False)


def read_list_of_data_in_yaml(data: list, graphic_name: str, type_NN: str, add_param: str):
    with open(
            '/home/rain/vs_code/relize/saves_loss_and_acc_for_NNs/' + '/' + type_NN + '/' + add_param + '_' + graphic_name + '.yaml',
            'r') as fr:
        print(fr.read())

# return

# def write_loss_and_acc_in_yaml(loos_and_acc: tuple, filename: str):
#     with open(filename + '.yaml', 'w') as fw:
#         yaml.dump(loos_and_acc, fw, sort_keys=False,
#                   default_flow_style=False)
#
#
# def read_loss_and_acc_in_yaml(loos_and_acc: tuple, filename: str):
#     with open(filename + '.yaml', 'r') as fr:
#         print(fr.read())
