# standart libs
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml

# my libs
from visualization_def.test_view import view_2_grahic
from architecture.my_CNN import ConvNet
from Custon_dataloader.Custom_dataloader import CustomDataset
import data_split.split_data as splt
from optimizer.my_optimizers import get_optimizer
# import optimizer.my_optimizers as get_opt
# from data_split import split_data
import utils.get_arch_NN as get_arch_NN
from utils.utils import count_parameters
from metrics.metrics import accuracy


# добавить метрики f1 f2 recall precision


def load_settings(path: str = '/home/rain/vs_code/relize/yaml_config/config.yml') -> dict:
    option_path = path
    with open(option_path, 'r') as file_option:
        option = yaml.safe_load(file_option)
    return option


def save_weights_nn(model_state_dict: dict,
                    name_model: str = None,
                    network_settings: dict = None,
                    optional: dict = None) -> None:
    optional = optional['in_channels']
    if (name_model and network_settings) is None:
        raise Exception('Имя модели или настройки не заданы')
    else:
        torch.save(model_state_dict, 'saves_weights/' + name_model + '/' + name_model + '_')


if __name__ == '__main__':
    # list_settings = {}
    all_settings = load_settings()
    model = get_arch_NN.get_arch_NN(all_settings)()
    print(model)
    # splt.split_dataset()
    train_dir = all_settings['path_to_split_datasets']['train_dir']
    val_dir = all_settings['path_to_split_datasets']['val_dir']
    test_dir = all_settings['path_to_split_datasets']['test_dir']
    # print(all_settings)
    # print(model)
    print()
    # print(train_dir,val_dir,test_dir,sep='/n')

    data_train = CustomDataset(train_dir)
    data_test = CustomDataset(test_dir)
    data_val = CustomDataset(val_dir)
    # print(len(data_train))
    BATCH_SIZE = all_settings['batch_size']
    # list_settings.update({'batch_size': BATCH_SIZE})
    data_loader_train = DataLoader(data_train, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=6,
                                   drop_last=True, pin_memory=True)
    data_loader_test = DataLoader(data_test, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=6,
                                  drop_last=True, pin_memory=True)
    data_loader_val = DataLoader(data_val, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=6,
                                 drop_last=True, pin_memory=True)
    # print(len(data_loader_train))

    print(f'Кол-во параметров модели {count_parameters(model)}')

    # print(type(model.parameters()))

    # ВРЕМЕННО
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model.parameters(),
                              all_settings['network_settings']['learning_setting']['optimizer'])()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=all_settings['network_settings']['learning_setting']['learning_rate'],
    #                               betas=(all_settings['network_settings']['learning_setting']['betas']['betas1'],
    #                                      all_settings['network_settings']['learning_setting']['betas']['betas2']),
    #                               weight_decay=0.01)
    # print(optimizer)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=all_settings['network_settings']['learning_setting'][
                                                           'scheduler_settings']['gamma']
                                                       )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    use_amp = bool(all_settings['network_settings']['AMP_settings']['use_amp'])
    scaler = torch.cuda.amp.GradScaler()

    torch.backends.cudnn.benchmark = bool(all_settings['network_settings']['torch.backends.cudnn.benchmark'])
    torch.backends.cudnn.deterministic = bool(all_settings['network_settings']['torch.backends.cudnn.deterministic'])

    epochs = all_settings['network_settings']['epochs']  # перенести в learning settings
    loss_item_list_train = []
    acc_item_list_train = []
    loss_epochs_list_train = []
    acc_epochs_list_train = []
    for epoch in range(epochs):
        print()
        print(f'Эпоха - {epoch}')
        loss_val = 0
        acc_val = 0
        for sample in (pbar := tqdm(data_loader_train)):
            img, label = sample['img'], sample['label']
            label = label.reshape(BATCH_SIZE)
            label = F.one_hot(label, all_settings['network_settings']['out_channels']).float()
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # test = torch.flatten(label)

            # break
            with autocast(use_amp):
                pred = model(img)
                loss = loss_fn(pred, label)

            scaler.scale(loss).backward()
            loss_item = loss.item()
            loss_val += loss_item

            scaler.step(optimizer)
            scaler.update()

            acc_current = accuracy(pred.cpu().float(), label.cpu().float())
            acc_val += acc_current

            loss_item_list_train += [loss_item]
            acc_item_list_train += [acc_current]

            pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        print()
        scheduler.step()
        loss_epochs_list_train += [loss_val / len(data_loader_train)]
        acc_epochs_list_train += [acc_val / len(data_loader_train)]
        print(f'avg_loss: {loss_epochs_list_train[-1]}')
        print(f'avg acc: {acc_epochs_list_train[-1]}')

        import json

        open("loss.json", "w").write(json.dumps(loss_item_list_train))
        open("acc.json", "w").write(json.dumps(acc_item_list_train))

        open("avg_loss.json", "w").write(json.dumps(loss_epochs_list_train))
        open("avg_acc.json", "w").write(json.dumps(acc_epochs_list_train))

        view_2_grahic(data=(loss_item_list_train, acc_item_list_train),
                      сhart_name='loss_and_acc',
                      type_NN=all_settings['architecture'],
                      epochs=all_settings['network_settings']['epochs'],
                      )
        view_2_grahic(data=(loss_epochs_list_train, acc_epochs_list_train),
                      сhart_name='avg_loss_and_acc',
                      type_NN=all_settings['architecture'],
                      epochs=all_settings['network_settings']['epochs'],
                      )

        # break
