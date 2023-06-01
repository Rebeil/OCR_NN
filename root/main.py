# My classes
from architecture.my_CNN import ConvNet
from Custon_dataloader.Custom_dataloader import CustomDataset
import data_split.split_data as splt
# from data_split import split_data
# libs
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torch.cuda.amp import autocast, GradScaler
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.get_arch_NN as get_arch_NN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


def load_settings(path: str = '/home/rain/vs_code/relize/yaml_config/config.yml') -> dict:
    option_path = path
    with open(option_path, 'r') as file_option:
        option = yaml.safe_load(file_option)
    return option


if __name__ == '__main__':
    all_settings = load_settings()
    model = get_arch_NN.get_arch_NN(all_settings)
    # splt.split_dataset()
    train_dir = all_settings['path_to_split_datasets']['train_dir']
    val_dir = all_settings['path_to_split_datasets']['val_dir']
    test_dir = all_settings['path_to_split_datasets']['test_dir']
    #print(all_settings)
    print(model)
    print()
    #print(train_dir,val_dir,test_dir,sep='/n')

    data_train = CustomDataset(train_dir)
    data_test = CustomDataset(test_dir)
    data_val = CustomDataset(val_dir)
    # print(len(data_train))
    BATCH_SIZE = all_settings['batch_size']
    data_loader_train = DataLoader(data_train, batch_size=BATCH_SIZE,
                                   shuffle=True, pin_memory=True,
                                   num_workers=6, drop_last=True)
    data_loader_test = DataLoader(data_test, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True,
                                  num_workers=6, drop_last=True)
    data_loader_val = DataLoader(data_val, batch_size=BATCH_SIZE,
                                 shuffle=True, pin_memory=True,
                                 num_workers=6, drop_last=True)
    # print(len(data_loader_train))

    print(f'Кол-во параметров модели {count_parameters(model)}')

    # ВРЕМЕННО
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=all_settings['network_settings']['learning_setting']['learning_rate'],
                                  betas=(all_settings['network_settings']['learning_setting']['betas']['betas1'],
                                         all_settings['network_settings']['learning_setting']['betas']['betas2']),
                                         weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma = all_settings['network_settings']['scheduler_settings']['gamma']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    use_amp = bool(all_settings['network_settings']['AMP_settings']['use_amp'])
    scaler = torch.cuda.amp.GradScaler()

    torch.backends.cudnn.benchmark = bool(all_settings['network_settings']['AMP_settings']['torch.backends.cudnn.benchmark'])
    torch.backends.cudnn.deterministic = bool(all_settings['network_settings']['AMP_settings']['torch.backends.cudnn.deterministic'])

    epochs = all_settings['network_settings']['epochs']
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
            test = torch.flatten(label)

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
            pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        print()
        loss_epochs_list_train += [loss_val/len(data_loader_train)]
        acc_epochs_list_train += [acc_val/len(data_loader_train)]
        print(f'avg_loss: {loss_val / len(data_loader_train)}')
        print(f'avg acc: {acc_val / len(data_loader_train)}')
