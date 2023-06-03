# standart libs
from tqdm.autonotebook import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F
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


def train(model, epochs: int, data_loader_train, BATCH_SIZE: int, out_channels: int, device: str, optimizer,
          use_amp: bool, loss_fn: str, scaler, scheduler): # train(model, params)
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
            label = F.one_hot(label, out_channels).float()
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

            loss_item_list_train += [loss_item]
            acc_item_list_train += [acc_current]

            pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        print()
        scheduler.step()
        loss_epochs_list_train += [loss_val / len(data_loader_train)]
        acc_epochs_list_train += [acc_val / len(data_loader_train)]
        print(f'avg_loss: {loss_epochs_list_train[-1]}')
        print(f'avg acc: {acc_epochs_list_train[-1]}')

        # +функция созранения листов и визуализации
