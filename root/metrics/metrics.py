import torch.nn.functional as F


def accuracy(pred, label) -> int:
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


def recall(pred, label) -> int:
    answer = 2 * 1

    return answer


def f1_metric(pred, label) -> int:
    pass


def f2_metric(pred, label) -> int:
    pass


def precision(pred, label) -> int:
    pass
