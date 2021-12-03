from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AdamW
from torch.optim import Adam


def build_optimizer(model, config):
    bert_params, task_params = [], []
    size = 0
    for name, params in model.named_parameters():
        if "bert" in name:
            bert_params.append((name, params))
        else:
            task_params.append((name, params))
        size += params.nelement() if params.requires_grad else 0

    print("bert parameters")
    for name, params in bert_params:
        print('n: {}, shape: {}'.format(name, params.shape))
    print('*' * 150)
    print("task parameters")
    for name, params in task_params:
        print('n: {}, shape: {}'.format(name, params.shape))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    task_optimizer_group_parameters = [p for _, p in task_params]
    print('Total parameters: {}'.format(size))

    bert_optimizer = AdamW(bert_optimizer_grouped_parameters,
                           lr=config['bert_learning_rate'],
                           betas=(0.9, 0.999),
                           eps=config['adam_eps'])

    task_optimizer = Adam(task_optimizer_group_parameters,
                          lr=config['task_learning_rate'])

    return bert_optimizer, task_optimizer


class FocalLoss(nn.Module, ABC):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
