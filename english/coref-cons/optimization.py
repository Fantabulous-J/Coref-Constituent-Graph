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
        size += params.nelement()

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
