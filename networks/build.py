import torch
from networks.cnn2 import CNN
from networks.nn import NN


def build_model(cfg):
    assert cfg.arch in ['NN', 'CNN'], 'Unrecognized model name'
    # print("=====", cfg.CNN.num_classes)
    if cfg.arch == 'CNN':
        model = CNN(
            num_classes=cfg.CNN.num_classes,
            conv_dims=cfg.CNN.conv_dims,
            fc_dims=cfg.CNN.fc_dims
        )
    
    else: # cfg.arch == 'NN':
        model = NN(
            num_classes=cfg.NN.num_classes,
            fc_dims=cfg.NN.fc_dims
        )
    # elif cfg.arch == 'CNN':
    #     model = CNN(
    #         num_classes=cfg.CNN.num_classes,
    #         conv_dims=cfg.CNN.conv_dims,
    #         fc_dims=cfg.CNN.fc_dims
    #     )
    # else:
    #     model = RNN(
    #         arch=cfg.arch,
    #         in_dim=cfg.RNN.in_dim,
    #         num_classes=cfg.RNN.num_classes,
    #         hidden_size=cfg.RNN.hidden_size,
    #         num_layers=cfg.RNN.num_layers
    #     )

    return model

def build_optimizer(cfg, model):
    assert cfg.optim in ['SGD', 'Adam', 'AdamW'], 'Unrecognized optimizer name'
    if cfg.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)

    return optimizer

def build_scheduler(cfg, optimizer):
    assert cfg.scheduler in ['StepLR', 'Cosine'], 'Unrecognized scheduler name'
    if cfg.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    return scheduler