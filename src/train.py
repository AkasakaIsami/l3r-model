import torch
from torch_geometric.loader import DataLoader

from model import StatementClassfier


# from torch.utils.data import DataLoader

def train(train_dataset, validate_dataset):
    # 定义一些超参
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    EPOCHS = 1
    BATCH_SIZE = 2
    USE_GPU = False

    # loader = DataLoader()
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    dev_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = StatementClassfier()

    # 训练的配置
    # parameters = model.parameters()
    # optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    # 一些评估指标
    train_loss_ = []
    val_loss_ = []

    train_acc_ = []
    val_acc_ = []

    best_acc = 0.0

    best_model = model

    print('Start training...')
    for epoch in range(EPOCHS):
        for i, data in enumerate(train_loader):
            result = model(data, BATCH_SIZE)

# model.train()
