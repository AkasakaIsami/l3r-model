import os.path

import torch
from fvcore.nn import sigmoid_focal_loss
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from model import StatementClassfier
from util import float_to_percent

import warnings

warnings.filterwarnings("ignore")


def train(train_dataset, validate_dataset, model_path):
    # 定义一些超参
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    EPOCHS = 5
    BATCH_SIZE = 64
    USE_GPU = False

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = StatementClassfier()

    # 训练的配置
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = sigmoid_focal_loss

    # 用于寻找效果最好的模型
    best_acc = 0.0
    best_model = model

    # 控制日志打印的一些参数
    total_train_step = 0

    for epoch in range(EPOCHS):
        print(f'------------第 {epoch + 1} 轮训练开始------------')

        model.train()
        for i, data in enumerate(train_loader):
            y_hat = model(data)
            y = data.y
            loss = loss_function(y_hat, y, alpha=0.75, gamma=1.25, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 1 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")

        # 验证集用以下几个指标为超参调整提供参考：
        #   1. Loss 验证集中所有数据表现出的损失值Loss
        #   2. Accuracy 准确率
        #   3. Balanced Accuracy 均衡准确率
        #   4. Precision
        #   5. Recall
        #   6. F1

        total_val_loss = 0.0
        total_acc = 0.0
        val_data_size = 0

        # 但指标得把验证集上所有数据拼在一起来训练
        y_hat_total = torch.randn(0)
        y_total = torch.randn(0)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                y_hat = model(data)
                y = data.y
                y_hat_trans = y_hat.argmax(1)
                y_trans = y.argmax(1)

                # 拼接
                y_hat_total = torch.cat([y_hat_total, y_hat_trans])
                y_total = torch.cat([y_total, y_trans])

                loss = loss_function(y_hat, y, alpha=0.75, gamma=1.25, reduction='mean')
                total_val_loss += loss.item()

                val_data_size = val_data_size + len(y)
                acc = (y_hat_trans == y_trans).sum()
                total_acc = total_acc + acc

        print(f"验证集整体Loss: {total_val_loss}")

        total_acc = total_acc / val_data_size
        total_acc_str = "%.2f%%" % (total_acc * 100)
        print(f"验证集Accuracy: {total_acc_str}")

        acc = accuracy_score(y_total, y_hat_total)
        balanced_acc = balanced_accuracy_score(y_total, y_hat_total)
        ps = precision_score(y_total, y_hat_total)
        rc = recall_score(y_total, y_hat_total)
        f1 = f1_score(y_total, y_hat_total)

        print(f"验证集 accuracy_score: {float_to_percent(acc)}")
        print(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
        print(f"验证集 precision_score: {float_to_percent(ps)}")
        print(f"验证集 recall_score: {float_to_percent(rc)}")
        print(f"验证集 f1_score: {float_to_percent(f1)}")

        # 主要看balanced_accuracy_score
        if balanced_acc > best_acc:
            best_model = model

    # 模型的保存 会报错……^_^
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # model_file = os.path.join(model_path, f"model.pth")
    # if not os.path.exists(model_file):
    #     torch.save(best_model, model_file)

    return best_model
