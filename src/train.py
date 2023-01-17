import configparser
import os
from datetime import datetime

import torch
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from torchvision.ops import sigmoid_focal_loss

from model import StatementClassfier
from util import float_to_percent

import warnings

warnings.filterwarnings("ignore")


def train(train_dataset, validate_dataset, model_path) -> str:
    # 读取一些超参
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    EPOCHS = cf.getint('train', 'epoch')
    BATCH_SIZE = cf.getint('train', 'batchSize')

    HIDDEN_DIM = cf.getint('train', 'hiddenDIM')
    ENCODE_DIM = cf.getint('train', 'encodeDIM')
    C_NUM_LAYERS = cf.getint('train', 'STClassifierNumLayers')
    E_NUM_LAYERS = cf.getint('train', 'STEncoderNumLayers')
    DROP = cf.getfloat('train', 'dropout')

    USE_GPU = cf.getboolean('environment', 'useGPU') and torch.cuda.is_available()

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练的配置
    model = StatementClassfier(encode_dim=ENCODE_DIM, hidden_dim=HIDDEN_DIM, num_layers=C_NUM_LAYERS, dropout=DROP,
                               use_gpu=USE_GPU)
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = sigmoid_focal_loss

    if USE_GPU:
        '''
        使用GPU的话，model需要转cuda
        '''
        model = model.cuda()

    # 用于寻找效果最好的模型
    best_acc = 0.0
    best_model = model

    # 控制日志打印的一些参数
    total_train_step = 0

    for epoch in range(EPOCHS):
        print(f'------------第 {epoch + 1} 轮训练开始------------')

        model.train()
        for i, data in enumerate(train_loader):
            if USE_GPU:
                data = data.cuda()

            y_hat = model(data)
            y = data.y.float()
            # FIXME: 要关注正样本的话，alpha到底应该设置成>0.5还是<0.5？
            loss = loss_function(y_hat, y, alpha=0.75, reduction="mean")

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

        if USE_GPU:
            y_hat_total = y_hat_total.cuda()
            y_total = y_total.cuda()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                if USE_GPU:
                    data = data.cuda()

                y_hat = model(data)

                if USE_GPU:
                    y_hat = y_hat.cuda()

                y = data.y.float()
                y_hat_trans = y_hat.argmax(1)
                y_trans = y.argmax(1)

                # 拼接
                y_hat_total = torch.cat([y_hat_total, y_hat_trans])
                y_total = torch.cat([y_total, y_trans])

                loss = loss_function(y_hat, y, alpha=0.75, reduction='mean')
                total_val_loss += loss.item()

                val_data_size = val_data_size + len(y)
                acc = (y_hat_trans == y_trans).sum()
                total_acc = total_acc + acc

        print(f"验证集整体Loss: {total_val_loss}")

        total_acc = total_acc / val_data_size
        total_acc_str = "%.2f%%" % (total_acc * 100)
        print(f"验证集Accuracy: {total_acc_str}")

        acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
        balanced_acc = balanced_accuracy_score(y_total.cpu(), y_hat_total.cpu())
        ps = precision_score(y_total.cpu(), y_hat_total.cpu())
        rc = recall_score(y_total.cpu(), y_hat_total.cpu())
        f1 = f1_score(y_total.cpu(), y_hat_total.cpu())

        print(f"验证集 accuracy_score: {float_to_percent(acc)}")
        print(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
        print(f"验证集 precision_score: {float_to_percent(ps)}")
        print(f"验证集 recall_score: {float_to_percent(rc)}")
        print(f"验证集 f1_score: {float_to_percent(f1)}")

        # 主要看balanced_accuracy_score
        if balanced_acc > best_acc:
            best_model = model

    def save_model(best_model, model_path: str) -> str:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        curr_time = datetime.now()
        time_str = datetime.strftime(curr_time, '%Y-%m-%d_%H:%M:%S')
        file_name = 'model_' + time_str + '_' + float_to_percent(balanced_acc) + '.pth'
        save_path = os.path.join(model_path, file_name)

        torch.save(model, save_path)
        print('模型保存成功！')
        return save_path

    return save_model(best_model, model_path)
