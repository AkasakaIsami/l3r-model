import configparser
import os
from datetime import datetime
import time

import torch
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from torchinfo import summary
from torchvision.ops import sigmoid_focal_loss

from model import StatementClassfier
from util import float_to_percent

import warnings

warnings.filterwarnings("ignore")


def train(train_dataset, validate_dataset, model_path) -> (str, str):
    """
    开始训练的函数，输入训练集和验证集就能开始训练了

    :param train_dataset: 训练集
    :param validate_dataset: 验证集
    :param model_path: 最佳模型的保存目录
    :return: 模型文件路径 记录训练信息的日志文件路径
    """

    # 读取一些超参
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    EPOCHS = cf.getint('train', 'epoch')
    BATCH_SIZE = cf.getint('train', 'batchSize')

    LR = cf.getfloat('train', 'learningRate')

    C_NUM_LAYERS = cf.getint('train', 'STClassifierNumLayers')
    E_NUM_LAYERS = cf.getint('train', 'STEncoderNumLayers')

    HIDDEN_DIM = cf.getint('train', 'hiddenDIM')
    ENCODE_DIM = cf.getint('train', 'encodeDIM')

    DROP = cf.getfloat('train', 'dropout')

    USE_GPU = cf.getboolean('environment', 'useGPU') and torch.cuda.is_available()

    # 在正式开始训练前，先设置一下日志持久化的配置
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%Y-%m-%d_%H:%M:%S')

    # 定义日志保存文件
    record_file_name = start_time_str + 'train_info_' + '.txt'
    record_file = open(os.path.join(model_path, record_file_name), 'w')
    record_file.write(f"本次实验开始时间：{start_time_str}\n")

    record_file.write(f"数据集信息如下：\n")
    record_file.write(f"    - 训练集函数级数据量：{len(train_dataset)}\n")
    record_file.write(f"    - 训练集函数级正样本比例：{len(train_dataset)}\n")
    record_file.write(f"    - 验证集函数级数据量：{-1}\n")
    record_file.write(f"    - 验证集函数级正样本比例：{-1}\n")

    record_file.write(f"    - 训练集语句级数据量：{-1}\n")
    record_file.write(f"    - 训练集语句级正样本比例：{-1}\n")
    record_file.write(f"    - 验证集语句级数据量：{-1}\n")
    record_file.write(f"    - 验证集语句级正样本比例：{-1}\n")

    record_file.write(f"模型配置如下：\n")
    record_file.write(f"    - EPOCHS：{EPOCHS}\n")
    record_file.write(f"    - BATCH_SIZE：{BATCH_SIZE}\n")
    record_file.write(f"    - LEARNING_RATE：{LR}\n")
    record_file.write(f"    - 语句编码器层数：{E_NUM_LAYERS}\n")
    record_file.write(f"    - 语句编码维度：{ENCODE_DIM}\n")
    record_file.write(f"    - 语句分类器层数：{C_NUM_LAYERS}\n")
    record_file.write(f"    - 隐藏层维度：{HIDDEN_DIM}\n")
    record_file.write(f"    - dropout率：{DROP}\n")

    # 正式开始训练！
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练的配置
    model = StatementClassfier(encode_dim=ENCODE_DIM, hidden_dim=HIDDEN_DIM, encoder_num_layers=ENCODE_DIM,
                               classifier_num_layers=C_NUM_LAYERS, dropout=DROP,
                               use_gpu=USE_GPU)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_function = sigmoid_focal_loss

    if USE_GPU:
        '''
        使用GPU的话，model需要转cuda
        '''
        model = model.cuda()

    record_file.write(f"模型结构如下：\n")
    record_file.write(str(summary(model)) + '\n')

    # 用于寻找效果最好的模型
    best_acc = 0.0
    best_model = model

    # 控制日志打印的一些参数
    total_train_step = 0

    start = time.time()
    record_file.write(f"开始训练！\n")
    for epoch in range(EPOCHS):
        print(f'------------第 {epoch + 1} 轮训练开始------------')
        record_file.write(f'------------第 {epoch + 1} 轮训练开始------------\n')

        model.train()
        for i, data in enumerate(train_loader):
            if USE_GPU:
                data = data.cuda()

            y_hat = model(data)
            y = data.y.float()
            loss = loss_function(y_hat, y, alpha=0.75, gamma=1.5, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 1 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")

            if total_train_step % 50 == 0:
                record_file.write(f"训练次数: {total_train_step}, Loss: {loss.item()}\n")

        total_val_loss = 0.0

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

                y_hat_total = torch.cat([y_hat_total, y_hat_trans])
                y_total = torch.cat([y_total, y_trans])

                loss = loss_function(y_hat, y, alpha=0.75, gamma=1.5, reduction='mean')
                total_val_loss += loss.item()

        print(f"验证集整体Loss: {total_val_loss}")
        record_file.write(f"验证集整体Loss: {total_val_loss}\n")

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

        record_file.write(f"验证集 accuracy_score: {float_to_percent(acc)}\n")
        record_file.write(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}\n")
        record_file.write(f"验证集 precision_score: {float_to_percent(ps)}\n")
        record_file.write(f"验证集 recall_score: {float_to_percent(rc)}\n")
        record_file.write(f"验证集 f1_score: {float_to_percent(f1)}\n")

        # 主要看balanced_accuracy_score
        if balanced_acc > best_acc:
            record_file.write(f"***当前模型的平衡准确率表现最好，被记为表现最好的模型***\n")
            best_model = model
            best_acc = balanced_acc

    end = time.time()
    print(f"训练完成，共耗时{end - start}秒。最佳balanced accuracy是{float_to_percent(best_acc)}。现在开始保存数据...")
    record_file.write(f"训练完成，共耗时{end - start}秒。最佳balanced accuracy是{best_acc}\n")
    record_file.write(
        f"——————————只有看到这条语句，并且对应的模型文件也成功保存了，这个日志文件的内容才有效！（不然就是中断了）——————————")

    def save_model(best_model, model_path: str, time_str) -> str:
        file_name = time_str + '_model@' + float_to_percent(best_acc) + '.pth'
        save_path = os.path.join(model_path, file_name)

        torch.save(best_model, save_path)
        print('模型保存成功！')
        return save_path

    model_file = save_model(best_model, model_path, start_time_str)
    return model_file
