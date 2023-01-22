import configparser
import os

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score, accuracy_score, \
    confusion_matrix

from util import float_to_percent
import warnings

warnings.filterwarnings("ignore")


def test(model, test_dataset, record_file_path: str):
    """
    有几个指标
        1. Accuracy 准确率
        2. Balanced Accuracy 均衡准确率
        3. Precision
        4. Recall
        5. F1
    :param model_path: 用于测试的模型
    :param test_dataset: 用于测试的测试集
    :return:
    """
    record_file = open(os.path.join(record_file_path), 'a')

    cf = configparser.ConfigParser()
    cf.read('config.ini')
    BATCH_SIZE = cf.getint('train', 'batchSize')
    USE_GPU = cf.getboolean('environment', 'useGPU') and torch.cuda.is_available()
    HAVE_TO_SAMPLE = cf.getint('data', 'negativeRatio') > 0

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 指标得把测试集上所有数据拼在一起来计算
    y_hat_total = torch.randn(0)
    y_total = torch.randn(0)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if USE_GPU:
                data = data.cuda()

            y_hat = model(data)

            if USE_GPU:
                y_hat = y_hat.cpu()

            y = data.y
            if USE_GPU:
                y = y.cpu()

            if HAVE_TO_SAMPLE:
                sample = data['sample']
                size = sample.shape[0]
                indices = []
                for i in range(size):
                    if sample[i] == 1:
                        indices.append(i)

                y_hat = torch.index_select(y_hat, dim=0, index=torch.tensor(indices))
                y = torch.index_select(y, dim=0, index=torch.tensor(indices))

            y_hat_trans = y_hat.argmax(1)
            y_trans = y.argmax(1)

            # 拼接
            y_hat_total = torch.cat([y_hat_total, y_hat_trans])
            y_total = torch.cat([y_total, y_trans])

    acc = accuracy_score(y_total, y_hat_total)
    balanced_acc = balanced_accuracy_score(y_total, y_hat_total)
    ps = precision_score(y_total, y_hat_total)
    rc = recall_score(y_total, y_hat_total)
    f1 = f1_score(y_total, y_hat_total)
    c = confusion_matrix(y_total, y_hat_total, labels=[0, 1])

    print(f"测试集 accuracy_score: {float_to_percent(acc)}")
    print(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
    print(f"测试集 precision_score: {float_to_percent(ps)}")
    print(f"测试集 recall_score: {float_to_percent(rc)}")
    print(f"测试集 f1_score: {float_to_percent(f1)}")
    print(f"测试集 混淆矩阵:\n {c}")

    record_file.write("下面是测试集结果：\n")
    record_file.write(f"测试集 accuracy_score: {float_to_percent(acc)}\n")
    record_file.write(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}\n")
    record_file.write(f"测试集 precision_score: {float_to_percent(ps)}\n")
    record_file.write(f"测试集 recall_score: {float_to_percent(rc)}\n")
    record_file.write(f"测试集 f1_score: {float_to_percent(f1)}\n")
    record_file.write(f"测试集 混淆矩阵:\n {c}")

    record_file.close()
