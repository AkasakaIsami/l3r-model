"""
这文件虽然叫eval
但是是用来实现一些用于实验的函数的
"""
import configparser

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import DataLoader

from util import float_to_percent, random_unit


def random_guess(test_dataset, project: str):
    """
    用随机预测算法来预测测试集里的数据
    :param test_dataset:    测试集数据
    :param project:    测试的项目
    :return:
    """
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    USE_GPU = cf.getboolean('environment', 'useGPU') and torch.cuda.is_available()
    lst_rate = cf.getfloat(project, 'loggedStatementRate')

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    y_hat_total = torch.randn(0)
    y_total = torch.randn(0)

    def guess(size, lst_rate):
        result = torch.zeros(size)
        for i in range(size):
            if random_unit(lst_rate):
                result[i] = 1
        return result

    for i, data in enumerate(test_loader):
        y = data.y
        if USE_GPU:
            y = y.cpu()

        y_hat = guess(y.shape[0], lst_rate)

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

    print(f"RG在测试集上的accuracy_score: {float_to_percent(acc)}")
    print(f"RG在测试集上的balanced_accuracy_score: {float_to_percent(balanced_acc)}")
    print(f"RG在测试集上的precision_score: {float_to_percent(ps)}")
    print(f"RG在测试集上的recall_score: {float_to_percent(rc)}")
    print(f"RG在测试集上的f1_score: {float_to_percent(f1)}")
