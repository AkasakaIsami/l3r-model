import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score, accuracy_score

from util import float_to_percent
import warnings

warnings.filterwarnings("ignore")


def test(model, test_dataset):
    """
    有几个指标要写
        1. Loss 验证集中所有数据表现出的损失值Loss
        2. Accuracy 准确率
        3. Balanced Accuracy 均衡准确率
        4. Precision
        5. Recall
        6. F1

    :param model:
    :param test_dataset:
    :return:
    """
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    total_acc = 0.0
    test_data_size = 0

    # 指标得把测试集上所有数据拼在一起来计算
    y_hat_total = torch.randn(0)
    y_total = torch.randn(0)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            y_hat = model(data)
            y = data.y
            y_hat_trans = y_hat.argmax(1)
            y_trans = y.argmax(1)

            # 拼接
            y_hat_total = torch.cat([y_hat_total, y_hat_trans])
            y_total = torch.cat([y_total, y_trans])

            test_data_size = test_data_size + len(y)
            acc = (y_hat_trans == y_trans).sum()
            total_acc = total_acc + acc

    total_acc = total_acc / test_data_size
    print(f"测试集 Accuracy: {float_to_percent(total_acc)}")

    acc = accuracy_score(y_total, y_hat_total)
    balanced_acc = balanced_accuracy_score(y_total, y_hat_total)
    ps = precision_score(y_total, y_hat_total)
    rc = recall_score(y_total, y_hat_total)
    f1 = f1_score(y_total, y_hat_total)

    print(f"测试集 accuracy_score: {float_to_percent(acc)}")
    print(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
    print(f"测试集 precision_score: {float_to_percent(ps)}")
    print(f"测试集 recall_score: {float_to_percent(rc)}")
    print(f"测试集 f1_score: {float_to_percent(f1)}")
