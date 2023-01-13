import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score


def test(model, test_dataset):
    """
    有几个指标要写
        1. Loss 验证集中所有数据表现出的损失值Loss
        2. Accuracy 准确率
        3. TODO: Balanced Accuracy 均衡准确率
        4. TODO: Precision
        5. TODO: Recall
        6. TODO: F1

    :param model:
    :param test_dataset:
    :return:
    """
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    total_acc = 0.0
    test_data_size = 0

    # 写起来真的很麻烦 我打算把batchsize设置的大一点然后每个batch来计算指标
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            y_hat = model(data)
            y = data.y

            y_trans = y_hat.argmax(1)

            balanced_acc = balanced_accuracy_score(y, y_trans)
            ps = precision_score(y, y_trans)
            rc = recall_score(y, y_trans)
            f1 = f1_score(y, y_trans)

            print(f"测试集第 {i + 1} 个 batch")
            print(f"balanced_accuracy_score: {float_to_percent(balanced_acc)}")
            print(f"precision_score: {float_to_percent(ps)}")
            print(f"recall_score: {float_to_percent(rc)}")
            print(f"f1_score: {float_to_percent(f1)}")

            test_data_size = test_data_size + len(y)
            acc = (y_trans == y).sum()
            total_acc = total_acc + acc

    total_acc = total_acc / test_data_size
    print(f"测试集Accuracy: {float_to_percent(total_acc)}")


def float_to_percent(num) -> str:
    # 浮点到百分比表示 保留两位小数
    return "%.2f%%" % (num * 100)
