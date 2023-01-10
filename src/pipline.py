import os

import pandas as pd
from dataset import SingleProjectDataset


class Pipeline:
    """
    pipline类，完成训练模型前的准备工作，包括
        预处理数据 把java得到的数据集转换成pytorch可以处理的数据格式
        切分数据集
        对于当前项目 准备embedding矩阵并写入文件

    Args:
    ratio (str): 训练集、验证集、测试集比例
    project (str): 指定作为目标数据的项目
    src_path (str): 读取数据的文件夹
    target_path (str): 读取数据的文件夹
    """

    def __init__(self, ratio: str, project: str, src_path: str, target_path: str):
        self.embedding_size = None
        self.ratio = ratio
        self.project = project
        self.src_path = os.path.join(src_path, project)
        self.target_path = target_path

    def split_data(self):
        """
        读取目标项目下的所有的函数目录 然后按比例进行切割

        :return:
        """
        ratios = [int(r) for r in self.ratio.split(':')]

        all_methods = []
        classes = os.listdir(self.src_path)
        for clz in classes:
            if clz == ".DS_Store":
                continue
            if clz == self.project + '_corpus.txt' or clz == self.project + '_w2v_128.model':
                continue

            method_dir = os.path.join(self.src_path, clz)
            methods = os.listdir(method_dir)
            for method in methods:
                if method == ".DS_Store":
                    continue
                all_methods.append((clz, method))

        method_num = len(all_methods)

        train_split = int(ratios[0] / sum(ratios) * method_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * method_num)

        data = pd.Series(all_methods)
        data.sample(frac=1, random_state=666)

        train = data.loc[:train_split]
        dev = data.loc[train_split:val_split]
        test = data.loc[val_split:]

        return train, dev, test

    def make_dataset(self, train, dev, test):
        train_dataset = SingleProjectDataset(root="../data", project=self.project, dataset_type="train", methods=train)
        validate_dataset = SingleProjectDataset(root="../data", project=self.project, dataset_type="validate",
                                                methods=dev)
        test_dataset = SingleProjectDataset(root="../data", project=self.project, dataset_type="test", methods=test)

        print(f"{len(train_dataset)=} {len(validate_dataset)=} {len(test_dataset)=}")
        return test_dataset, validate_dataset, test_dataset

    def dictionary_and_embedding(self, project, train_data, embedding_size):
        """
        construct dictionary and train word embedding

        :param project: 输入要计算embedding的项目
        :param train_data: 基于训练集构建语料库（懒得写了 暂时先用整个项目做语料库吧）
        :param embedding_size: 要训练的词嵌入大小
        """
        self.embedding_size = embedding_size
        corpus_file_path = os.path.join(self.src_path, project + '_corpus.txt')

        from gensim.models import word2vec

        corpus = word2vec.LineSentence(corpus_file_path)
        w2v = word2vec.Word2Vec(corpus, vector_size=embedding_size, workers=16, sg=1, min_count=3)

        model_file_name = project + "_w2v_" + str(embedding_size) + '.model'

        save_path = os.path.join(self.src_path, model_file_name)
        if not os.path.exists(save_path):
            w2v.save(save_path)

    def run(self):
        print('切分数据...')
        train, dev, test = self.split_data()

        print('词嵌入训练...')
        self.dictionary_and_embedding("zookeeper", train, 128)

        print('制作数据集...')
        self.make_dataset(train, dev, test)

        print('开始训练...')


ppl = Pipeline('3:1:1', 'zookeeper', '../data/raw', '../data/processed')
ppl.run()
