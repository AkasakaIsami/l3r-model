import configparser
import os
import pickle

import pandas as pd
import torch
from gensim import corpora, models
from pandas import Series

from dataset import SingleProjectDataset
from test import test
from train import train

'''
因为数据集构建的逻辑要重构 
不想覆盖原来的代码
所以直接重写一遍
'''


class Pipeline:

    def __init__(self, ratio: str, project: str, root: str):
        self.root = root
        self.src_path = os.path.join(root, 'raw')
        self.target_path = os.path.join(root, 'processed')

        use_gpu = cf.getboolean('environment', 'useGPU')
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_gpu = True
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False

        self.embedding_size = None
        self.ratio = ratio
        self.project = project

    def dictionary_and_embedding(self, project, embedding_size):
        """
        :param project: 输入要计算embedding的项目
        :param train_data: 基于训练集构建语料库（懒得写了 暂时先用整个项目做语料库吧）
        :param embedding_size: 要训练的词嵌入大小
        """

        def build_w2v_model(project, embedding_size):
            self.embedding_size = embedding_size
            # 创建过就不要训练了
            corpus_file_path = os.path.join(self.src_path, project, project + '_corpus.txt')
            model_file_name = project + "_w2v_" + str(embedding_size) + '.model'

            save_path = os.path.join(self.src_path, project, model_file_name)
            if os.path.exists(save_path):
                return

            from gensim.models import word2vec

            corpus = word2vec.LineSentence(corpus_file_path)
            w2v = word2vec.Word2Vec(corpus, vector_size=embedding_size, workers=16, sg=1, min_count=3)
            w2v.save(save_path)

        def build_tfidf_model(project):
            docs = []
            # 每个函数和他对应的文档
            # 后面读的时候需要用到他来构建词袋
            name2doc = {}
            corpus_tfidf_file_path = os.path.join(self.src_path, project, project + '_corpus_tfidf.txt')

            info_file_name = project + 'tfidf_info.pickle'
            tfidf_model_file_name = project + '_model.tfidf'

            save_path_info = os.path.join(self.src_path, project, info_file_name)
            save_path_model = os.path.join(self.src_path, project, tfidf_model_file_name)

            if os.path.exists(save_path_info) and os.path.exists(save_path_model):
                return

            with open(corpus_tfidf_file_path, 'r') as file:
                for line in file.readlines():
                    line = line.strip().split(' ')

                    method_name = line[0]
                    method_doc = line[1:]

                    name2doc[method_name] = method_doc
                    docs.append(method_doc)

            # 词典 为语料库中每个单词都赋予一个索引
            dictionary = corpora.Dictionary(docs)
            new_corpus = [dictionary.doc2bow(doc) for doc in docs]
            tfidf = models.TfidfModel(new_corpus)
            tfidf.save(save_path_model)

            # 还要保存dictionary和name2doc
            with open(save_path_info, 'wb') as file:
                pickle.dump((dictionary, name2doc), file)

        build_w2v_model(project, embedding_size)
        # build_tfidf_model(project)

    def get_data(self) -> Series:
        """
        最开始看astnn的源码 看他们是先切数据集所以才能建语料库的
        我上当了
        这个函数不再做切分了
        但是要丢弃所有后缀为Test的类

        :return: 返回所有的函数列表
        """

        all_methods = []
        project_dir = os.path.join(self.src_path, self.project)
        classes = os.listdir(project_dir)

        for clz in classes:

            method_dir = os.path.join(project_dir, clz)
            if os.path.isfile(method_dir):
                continue

            if clz.endswith('Test'):
                continue

            methods = os.listdir(method_dir)
            for method in methods:
                if method == ".DS_Store":
                    continue
                all_methods.append((clz, method))

        data = pd.Series(all_methods)
        data = data.sample(frac=1)

        return data

    def make_dataset(self, methods):
        """
        根据所有函数的名称制作数据集
        默认保留所有不包含日志的函数
        可以设置丢弃一部分不包含日志的函数

        :param methods: 函数列表
        :param drop: 要丢掉多少比例不包含日志的函数
        :return: 返回结果
        """
        cf = configparser.ConfigParser()
        cf.read('config.ini')
        negative_ratio = cf.getint('data', 'negativeRatio')
        drop = cf.getfloat('data', 'drop')

        train_dataset = SingleProjectDataset(root=self.root, project=self.project, dataset_type="train",
                                             methods=methods, ratio=self.ratio, drop=drop,
                                             negative_ratio=negative_ratio)
        # 第一次获取的时候就创建好了 所以不用再传了
        validate_dataset = SingleProjectDataset(root=self.root, project=self.project, dataset_type="validate",
                                                drop=drop, negative_ratio=negative_ratio)
        test_dataset = SingleProjectDataset(root=self.root, project=self.project, dataset_type="test", drop=drop,
                                            negative_ratio=negative_ratio)
        print(f"{len(train_dataset)=} {len(validate_dataset)=} {len(test_dataset)=}")
        return train_dataset, validate_dataset, test_dataset, train_dataset.processed_paths[5]

    def run(self):
        print(f'开始数据预处理（目标项目为{self.project}）...')

        print('词嵌入训练...')
        self.dictionary_and_embedding(self.project, 128)

        print('获取源数据...')
        method_list = self.get_data()

        print('制作数据集...')
        # 这里开始不一样了 切分数据集的工作交给DataSet去做
        train_dataset, validate_dataset, test_dataset, dataset_info = self.make_dataset(method_list)

        # 后面的函数不要了 这里只制作数据集 不训练模型
        print('开始训练...')
        model = train(train_dataset, validate_dataset, os.path.join(self.root, 'model', self.project),
                      dataset_info)

        print('开始测试...')
        # model_file要指定完整路径

        # model_file = ''
        test(model, test_dataset)


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    ratio = cf.get('data', 'datasetRadio')
    project_name = cf.get('data', 'projectName')
    data_src = cf.get('data', 'dataSrc')

    ppl = Pipeline(ratio=ratio, project=project_name, root=data_src)
    ppl.run()
