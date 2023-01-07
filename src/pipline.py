import os

import pandas as pd


class Pipeline:
    """
    pipline类，完成训练模型前的准备工作，包括
        预处理数据 把java得到的数据集转换成pytorch可以处理的数据格式
        切分数据集
        对于当前项目 准备embedding矩阵并写入文件

    Args:
    ratio (str): 训练集、验证集、测试集比例
    src_path (str): 读取数据的文件夹
    target_path (str): 读取数据的文件夹
    """

    def __init__(self, ratio: str, src_path: str, target_path: str):
        self.embedding_size = None
        self.ratio = ratio
        self.src_path = src_path
        self.target_path = target_path

    def preprocess_data(self):
        return

    def split_data(self):
        ratios = [int(r) for r in self.ratio.split(':')]

    def dictionary_and_embedding(self, project, embedding_size):
        """
        construct dictionary and train word embedding

        :param project: 输入要计算embedding的项目
        :param embedding_size: 要训练的词嵌入大小
        """
        self.embedding_size = embedding_size
        corpus_file_path = os.path.join(self.src_path, project, 'corpus.txt')

        from gensim.models import word2vec

        corpus = word2vec.LineSentence(corpus_file_path)
        w2v = word2vec.Word2Vec(corpus, size=embedding_size, workers=16, sg=1, min_count=3)

        save_path = os.path.join(self.target_path, project, "w2v_" + str(embedding_size) + ".model")
        w2v.save(save_path)

    def update_ast_node_feature(self, project):
        """
        将当前项目训练集中所有的特征都转换成word embedding

        :param project: 待处理的项目
        :return:
        """
        pass

    def run(self):
        print('读取java模块处理完的数据...')

        projects = os.listdir(self.src_path)

        # all_classes, 存储项目名和对应的类列表
        # e.g. {zookeeper : [zookeeper_AckRequestProcessor, ...]}
        all_classes = {}

        for project in projects:
            project_dir = os.path.join(self.src_path, project)
            classes = os.listdir(project_dir)

            all_classes.update({project: classes})

            for _class in classes:
                class_dir = os.path.join(self.src_path, project, _class)
                methods = os.listdir(class_dir)

        print('切分数据...')

        print('词嵌入训练...')
        self.dictionary_and_embedding("zookeeper", 128)

        print('更新AST的特征矩阵...')
        self.update_ast_node_feature("zookeeper")


ppl = Pipeline('3:1:1', '../data/originalData', '../data/dataset')
ppl.run()
