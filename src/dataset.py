import functools
import os.path
import random
import time
from typing import Optional, Callable, Union, List, Tuple
import numpy as np
import pydot as pydot
import torch
from gensim.models import Word2Vec
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from util import cut_word, random_unit, float_to_percent


class SingleProjectDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, project=None, dataset_type="train",
                 methods=None, ratio='8:1:1'):
        self.word2vec = None
        self.embeddings = None
        self.project = project
        self.methods = methods
        self.ratio = ratio

        super(SingleProjectDataset, self).__init__(root, transform, pre_transform)

        if dataset_type == "train":
            print(f"{dataset_type} using {self.processed_paths[1]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[1])

        elif dataset_type == "validate":
            print(f"{dataset_type} using {self.processed_paths[2]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[2])

        elif dataset_type == "test":
            print(f"{dataset_type} using {self.processed_paths[3]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[3])

    @property
    def raw_file_names(self) -> List[str]:
        paths = [self.project]
        return paths

    @property
    def processed_file_names(self) -> List[str]:
        processed_train_path = os.path.join(self.project, "train_.pt")
        processed_dev_path = os.path.join(self.project, "dev_.pt")
        processed_test_path = os.path.join(self.project, "test_.pt")
        return [self.project, processed_train_path, processed_dev_path, processed_test_path]

    def download(self):
        pass

    def process(self):
        project_root = self.raw_paths[0]

        # 先导入词嵌入矩阵
        word2vec_path = os.path.join(project_root, self.project + '_w2v_128.model')
        word2vec = Word2Vec.load(word2vec_path).wv
        embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
        embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
        self.word2vec = word2vec
        self.embeddings = embeddings

        # 开始根据传入的函数制作数据集
        mbar = tqdm(self.methods,
                    total=len(self.methods),
                    leave=True,
                    unit_scale=False,
                    colour="red")

        # 逻辑就是不再像之前一样分成三份分别做
        # 而是做成一大份
        # 但这一大份会被分成两小份
        # 一份里面全装没日志的函数
        # 还有一份里装有日志的函数
        # 做完这一大份 再去分三份

        # 这里不存data列表而存字典是因为要知道每条数据对应哪个函数
        logged_data_dict = {}
        unlogged_data_dict = {}

        start = time.time()
        for _, item in enumerate(mbar):
            clz = item[0]
            if clz.endswith('Test'):
                continue

            method = item[1]
            path = os.path.join(project_root, clz, method)
            mbar.set_postfix_str(f"{clz}.{method}")

            files = os.listdir(path)
            # 这里的优化是 强制先解析函数文件
            # 如果是无日志的函数 40%概率丢弃这条数据

            graph_data = {}

            method_graph_file = None
            statement_graphs_file = None

            for file in files:
                if file == '.DS_Store':
                    continue
                elif file.startswith('statements'):
                    statement_graphs_file = file
                else:
                    method_graph_file = file

            # 开始解析函数图
            method_graph_path = os.path.join(path, method_graph_file)
            method_graphs = pydot.graph_from_dot_file(method_graph_path)
            method_graph = method_graphs[0]

            is_all_negative, x, cfg_edge_index, dfg_edge_index, y = self.process_method_dot(method_graph)

            # 如果is_all_negative是True，意味着当前函数不存在日志语句
            # 30%的概率丢弃当前函数
            if is_all_negative:
                if random_unit(0.5):
                    continue

            # 这个语句在最后将单条数据添加到数据集的时候要用 训练的时候不用
            graph_data['is_all_negative'] = is_all_negative

            graph_data['x'] = x
            graph_data['edge_index'] = torch.cat([cfg_edge_index, dfg_edge_index], 1).long()

            len_1 = cfg_edge_index.shape[1]
            len_2 = dfg_edge_index.shape[1]
            edge_type_1 = torch.zeros(len_1, )
            edge_type_2 = torch.ones(len_2, )
            edge_type = torch.cat([edge_type_1, edge_type_2], -1).long()

            graph_data['edge_type'] = edge_type
            graph_data['y'] = y.long()

            # 解析所有语句图
            statements_path = os.path.join(path, statement_graphs_file)
            statement_graphs = pydot.graph_from_dot_file(statements_path)

            # 简单做个验证
            if len(statement_graphs) != len(graph_data['x']):
                print(f"!!!!!!!!!!!!!!!!!!{clz}的{method}解析的有问题！！！")

            ast_x_list = []
            ast_edge_index_list = []
            for statement_graph in statement_graphs:
                ast_x, ast_edge_index = self.process_statement_dot(graph=statement_graph)
                ast_x_list.append(ast_x)
                ast_edge_index_list.append(ast_edge_index)

            def list_to_matrix(ast_x_list, ast_edge_index_list):
                n = []
                for ast_x in ast_x_list:
                    temp_n = len(ast_x)
                    n.append([temp_n])
                n = torch.tensor(n)

                ast_x_matrix = torch.cat(ast_x_list, 0)
                ast_edge_index_matrix = torch.cat(ast_edge_index_list, 1).int()

                return n, ast_x_matrix, ast_edge_index_matrix

            n, ast_x_matrix, ast_edge_index_matrix = list_to_matrix(ast_x_list, ast_edge_index_list)
            graph_data['n'] = n
            graph_data['ast_x_matrix'] = ast_x_matrix
            graph_data['ast_edge_index_matrix'] = ast_edge_index_matrix

            key = clz + '@' + method
            if graph_data['is_all_negative']:
                del graph_data['is_all_negative']
                graph_data = Data.from_dict(graph_data)
                unlogged_data_dict[key] = graph_data
            else:
                del graph_data['is_all_negative']
                graph_data = Data.from_dict(graph_data)
                logged_data_dict[key] = graph_data

        score = len(logged_data_dict) / (len(unlogged_data_dict) + len(logged_data_dict))
        print(
            f"完成数据读取，正数据分别是{len(unlogged_data_dict)}和{len(logged_data_dict)}，正样本比例为{float_to_percent(score)}")

        def split_data(unlogged_data_dict: dict, logged_data_dict: dict):
            train_datalist = []
            dev_datalist = []
            test_datalist = []

            test_methods = []

            ratios = [int(r) for r in self.ratio.split(':')]

            n_unlogged = len(unlogged_data_dict)
            n_logged = len(logged_data_dict)

            train_split_unlogged = int(ratios[0] / sum(ratios) * n_unlogged)
            val_split_unlogged = train_split_unlogged + int(ratios[1] / sum(ratios) * n_unlogged)
            train_datalist.extend(list(unlogged_data_dict.values())[:train_split_unlogged])
            dev_datalist.extend(list(unlogged_data_dict.values())[train_split_unlogged:val_split_unlogged])
            test_datalist.extend(list(unlogged_data_dict.values())[val_split_unlogged:])
            test_methods.extend(list(unlogged_data_dict.keys())[val_split_unlogged:])

            train_split_logged = int(ratios[0] / sum(ratios) * n_logged)
            val_split_logged = train_split_logged + int(ratios[1] / sum(ratios) * n_logged)
            train_datalist.extend(list(logged_data_dict.values())[:train_split_logged])
            dev_datalist.extend(list(logged_data_dict.values())[train_split_logged:val_split_logged])
            test_datalist.extend(list(logged_data_dict.values())[val_split_logged:])
            test_methods.extend(list(logged_data_dict.keys())[val_split_logged:])

            # 因为是根据标签顺序拼接的 所以打乱一下train_datalist 和 dev_datalist
            random.shuffle(train_datalist)
            random.shuffle(dev_datalist)

            return train_datalist, dev_datalist, test_datalist, test_methods

        def save_data(train_datalist: list, dev_datalist: list, test_datalist: list, test_methods: list):
            # 2023.01.11 2:46 am 感谢维饶帮我debug到凌晨三点
            # 特写此注释 以表感谢
            # 等你回上海 我请你吃生蚝鸡煲
            # 没有阴阳怪气！！
            if not os.path.exists(self.processed_paths[0]):
                os.makedirs(self.processed_paths[0])

            end = time.time()
            print(f"数据集制作完成，共耗时{end - start}秒。现在开始保存数据...")

            print("collating train data")
            data, slices = self.collate(train_datalist)
            torch.save((data, slices), self.processed_paths[1])

            print("collating validate data")
            data, slices = self.collate(dev_datalist)
            torch.save((data, slices), self.processed_paths[2])

            print("collating test data")
            data, slices = self.collate(test_datalist)
            torch.save((data, slices), self.processed_paths[3])

            # TODO: 函数名的保存以后再写吧！

        train_datalist, dev_datalist, test_datalist, test_methods = split_data(unlogged_data_dict, logged_data_dict)
        print(
            f"完成数据集切分，训练集数据量{len(train_datalist)},验证集数据量{len(dev_datalist)},测试集数据量{len(test_datalist)}")
        save_data(train_datalist, dev_datalist, test_datalist, test_methods)

    def process_method_dot(self, graph):
        """
        处理函数的dot，返回当前函数的图结构
        首个参数意味着这个函数是不是全是负样本函数……也就是说不存在日志语句

        :param graph:
        :return:
        """
        nodes = graph.get_node_list()
        if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
            nodes = graph.get_node_list()[:-1]
        node_num = len(nodes)

        # 初始化特征为128维的空向量
        # x: n * 128
        x = torch.zeros([node_num, 128], dtype=torch.float)

        is_all_negative = True
        y = []
        for node in nodes:
            label = 'true' in node.get_attributes()['isLogged']
            if label:
                y.append([0, 1])
                is_all_negative = False
            else:
                y.append([1, 0])

        y = torch.as_tensor(y)

        edges = graph.get_edge_list()

        edge_0_cfg = []
        edge_1_cfg = []
        edge_0_dfg = []
        edge_1_dfg = []

        for edge in edges:
            source = int(edge.get_source()[1:])
            destination = int(edge.get_destination()[1:])
            color = edge.get_attributes()['color']

            if color == 'red':
                edge_0_cfg.append(source)
                edge_1_cfg.append(destination)
            elif color == 'green':
                edge_0_dfg.append(source)
                edge_1_dfg.append(destination)

        edge_0_cfg = torch.as_tensor(edge_0_cfg)
        edge_1_cfg = torch.as_tensor(edge_1_cfg)
        edge_0_cfg = edge_0_cfg.reshape(1, len(edge_0_cfg))
        edge_1_cfg = edge_1_cfg.reshape(1, len(edge_1_cfg))

        edge_0_dfg = torch.as_tensor(edge_0_dfg)
        edge_1_dfg = torch.as_tensor(edge_1_dfg)
        edge_0_dfg = edge_0_dfg.reshape(1, len(edge_0_dfg))
        edge_1_dfg = edge_1_dfg.reshape(1, len(edge_1_dfg))

        cfg_edge_index = torch.cat([edge_0_cfg, edge_1_cfg], dim=0)
        dfg_edge_index = torch.cat([edge_0_dfg, edge_1_dfg], dim=0)

        return is_all_negative, x, cfg_edge_index, dfg_edge_index, y

    def process_statement_dot(self, graph):
        """
        这个函数返回ST-AST的特征矩阵和邻接矩阵
        特征矩阵需要根据语料库构建……

        :param graph: ST-AST
        :return: 特征矩阵和邻接矩阵
        """

        def word_to_vec(token):
            """
            词转词嵌入
            :param token:
            :return: 返回一个代表词嵌入的ndarray
            """
            max_token = self.word2vec.vectors.shape[0]
            index = [self.word2vec.key_to_index[token] if token in self.word2vec.key_to_index else max_token]
            return self.embeddings[index]

        def tokens_to_embedding(tokens):
            """
            对于多token组合的节点 可以有多种加权求和方式
            这里简单的求平均先

            :param tokens:节点的token序列
            :return: 最终的节点向量
            """
            result = torch.zeros([1, 128], dtype=torch.float)

            for token in tokens:
                token_embedding = torch.from_numpy(word_to_vec(token))
                result = result + token_embedding

            count = len(tokens)
            result = result / count
            return result

        x = []
        nodes = graph.get_node_list()
        if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
            nodes = graph.get_node_list()[:-1]

        # 没节点就一个随机的
        if len(nodes) == 0:
            return torch.zeros([1, 128]), torch.zeros([2, 0])

        for node in nodes:
            node_str = node.get_attributes()['label']
            # token 可能是多种形势，要先切分
            tokens = cut_word(node_str)
            # 多token可以考虑不同的合并方式
            node_embedding = tokens_to_embedding(tokens)
            x.append(node_embedding)

        x = torch.cat(x)

        edges = graph.get_edge_list()
        edge_0 = []
        edge_1 = []

        for edge in edges:
            source = int(edge.get_source()[1:])
            destination = int(edge.get_destination()[1:])
            edge_0.append(source)
            edge_1.append(destination)

        edge_0 = torch.as_tensor(edge_0, dtype=torch.int)
        edge_1 = torch.as_tensor(edge_1, dtype=torch.int)
        edge_0 = edge_0.reshape(1, len(edge_0))
        edge_1 = edge_1.reshape(1, len(edge_1))

        edge_index = torch.cat([edge_0, edge_1], dim=0)

        return x, edge_index
