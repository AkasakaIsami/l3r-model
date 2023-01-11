import os.path
import pickle
from typing import Optional, Callable, Union, List, Tuple
import numpy as np
import pydot as pydot
import torch
from gensim.models import Word2Vec
from torch_geometric.data import InMemoryDataset, Data

from util import cut_word


class SingleProjectDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, project=None, dataset_type="train",
                 train_methods=None, dev_methods=None, test_methods=None):
        self.word2vec = None
        self.embeddings = None
        self.project = project
        self.train_methods = train_methods
        self.dev_methods = dev_methods
        self.test_methods = test_methods

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
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        paths = [self.project]

        return paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        processed_train_path = os.path.join(self.project, "train_.pt")
        processed_dev_path = os.path.join(self.project, "dev_.pt")
        processed_test_path = os.path.join(self.project, "test_.pt")
        return [self.project, processed_train_path, processed_dev_path, processed_test_path]

    def download(self):
        pass

    def process(self):
        """
        每个data对应一个函数
        所以一个data应该包括：
            1 一个代表函数的特征矩阵
            2 一个代表控制流边的边矩阵
            3 一个代表数据流边的边矩阵
            4 一个特征矩阵列表，包含每个语句的AST特征矩阵
            5 一个边矩阵列表，包含每个语句的AST边矩阵
        :return:
        """
        project_root = self.raw_paths[0]

        # 先导入词嵌入矩阵
        word2vec_path = os.path.join(project_root, self.project + '_w2v_128.model')
        word2vec = Word2Vec.load(word2vec_path).wv
        embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
        embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
        self.word2vec = word2vec
        self.embeddings = embeddings

        def build_datalist(methods):
            datalist = []
            for item in methods:
                clz = item[0]
                method = item[1]
                path = os.path.join(project_root, clz, method)

                graph_dict = {}

                files = os.listdir(path)
                for file in files:
                    if file != 'statements':
                        method_graph_file = os.path.join(path, file)
                        method_graph = pydot.graph_from_dot_file(method_graph_file)
                        method_graph = method_graph[0]
                        x, cfg_edge_index, dfg_edge_index, y = self.process_method_dot(method_graph)
                        graph_dict['x'] = x
                        graph_dict['edge_index'] = cfg_edge_index
                        graph_dict['y'] = y
                    else:
                        statement_dir = os.path.join(path, 'statements')
                        # TODO 排序
                        statement_files = os.listdir(statement_dir)

                        n = len(statement_files)

                        ast_x_list = []
                        ast_edge_index_list = []

                        for i in range(n):
                            statement_file = statement_files[i]

                            statement_ast = os.path.join(statement_dir, statement_file)
                            statement_graph = pydot.graph_from_dot_file(statement_ast)
                            statement_graph = statement_graph[0]

                            # 初始化节点特征的时候用了w2v 所以把矩阵导入
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
                            ast_edge_index_matrix = torch.cat(ast_edge_index_list, 1)

                            return n, ast_x_matrix, ast_edge_index_matrix

                        n, ast_x_matrix, ast_edge_index_matrix = list_to_matrix(ast_x_list, ast_edge_index_list)
                        graph_dict['n'] = n
                        graph_dict['ast_x_matrix'] = ast_x_matrix
                        graph_dict['ast_edge_index_matrix'] = ast_edge_index_matrix

                graph_data = Data.from_dict(graph_dict)
                datalist.append(graph_data)
            return datalist

        train_datalist = build_datalist(self.train_methods)
        dev_datalist = build_datalist(self.dev_methods)
        test_datalist = build_datalist(self.test_methods)

        if not os.path.exists(self.processed_paths[0]):
            os.makedirs(self.processed_paths[0])

        # 2023.01.11 2:46 am 感谢维饶帮我debug到凌晨三点
        # 特写此注释 以表感谢
        # 等你回上海 我请你吃生蚝鸡煲
        # 没有阴阳怪气！！
        print("collating train data")
        data, slices = self.collate(train_datalist)
        torch.save((data, slices), self.processed_paths[1])

        print("collating validate data")
        data, slices = self.collate(dev_datalist)
        torch.save((data, slices), self.processed_paths[2])

        print("collating test data")
        data, slices = self.collate(test_datalist)
        torch.save((data, slices), self.processed_paths[3])

    def process_method_dot(self, graph):
        nodes = graph.get_node_list()[:-1]
        node_num = len(nodes)

        # 初始化特征为128维的空向量
        # x: n * 128
        x = torch.zeros([node_num, 128], dtype=torch.float)

        y = []
        for node in nodes:
            label = node.get_attributes()['isLogged'] == '"true"'
            if label:
                y.append([1])
            else:
                y.append([0])

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

        return x, cfg_edge_index, dfg_edge_index, y

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
        nodes = graph.get_node_list()[:-1]

        # 没节点就返回空的
        if len(nodes) == 0:
            return torch.zeros([0, 128]), torch.zeros([2, 0])

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

        edge_0 = torch.as_tensor(edge_0)
        edge_1 = torch.as_tensor(edge_1)
        edge_0 = edge_0.reshape(1, len(edge_0))
        edge_1 = edge_1.reshape(1, len(edge_1))

        edge_index = torch.cat([edge_0, edge_1], dim=0)

        return x, edge_index


class AllProjectsDataset(InMemoryDataset):

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def download(self):
        pass

    def process(self):
        pass
