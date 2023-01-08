import os.path
from typing import Optional, Callable, Union, List, Tuple

import numpy
import pydot as pydot
import torch
from torch_geometric.data import InMemoryDataset, Data


class SingleProjectDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, project=None, dataset_type="train", methods=None):
        self.project = project
        self.methods = methods

        super(SingleProjectDataset, self).__init__(root, transform, pre_transform)

        if dataset_type == "train":
            print(f"{dataset_type} using {self.processed_paths[0]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[0])

        elif dataset_type == "validate":
            print(f"{dataset_type} using {self.processed_paths[1]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[1])

        elif dataset_type == "test":
            print(f"{dataset_type} using {self.processed_paths[2]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        paths = []
        for item in self.methods.values:
            clz = item[0]
            method = item[1]
            path = os.path.join(self.project, clz, method)
            paths.append(path)

        return paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        processed_train_path = os.path.join(self.project, "train", "train_.pt")
        processed_dev_path = os.path.join(self.project, "dev", "dev_.pt")
        processed_test_path = os.path.join(self.project, "test", "test_.pt")
        return [processed_train_path, processed_dev_path, processed_test_path]

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

        datalist = []

        for path in self.raw_paths:
            # 每次遍历对单个函数进行处理 一个函数就是一条数据
            X = None
            cfg_edge_index = None
            dfg_edge_index = None
            Y = None
            ast_x_list = []
            ast_edge_index_list = []

            files = os.listdir(path)
            for file in files:
                if file != 'statements':
                    method_graph_file = os.path.join(path, file)
                    method_graph = pydot.graph_from_dot_file(method_graph_file)
                    method_graph = method_graph[0]
                    x, cfg_edge_index, dfg_edge_index, y = self.process_method_dot(method_graph)

                else:
                    statement_dir = os.path.join(path, 'statements')
                    statement_files = os.listdir(statement_dir)

                    for statement_file in statement_files:
                        statement_ast = os.path.join(statement_dir, statement_file)
                        statement_graph = pydot.graph_from_dot_file(statement_ast)
                        statement_graph = statement_graph[0]
                        ast_x, ast_edge_index = self.process_statement_dot(graph=statement_graph)

                        ast_x_list.append(ast_x)
                        ast_edge_index_list.append(ast_edge_index)

            graph_dict = {
                'x': X,
                'edge_index': cfg_edge_index,
                'y': Y,
                'ast_x_list': ast_x_list,
                'ast_edge_index_list': ast_edge_index_list,
            }

            graph_data = Data.from_dict(graph_dict)
            datalist.append(graph_data)

        if self.dataset_type == "train":
            print("collating train data")
            data, slices = self.collate(datalist)
            torch.save((data, slices), self.processed_paths[0])

        elif self.dataset_type == "validate":
            print("collating validate data")
            data, slices = self.collate(datalist)
            torch.save((data, slices), self.processed_paths[1])

        elif self.dataset_type == "test":
            print("collating test data")
            data, slices = self.collate(datalist)
            torch.save((data, slices), self.processed_paths[2])

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
        torch.as_tensor(numpy.array([1, 2, 4, 3, 6, 3, 15]))
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
        nodes = graph.get_node_list()[:-1]
        node_num = len(nodes)




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
