import configparser
import os.path
import pickle
import random
import time
from typing import List
import numpy as np
import pandas as pd
import pydot as pydot
import torch
from gensim import models
from gensim.models import Word2Vec
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from util import cut_word, random_unit, float_to_percent


class SingleProjectDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, project=None, dataset_type="train",
                 methods=None, ratio='8:1:1', drop=0, negative_ratio=4, drop_ast=0):
        self.word2vec = None
        self.embeddings = None
        self.project = project
        self.methods = methods
        self.ratio = ratio
        self.drop = drop
        self.negative_ratio = negative_ratio

        self.LOC = 0
        self.LLOC = 0

        self.drop_ast = drop_ast == 1

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
        dir_name_0 = 'abandon@' + float_to_percent(self.drop)
        dir_name_1 = '1-' + str(self.negative_ratio)
        if self.drop_ast:
            dir_name_1 += '-dropAST'

        dir_name = os.path.join(dir_name_0, dir_name_1)
        processed_train_path = os.path.join(self.project, dir_name, f"train_.pt")
        processed_dev_path = os.path.join(self.project, dir_name, "dev_.pt")
        processed_test_path = os.path.join(self.project, dir_name, "test_.pt")
        processed_test_info = os.path.join(self.project, dir_name, "test_info.pkl")
        processed_dataset_info = os.path.join(self.project, dir_name, 'dataset_info.txt')

        return [os.path.join(self.project, dir_name), processed_train_path, processed_dev_path, processed_test_path,
                processed_test_info, processed_dataset_info]

    def download(self):
        pass

    def process(self):

        project_root = self.raw_paths[0]

        if not os.path.exists(self.processed_paths[0]):
            os.makedirs(self.processed_paths[0])

        record_file = open(self.processed_paths[5], 'w')
        record_file.write(f"??????????????????????????????{self.project}\n")
        record_file.write(f"    -???????????????{len(self.methods)}\n")
        record_file.write(f"    -????????????????????????{float_to_percent(self.drop)}\n")

        # ????????????????????????
        word2vec_path = os.path.join(project_root, self.project + '_w2v_128.model')
        word2vec = Word2Vec.load(word2vec_path).wv
        embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
        embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
        self.word2vec = word2vec
        self.embeddings = embeddings

        # # ????????????tfidf??????
        # info_file_name = self.project + 'tfidf_info.pickle'
        # tfidf_model_file_name = self.project + '_model.tfidf'
        #
        # info_file = os.path.join(self.project_root, self.project, info_file_name)
        # model_file = os.path.join(self.project_root, self.project, tfidf_model_file_name)
        #
        # with open(info_file, 'rb') as file:  # ???with?????????????????????????????????????????????
        #     data = pickle.load(file)
        #     dictionary = data[0]
        #     name2doc = data[1]
        #
        # tfidf = models.TfidfModel.load(model_file)

        # ??????????????????????????????????????????
        mbar = tqdm(self.methods,
                    total=len(self.methods),
                    leave=True,
                    unit_scale=False,
                    colour="red")

        logged_data_dict = pd.DataFrame(columns=['Name', 'Data', 'LOC', 'LLOC', 'Lrate'])
        unlogged_data_dict = pd.DataFrame(columns=['Name', 'Data', 'LOC', 'LLOC', 'Lrate'])

        start = time.time()
        for _, item in enumerate(mbar):
            clz = item[0]
            if clz.endswith('Test'):
                continue

            method = item[1]
            path = os.path.join(project_root, clz, method)
            mbar.set_postfix_str(f"{clz}.{method}")

            # doc_name = clz + '@' + method
            # doc_tfidf = tfidf[dictionary.doc2bow(name2doc[doc_name])]
            # # ??????????????????????????????????????????token????????????
            # doc_dict = {}
            # for id, value in doc_tfidf:
            #     word = dictionary.get(id)
            #     doc_dict[word] = value

            graph_data = {}

            files = os.listdir(path)
            method_graph_file = None
            statement_graphs_file = None
            for file in files:
                if file == '.DS_Store':
                    continue
                elif file.startswith('statements'):
                    statement_graphs_file = file
                else:
                    method_graph_file = file

            # ?????????????????????
            method_graph_path = os.path.join(path, method_graph_file)
            method_graphs = pydot.graph_from_dot_file(method_graph_path)
            method_graph = method_graphs[0]

            tempLOC, tempLLOC, x, cfg_edge_index, dfg_edge_index, y, sample, lines = self.process_method_dot(
                method_graph)
            self.LOC = self.LOC + tempLOC
            self.LLOC = self.LLOC + tempLLOC
            is_all_negative = tempLLOC == 0

            # 30%?????????????????????????????????????????????
            if is_all_negative:
                if random_unit(self.drop):
                    continue

            # ????????????????????????????????????????????????????????????????????? ?????????????????????
            graph_data['is_all_negative'] = is_all_negative

            graph_data['x'] = x
            graph_data['edge_index'] = torch.cat([cfg_edge_index, dfg_edge_index], 1).long()
            graph_data['lines'] = lines

            len_1 = cfg_edge_index.shape[1]
            len_2 = dfg_edge_index.shape[1]
            edge_type_1 = torch.zeros(len_1, )
            edge_type_2 = torch.ones(len_2, )
            edge_type = torch.cat([edge_type_1, edge_type_2], -1).long()

            graph_data['edge_type'] = edge_type
            graph_data['y'] = y.long()
            graph_data['sample'] = sample

            # ?????????????????????
            statements_path = os.path.join(path, statement_graphs_file)
            statement_graphs = pydot.graph_from_dot_file(statements_path)

            # ??????????????????
            if len(statement_graphs) != len(graph_data['x']):
                print(f"!!!!!!!!!!!!!!!!!!{clz}???{method}???????????????????????????")

            ast_x_list = []
            ast_edge_index_list = []

            num_statements = len(statement_graphs)
            for i in range(num_statements):
                statement_graph = statement_graphs[i]
                ast_x, ast_edge_index = self.process_statement_dot(graph=statement_graph, weight=None)
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
            graph_data['id'] = key
            if graph_data['is_all_negative']:
                del graph_data['is_all_negative']
                graph_data = Data.from_dict(graph_data)
                temp_data = [key, graph_data, tempLOC, tempLLOC, tempLLOC / tempLOC]
                unlogged_data_dict.loc[len(unlogged_data_dict)] = temp_data

            else:
                del graph_data['is_all_negative']
                graph_data = Data.from_dict(graph_data)
                temp_data = [key, graph_data, tempLOC, tempLLOC, tempLLOC / tempLOC]
                logged_data_dict.loc[len(logged_data_dict)] = temp_data

        score1 = len(logged_data_dict) / (len(unlogged_data_dict) + len(logged_data_dict))
        score2 = self.LLOC / self.LOC
        print(
            f"?????????????????????")
        print(
            f"???????????????????????????{len(unlogged_data_dict)}???{len(logged_data_dict)}?????????????????????{float_to_percent(score1)}")
        print(
            f"???????????????????????????{self.LOC - self.LLOC}???{self.LLOC}?????????????????????{float_to_percent(score2)}")

        record_file.write(f"?????????????????????\n")
        record_file.write(f"    -???????????????????????????{len(unlogged_data_dict)}\n")
        record_file.write(f"    -???????????????????????????{len(logged_data_dict)}\n")
        record_file.write(f"    -??????????????????????????????{float_to_percent(score1)}\n")

        record_file.write(f"    -???????????????????????????{self.LOC - self.LLOC}\n")
        record_file.write(f"    -???????????????????????????{self.LLOC}\n")
        record_file.write(f"    -??????????????????????????????{float_to_percent(score2)}\n")

        def split_data(unlogged_data_dict: pd.DataFrame, logged_data_dict: pd.DataFrame):
            train_datalist = []
            dev_datalist = []
            test_datalist = []

            test_methods = pd.DataFrame(columns=['Name', 'Data', 'LOC', 'LLOC'])

            ratios = [int(r) for r in self.ratio.split(':')]

            n_unlogged = len(unlogged_data_dict)
            n_logged = len(logged_data_dict)

            train_split_unlogged = int(ratios[0] / sum(ratios) * n_unlogged)
            val_split_unlogged = train_split_unlogged + int(ratios[1] / sum(ratios) * n_unlogged)

            train_datalist.extend(unlogged_data_dict[:train_split_unlogged]['Data'].tolist())
            dev_datalist.extend(unlogged_data_dict[train_split_unlogged:val_split_unlogged]['Data'].tolist())
            test_datalist.extend(unlogged_data_dict[val_split_unlogged:]['Data'].tolist())
            test_methods.append(unlogged_data_dict[val_split_unlogged:])
            # test?????????????????????LOC
            test_LOC = unlogged_data_dict[val_split_unlogged:]['LOC'].sum()

            train_split_logged = int(ratios[0] / sum(ratios) * n_logged)
            val_split_logged = train_split_logged + int(ratios[1] / sum(ratios) * n_logged)

            train_datalist.extend(logged_data_dict[:train_split_logged]['Data'].tolist())
            dev_datalist.extend(logged_data_dict[train_split_logged:val_split_logged]['Data'].tolist())
            test_datalist.extend(logged_data_dict[val_split_logged:]['Data'].tolist())
            test_methods.append(logged_data_dict[val_split_logged:])
            # test?????????????????????LOC???LLOC
            test_LOC += logged_data_dict[val_split_logged:]['LOC'].sum()
            test_LLOC = logged_data_dict[val_split_logged:]['LLOC'].sum()

            return train_datalist, dev_datalist, test_datalist, test_methods, test_LOC, test_LLOC

        def save_data(train_datalist: list, dev_datalist: list, test_datalist: list, test_methods: pd.DataFrame):
            # 2023.01.11 2:46 am ??????????????????debug???????????????
            # ??????????????? ????????????
            # ??????????????? ????????????????????????
            # ????????????????????????

            end = time.time()
            print(f"?????????????????????????????????{end - start}??????????????????????????????...")

            print("collating train data")
            data, slices = self.collate(train_datalist)
            torch.save((data, slices), self.processed_paths[1])

            print("collating validate data")
            data, slices = self.collate(dev_datalist)
            torch.save((data, slices), self.processed_paths[2])

            print("collating test data")
            data, slices = self.collate(test_datalist)
            torch.save((data, slices), self.processed_paths[3])

            print("saving test info")
            test_methods.to_pickle(self.processed_paths[4])

        train_datalist, dev_datalist, test_datalist, test_methods, test_LOC, test_LLOC = split_data(unlogged_data_dict,
                                                                                                    logged_data_dict)
        print(
            f"??????????????????????????????????????????{len(train_datalist)},??????????????????{len(dev_datalist)},??????????????????{len(test_datalist)}")
        record_file.write(
            f"??????????????????????????????????????????{len(train_datalist)},??????????????????{len(dev_datalist)},??????????????????{len(test_datalist)}\n")
        record_file.write(
            f"?????????????????????{test_LOC}???????????????????????????{test_LLOC}?????????????????????{float_to_percent(test_LLOC / test_LOC)}")
        save_data(train_datalist, dev_datalist, test_datalist, test_methods)
        record_file.close()

    def process_method_dot(self, graph):
        """
        ???????????????dot?????????????????????????????????

        :param graph:
        :return:
        """
        nodes = graph.get_node_list()
        if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
            nodes = graph.get_node_list()[:-1]
        node_num = len(nodes)
        x = torch.zeros([node_num, 128], dtype=torch.float)

        tempLOC = 0
        tempLLOC = 0
        y = []

        # ??????????????????????????? ???????????????????????????
        lines = []

        # indices?????????????????????????????????
        # ??????????????????????????????
        # ??????????????????????????????????????????loss???????????????????????????
        node_num = len(nodes)
        indices_all = list(range(node_num))
        indices_sampled = []

        for i in range(node_num):
            node = nodes[i]

            line = node.get_attributes()['line']
            lines.append([int(line)])

            tempLOC = tempLOC + 1
            if 'true' in node.get_attributes()['isLogged']:
                indices_all.remove(i)
                indices_sampled.append(i)
                tempLLOC = tempLLOC + 1
                y.append([0, 1])
            else:
                y.append([1, 0])

        if self.negative_ratio < 0:
            num_negative = len(indices_all)
        else:
            num_negative = tempLLOC * self.negative_ratio

        indices_sampled.extend(
            indices_all if len(indices_all) <= num_negative else random.sample(indices_all, num_negative))

        sample = []
        for i in range(node_num):
            if i not in indices_sampled:
                sample.append([0])
            else:
                sample.append([1])

        y = torch.as_tensor(y)
        sample = torch.as_tensor(sample)
        lines = torch.as_tensor(lines)

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

        return tempLOC, tempLLOC, x, cfg_edge_index, dfg_edge_index, y, sample, lines

    def process_statement_dot(self, graph, weight):
        """
        ??????????????????ST-AST??????????????????????????????
        ?????????????????????????????????????????????

        :param weight:
        :param graph: ST-AST
        :return: ???????????????????????????
        """

        def word_to_vec(token):
            """
            ???????????????
            :param token:
            :return: ??????????????????????????????ndarray
            """
            max_token = self.word2vec.vectors.shape[0]
            index = [self.word2vec.key_to_index[token] if token in self.word2vec.key_to_index else max_token]
            return self.embeddings[index]

        def tokens_to_embedding(tokens, weight):
            """
            ?????????token??????????????? ?????????????????????????????????
            ???????????????????????????

            :param tokens:?????????token??????
            :return: ?????????????????????
            """
            result = torch.zeros([1, 128], dtype=torch.float)

            for token in tokens:
                token_embedding = torch.from_numpy(word_to_vec(token))
                if weight is not None:
                    token_weight = weight[token] if weight.has_key(token) else 0
                    token_embedding = token_embedding * token_weight
                result = result + token_embedding

            count = len(tokens)
            result = result / count
            return result

        if self.drop_ast is not True:
            x = []
            nodes = graph.get_node_list()
            if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
                nodes = graph.get_node_list()[:-1]

            # ???????????????????????????
            if len(nodes) == 0:
                return torch.zeros([1, 128]), torch.zeros([2, 0])

            for node in nodes:
                node_str = node.get_attributes()['label']
                # token ????????????????????????????????????
                tokens = cut_word(node_str)
                # ???token?????????????????????????????????
                node_embedding = tokens_to_embedding(tokens, weight)
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

        else:
            # ????????????AST ?????????????????????????????????
            nodes = graph.get_node_list()
            if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
                nodes = graph.get_node_list()[:-1]

            # ???????????????????????????
            if len(nodes) == 0:
                return torch.zeros([1, 128]), torch.zeros([2, 0])

            node = nodes[0]
            node_str = node.get_attributes()['label']
            # token ????????????????????????????????????
            tokens = cut_word(node_str)
            # ???token?????????????????????????????????
            node_embedding = tokens_to_embedding(tokens, weight)

            x = node_embedding
            edge_index = torch.zeros(2, 0).long()
            return x, edge_index
