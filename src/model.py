import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GATConv, BatchNorm, MLP, TopKPooling, global_mean_pool, RGCNConv


class StatementClassfier(nn.Module):
    def __init__(self, embedding_dim=128, encode_dim=128, hidden_dim=32, num_layers=3, dropout=0.2,
                 use_gpu=False) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim  # 128
        self.encode_dim = encode_dim  # 128

        self.hidden_dim = hidden_dim  # 也128吧
        self.num_layers = num_layers  # 3层？
        self.dropout = dropout  # 0.2吧

        self.gpu = use_gpu
        self.th = torch.cuda if use_gpu else torch

        # 网络结构的定义
        # 就用RGCN吧 两层RGCN/RGAT都可以试试看
        self.encoder = StatementEncoder(self.embedding_dim, self.hidden_dim, self.encode_dim, self.gpu, self.dropout)
        self.layer_0 = Sequential('x, edge_index, edge_type', [
            (RGCNConv(in_channels=self.encode_dim, out_channels=self.hidden_dim, num_relations=2, is_sorted=True),
             'x, edge_index,edge_type -> x'),
            nn.ReLU(),
            BatchNorm(self.hidden_dim)
        ])

        self.layer_1 = Sequential('x, edge_index, edge_type', [
            (RGCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, num_relations=2, is_sorted=True),
             'x, edge_index,edge_type -> x'),
            nn.ReLU(),
            BatchNorm(self.hidden_dim)
        ])

        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.hidden_dim, 2))

        self.sm = nn.Softmax(dim=1)

    def forward(self, data):

        def extract(n, ast_x_matrix, ast_edge_index_matrix):
            n = n.transpose(0, 1)[0].tolist()
            ast_x_list = []
            ast_edge_index_list = []

            bias_1 = 0
            bias_2 = 0

            for size in n:
                ast_x = ast_x_matrix[bias_1:bias_1 + size]
                ast_edge_index = ast_edge_index_matrix[:, bias_2:bias_2 + size - 1 if size != 0 else bias_2 + size]
                ast_x_list.append(ast_x)
                ast_edge_index_list.append(ast_edge_index)
                bias_1 = bias_1 + size
                bias_2 = bias_2 + size - 1 if size != 0 else bias_2 + size

            return ast_x_list, ast_edge_index_list

        size = len(data)
        for i in range(size):
            # 一条data 里面有x、edge_index、y，以及多个ast树的特征矩阵和邻接矩阵
            n = data[i]["n"]
            ast_x_matrix = data[i]["ast_x_matrix"]
            ast_edge_index_matrix = data[i]["ast_edge_index_matrix"]

            ast_x_list, ast_edge_index_list = extract(n, ast_x_matrix, ast_edge_index_matrix)

            # 根据获得的多个ast树的特征矩阵和邻接矩阵来更新每个statement节点的特征
            size = len(ast_x_list)
            for j in range(size):
                ast_x = ast_x_list[j]
                ast_edge_index = ast_edge_index_list[j]

                statement_vec = self.encoder(ast_x, ast_edge_index)
                for k in range(self.encode_dim):
                    data[i]['statement'].x[j][k] = statement_vec[0][k]

        # 节点初始化特征学习完以后 开始外层GNN
        x = data['statement'].x
        edge_index = torch.cat([data['cfg']['edge_index'], data['dfg']['edge_index']], 1).long()

        len_1 = data['cfg'].num_edges
        len_2 = data['dfg'].num_edges

        edge_type_1 = torch.zeros(len_1, )
        edge_type_2 = torch.ones(len_2, )
        edge_type = torch.cat([edge_type_1, edge_type_2], -1).int()

        h = self.layer_0(x, edge_index, edge_type)
        h = self.layer_1(h, edge_index, edge_type)
        out = self.mlp(h)
        return out


class StatementEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, encode_dim, use_gpu, dropout) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encode_dim = encode_dim
        self.use_gpu = use_gpu
        self.dropout = dropout

        # 网络结构定义
        # 还是用GNN
        # 用两层GAT
        # 池化暂时先用最简单的gmp看看效果 之后可以改成 TopKPooling/MaxPooling
        self.layer_0 = Sequential('x, edge_index', [
            (GATConv(in_channels=self.embedding_dim, out_channels=self.embedding_dim, heads=3,
                     dropout=self.dropout), 'x, edge_index -> x'),
            nn.ReLU(),
            BatchNorm(self.embedding_dim * 3)
        ])
        self.layer_1 = Sequential('x, edge_index', [
            (GATConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim, heads=1,
                     dropout=self.dropout), 'x, edge_index -> x'),
            nn.ReLU(),
            BatchNorm(self.embedding_dim)
        ])

        self.mlp = nn.Sequential(nn.Linear(self.embedding_dim, self.hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.hidden_dim, self.encode_dim))

    def forward(self, x, edge_index):
        """
        输入是一个N*128的特征矩阵 + 2*(N-1)的邻接矩阵
        需要输出一个 1*128的特征向量来表示这个statement

        :param x:
        :param edge_index:
        :param batch_size:
        :return:
        """

        # 考虑到有的ast树只有一个节点没有边
        # 这样的ast直接把节点的嵌入返回即可
        size = len(x)
        if size > 1:
            batch = torch.zeros(size, ).long()

            h = self.layer_0(x, edge_index)
            h = self.layer_1(h, edge_index)
            h = global_mean_pool(h, batch)
            h = h.relu()
            out = self.mlp(h)
        else:
            out = self.mlp(x)

        return out
