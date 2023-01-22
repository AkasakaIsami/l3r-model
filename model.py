import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GATConv, BatchNorm, TopKPooling, RGCNConv, Linear, \
    global_max_pool, RGATConv


class StatementClassifier(nn.Module):
    def __init__(self, embedding_dim=128, encode_dim=128, hidden_dim=32, encoder_num_layers=2, classifier_num_layers=3,
                 dropout=0.2, use_gpu=False) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.classifier_num_layers = classifier_num_layers
        self.dropout = dropout

        self.gpu = use_gpu

        self.encoder = StatementEncoder(self.embedding_dim, self.hidden_dim, self.encode_dim, self.dropout, self.gpu)
        self.layer_0 = Sequential('x, edge_index, edge_type', [
            (RGCNConv(in_channels=self.encode_dim, out_channels=self.hidden_dim, num_relations=2, is_sorted=True),
             'x, edge_index,edge_type -> x'),
            nn.Sigmoid(),
            BatchNorm(self.hidden_dim)
        ])
        self.layer_1 = Sequential('x, edge_index, edge_type', [
            (RGCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, num_relations=2, is_sorted=True),
             'x, edge_index,edge_type -> x'),
            nn.Sigmoid(),
            BatchNorm(self.hidden_dim)
        ])

        self.conv_0 = Sequential('x, edge_index, edge_type', [
            (RGATConv(self.encode_dim, self.hidden_dim, num_relations=2, heads=3, dropout=self.dropout),
             'x, edge_index,edge_type -> x'),
            nn.Tanh(),
            BatchNorm(self.hidden_dim * 3)
        ])
        self.conv_1 = Sequential('x, edge_index, edge_type', [
            (RGATConv(self.hidden_dim * 3, self.hidden_dim, num_relations=2, heads=1, dropout=self.dropout),
             'x, edge_index,edge_type -> x'),
            nn.Tanh(),
            BatchNorm(self.hidden_dim)
        ])

        self.mlp = nn.Sequential(Linear(self.hidden_dim, self.hidden_dim, weight_initializer='kaiming_uniform'),
                                 nn.ReLU(),
                                 Linear(self.hidden_dim, 2, weight_initializer='kaiming_uniform'))

        # 第二个mlp 对于只有一个节点的图那就过个全连接吧- -
        self.mlp_2 = nn.Sequential(Linear(self.encode_dim, self.hidden_dim, weight_initializer='kaiming_uniform'),
                                   nn.ReLU(),
                                   Linear(self.hidden_dim, 2, weight_initializer='kaiming_uniform'))

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

        def n2batch(n):
            batch = []
            index = 0
            for num in n:
                for k in range(num):
                    batch.append(index)
                index += 1

            batch = torch.tensor(batch).long().cuda() if self.gpu else torch.tensor(batch).long()
            return batch

        statements_vec = self.encoder(data["ast_x_matrix"], data["ast_edge_index_matrix"].long(), n2batch(data['n']))
        data.x = statements_vec
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type
        if x.shape[0] > 1:
            # h = self.layer_0(x, edge_index, edge_type)
            # h = self.layer_1(h, edge_index, edge_type)
            h = self.conv_0(x, edge_index, edge_type)
            h = self.conv_1(h, edge_index, edge_type)
            out = self.mlp(h)
        else:
            out = self.mlp_2(x)
        return out


class StatementEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, encode_dim, dropout, use_gpu) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encode_dim = encode_dim
        self.use_gpu = use_gpu
        self.dropout = dropout

        self.layer_0 = Sequential('x, edge_index', [
            (GATConv(in_channels=self.embedding_dim, out_channels=self.embedding_dim, heads=3,
                     dropout=self.dropout), 'x, edge_index -> x'),
            nn.Tanh(),
            BatchNorm(self.embedding_dim * 3)
        ])
        self.pooling_0 = TopKPooling(self.embedding_dim * 3, ratio=0.7)
        self.layer_1 = Sequential('x, edge_index', [
            (GATConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim, heads=1,
                     dropout=self.dropout), 'x, edge_index -> x'),
            nn.Tanh(),
            BatchNorm(self.embedding_dim)
        ])
        self.pooling_1 = TopKPooling(self.embedding_dim, ratio=0.5)
        self.mlp = nn.Sequential(Linear(self.embedding_dim, self.hidden_dim, weight_initializer='kaiming_uniform'),
                                 nn.ReLU(),
                                 Linear(self.hidden_dim, self.encode_dim, weight_initializer='kaiming_uniform'))

    def forward(self, x, edge_index, batch=None):
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
            if self.use_gpu:
                batch = batch.cuda()
            h = self.layer_0(x, edge_index)
            h, edge_index, _, batch, _, _ = self.pooling_0(h, edge_index, None, batch)
            h = h.relu()
            h = self.layer_1(h, edge_index)
            h, edge_index, _, batch, _, _ = self.pooling_1(h, edge_index, None, batch)
            h = global_max_pool(h, batch)
            h = h.relu()
            out = self.mlp(h)
        else:
            out = self.mlp(x)

        return out
