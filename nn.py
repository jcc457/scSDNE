import itertools
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch.optim as optim


def get_laplacian(adj):
    # 计算度矩阵
    deg = torch.sum(adj, dim=1)  # 对每一行求和，得到节点的度
    # 构建度矩阵
    deg_mat = torch.diag(deg)
    # 计算拉普拉斯矩阵
    L = deg_mat - adj
    return L

def loss_1st(y_true, y_pred, alpha):
    L = y_true  # 拉普拉斯矩阵
    Y = y_pred  # 低维潜在特征
    batch_size = L.shape[0]
    # 计算 Y^T * L * Y
    product = torch.chain_matmul(Y.transpose(0, 1), L, Y)
    # 计算迹
    loss = alpha * 2 * torch.trace(product) / batch_size
    return loss

def skew(M,Z):
    ''' return the skew-symmetric part of $M^T Z$'''
    return 0.5 * (M.t()@Z - Z.t()@M)

def proj_stiefel(M,Z):

    MskewMTZ = M@skew(M,Z)
    IMMTZ = (torch.eye(len(M)) - M@M.t())@Z
    return MskewMTZ + IMMTZ

def l2_regularization(model, l2_lambda):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)
    return l2_lambda * l2_loss


# 定义 L1 正则化函数
def l1_regularization(model, l1_lambda):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return l1_lambda * l1_loss

class GraphAutoencoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, device="cpu"):
        super(GraphAutoencoder, self).__init__()
        self.device = device
        input_dim_copy = input_dim
        layers = []
        for layer_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            #layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Sigmoid())
            input_dim = layer_dim
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for layer_dim in reversed(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            #layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Sigmoid())
            input_dim = layer_dim
        # 最后加一层输入的维度
        layers.append(torch.nn.Linear(input_dim, input_dim_copy))
        #layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*layers)
        # torch中的只对weight进行正则真难搞啊
        # self.regularize = Regularization(self.encoder, weight_decay=gamma).to(self.device) + Regularization(self.decoder,weight_decay=gamma).to(self.device

    def forward(self, A):
        '''
        输入节点的领接矩阵和拉普拉斯矩阵，主要计算方式参考论文
        :param A: adjacency_matrix, dim=(m, n)
        :param L: laplace_matrix, dim=(m, m)
        :return:
        '''
        Y = self.encoder(A)
        A_hat = self.decoder(Y)
        return Y, A_hat

class SDNE: # trainer
    def __init__(self,
                 #data_arr,
                 w,
                 dim,
                 n_dim,
                 layers):
        # TODO: seed
        self.w = w
        self.n_dim = n_dim
        self.input_dim = dim
        np.fill_diagonal(self.w, 0)
        self.A = torch.tensor(self.w, dtype=torch.float)
        X = np.eye(self.input_dim)
        self.X = torch.tensor(X, dtype=torch.float)
        self.L = get_laplacian(self.A)

    def train(self,
              num_epochs = 200,
              lr = 0.001,
              verbose = False,
              **optim_kwargs):
        self.losses = []
        criterion = nn.MSELoss()
        gae = GraphAutoencoder(self.input_dim, [self.n_dim])
        optimizer = optim.Adam(gae.parameters(), lr=lr)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            node_embedding, reconstructed_X = gae(self.A)
            # u, _, v = torch.svd(node_embedding, some=True)
            # node_embedding = u @ v.t()
            # loss = criterion(reconstructed_X, A)
            loss = criterion(reconstructed_X, self.A) + loss_1st(self.L, node_embedding, alpha=2) + l2_regularization(gae,l2_lambda=1e-3) + l1_regularization(gae, l1_lambda=1e-3)

            if epoch == 0 or epoch % 10 == 9:
                if verbose:
                    print(epoch + 1, loss.item())
            self.losses.append(loss.item())

            node_embedding.retain_grad()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)

            rgrad = proj_stiefel(node_embedding, node_embedding.grad)
            optimizer.zero_grad()
            # Backpropogate the Rimannian gradient w.r.t proj_outputs
            node_embedding.backward(rgrad)  # backprop(pt)
            optimizer.step()

            """if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')"""
        node_embedding, _ = gae(self.A)
        self.node_embedding = node_embedding.detach().numpy()
        return self.node_embedding
    def plot_losses(self, file_name=None):
        '''plot loss every 100 steps'''
        plt.figure(figsize=(6, 5), dpi=80)
        plt.plot(np.arange(len(self.losses)), self.losses)
        if file_name is not None:
            plt.savefig(file_name, dpi=80)
        plt.show()