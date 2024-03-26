import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, dim_k=None, dim_v=None):
        """
        :param embed_dim: 每个特征对应的向量维度
        :param dim_k: 矩阵W^k 和 W^q 的维度
        """
        super(SelfAttention, self).__init__()

        self.embed_dim = embed_dim
        if dim_k is None:
            dim_k = embed_dim
        if dim_v is None:
            dim_v = embed_dim

        # 实际中通常使用线性层来表示需要训练的矩阵
        self.W_q = nn.Linear(embed_dim, dim_k, bias=False)
        self.W_k = nn.Linear(embed_dim, dim_k, bias=False)
        self.W_v = nn.Linear(embed_dim, dim_v, bias=False)

        # 根号d_k
        self._norm_fact = 1 / np.sqrt(dim_k)

    def forward(self, x):
        """
        进行前向传播
        :param x: 输入的特征向量，size为(batch_size, input_num, embed_dim)
        """

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # permute 交换维度
        K_T = K.permute(0, 2, 1)

        # bmm为batch矩阵乘法
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K_T) * self._norm_fact)

        # 最后再乘以V
        output = torch.bmm(atten, V)

        return output


if __name__ == "__main__":
    # embed_dim 为 32
    model = SelfAttention(32, 5, 4)
    # batch_size 为 64, input_num 为 10, embed_dim 为 32
    input = torch.randn(64, 10, 32)
    output = model(input)
    print(output.shape)
