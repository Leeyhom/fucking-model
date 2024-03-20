import torch
from torch import nn
import random
import numpy as np
import copy


class PReLU(nn.Module):
    def __init__(self, size):
        super(PReLU, self).__init__()
        self.name = "PReLU"
        self.alpha = torch.zeros((size,))
        self.relu = nn.ReLU()

    def forward(self, x):
        # 正数部分
        pos = self.relu(x)
        neg = self.alpha * (x - abs(x)) * 0.5
        return pos + neg


class Dice(nn.Module):
    def __init__(self, emb_size, eps=1e-8, dim=3):
        super(Dice, self).__init__()
        self.name = "dice"
        self.dim = dim
        # assert self.dim == 2 or self.dim == 3
        self.bn = nn.BatchNorm1d(emb_size, eps=eps)
        self.sig = nn.Sigmoid()
        if dim == 2:  # [B,C] 维度为2时只有batch_size 和 特征数
            self.alpha = torch.zeros((emb_size,))
            self.beta = torch.zeros((emb_size,))
        elif dim == 3:  # [B,C,E] C 为特征数 E为embedding维数
            self.alpha = torch.zeros((emb_size, 1))
            self.beta = torch.zeros((emb_size, 1))

        # 仅用于测试
        elif dim == 1:  # [C] 只有特征数
            self.alpha = torch.zeros((1,))
            self.beta = torch.zeros((1,))

    def forward(self, x):
        if self.dim == 2:
            x_n = self.sig(self.beta * self.bn(x))
            return self.alpha * (1 - x_n) * x + x_n * x
        elif self.dim == 3:
            x = torch.transpose(x, 1, 2)  # 需要将第一维和第二维转置一下，让需要求均值和方差的东西放在列上面，也就是说特征处于列上面
            x_n = self.sig(self.beta * self.bn(x))
            output = self.alpha * (1 - x_n) * x + x_n * x
            output = torch.transpose(output, 1, 2)
            return output
        # 仅用于测试
        elif self.dim == 1:
            x_n = self.sig(self.beta * x)
            return self.alpha * (1 - x_n) * x + x_n * x


class AttentionUnit(nn.Module):
    def __init__(self, inSize, af="Dice", hidden_size=36):
        super(AttentionUnit, self).__init__()
        self.name = "attention_unit"
        self.linear1 = nn.Linear(inSize, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        if af == "Dice":
            self.af = Dice(hidden_size, dim=1)
        elif af == "PReLU":
            self.af = PReLU()
        else:
            print("only dice and prelu can be chosen for activation funtion")

    def forward(self, item1, item2):
        x = torch.cat([item1, item1 * item2, item2, item1 - item2], -1)
        x = self.linear1(x)
        x = self.af(x)
        x = self.linear2(x)
        return x


class DIN(nn.Module):
    def __init__(self, user_num, item_num, cate_num, emb_size=64):
        """
        DIN input parameters
        :param user_num: int numbers of users
        :param item_num: int numbers of items
        :param cata_num: int numbers of categories
        :param emb_size: embedding_size
        """

        super(DIN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.u_emb = nn.Embedding(user_num, emb_size)
        self.i_emb = nn.Embedding(item_num, emb_size)
        self.c_emb = nn.Embedding(cate_num, emb_size)
        self.linear = nn.Sequential(nn.Linear(emb_size * 4, 80), Dice(80, dim=1), nn.Linear(80, 40), nn.Linear(40, 1), nn.Sigmoid())
        self.au = AttentionUnit(4 * emb_size)

    def forword(self, user_id, hist, item_id, cate_id):
        user = self.u_emb(user_id).squeeze()
        item = self.i_emb(item_id).squeeze()
        cate = self.c_emb(cate_id).squeeze()
        h = []
        weights = []
        for i in range(len(hist)):
            hist_i = self.i_emb(hist[i]).squeeze()
            h.append(hist_i.detach())
            weight = self.au(hist_i, item)
            weights.append(weight)

        cur = torch.zeros_like(h[0])
        for i in range(len(h)):
            cur += torch.tensor(weights[i] * h[i], dtype=torch.float32)

        res = torch.cat([user, item, cate, cur], -1)
        res = self.linear(res)
        return res


if __name__ == "__main__":
    # 测试DIN模型
    user_num = 1
    item_num = 3
    cate_num = 1
    emb_size = 64

    model = DIN(user_num, item_num, cate_num, emb_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    user_id = torch.tensor([0])
    item_id = torch.tensor([0])
    cate_id = torch.tensor([0])
    hist = [torch.tensor([1]), torch.tensor([2])]

    for epoch in range(5):
        optimizer.zero_grad()
        output = model.forword(user_id, hist, item_id, cate_id)
        # 定义交叉熵损失函数
        loss = torch.nn.BCELoss()(output, torch.tensor([1.0]))
        loss.backward()
        optimizer.step()
        print("epoch:", epoch, "loss:", loss.item())
