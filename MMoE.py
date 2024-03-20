import torch
import torch.nn as nn


class MMoE(nn.Module):
    def __init__(self, expert_num, task_num, hidden_size, input_size):
        super(MMoE, self).__init__()
        self.expert_num = expert_num
        self.task_num = task_num
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.experts = nn.ModuleList([nn.Linear(input_size, hidden_size) for i in range(expert_num)])
        self.gates = nn.ModuleList([nn.Linear(input_size, expert_num) for i in range(task_num)])
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(task_num)])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):  # [user_embedding, item_embedding]
        expert_outputs = []
        gate_outputs = []

        # 每个专家塔的输出
        for i in range(self.expert_num):
            expert_outputs.append(self.relu(self.experts[i](x)))

        # 每个任务对应的门控
        for i in range(self.task_num):
            gate_outputs.append(self.softmax(self.gates[i](x)))

        expert_outputs = torch.stack(expert_outputs)
        gate_outputs = torch.stack(gate_outputs)
        res = []

        # 对每个任务求专家塔的组合
        for i in range(self.task_num):
            tmp = torch.zeros(self.hidden_size)
            for j in range(self.expert_num):
                tmp += gate_outputs[i][j] * expert_outputs[j]
            res.append(tmp)

        res = torch.stack(res)
        out = []

        # 对每个任务塔最后过一个mlp
        for i in range(self.task_num):
            out.append(self.fcs[i](res[i]))

        out = torch.stack(out)
        return out


if __name__ == "__main__":
    user_embedding = torch.tensor([1.0, 2.0, 4.0, 3.0, 7.0])
    item_embedding = torch.tensor([5.0, 7.0, 3.0, 8.0, 1.0])
    input = torch.cat([user_embedding, item_embedding])

    model = MMoE(5, 3, 32, input.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input)
        loss = nn.MSELoss()(output.squeeze(), torch.tensor([2.0, 3.0, 1.0]))
        loss.backward()
        optimizer.step()
        print("epoch: {}, loss: {}".format(epoch + 1, loss))
