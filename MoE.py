import torch
import torch.nn as nn


class MoE(nn.Module):
    def __init__(self, expert_num, hidden_size, input_size):
        super(MoE, self).__init__()
        self.expert_num = expert_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList([nn.Linear(input_size, hidden_size) for i in range(expert_num)])
        self.gate = nn.Linear(input_size, expert_num)
        self.fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x):  # [user_embedding, item_embedding]
        expert_outputs = []
        for i in range(self.expert_num):
            expert_outputs.append(self.relu(self.experts[i](x)))

        expert_outputs = torch.stack(expert_outputs)
        gate_output = self.softmax(self.gate(x))
        res = torch.zeros(self.hidden_size)
        for i in range(self.expert_num):
            res += gate_output[i] * expert_outputs[i]

        res = self.fc(res)
        return res


if __name__ == "__main__":
    user_embedding = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    item_embedding = torch.tensor([3.0, 6.0, 3.0, 4.0, 8.0])
    input = torch.cat([user_embedding, item_embedding])

    model = MoE(3, 32, input.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input)
        loss = nn.MSELoss()(output, torch.tensor([5.0]))
        loss.backward()
        optimizer.step()
        print("epoch: {}, loss: {} ".format(epoch + 1, loss))
