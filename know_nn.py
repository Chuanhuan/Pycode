import torch.nn as nn
import torch.nn.functional as F
import torch

# MaxPool2d
# m = nn.MaxPool2d((3, 2), stride=(2, 1))
m = nn.MaxPool2d(2, stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m(input)
print(output.shape)
# torch.Size([20, 16, 24, 31])

# relu
m = nn.ReLU()
input = torch.randn(2)
print(input, m(input))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 16 * 6 * 6)
        print(x.shape)
        # x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
print(model)
# input is (1,28,28)
x = torch.randn(1,28,28)
model(x)

