import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class myNN(nn.Module):
    def __init__(self) -> None:
        super(myNN, self).__init__()
        self.layer1 = nn.Linear(10, 32)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


model = myNN()
# the input must be (* , 10)
x = torch.randn(20, 10)
print(model(x))
print(model.parameters())


plt.plot([1, 2, 3, 4], [11, 33, 22, 11])
plt.show()
