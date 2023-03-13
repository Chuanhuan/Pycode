# ref:
# [CIFAR10 CNN](https://tomroth.com.au/pytorch-cnn/)
# [hw3](https://colab.research.google.com/drive/15hMu9YiYjE_6HY99UXon2vKGk2KwugWu)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision import datasets
from torchvision.transforms import ToTensor


# test
# a = torch.randn([2,3,4])
# print(a.shape)
# print(a.view(-1,4).shape)

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,                         
    transform = train_tfm, 
    download = True,            
)
test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    transform = test_tfm,
)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                         shuffle=False, num_workers=2)
d_class2idx = train_data.class_to_idx
d_idx2class = dict(zip(d_class2idx.values(), d_class2idx.keys()))


# image (1,28,28)
# check what's diff between train*
print('train_data:', train_data.__dict__)
# print(dir(trainloader))
print('trainlader:', trainloader.__dict__)



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
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 16 * 6 * 6)
        # print(x.shape)
        # x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()




# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# print()

# Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

model.train()
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader): # trainloader has batch_size
    # for i, data in enumerate(train_data, 0):
        # get the inputs
        inputs, labels  = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # print(inputs)
        # break
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            # break

print('Finished Training')

# test data
model.eval()

total = 0  # keeps track of how many images we have processed 
correct = 0  # keeps track of how many correct images our net predicts
with torch.no_grad():
    for i, data in enumerate(testloader): 
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()
    
print("Accuracy: ", correct/total)

class_correct = list(0 for i in range(10))  # Holds how many correct images for the class
class_total = list(0 for i in range(10))  # Holds total images for the class 


with torch.no_grad(): 
    for i, data in enumerate(testloader): 
        images, labels = data 
        outputs = model(images) 
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels)
        for j in range(4): 
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
            
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        d_idx2class[i], 100 * class_correct[i] / class_total[i]))
