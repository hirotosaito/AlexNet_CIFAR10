import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt


train_dataset = torchvision.datasets.CIFAR10(root      = 'data',\
                                            train      = True,\
                                            transform  = transforms.ToTensor(),\
                                            download   = False)

test_dataset  = torchvision.datasets.CIFAR10(root      = 'data',\
                                            train      = False,\
                                            transform  = transforms.ToTensor(),\
                                            download   = False)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers = 2)
test_loader  = torch.utils.data.DataLoader(dataset = test_dataset,  batch_size = 64, shuffle = True, num_workers = 2)
num_classes  = 10

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 5),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = 'cpu'
net = AlexNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)

num_epochs = 20

train_loss_list = []
train_acc_list  = []
val_loss_list   = []
val_acc_list    = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc  = 0
    val_loss   = 0
    val_acc    = 0

    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs     = net(images)
        loss        = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc  += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc  = train_acc  / len(train_loader.dataset)

    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs   = net(images)
            loss      = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc  += (outputs.max(1)[1] == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc  = val_acc  / len(test_loader.dataset)

    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    train_acc_list.append(avg_train_acc)
    val_acc_list.append(avg_val_acc)


    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)


    print(epoch,avg_train_loss,avg_val_loss)
    print(epoch,avg_train_acc,avg_val_acc)




