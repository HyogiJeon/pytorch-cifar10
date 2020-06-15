import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

import torchvision
import torchvision.transforms as transforms

from model.CNN import *
from model.VGG import *
from model.ResNet import *



os.environ["CUDA_VISIBLE_DEVICES"]= "0"

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=False)



# model = CNN().to(device)
model = VGG16().to(device)      # VGG11, VGG13, VGG16, VGG19
# model = ResNet18().to(device)   # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


# model = nn.DataParallel(model)



epochs = 100
learning_rate = 5e-3
momentum = 0.9


optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_function = nn.CrossEntropyLoss()


def train_process(epoch):
    model.train()

    train_loss = 0
    train_correct = 0

    for idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        train_loss += loss
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1, keepdims=True)
        train_correct += pred.eq(labels.view_as(pred)).sum().item()

        if idx % 20 == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}'.format(
                epoch, idx * len(images), len(trainloader.dataset),
                       100 * idx / len(trainloader), loss.item()
            ))

    scheduler.step()

    train_loss /= idx

    print('Train set - Average Loss : {:.4f}, Accuracy : {}/{} ({:.2f}%)'.format(
        train_loss, train_correct, len(trainloader.dataset), 100 * train_correct / len(trainloader.dataset)))


def test_process():
    model.eval()

    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            test_loss += loss_function(outputs, labels)

            pred = outputs.argmax(dim=1, keepdims=True)
            test_correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= idx

    print('Test set - Average Loss : {:.4f}, Accuracy : {}/{} ({:.2f}%)'.format(
        test_loss, test_correct, len(testloader.dataset), 100 * test_correct / len(testloader.dataset)))


for epoch in range(1, epochs + 1):
    train_process(epoch)
    test_process()