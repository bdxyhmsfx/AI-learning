import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

train_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=5,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=5,out_channels=5,kernel_size=5)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear=nn.Linear(5*4*4,10)
        self.sigmoid = nn.Sigmoid()
        self.flatten=nn.Flatten()
    def forward(self,x):
        x=self.conv1(x)
        x=self.sigmoid(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x = self.sigmoid(x)
        x=self.pool2(x)
        x=self.flatten(x)
        logits=self.linear(x)
        return logits
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn=nn.CrossEntropyLoss()
small_CNN=CNN().to(device)
optimizer=torch.optim.SGD(small_CNN.parameters(),lr=0.01)

for epoch in range(5):
    small_CNN.train()
    total_loss=0

    for batch_idx,(data,labels) in enumerate(train_loader):
        data,labels=data.to(device),labels.to(device)
        logits=small_CNN(data)
        loss=loss_fn(logits,labels)
        total_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch{epoch+1}loss:{total_loss/len(train_loader)}")