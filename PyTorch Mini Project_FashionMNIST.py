import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_dataloader=DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits
device=(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model=Net().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
def training_loop(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx,(data,labels)in enumerate(dataloader):
        pred=model(data)
        loss=loss_fn(pred,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx%100==0:
            loss=loss.item()
            current=batch_idx*64+len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test_loop(dataloader,model,loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch_idx,(X,y)in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs=5
for epoch in range(epochs):
    training_loop(training_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)
torch.save(model.state_dict(),"FashionMNIST.pth")
