import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
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

state_dict = torch.load("FashionMNIST.pth")
load_model=Net()
load_model.load_state_dict(state_dict)
load_model.eval()

image,label=test_data[0]
with torch.inference_mode():
    pred=load_model(image)
pred_label=pred.argmax(1).item()
print("预测类别：",classes[pred_label])