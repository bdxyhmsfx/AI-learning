import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import Net

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

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

model = Net()
model.load_state_dict(
    torch.load(
        "FashionMNIST.pth",
        map_location="cpu"
    )
)
model.eval()
image, label = test_data[0]
image = image.unsqueeze(0)
with torch.no_grad():
    pred = model(image)
pred_label = pred.argmax(1).item()
print("预测：", classes[pred_label])
print("真实：", classes[label])
print("预测完成")