import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from liquidnet.vision import VisionLiquidNet

device = "cuda" if torch.cuda.is_available() else "cpu"

num_units = 64
num_classes = 10
num_epochs = 10
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = VisionLiquidNet(num_units=num_units, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.hidden_state = model.hidden_state.detach()

        if (i + 1) % 1000 == 0:
            print(
                f"epoch: [{epoch+1}/{num_epochs}], step: [{i+1}/{len(train_loader)}], loss: {loss.item():.4f}"
            )
        
        torch.save(model.state_dict(), "cifar_liquidnet.ckpt")