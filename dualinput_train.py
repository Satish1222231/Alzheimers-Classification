import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_vgg = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_eff = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data_vgg = datasets.ImageFolder("data/train", transform=transform_vgg)
train_data_eff = datasets.ImageFolder("data/train", transform=transform_eff)
val_data_vgg = datasets.ImageFolder("data/val", transform=transform_vgg)
val_data_eff = datasets.ImageFolder("data/val", transform=transform_eff)

train_loader_vgg = DataLoader(train_data_vgg, batch_size=8, shuffle=True)
train_loader_eff = DataLoader(train_data_eff, batch_size=8, shuffle=True)
val_loader_vgg = DataLoader(val_data_vgg, batch_size=8, shuffle=False)
val_loader_eff = DataLoader(val_data_eff, batch_size=8, shuffle=False)

vgg = models.vgg16(pretrained=True)
eff = models.efficientnet_b4(pretrained=True)

for param in vgg.parameters():
    param.requires_grad = False
for param in eff.parameters():
    param.requires_grad = False

vgg.classifier = nn.Identity()
eff.classifier = nn.Identity()

class DualInput(nn.Module):
    def __init__(self, vgg, eff, num_classes=4):
        super(DualInput, self).__init__()
        self.vgg = vgg
        self.eff = eff
        self.fc = nn.Sequential(
            nn.Linear(25088 + 1792, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x1, x2):
        x1 = self.vgg.features(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self.eff.features(x2)
        x2 = torch.flatten(x2, 1)
        out = torch.cat((x1, x2), dim=1)
        return self.fc(out)

model = DualInput(vgg, eff).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    for (i1, l1), (i2, l2) in zip(train_loader_vgg, train_loader_eff):
        i1, l1, i2 = i1.to(device), l1.to(device), i2.to(device)
        optimizer.zero_grad()
        outputs = model(i1, i2)
        loss = criterion(outputs, l1)
        loss.backward()
        optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (i1, l1), (i2, l2) in zip(val_loader_vgg, val_loader_eff):
            i1, l1, i2 = i1.to(device), l1.to(device), i2.to(device)
            outputs = model(i1, i2)
            _, predicted = torch.max(outputs, 1)
            total += l1.size(0)
            correct += (predicted == l1).sum().item()
    print(f"Epoch {epoch+1}, Val Accuracy: {100 * correct / total:.2f}%")

torch.save(model, "dualinput_alzheimers_model.pt")
