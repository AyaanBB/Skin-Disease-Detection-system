import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from preprocess import processing
from test_model import test

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=(224,224),antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224,224)),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class SkinDataset(Dataset):
    def __init__(self,paths,labels,transform=None):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path = self.paths[index]
        image = cv2.imread(img_path)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, label            

X_train,X_test,y_train,y_test = processing()

train_loader = DataLoader(SkinDataset(X_train,y_train,train_transform), batch_size=32, shuffle=True)
test_loader = DataLoader(SkinDataset(X_test,y_test,test_transform),batch_size=32,shuffle=True)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = torch.nn.Linear(model.last_channel, 7)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr =0.001)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for i,(images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print(f'Epoch: {epoch+1}, Batch {i+1}, Loss: {running_loss / 50:.3f}')
            running_loss = 0.0

torch.save(model.state_dict(),'skin_model.pth')
test()

