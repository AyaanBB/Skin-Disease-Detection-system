import torch
from torchvision import models
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from preprocess import processing

model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 7)

def test() -> None: 
    try:
        model.load_state_dict(torch.load('skin_model.pth', map_location=torch.device('cpu')))
        model.eval()
        print('.pth file Successfully loaded')
        print(f'Sample weights: {model.classifier[1].weight[0][:5]}')
    except Exception as e:
        print(f'Error: {e}')

test()

_, X_test, _, y_test = processing()

def predict(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to('cpu')

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(prob,1)

    classes = ['akiec','bcc','bkl','df','mel','nv','vasc']
    plt.imshow(image)
    plt.title(f'{classes[pred.item()]} ({conf.item()*100:.1f}%)')
    plt.show()

predict(X_test[0])

