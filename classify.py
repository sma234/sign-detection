import torch
import cv2
import numpy as np

from model import SmallSignNet
from classes import CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallSignNet(num_classes=len(CLASSES))
model.load_state_dict(torch.load("traffic_signs.pth", map_location=device))
model.to(device)
model.eval()


def classify(roi):
    img = cv2.resize(roi, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        conf, idx = probs.max(0)

    return CLASSES[idx.item()], float(conf)
