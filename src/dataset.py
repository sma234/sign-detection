import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from classes import CLASSES

class SignDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.root = root
        self.classes = CLASSES
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            folder = os.path.join(root, cls)
            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(folder, file), self.class_to_index[cls])
                    )

        self.transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
