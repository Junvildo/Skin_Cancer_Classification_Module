from torch.utils.data import Dataset
import torch
import numpy as np

class ClassificationDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        original_image = np.array(item["image"])
        label = item["label"]

        transformed = self.transform(image=original_image)
        image = torch.tensor(transformed['image'])

        # convert to C, H, W
        image = image.permute(2, 0, 1)

        return image, torch.tensor(label, dtype=torch.long), original_image