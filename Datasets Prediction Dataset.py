import torch
from torch.utils.data import Dataset

class PredictionDataset(Dataset):
    def __init__(self, dataset):
        """
        Initialize the prediction dataset.

        :param dataset: A list of dictionaries containing features and image data, formatted as:
                        [{'features': [...], 'image_data': [...], 'coordinates': [...]}, ...]
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a single sample, including 1D features, 2D image data, and its coordinates.

        :param idx: Index of the sample
        :return: Features (1D), image data (2D), and coordinates
        """
        sample = self.dataset[idx]
        features_1d = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        image_data_2d = torch.tensor(sample['image_data'], dtype=torch.float32).permute(2, 0, 1)  # Adjust channel dimensions
        return features_1d, image_data_2d, sample['coordinates']