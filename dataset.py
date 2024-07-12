import os
import glob
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class PalettedImageDataset(Dataset):
    def __init__(self, folder, square_length):
        """
        Initializes the dataset by preloading all images and converting them to palette indices.
        
        :param folder: The folder containing PNG images.
        :param square_length: The length of the square region around y.
        """
        if square_length % 2 == 0:
            raise ValueError("Square length must be odd.")
        
        self.images = []
        self.square_length = square_length
        self.half_length = square_length // 2

        png_files = glob.glob(os.path.join(folder, "*.png"))
        if not png_files:
            raise ValueError("No PNG files found in the folder.")

        for file in png_files:
            with Image.open(file) as image:
                if image.mode != "P":
                    raise ValueError(f"Image {file} is not a palette image.")
                self.images.append(np.array(image))

        self.image_shapes = [(image.shape[0], image.shape[1]) for image in self.images]
    
    def __len__(self):
        return len(self.images) * sum([height * width for height, width in self.image_shapes])
    
    def __getitem__(self, index):
        image_index = random.randint(0, len(self.images) - 1)
        image = self.images[image_index]
        height, width = self.image_shapes[image_index]
        
        y_pos = random.randint(0, height - 1)
        x_pos = random.randint(0, width - 1)
        
        y = image[y_pos, x_pos]
        
        x = np.zeros((self.square_length, self.square_length), dtype=int)
        
        for i in range(-self.half_length, self.half_length + 1):
            for j in range(-self.half_length, self.half_length + 1):
                if i == 0 and j == 0:
                    continue
                x_idx = (x_pos + i) % width
                y_idx = (y_pos + j) % height
                x[i + self.half_length, j + self.half_length] = image[y_idx, x_idx]
        
        x = np.delete(x, self.half_length * self.square_length + self.half_length)  # Remove the center pixel
        return x, y

class PalettedImageDataModule(pl.LightningDataModule):
    def __init__(self, folder, square_length, batch_size, num_workers=4):
        super().__init__()
        self.folder = folder
        self.square_length = square_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = PalettedImageDataset(self.folder, self.square_length)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

# Example usage
if __name__ == "__main__":
    data_module = PalettedImageDataModule(folder="data", square_length=5, batch_size=32)
    data_module.setup()
    for x, y in data_module.train_dataloader():
        print(x.shape, y.shape)
        break
