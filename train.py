import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from dataset import PalettedImageDataset, PalettedImageDataModule
from model import get_model
import fire

class NeuralCATrainer(pl.LightningModule):
    def __init__(self, config):
        super(NeuralCATrainer, self).__init__()
        self.model = get_model(config)
        self.lr = config["learning_rate"]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        dataset = PalettedImageDataset(folder=self.config["data_folder"], square_length=self.config["square_size"])
        sampler = RandomSampler(dataset, replacement=True)
        batch_sampler = BatchSampler(sampler, batch_size=self.config["batch_size"], drop_last=True)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.config["num_workers"])

class TrainCallback(Callback):
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step >= self.max_steps:
            trainer.should_stop = True

def main(
        data_folder="data",
        square_size=5,
        embedding_size=64,
        num_heads=4,
        num_encoders=2,
        num_classes=256,
        batch_size=32,
        num_workers=4,
        learning_rate=0.001,
        max_steps=10000,
        log_dir="logs"
    ):

    config = {
        "square_size": square_size,
        "embedding_size": embedding_size,
        "num_heads": num_heads,
        "num_encoders": num_encoders,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "data_folder": data_folder,
        "batch_size": batch_size,
        "num_workers": num_workers
    }

    # Initialize the model
    model = NeuralCATrainer(config)

    # Initialize logger and checkpoint callback
    logger = TensorBoardLogger(log_dir, name="NeuralCA")
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=log_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min"
    )

    # Initialize the trainer with a callback to stop after max_steps
    trainer = pl.Trainer(
        max_steps=max_steps,
        logger=logger,
        callbacks=[checkpoint_callback, TrainCallback(max_steps)]
    )

    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    fire.Fire(main)
