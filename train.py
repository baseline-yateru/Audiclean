from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import AudioDataset
from src.model import AudioDenoiser
from src.trainer import Trainer


def main():

    config = Config()

    # Training dataset
    train_dataset = AudioDataset(
        clean_dir="data/train/clean",
        noisy_dir="data/train/noisy",
        config=config,
        training=True
    )

    # Validation dataset
    val_dataset = AudioDataset(
        clean_dir="data/test/clean",
        noisy_dir="data/test/noisy",
        config=config,
        training=False
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = AudioDenoiser()

    # Trainer
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config
    )

    # Start training
    trainer.fit()


if __name__ == "__main__":
    main()