
import os
import torch
import torch.nn as nn

from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config
    ):

        self.model = model.to(config.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        os.makedirs(
            config.checkpoint_dir,
            exist_ok=True
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        for noisy, clean in tqdm(
            self.train_loader,
            desc="Training"
        ):

            noisy = noisy.to(self.config.device)
            clean = clean.to(self.config.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            predicted = self.model(noisy)

            # Calculate loss
            loss = self.criterion(
                predicted, clean
            )

            # Backpropagation
            loss.backward()

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0

        for noisy, clean in self.val_loader:

            noisy = noisy.to(self.config.device)
            clean = clean.to(self.config.device)

            predicted = self.model(noisy)

            loss = self.criterion(
                predicted, clean
            )

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch):

        path = os.path.join(
            self.config.checkpoint_dir,
            "best_model.pth"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)

    def fit(self):

        best_val_loss = float("inf")

        for epoch in range(self.config.epochs):

            train_loss = self.train_epoch()

            val_loss = self.validate()

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train: {train_loss:.6f} | "
                f"Validation: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                self.save_checkpoint(epoch)

                print("Saved best model!")