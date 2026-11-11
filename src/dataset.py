# src/dataset.py

import os
import torch
import librosa
import numpy as np

from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(
        self,
        clean_dir,
        noisy_dir,
        config,
        training=True
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.config = config
        self.training = training

        self.files = sorted([
            f for f in os.listdir(clean_dir)
            if f.endswith(".wav")
        ])

        # Verify matching files
        self.files = [
            f for f in self.files
            if os.path.exists(
                os.path.join(noisy_dir, f)
            )
        ]

        self.window = torch.hann_window(
            config.win_length
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        clean_path = os.path.join(
            self.clean_dir, filename
        )

        noisy_path = os.path.join(
            self.noisy_dir, filename
        )

        # Load audio
        clean, _ = librosa.load(
            clean_path,
            sr=self.config.sample_rate,
            mono=True
        )

        noisy, _ = librosa.load(
            noisy_path,
            sr=self.config.sample_rate,
            mono=True
        )

        # Ensure equal lengths
        length = min(len(clean), len(noisy))

        clean = clean[:length]
        noisy = noisy[:length]

        segment_length = self.config.segment_length

        # Crop corresponding segments
        if length > segment_length:

            if self.training:
                start = np.random.randint(
                    0, length - segment_length + 1
                )
            else:
                start = 0

            end = start + segment_length

            clean = clean[start:end]
            noisy = noisy[start:end]

        else:
            # Pad shorter recordings
            padding = segment_length - length

            clean = np.pad(clean, (0, padding))
            noisy = np.pad(noisy, (0, padding))

        clean = torch.tensor(
            clean, dtype=torch.float32
        )

        noisy = torch.tensor(
            noisy, dtype=torch.float32
        )

        # STFT
        S = torch.stft(
            clean,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            return_complex=True
        )

        X = torch.stft(
            noisy,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            return_complex=True
        )

        # Log magnitude spectrogram
        S_mag = torch.log1p(torch.abs(S))
        X_mag = torch.log1p(torch.abs(X))

        # Add channel dimension
        S_mag = S_mag.unsqueeze(0)
        X_mag = X_mag.unsqueeze(0)

        return X_mag, S_mag