# src/model.py

import torch
import torch.nn as nn


class AudioDenoiser(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input: 1 channel
            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                16, 1,
                kernel_size=3,
                padding=1
            ),

            # Ensure non-negative magnitude
            nn.ReLU()
        )

    def forward(self, x):

        return self.network(x)