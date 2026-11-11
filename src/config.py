import torch

class Config:
    # Audio parameters
    sample_rate = 16000
    n_fft = 512
    win_length = 512
    hop_length = 128

    # Dataset
    segment_length = 16000 * 2  # 2 seconds

    # Model
    in_channels = 1
    out_channels = 1

    # Training
    batch_size = 16
    learning_rate = 1e-3
    epochs = 30

    # Training settings
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_dir = "checkpoints"