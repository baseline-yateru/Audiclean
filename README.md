# Audiclean

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[audio]"
```

Run it with either command:

```

## Importing audio processing in a notebook

The [`scripts/audio.py`](scripts/audio.py) module is included automatically
because `pyproject.toml` discovers the `scripts` package. After installing the
project, select the same `.venv` interpreter as the notebook kernel and use:

```python
from scripts.audio import AudioProcessor

processor = AudioProcessor("path/to/audio.wav")
processor.plot_waveform()
mfccs = processor.get_mfcc()
```

If the notebook is inside this project, install the project in editable mode
once with `python -m pip install -e ".[audio]"`. Editable installation means
future changes to `scripts/audio.py` are available after restarting the
notebook kernel; reinstalling is not needed for each source change.bash

To have IPython reload changed source files automatically while the notebook
kernel is running, put this in the first notebook cell:

```python
%load_ext autoreload
%autoreload 2

from scripts.audio import AudioProcessor
```

`%autoreload 2` reloads imported modules before executing each cell. It avoids
stale `scripts.audio` code during development, but restart the kernel after
changing class definitions or dependencies if the notebook still contains
objects created from the previous version.bash
audiclean
python -m scripts.main
```

The [`scripts/__init__.py`](scripts/__init__.py) file initializes the Python
package. Project metadata and dependencies belong in [`pyproject.toml`](pyproject.toml),
not in `__init__.py`.

## Fixing Torch version conflicts

The project pins the current compatible PyTorch family in the `audio` extra:
`torch 2.15.0`, `torchvision 0.29.2`, and `torchaudio 2.11.1`. NumPy is also
constrained below `2.0.0` because TensorFlow 2.16.1 requires that range on
Python 3.11.

If the environment already contains incompatible packages, recreate it rather
than layering another Torch version on top:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[audio]"
python -m pip check
```

On Apple Silicon, this installs the standard macOS build, which can use the
MPS device when supported by the operation:

```python
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
```
