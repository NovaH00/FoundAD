# FoundAD

Self-supervised visual anomaly detection using foundation vision models and trainable projectors.

FoundAD uses a frozen vision transformer backbone ([DINOv2](https://github.com/facebookresearch/dinov2)) to extract patch representations, and trains a lightweight projector with CutPaste self-supervision on clean normal images. During inference, distance between projected features and clean embeddings yields pixel-level anomaly heatmaps and image-level defect scores.

---

## Installation

This repository uses [uv](https://docs.astral.sh/uv/) for Python package management:

```bash
git clone https://github.com/NovaH00/FoundAD.git
cd foundad
uv sync
```

---

## Python API

### 1. Initialize the Model

Specify the encoder and projector explicitly:

```python
from foundad import DinoV2Encoder, FoundAD, TransformerProjector

# 1. Initialize encoder ('small', 'base', or 'large')
encoder = DinoV2Encoder("small")

# 2. Initialize projector
projector = TransformerProjector(
    embed_dim=encoder.get_embed_dim(),
    num_heads=12,
    num_blocks=6,
    mlp_ratio=4,
)

# 3. Assemble FoundAD model
model = FoundAD(encoder, projector)
```

### 2. Train the Model

Train the projector using clean images (4 to 10 normal images recommended):

```python
model.train_model(
    dataset_dir="data/images",
    epochs=100,
    batch_size=32,
    batches=16,
    learning_rate=1e-4,
    device="cuda",
    output_dir="results/exp_1",
)
```

> [!NOTE]
> `.train(mode: bool = True)` is reserved for standard PyTorch mode switching (`model.eval()` / `model.train()`). Use `.train_model(...)` to run the training process.

### 3. Save and Load Checkpoints

Checkpoints store **only the projector weights** and configuration metadata, keeping files under 25 MB:

```python
# Save projector weights and architecture metadata
model.save("checkpoints/projector.pt")

# Load weights into a compatible model instance
model.load("checkpoints/projector.pt")
```

If the checkpoint architecture or hyperparameters do not match the target model (for example, different projector types or mismatched attention heads), `model.load()` raises a `ValueError` before loading weights.

You can also rebuild a complete model directly from a checkpoint or run folder:

```python
model = FoundAD.from_checkpoint("results/exp_1")
# Or from a specific weight file:
model = FoundAD.from_checkpoint("checkpoints/projector.pt")
```

### 4. Run Inference

Predict anomaly maps and scores on a single image, file path, list of images, or tensor:

```python
result = model.predict("data/images/000000000139.jpg")

print("Image anomaly score:", result.anomaly_score.item())
print("Heatmap shape:", result.anomaly_map.shape)  # (1, 1, H, W)
```

The returned `InferenceResult` object contains:
- `anomaly_score`: Image-level anomaly score (`Tensor` of shape `(B,)`).
- `anomaly_map`: Pixel-level defect heatmap (`Tensor` of shape `(B, 1, H, W)`).
- `patch_anomaly`: Un-interpolated token anomaly scores (`Tensor` of shape `(B, L)`).

---

## Command Line Interface (CLI)

### Training (`scripts/train.py`)

Train a model from the terminal:

```bash
uv run scripts/train.py \
  --dataset-dir data/images \
  --encoder-size small \
  --epochs 100 \
  --batch-size 32 \
  --batches 16 \
  --learning-rate 1e-4 \
  --device auto \
  --output-dir results/exp_1
```

Options:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset-dir` | `str` | *required* | Path to folder containing normal training images. |
| `--encoder-size` | `str` | `small` | DINOv2 model size (`small`, `base`, `large`). |
| `--epochs` | `int` | `100` | Number of training epochs. |
| `--batches` | `int` | `16` | Number of batches per epoch. |
| `--batch-size` | `int` | `32` | Batch size. |
| `--learning-rate`| `float` | `1e-4` | Learning rate for AdamW optimizer. |
| `--num-heads` | `int` | `12` | Projector attention heads. |
| `--num-blocks` | `int` | `6` | Projector transformer blocks. |
| `--mlp-ratio` | `int` | `4` | Projector MLP expansion ratio. |
| `--output-dir` | `str` | `results/exp_<timestamp>` | Output directory for checkpoints and logs. |

---

### Inference (`scripts/inference.py`)

The inference script provides both a command line batch processor and a REST API server.

#### Option A: Batch File Processing (`cli`)

Process a single image or an entire folder of images:

```bash
# Print JSON results to stdout
uv run scripts/inference.py cli \
  -c results/exp_1 \
  -i data/images/000000000139.jpg \
  --threshold 0.01

# Process a directory and save visual heatmaps
uv run scripts/inference.py cli \
  -c results/exp_1 \
  -i data/images \
  -o results/inferences \
  --save-heatmaps \
  --batch-size 4
```

`results/inferences/results.json` output:
```json
[
  {
    "file": "/path/to/data/images/000000000139.jpg",
    "anomaly_score": 0.010116,
    "is_anomaly": true,
    "threshold": 0.01
  }
]
```

#### Option B: REST API Server (`api`)

Start a FastAPI inference service:

```bash
uv run scripts/inference.py api -c results/exp_1 --host localhost --port 9999
```

API Endpoints:
- `GET /health`: Health and device status.
- `GET /config`: Loaded model configuration.
- `POST /predict`: Upload image file (`multipart/form-data`) and receive JSON anomaly scores:
  ```bash
  curl -X POST "http://localhost:9999/predict" -F "file=@data/images/000000000139.jpg"
  ```
- `POST /predict/heatmap`: Upload image file and download blended PNG heatmap overlay:
  ```bash
  curl -X POST "http://localhost:9999/predict/heatmap" -F "file=@data/images/000000000139.jpg" -o heatmap.png
  ```

---

## Results Directory Structure

Each training run generates a self-contained output directory:

```text
results/
└── exp_<timestamp>/
    ├── config.json                        # Model, training, and dataset metadata
    ├── checkpoints/
    │   ├── best.pt                        # Checkpoint with lowest epoch loss
    │   ├── last.pt                        # Latest epoch checkpoint for training resumption
    │   └── projector_final.pt             # Standalone projector weights
    └── logs/
        ├── train.log                      # Plaintext training and epoch log
        └── tensorboard/                   # TensorBoard event files
```

---

## Custom Architectures

Every `Encoder` and `Projector` subclass implements the `get_config()` and `@classmethod from_config(config)` protocol. To implement a new projector (e.g. `CNNProjector`):

```python
from typing import Any
from torch import Tensor
from foundad.modules.projector.base import Projector

class CNNProjector(Projector):
    def __init__(self, embed_dim: int, kernel_size: int = 3, num_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        # ... define layers ...

    def project(self, x: Tensor) -> Tensor:
        # ... forward projection ...
        return x

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "embed_dim": self.embed_dim,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
        }
```

The new component inherits `from_config()` automatically and integrates seamlessly with `FoundAD.save()`, `FoundAD.load()`, and checkpoint hyperparameter validation.
