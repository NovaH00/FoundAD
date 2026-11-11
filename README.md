outputs/runs/<run_id>/train/
    weights/{best,last,epoch_05,...}.pt
    config.json

For config.json
```json
{
    "dataset": {
        "path": "/absolute/path/to/dataset",
        "samples": 4
    },
    "model": {
        "encoder": {},
        "projector": {
            "num_heads": 12,
            "num_blocks": 6,
            "mlp_ratio": 4
        }
    },
    "training": {
        "epochs": 20,
        "learning_rate": 0.00004,
        "batches": 128,
        "batch_size": 32,
        "device": "auto"
    }
}
```
