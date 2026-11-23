from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Literal


def run(
    *,
    dataset_dir: str,
    encoder_size: Literal["small", "base", "large"] = "small",
    learning_rate: float = 1e-4,
    epochs: int = 100,
    batches: int = 16,
    batch_size: int = 32,
    device: str = "auto",
    num_heads: int = 12,
    num_blocks: int = 6,
    mlp_ratio: int = 4,
    output_dir: str | None = None,
    num_workers: int = 0,
) -> None:

    from foundad import DinoV2Encoder, FoundAD, TransformerProjector


    # 1. Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = Path("results") / f"exp_{timestamp}"
    else:
        run_output_dir = Path(output_dir)

    print(f"Results and checkpoints will be saved to: {run_output_dir}")

    # 2. Build model with user-selected encoder and projector
    print(f"Initializing DinoV2Encoder('{encoder_size}')...")
    encoder = DinoV2Encoder(encoder_size)

    print("Initializing TransformerProjector...")
    projector = TransformerProjector(
        embed_dim=encoder.get_embed_dim(),
        num_heads=num_heads,
        num_blocks=num_blocks,
        mlp_ratio=mlp_ratio,
    )

    model = FoundAD(encoder, projector)

    # 3. Train using programming interface
    model.train_model(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        batches=batches,
        learning_rate=learning_rate,
        device=None if device == "auto" else device,
        output_dir=run_output_dir,
        num_workers=num_workers,
    )

    # 4. Save final projector weights checkpoint
    projector_final_path = run_output_dir / "checkpoints" / "projector_final.pt"
    model.save(projector_final_path)
    print(f"Final projector weights saved to: {projector_final_path}")


def main() -> None:
    parser = ArgumentParser(description="Train FoundAD anomaly detection model.")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to folder containing normal training images.",
    )

    parser.add_argument(
        "--encoder-size",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="DinoV2 encoder size ('small', 'base', 'large'). Default: small.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (Default: 1e-4).",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (Default: 100).",
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=16,
        help="Number of batches in a single training epoch (Default: 16).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per training step (Default: 32).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device ('cpu', 'cuda', 'auto'). Default: auto.",
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Number of attention heads in TransformerProjector (Default: 12).",
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=6,
        help="Number of transformer blocks in TransformerProjector (Default: 6).",
    )

    parser.add_argument(
        "--mlp-ratio",
        type=int,
        default=4,
        help="MLP expansion ratio in TransformerProjector (Default: 4).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save logs and checkpoints (Default: results/exp_<timestamp>).",
    )

    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=0,
        help="Number of data loader worker processes (Default: 0).",
    )

    args = parser.parse_args()

    run(
        dataset_dir=args.dataset_dir,
        encoder_size=args.encoder_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batches=args.batches,
        batch_size=args.batch_size,
        device=args.device,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        mlp_ratio=args.mlp_ratio,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
