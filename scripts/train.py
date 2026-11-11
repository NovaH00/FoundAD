def run(
    *,
    dataset_dir: str,
    learning_rate: float,
    epochs: int,
    batches: int,
    batch_size: int,
    device: str,
    num_heads: int,
    num_blocks: int,
    mlp_ratio: int,
):
    from pathlib import Path

    import torch
    from PIL import Image
    from torch import nn
    from torch.optim import AdamW

    from foundad.data.dataloader import get_train_loader
    from foundad.data.dataset import TrainDataset
    from foundad.engines.trainer import FoundADTrainer
    from foundad.modules import FoundAD
    from foundad.modules.cutpaste import CutPasteRectBatch
    from foundad.modules.encoder.impl import DinoV2Encoder
    from foundad.modules.projector.impl import TransformerProjector


    supported_image_extensions = list(
        Image
            .registered_extensions()
            .keys()
    )

    directory = Path(dataset_dir)

    image_paths = [
        file
        for file in directory.iterdir()
        if (
            file.is_file()
            and file.suffix.lower() in supported_image_extensions
        )
    ]

    if len(image_paths) <= 0:
        raise FileNotFoundError(
            f"There are no image files in {dataset_dir}.\n"
          + f"Supported image extensions: {", ".join(supported_image_extensions)}"
        )

    images = [
        Image.open(p)
        for p in image_paths
    ]

    print(f"Loaded {len(images)} image(s) to dataset")

    # In the paper, they tested with 4 and 10 samples
    # So we use 4 as the lower bound.
    if len(images) < 4:
        print(
            f"WARNING: Amount of available image ({len(images)}) "
             "is less than recommended amount of images (4). "
             "Consider collecting 4 to 10 samples and re-train if accuracy is low"
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but CUDA is not available"
        )

    train_device: torch.device
    if (device == "cuda" or device == "auto") and torch.cuda.is_available():
        train_device = torch.device("cuda")
    else:
        train_device = torch.device("cpu")

    print(f"Using device: {train_device}")

    encoder = DinoV2Encoder()
    projector = TransformerProjector(
        encoder.get_embed_dim(),
        num_heads,
        num_blocks,
        mlp_ratio
    )
    model = FoundAD(encoder, projector)

    model.to(train_device)

    dataset = TrainDataset(images, model.process_image)

    train_loader = get_train_loader(
        dataset=dataset,
        batch_size=batch_size,
        batch_per_epoch=batches
    )

    trainer = FoundADTrainer(
        model=model,
        optimizer_factory=lambda params: AdamW(
            params,
            learning_rate
        ),
        criterion=nn.MSELoss(),
        cutpaste=CutPasteRectBatch(),
        device=train_device
    )

    # trainer.run(train_loader, max_epochs=epochs)

def main():
    import sys
    from argparse import ArgumentParser


    parser = ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the folder containing 4 to 10 normal images."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=False,
        default=1e-4,
        help="Learning rate (Default: 1e-4)."
    )

    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="auto",
        help="Device to train the model. Options: cpu, cuda, auto (Default: auto)."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to train the model."
    )

    parser.add_argument(
        "--batches",
        type=int,
        required=True,
        help="Number of batches in a single training epoch."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Number of samples in a single batch."
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        required=False,
        default=12,
        help="Number of heads in the projector's self-attention layer (Default: 12)."
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        required=False,
        default=6,
        help="Number of transformer blocks in the projector (Default: 6)."
    )

    parser.add_argument(
        "--mlp-ratio",
        type=int,
        required=False,
        default=4,
        help=(
            "Controls the expansion ratio of the hidden dimension in the MLP (feed-forward) block "
          + "relative to the input embedding dimension (Default: 4)."
        )
    )

    args = parser.parse_args(sys.argv[1:])

    run(
        dataset_dir   = args.dataset_dir,
        epochs        = args.epochs,
        learning_rate = args.learning_rate,
        batches       = args.batches,
        batch_size    = args.batch_size,
        device        = args.device,
        num_heads     = args.num_heads,
        num_blocks    = args.num_blocks,
        mlp_ratio     = args.mlp_ratio
    )


if __name__ == "__main__":
    main()
