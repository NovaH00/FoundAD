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


def main():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = FoundAD(
        DinoV2Encoder(),
        TransformerProjector()
    )

    model.to(device)

    images = [
        Image.open(p)
        for p in [
            "data/images/000000000139.jpg",
            "data/images/000000039769.jpg",
            "data/images/000000391895.jpg"
        ]
    ]

    dataset = TrainDataset(images, model.process_image)

    train_loader = get_train_loader(
        dataset=dataset,
        batch_size=64,
        batch_per_epoch=32
    )

    trainer = FoundADTrainer(
        model=model,
        optimizer_factory=lambda params: AdamW(
            params,
        ),
        criterion=nn.MSELoss(),
        cutpaste=CutPasteRectBatch(),
        device=device
    )

    trainer.run(train_loader, max_epochs=20)

if __name__ == "__main__":
    main()
