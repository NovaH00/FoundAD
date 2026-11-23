import io
import json
from argparse import ArgumentParser
from pathlib import Path


def run_cli(
    *,
    checkpoint: str,
    input_path: str,
    batch_size: int = 1,
    output: str = "stdout",
    device: str = "auto",
    threshold: float | None = None,
    save_heatmaps: bool = False,
    num_workers: int = 0,
) -> None:

    import torch
    from PIL import Image

    from foundad import FoundAD


    def save_heatmap_overlay(
        image: Image.Image,
        anomaly_map: torch.Tensor,
        output_path: Path,
    ) -> None:
        """Saves a blended anomaly heatmap overlay image to disk."""
        import numpy as np
        from matplotlib import colormaps

        # Resize original image to match anomaly map if necessary
        h, w = anomaly_map.shape[-2], anomaly_map.shape[-1]
        orig_img = image.resize((w, h)).convert("RGB")
        orig_np = np.array(orig_img, dtype=np.float32) / 255.0

        # Normalize heatmap to [0, 1]
        heatmap_np = anomaly_map.squeeze().cpu().numpy()
        h_min, h_max = heatmap_np.min(), heatmap_np.max()
        norm_heatmap = (heatmap_np - h_min) / (h_max - h_min + 1e-8)

        # Colorize with jet colormap
        cmap = colormaps["jet"]
        colored_heatmap = cmap(norm_heatmap)[:, :, :3]

        # Alpha blend: 60% original image + 40% heatmap
        blended = (0.6 * orig_np + 0.4 * colored_heatmap).clip(0, 1)
        result_img = Image.fromarray((blended * 255).astype(np.uint8))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_img.save(output_path)


    infer_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )

    print(f"Loading FoundAD model from: {checkpoint} (Device: {infer_device})")
    model = FoundAD.from_checkpoint(checkpoint, device=infer_device)

    input_p = Path(input_path)
    supported_exts = set(Image.registered_extensions().keys())

    if input_p.is_file():
        image_files = [input_p]
    elif input_p.is_dir():
        image_files = [
            f for f in sorted(input_p.iterdir())
            if f.is_file() and f.suffix.lower() in supported_exts
        ]
        if not image_files:
            raise FileNotFoundError(f"No valid images found in directory: {input_p}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_p}")

    print(f"Running inference on {len(image_files)} image(s) (num_workers={num_workers})...")

    from foundad.data import EvaluateDataset, get_inference_loader

    dataset = EvaluateDataset(image_files, transform=model.get_transform())
    dataloader = get_inference_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(infer_device.type == "cuda"),
    )

    results: list[dict] = []
    output_dir = Path(output) if output != "stdout" else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for batch_paths, batch_tensors in dataloader:
            pred = model.predict(batch_tensors)

            for j, file_path_str in enumerate(batch_paths):
                score = float(pred.anomaly_score[j].item())
                file_path = Path(file_path_str)
                item = {
                    "file": str(file_path.resolve()),
                    "anomaly_score": round(score, 6),
                }
                if threshold is not None:
                    item["is_anomaly"] = score >= threshold
                    item["threshold"] = threshold

                results.append(item)

                if output_dir is not None and save_heatmaps:
                    heatmap_file = output_dir / f"heatmap_{file_path.stem}.png"
                    orig_image = Image.open(file_path).convert("RGB")
                    save_heatmap_overlay(
                        orig_image,
                        pred.anomaly_map[j : j + 1],
                        heatmap_file,
                    )
    finally:
        iterator = getattr(dataloader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    if output == "stdout":
        print(json.dumps(results, indent=2))
    else:
        assert output_dir is not None
        json_path = output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Inference complete. Results saved to: {json_path}")
        if save_heatmaps:
            print(f"Heatmaps saved to: {output_dir}/")


def run_api(
    *,
    checkpoint: str,
    host: str = "localhost",
    port: int = 9999,
    device: str = "auto",
    threshold: float | None = None,
) -> None:
    from typing import Annotated

    import torch
    import uvicorn
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import Response
    from PIL import Image

    from foundad import FoundAD

    infer_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )

    print(f"Initializing FoundAD API from checkpoint: {checkpoint} (Device: {infer_device})")
    model = FoundAD.from_checkpoint(checkpoint, device=infer_device)

    app = FastAPI(
        title="FoundAD Anomaly Detection API",
        version="0.1.0",
        description="REST API for FoundAD anomaly detection model.",
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "device": str(infer_device),
            "projector": model.projector.__class__.__name__,
            "encoder": model.encoder.__class__.__name__,
        }

    @app.get("/config")
    def get_model_config():
        return {
            "projector": model.projector.get_config(),
            "encoder": model.encoder.get_config(),
        }

    @app.post("/predict")
    async def predict_image(
        file: Annotated[UploadFile, File(description="Image file (JPG/PNG) to analyze")],
    ):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

        pred = model.predict(image)
        score = float(pred.anomaly_score[0].item())

        resp = {
            "filename": file.filename,
            "anomaly_score": round(score, 6),
        }
        if threshold is not None:
            resp["is_anomaly"] = score >= threshold
            resp["threshold"] = threshold

        return resp

    @app.post("/predict/heatmap")
    async def predict_heatmap(
        file: Annotated[UploadFile, File(description="Image file to generate heatmap overlay for")],
    ):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

        pred = model.predict(image)

        # Generate blended heatmap in-memory
        import numpy as np
        from matplotlib import colormaps

        buf = io.BytesIO()
        h, w = pred.anomaly_map.shape[-2], pred.anomaly_map.shape[-1]
        orig_img = image.resize((w, h)).convert("RGB")
        orig_np = np.array(orig_img, dtype=np.float32) / 255.0

        heatmap_np = pred.anomaly_map[0, 0].cpu().numpy()
        h_min, h_max = heatmap_np.min(), heatmap_np.max()
        norm_heatmap = (heatmap_np - h_min) / (h_max - h_min + 1e-8)
        cmap = colormaps["jet"]
        colored_heatmap = cmap(norm_heatmap)[:, :, :3]

        blended = (0.6 * orig_np + 0.4 * colored_heatmap).clip(0, 1)
        result_img = Image.fromarray((blended * 255).astype(np.uint8))
        result_img.save(buf, format="PNG")

        return Response(content=buf.getvalue(), media_type="image/png")

    print(f"Starting API server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = ArgumentParser(description="FoundAD Inference CLI & API service.")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Command to run: 'cli' for batch/file processing, 'api' to start REST API server.",
    )

    # CLI subparser
    cli_parser = subparsers.add_parser("cli", help="Run inference on images from terminal.")
    cli_parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path or experiment run directory.",
    )
    cli_parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image file or folder of images.",
    )
    cli_parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size during inference (Default: 1).",
    )
    cli_parser.add_argument(
        "-o", "--output",
        type=str,
        default="stdout",
        help="Output destination ('stdout' or directory path).",
    )
    cli_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional anomaly score threshold to classify images as anomalous.",
    )
    cli_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('cpu', 'cuda', 'auto'). Default: auto.",
    )
    cli_parser.add_argument(
        "--save-heatmaps",
        action="store_true",
        help="When --output is a directory, saves visual heatmap PNG overlays.",
    )
    cli_parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading (Default: 0).",
    )

    # API subparser
    api_parser = subparsers.add_parser("api", help="Launch FastAPI REST server.")
    api_parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path or experiment run directory.",
    )
    api_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind server (Default: localhost).",
    )
    api_parser.add_argument(
        "-p", "--port",
        type=int,
        default=9999,
        help="Port to bind server (Default: 9999).",
    )
    api_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('cpu', 'cuda', 'auto'). Default: auto.",
    )
    api_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional anomaly classification threshold.",
    )

    args = parser.parse_args()

    if args.command == "cli":
        run_cli(
            checkpoint=args.checkpoint,
            input_path=args.input,
            batch_size=args.batch_size,
            output=args.output,
            device=args.device,
            threshold=args.threshold,
            save_heatmaps=args.save_heatmaps,
            num_workers=args.num_workers,
        )
    elif args.command == "api":
        run_api(
            checkpoint=args.checkpoint,
            host=args.host,
            port=args.port,
            device=args.device,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
