"""
src/export_onnx.py
Export the trained deepfake detector to ONNX for edge / cross-runtime deployment.

ONNX lets the model run under ONNX Runtime, TensorRT, OpenVINO, or in the
browser (onnxruntime-web) — no PyTorch dependency at inference time.

Usage:
    python -m src.export_onnx
    python -m src.export_onnx --checkpoint models/best_xception_celebdf.pth \
                              --output models/xception_deepfake.onnx --opset 17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import CHECKPOINT_PATH, IMG_SIZE, MODEL_NAME, MODELS_DIR
from src.logger import get_logger
from src.model import build_model, load_checkpoint

logger = get_logger(__name__)


def export_to_onnx(checkpoint_path: str = str(CHECKPOINT_PATH),
                   output_path: str = str(MODELS_DIR / "xception_deepfake.onnx"),
                   model_name: str = MODEL_NAME, img_size: int = IMG_SIZE,
                   opset: int = 17, verify: bool = True,
                   quantize: bool = False) -> Path:
    """Build the model, load weights, and export an ONNX graph. Returns the path."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model or set "
            f"--checkpoint (or DEEPFAKE_CHECKPOINT_URL / DEEPFAKE_HF_REPO)."
        )

    device = "cpu"  # export on CPU for maximum portability
    logger.info("Loading %s from %s", model_name, checkpoint_path)
    model = build_model(model_name, pretrained=False, device=device)
    model = load_checkpoint(model, str(checkpoint_path), device=device)
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    logger.info("Exporting to ONNX (opset %d) -> %s", opset, output_path)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )
    logger.info("ONNX export complete: %s", output_path)

    if verify:
        _verify_onnx(model, output_path, dummy)
    if quantize:
        quantize_onnx(output_path)
    return output_path


def quantize_onnx(onnx_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
    """
    Produce a dynamically-quantized INT8 ONNX model for edge devices.

    Dynamic quantization shrinks the model ~4x and speeds up CPU inference with
    minimal accuracy loss — ideal for laptops / edge / mobile. Requires
    onnxruntime; returns None (with a log message) if it isn't installed.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        logger.warning("onnxruntime not installed — cannot quantize "
                       "(pip install onnxruntime). Skipping.")
        return None

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    output_path = Path(output_path)

    logger.info("Quantizing (INT8, dynamic) -> %s", output_path)
    quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)

    fp32_mb = onnx_path.stat().st_size / 1e6
    int8_mb = output_path.stat().st_size / 1e6
    logger.info("Edge model ready: %.1f MB -> %.1f MB (%.1fx smaller)",
                fp32_mb, int8_mb, fp32_mb / max(int8_mb, 1e-6))
    return output_path


def _verify_onnx(torch_model, onnx_path: Path, dummy: torch.Tensor) -> None:
    """Compare PyTorch vs ONNX Runtime outputs (skips if onnxruntime absent)."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        logger.info("onnxruntime not installed — skipping numerical verification "
                    "(pip install onnxruntime to enable).")
        return

    with torch.no_grad():
        torch_out = torch_model(dummy).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.cpu().numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if max_diff < 1e-3:
        logger.info("Verification OK — max |PyTorch - ONNX| = %.2e", max_diff)
    else:  # pragma: no cover
        logger.warning("Verification WARNING — max diff %.2e exceeds 1e-3", max_diff)


def parse_args():
    p = argparse.ArgumentParser(description="Export the detector to ONNX")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    p.add_argument("--output", default=str(MODELS_DIR / "xception_deepfake.onnx"))
    p.add_argument("--model", default=MODEL_NAME, choices=["xception", "efficientnet", "vit"])
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--quantize", action="store_true",
                   help="Also produce an INT8 quantized model for edge devices")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        path = export_to_onnx(
            checkpoint_path=args.checkpoint, output_path=args.output,
            model_name=args.model, img_size=args.img_size, opset=args.opset,
            verify=not args.no_verify, quantize=args.quantize,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    print(f"ONNX model saved to: {path}")


if __name__ == "__main__":
    main()
