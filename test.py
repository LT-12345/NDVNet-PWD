"""
Review release testing script (NPY-based)

Default inputs:
- dataprepare/data_test.npy
- dataprepare/mask_test.npy
- dataprepare/ndvi_evi_test.npy (optional)
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

from models.UltraLight_VM_UNet import UltraLight_VM_UNet


MODEL_CONFIG = {
    "num_classes": 1,
    "input_channels": 3,
    "ndvi_evi_channels": 2,
    "c_list": [8, 16, 24, 32, 48, 64],
    "split_att": "fc",
    "bridge": True,
}


def compute_binary_metrics(y_true, y_pred):
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_pred = y_pred.astype(np.uint8).reshape(-1)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou_pos = tp / max(tp + fp + fn, 1)
    iou_neg = tn / max(tn + fp + fn, 1)
    miou = (iou_pos + iou_neg) / 2.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "miou": float(miou),
        "iou_pos": float(iou_pos),
        "iou_neg": float(iou_neg),
        "mcc": float(mcc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def normalize_rgb(x):
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return x


def build_model(weight_path, device):
    model = UltraLight_VM_UNet(
        num_classes=MODEL_CONFIG["num_classes"],
        input_channels=MODEL_CONFIG["input_channels"],
        ndvi_evi_channels=MODEL_CONFIG["ndvi_evi_channels"],
        c_list=MODEL_CONFIG["c_list"],
        split_att=MODEL_CONFIG["split_att"],
        bridge=MODEL_CONFIG["bridge"],
    )
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, "pred_masks")
    os.makedirs(pred_dir, exist_ok=True)

    data = np.load(args.data_npy)
    mask = np.load(args.mask_npy)
    vi = np.load(args.vi_npy) if args.use_multimodal and os.path.exists(args.vi_npy) else None

    if data.ndim != 4:
        raise ValueError(f"data_npy shape must be [N,H,W,C], got {data.shape}")
    if mask.ndim not in (3, 4):
        raise ValueError(f"mask_npy shape must be [N,H,W] or [N,H,W,1], got {mask.shape}")

    if mask.ndim == 4:
        mask = mask[..., 0]
    mask = (mask > 127).astype(np.uint8)

    if vi is not None and vi.ndim != 4:
        raise ValueError(f"vi_npy shape must be [N,H,W,2] or [N,2,H,W], got {vi.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.weight_path, device)

    all_pred = []
    all_true = []

    with torch.no_grad():
        for i in tqdm(range(len(data)), desc="Testing"):
            rgb = normalize_rgb(data[i])
            h, w = rgb.shape[:2]
            rgb_resized = cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            rgb_tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            if args.use_multimodal and vi is not None:
                vi_sample = vi[i]
                if vi_sample.ndim == 3 and vi_sample.shape[0] == 2:
                    vi_chw = vi_sample
                else:
                    vi_chw = vi_sample.transpose(2, 0, 1)
                vi0 = cv2.resize(vi_chw[0], (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                vi1 = cv2.resize(vi_chw[1], (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                vi_tensor = torch.from_numpy(np.stack([vi0, vi1], axis=0)).unsqueeze(0).float().to(device)
            else:
                vi_tensor = torch.zeros(
                    (1, MODEL_CONFIG["ndvi_evi_channels"], args.height, args.width), dtype=torch.float32, device=device
                )

            out = model(rgb_tensor, vi_tensor)[0, 0].cpu().numpy()
            pred = (out > args.threshold).astype(np.uint8)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, f"{i:04d}_pred.png"), pred * 255)

            all_pred.append(pred)
            all_true.append(mask[i])

    metrics = compute_binary_metrics(np.stack(all_true, axis=0), np.stack(all_pred, axis=0))
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Test finished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved masks to: {pred_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP release test script")
    parser.add_argument("--weight_path", type=str, default="checkpoints/best-epoch274-dice0.6147.pth")
    parser.add_argument("--data_npy", type=str, default="dataprepare/data_test.npy")
    parser.add_argument("--mask_npy", type=str, default="dataprepare/mask_test.npy")
    parser.add_argument("--vi_npy", type=str, default="dataprepare/ndvi_evi_test.npy")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results")
    parser.add_argument("--use_multimodal", action="store_true")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    main(parser.parse_args())
