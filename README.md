# PP Paper Open-Source Code (Minimal Reproducible Release)

This repository is prepared for peer-review reproducibility.  
It provides only the minimum files needed for testing/inference with released model weights.  
Training code is not included in this release.

## Repository Structure

- `checkpoints/`
  - Model weights (e.g., `best-epoch274-dice0.6147.pth`).
- `test.py`
  - Main NPY-based batch testing script.
- `models/UltraLight_VM_UNet.py`
  - Model architecture definition.
- `configs/config_setting.py`
  - Reference configuration file.
- `dataprepare/Prepare_your_dataset.py`
  - Data preprocessing script (converts raw images/masks/VI values to NPY files).
- `dataprepare/INPUT_FORMAT.md`
  - Input data format and folder organization (`images`, `masks`, `csv`).
- `requirements.txt`
  - Runtime dependencies.

## 1) Environment Setup

Recommended Python version: **3.9**

```bash
python --version
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Data Preparation (Optional)

You can directly use the NPY files already included in this release package.  
In that case, you can skip preprocessing and go straight to **Section 3 (Run Testing)**.

If you want to rebuild NPY files from your own `images/`, `masks/`, and `ndvi_evi_results.csv`, organize data according to `dataprepare/INPUT_FORMAT.md`, then run:

```bash
cd dataprepare
python Prepare_your_dataset.py
cd ..
```

This script will generate 9 NPY files for train/val/test splits (`data`, `mask`, `ndvi_evi`).

## 3) Run Testing (Main Workflow)

Run the following command from the repository root:

```bash
python test.py \
  --weight_path "checkpoints/best-epoch274-dice0.6147.pth" \
  --data_npy "dataprepare/data_test.npy" \
  --mask_npy "dataprepare/mask_test.npy" \
  --vi_npy "dataprepare/ndvi_evi_test.npy" \
  --use_multimodal \
  --output_dir "outputs/test_results"
```

## 4) Outputs

- Metrics file: `outputs/test_results/test_metrics.json`
- Predicted masks: `outputs/test_results/pred_masks/`

Reported metrics include:
- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `miou`
- `mcc`

## 5) Troubleshooting

- **Weight file not found**: check whether `--weight_path` is correct.
- **Unexpectedly low metrics**: ensure `data_test.npy`, `mask_test.npy`, and `ndvi_evi_test.npy` are generated in the same preprocessing run and have aligned sample order.
