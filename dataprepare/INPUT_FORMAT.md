# INPUT FORMAT (RGB + VI)

This document describes how to prepare data for `Prepare_your_dataset.py`.

## 1. Required folder structure

Place files under `dataprepare/` with this layout:

```text
dataprepare/
  images/
    0001.png
    0002.png
    ...
  masks/
    0001.png
    0002.png
    ...
  ndvi_evi_results.csv
  Prepare_your_dataset.py
```

## 2. Image and mask requirements

- `images/`: RGB images (PNG), one file per sample.
- `masks/`: binary segmentation masks (PNG), same basename as image.
- Pairing rule:
  - `images/0001.png` <-> `masks/0001.png`
- In current script, image list is read by:
  - `glob("images/*.png")`
- Mask path is derived by filename ID in script (`masks/{id}.png`), so naming consistency is required.

## 3. Vegetation-index CSV format

File: `ndvi_evi_results.csv`

Required columns:

- `filename` (e.g., `0001.png`)
- `ndvi_mean` (float)
- `evi_mean` (float)

Example:

```csv
filename,ndvi_mean,evi_mean
0001.png,0.4123,0.2871
0002.png,0.3659,0.2410
```

The script loads this CSV and builds two VI channels (NDVI, EVI) for each sample.

## 4. Run preprocessing

From `dataprepare/` directory, run:

```bash
python Prepare_your_dataset.py
```

## 5. Generated NPY files

After running, the script generates 9 files for training/testing:

- `data_train.npy`
- `ndvi_evi_train.npy`
- `mask_train.npy`
- `data_val.npy`
- `ndvi_evi_val.npy`
- `mask_val.npy`
- `data_test.npy`
- `ndvi_evi_test.npy`
- `mask_test.npy`

## 6. Notes

- The preprocessing script applies random shuffle and augmentation.
- Input size is resized to `256 x 256` in the current implementation.
- Training code is not included in this release for now.

## 7. Public release scope

- The existing `.npy` files are generated from our full internal dataset preprocessing pipeline and can be used for testing/inference reproduction.
- In this public release, we provide only a subset of **29 samples** with their corresponding:
  - RGB images
  - masks
  - vegetation-index entries (NDVI/EVI in `ndvi_evi_results.csv`)
- This subset is for review and reproducibility checks, not the full training dataset.
