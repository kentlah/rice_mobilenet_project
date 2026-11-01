# Rice disease classification: start here

This is a runnable **adapted reproduction of the frozen MobileNetV2 experiment** in Altabaji et al., *Comparative Analysis of Transfer Learning, LeafNet, and Modified LeafNet Models for Accurate Rice Leaf Diseases Classification*, IEEE Access 12, 36622–36635, 2024, DOI: 10.1109/ACCESS.2024.3373000. It does not reproduce all four experiments or promise the paper's accuracy on a different dataset.

## 1. Project setup (Windows PowerShell)

Install Python **3.12** from https://www.python.org/downloads/ if necessary. Extract this project into a normal writable directory, for example `Documents\rice-project`. Open PowerShell in that directory. Use the virtual-environment interpreter directly; activation is unnecessary.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CPU threads:', torch.get_num_threads())"
```

The CPU packages were verified together in the development environment. Your 32 GB RAM is ample for the model and batch size 16, but CPU training time must be measured on your laptop. This implementation always executes on CPU. Torchvision supplies the method-specific MobileNetV2 model/transforms, and Pillow supplies image loading. Document these additional packages in your user manual. NumPy, matplotlib, and scikit-learn are permitted by the specification. OpenCV and scikit-image are not used, so you need not install them for this method.

Put the **unzipped assignment dataset** beside the script, keeping its original internal folders:

```text
rice-project/
  group_name.py
  README.md
  requirements.txt
  Dhan-Shomadhan/
    [the original background and disease folders, unchanged]
```

All generated weights, inventories, checkpoints and figures go inside `Dhan-Shomadhan/_mobilenet/`. The code creates these directories. It supports disease folders including `Browon Spot`, `Leaf Scald`/`Leaf Scaled`, `Rice Turgro`, and `Shath Blight`; background ancestor folder names must contain `field` or `white`. Labels are inferred from folders, never image filenames.

## 2. Audit before modelling

```powershell
.\.venv\Scripts\python.exe group_name.py audit
```

Check the printed five-class × two-background counts and the total against the dataset documentation (expected 1,106 images). Open `Dhan-Shomadhan\_mobilenet\audit.json` and `inventory.csv`. A difference in the total is a reason to inspect the download/extraction, not to silently discard files.

Exact file duplicates are identified by SHA-256 and kept in the same split. Conflicting labels/backgrounds on identical files stop the run. Inspect visually for near-identical images, multiple photos of the same leaf, or related captures: byte hashing alone cannot prevent this source of leakage. If such groups are present, extend the split to keep those groups together **before producing final results**.

If folder names cannot be inferred, create a CSV `dataset_manifest.csv` with columns `path,class,background`. Paths are relative to `Dhan-Shomadhan`, classes exactly match the five names below, and background is `field` or `white`. Use `--manifest dataset_manifest.csv` on both `audit` and `train`. Do not reorganise the assignment dataset.

Classes in output order: Brown Spot, Leaf Scald, Rice Blast, Rice Tungro, Sheath Blight. No healthy class is included: the assignment assumes an image contains one of the five diseases.

## 3. Download the ImageNet starting weights

```powershell
.\.venv\Scripts\python.exe group_name.py download-weights
```

This downloads the official torchvision **IMAGENET1K_V1** weights, checks their published hash prefix, and places them at:

`Dhan-Shomadhan\_mobilenet\weights\mobilenet_v2-b0353104.pth`

Source: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth . Internet is needed for installation/download; training then uses the local file. Do not include these weights in the Moodle ZIP. Explain in the report that pretrained generic visual features allow supervised training on a small dataset without retraining the backbone.

## 4. First run: verify the complete pipeline

```powershell
.\.venv\Scripts\python.exe group_name.py train --run smoke --head paper --seeds 42 --epochs 2 --threads 4
```

This checks splitting, image reading, augmentation, frozen-backbone training, validation checkpoint selection, test evaluation and plots. It is **not** a report experiment. Each line prints epoch time. Investigate decoding errors or label/count discrepancies now. Check the learning-curve and confusion-matrix PNGs. Do not tune choices using these smoke-test test metrics; tune on validation only.

Estimate a run from your own timing: roughly `seconds_per_epoch × epochs × five splits`, plus evaluation/startup. Choose the thread count using timing, not a claim about core count. Try 4 versus 8 on a fresh smoke run name if useful. The program refuses to overwrite any existing run directory; use a new `--run` each time.

## 5. Lean experiment we recommend starting with

Retain the paper-style classifier, but predeclare a shorter training budget and validation early stopping:

```powershell
.\.venv\Scripts\python.exe group_name.py train --run paper_50 --head paper --epochs 50 --patience 8 --threads 4
.\.venv\Scripts\python.exe group_name.py summarize --run paper_50
```

Defaults: five seeds 42–46; batch 16; Adam learning rate 0.0001; dropout 0.5. Save the checkpoint with minimum validation cross-entropy, stop after eight non-improving epochs or 50 epochs, and evaluate that checkpoint once on the held-out test set. The backbone and its BatchNorm statistics stay frozen throughout.

This **50-epoch cap and early stopping are deviations from the paper's 400 epochs**. They are legitimate assignment choices if justified and explicitly reported; they must not be described as exact replication. Keep hyperparameters fixed across all five splits. Do not change the budget in response to test performance.

If you want the paper-length version after measuring runtime:

```powershell
.\.venv\Scripts\python.exe group_name.py train --run paper_400 --head paper --epochs 400 --patience 0 --threads 4
```


## 6. Optional smaller classifier comparison

```powershell
.\.venv\Scripts\python.exe group_name.py train --run pooled_50 --head pooled --epochs 50 --patience 8 --threads 4
.\.venv\Scripts\python.exe group_name.py summarize --run pooled_50
```

The paper head is `7×7×1280 → Flatten → Dense(64) → ReLU → Dropout → Dense(5)`. It has **4,014,469** trainable parameters. The optional pooled head is `7×7×1280 → Global Average Pool → Dense(64) → ReLU → Dropout → Dense(5)` with **82,309** trainable parameters (about 49× fewer). This changes the method and is a proposed adaptation/ablation, not the paper's architecture. It still computes the same frozen backbone each epoch; the parameter reduction does not imply a 49× runtime speedup. Do the paper head first unless timing makes its cost unreasonable.

To keep work lean, one carefully evaluated configuration is sufficient; the assignment does not require reproducing all four paper models. If comparing configurations, keep the same seeds/splits and budget, select the final method using validation evidence, and report completed experiments honestly.

## 7. Outputs and evaluation

Each run contains `config.json` (settings, package versions, inventory fingerprint, weight hash) and one folder per seed:

- `splits.csv`: exact train/validation/test allocation, including file hashes.
- `history.csv`, `learning_curves.png`: training and validation results and timing.
- `best.pt`: checkpoint selected on minimum validation loss.
- `test_predictions.csv`: image paths, labels and predictions for error inspection.
- `metrics.json`: accuracy, macro F1, per-class precision/recall/F1 and confusion counts.
- `confusion_white.png`, `confusion_field.png`, `confusion_mixed.png`.

`summarize` requires exactly five completed distinct splits and writes `summary.csv`/`summary.json`. It averages each split's accuracy and macro F1 separately, with **sample standard deviation (ddof=1)**. Values are fractions; multiply both mean and SD by 100 to report percentages. Do not pool repeated test appearances from the five overlapping random splits as independent samples.

The split is approximately **60% train / 20% validation / 20% test**, stratified jointly by disease and background. Integer rounding and exact-duplicate groups may shift image counts slightly. One model trains on mixed backgrounds per split, and the same held-out test allocation is evaluated on its white subset, field subset and full mixed set. These are the assignment's three required scenarios; three separate models are unnecessary.

Exact duplicates cannot cross the split, but the existing code does not infer leaf identity or capture sessions. State any remaining leakage limitation. Validation and test transforms are deterministic; only training images are randomly augmented. Test data never enters optimisation or checkpoint selection.

Prediction for a single image:

```powershell
.\.venv\Scripts\python.exe group_name.py predict --checkpoint .\Dhan-Shomadhan\_mobilenet\runs\paper_50\seed_42\best.pt --image "C:\path\to\leaf.jpg"
```

Softmax scores are not calibrated diagnostic confidence. The classifier assumes one of the five diseases and has no healthy/unknown rejection mechanism.

## 8. Submission and reproducibility

Before final experiments, freeze package versions with:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > environment-lock.txt
```

Document the exact used versions/install commands and all changes to the paper in the report. Config files capture versions too. Keep development files and outputs for your group; the Moodle submission must contain **only** `group_name.py` and `group_name.pdf` inside `group_name.zip` (replace the placeholder with your group name). Do not include images, weights, run directories or this development ZIP. Incorporate the setup/reproduction commands into the report's two-page user manual, including how to recreate the weights and generated directories.

See `METHOD_NOTES.md` for the paper-to-code mapping and `VALIDATION.md` for development checks. No real assignment-dataset result is supplied with this project.
