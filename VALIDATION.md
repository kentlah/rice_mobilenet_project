# Development validation — 8 October 2026

Checked on Python3.12 with CPU torch2.14.1, torchvision0.29.1, NumPy2.3.5, matplotlib3.10.8, scikit-learn1.8.0 and Pillow12.3.0. Windows installation/commands are provided but have not been executed on the user's laptop.

Completed checks:
- Python syntax compilation and command-line entry points.
- Download of official IMAGENET1K_V1 weights with published hash-prefix verification.
- Synthetic 100-image dataset, with 10 images per class/background combination: folder inference, counts, image validation and split feasibility.
- Five different, repeatable split allocations; each split retained all class/background strata.
- Duplicate-hash grouping prevented the same exact image crossing train/validation/test. Conflicting-label duplicates were rejected.
- Both head parameter counts verified: paper4,014,469; pooled82,309.
- Backpropagation reached head parameters only. Frozen feature parameters and BatchNorm running statistics stayed unchanged during training mode.
- Actual two-epoch paper-head training on all five synthetic splits, using downloaded pretrained features: validation-selected checkpoints, test predictions, scenario metrics, learning curves and confusion matrices generated.
- Actual one-epoch pooled-head training and checkpoint inference.
- Five-run mean/sample-SD summary generation; subgroup counts and confusion counts checked against test predictions.
- Existing experiment overwrite rejected; one-split summary rejected.
- Checkpoint single-image prediction and softmax-score sum verified.
- Generated confusion plot visually inspected for labels and legibility.

Synthetic noise images only validate execution. Their accuracy is meaningless and is not an assignment result. No Dhan-Shomadhan images were available to the implementation tests. Training time on the development CPU is not an estimate for the user's laptop. The real dataset folder names/counts, related-image leakage risk, convergence, performance and laptop runtime must still be checked.
