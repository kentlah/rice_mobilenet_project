# Mapping the paper to our implementation

Source: uploaded Altabaji et al. (2024), particularly Section III.E, Tables 1, 6 and 8, experimental setup and Section V. This is an adapted reproduction, not an exact numerical replication.

| Item | Paper | Project |
|---|---|---|
| Dataset | 2,658 Kaggle rice images; Brown Spot, Hispa, Leaf Blast, Healthy | Assignment's Dhan-Shomadhan, five diseased classes and two backgrounds |
| Input | MobileNetV2 224×224 RGB | 224×224 RGB; direct resize, EXIF orientation corrected |
| Initial weights | Keras ImageNet weights | Torchvision IMAGENET1K_V1; different checkpoint/framework |
| Backbone | Conv1 to out_relu frozen | Entire torchvision feature module frozen, including BatchNorm statistics |
| Head (Table 6) | Flatten 62,720; Dense 64; Dropout; Dense 4 | Flatten 62,720; Dense 64; ReLU; Dropout; Dense 5 |
| Ambiguous head details | Dropout rate and Dense-64 activation not specified in Table 6/text | ReLU and dropout 0.5 are declared assumptions, configurable dropout |
| Scaling | Image pixels scaled by 1/255 | Scale by 1/255, then ImageNet mean/std normalization required for torchvision weights |
| Zoom | Images first zoomed to expose lesions; no reproducible crop rule | No lesion crop: prevents subjective label-informed preprocessing; disclose omission |
| Augmentation | Rotation 30°, horizontal/vertical flips, size 80–120%, brightness 80–120% | Uniform rotation ±30°, flips p=0.5 each, isotropic scale 0.8–1.2 with black fill, brightness 0.8–1.2; train only |
| Geometry ambiguity | Height/width adjustment described without exact executable policy | Isotropic affine zoom is a declared approximation, not independent-axis reproduction; no translation added |
| Optimizer/loss | Adam or RMSprop comparisons; categorical cross entropy | Adam, lr=0.0001, cross entropy on logits (softmax applied only for prediction) |
| Learning rate ambiguity | Setup says 0.0001; results text describes curves with 0.001; Table 8 includes both | Fix 0.0001 based on setup; do not claim uniquely established best rate |
| Batch/epochs | MobileNet batch16; 400 epochs | Batch16; defaults400. Recommended lean experiment cap50/patience8 explicitly changes schedule |
| Checkpoint | Paper does not establish a complete reproducible model-selection/checkpoint rule | Minimum validation loss; declared protocol choice |
| Split | Table 1: 1,732 train, 434 validation, 492 test (about65.16/16.33/18.51%) | 60/20/20 joint class-background stratification; protocol adaptation |
| Repeated evaluation | Single listed split | Five different split seeds; mean and sample SD; white, field, mixed metrics |
| Optional pooled head | Not the paper's head | Proposed smaller architecture; optional comparison only |

The paper's MobileNetV2 test accuracy of 67.68% is **not a target or expected score** on our five-class dataset. Different classes, image backgrounds, checkpoint, scaling and splitting prevent direct numerical comparison. The method should be chosen for its feasibility and justified visual features, not on that cross-dataset score.

## Implementation sequence

1. Decode/verify images and record labels/backgrounds and hashes.
2. Make class×background-stratified 60/20/20 splits, grouping exact duplicates.
3. Resize RGB images to224×224. Augment training images only. Convert to tensors and normalize with mean (0.485,0.456,0.406), std (0.229,0.224,0.225).
4. Extract frozen ImageNet features without gradients. Keep backbone in evaluation mode.
5. Train only the classifier with Adam; use validation cross-entropy for checkpoint selection and optional stopping.
6. Load the best checkpoint, evaluate held-out data, then derive background subgroup results from the same predictions.
7. Repeat the whole model initialization and split procedure five times; average the five split-level metrics and compute SD.

Why this is a reasonable starting point: RGB CNN features capture colour/texture/shape, transfer learning limits optimisation cost, and a five-class head addresses the closed-set task. Potential weaknesses: ImageNet-to-rice domain shift, background shortcuts, small data, direct resizing distortion, uncropped lesions occupying little of an image, and overfitting in the Flatten head. Inspect error images to support these claims. A pooled head or carefully justified partial fine-tuning can be future method improvements; don't propose more data as the only improvement.

Report the actual image analysis, run settings and measured outcomes rather than copying this text as a finished report. Retain at least four relevant IEEE papers published in/after2022 for the related-work requirement. Choosing one implementation paper does not replace that literature requirement.

Official implementation sources:
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html
- https://docs.pytorch.org/vision/main/_modules/torchvision/models/mobilenetv2.html
- https://docs.pytorch.org/get-started/locally/
