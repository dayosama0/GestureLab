# Scientific methodology

## Assignment and source provenance

The authoritative assignment is `source/Assignment2.docx`: continue the Assignment 1 topic and data; integrate a modern CV method; compare with baselines; provide upload/camera prediction with a visual explanation; report metrics, training curves, an ablation, latency, environment and ethics; submit a 6–8-page report, a 3–5-minute recorded demo, repository access and an app link or local run guide. Its text allows a new/modern method beyond the suggested generators. Submission is due **7 November 2026, 23:59 Asia/Almaty**, with a practice-session defense.

The supplied `source/notebook.ipynb` defines SimpleCNN and frozen ImageNet ResNet18 baselines, with training disabled by default. Its referenced result/checkpoint and augmented-data folders were not supplied. The experiments are newly executed **A1 architecture/protocol reproductions**, not recovered original A1 metrics. Moderate transforms follow the notebook, but variants stay in memory rather than being JPEG-recompressed before training, so numerical equivalence to a historical notebook run is not claimed.

## Dataset and split integrity

The private archive `data/ready_dataset.zip` contains 400 decoded 224×224 RGB photographs. Each of `fist`, `open_palm`, `peace`, `thumbs_up` has 70 train, 15 validation and 15 test examples, for totals of 280/60/60. The class-to-index order is fixed in that order. The audit checks image integrity, dimensions/color, SHA-256 bytes and source IDs; it also screens cross-split 64-bit difference hashes at Hamming distance ≤3. The actual audit is stored in `artifacts/data_audit.json` with individual source hashes.

No source split is regenerated. In the November7 expansion, all 400 original UUIDs/labels were matched to official HaGRID annotations:400 distinct user IDs with no original split intersections. This establishes separation under upstream identifiers, not independent identity or personal consent. HaGRID custom licensing conditions are now cached; signed individual consent is not independently re-established. Source images and image-bearing report evidence must remain private until the owner verifies rights. No generative images/video are used; synthetic/generative share is zero.

## Shared crop and preprocessing

Training and serving use the same `ml.preprocessing` functions: EXIF orientation, RGB conversion, gray `(128,128,128)` letterboxing to 224 square, CHW float conversion and ImageNet normalization with mean `(0.485,0.456,0.406)` and standard deviation `(0.229,0.224,0.225)`. Transparency is composited onto gray. Camera alignment and upload crop controls request a hand ROI; no automatic hand localization or unrelated-object rejection is claimed.

The timm MobileNetV4 checkpoint's native recipe uses 256-pixel input and mean/std `(0.5,0.5,0.5)`. This project deliberately adapts it to the shared A1 224/ImageNet-normalized ROI contract for both training and inference. Results apply to that adaptation, not the official MobileNetV4 ImageNet benchmark recipe.

## Models and optimization

| Run | Backbone / head | Training data |
| --- | --- | --- |
| `simple_cnn` | Notebook three-block 16/32/64-channel CNN; adaptive 4×4 pooling; dense 128; dropout 0.3; four logits. All parameters trained from scratch. | Originals plus four fixed moderate variants |
| `resnet18` | Frozen torchvision ImageNet ResNet18; dropout 0.2 plus linear 512→4 head, matching notebook module structure. | Originals plus four fixed moderate variants |
| `mobilenetv4_noaug` | Frozen `mobilenetv4_conv_small.e3600_r256_in1k`; dropout 0.2 plus linear 1280→4 head. | Originals only |
| `mobilenetv4` | Same pretrained architecture and head as no-augmentation run. | Originals plus four fixed moderate variants |

All runs use seed 42, Adam learning rate 0.001, weight decay 0.0001, batch size 32, a maximum of 15 epochs, and early stopping after five non-improving validation-loss epochs. Minimum validation cross-entropy selects the saved checkpoint. Frozen backbones stay in evaluation mode while train/validation features are cached, so their pretrained batch-normalization statistics do not change. Training and benchmark device is CPU; actual versions/environment and run durations are recorded in metadata.

The four fixed moderate variants are mirror, small rotation, brightness/contrast/color change, and affine scale/translation. Only training receives them. There are 1,400 moderate training tensors versus 280 without augmentation. Validation and clean test never receive these variants. The MobileNet pair is a **fixed-epoch augmentation ablation**, not a compute-matched comparison: five times as many train tensors also yield more optimizer steps per epoch. Any observed difference combines transformed examples and increased optimization exposure. With a single seed and small test set, causal or significance claims are unwarranted.

## Evaluation and latency

Each chosen checkpoint is saved and strictly reloaded into a newly constructed model before clean held-out evaluation. Metrics are accuracy, macro precision/recall/F1, per-class scores and a 4×4 confusion matrix. Per-image predictions, histories, checkpoint hashes and comparison/ablation CSVs retain the evidence. The test set is not used for hyperparameter or epoch selection. One error changes overall test accuracy by about 1.67 percentage points; 15 examples per class make class-wise estimates unstable. Any binomial interval assumes independent observations, which unknown participant/session overlap cannot guarantee.

Deterministic stress conditions are evaluated separately on the same 60 test images: brightness multiplier 0.5, Gaussian blur radius 2 and JPEG quality 30. They are correlated perturbations of held-out images, not independent samples, motion synthesis, ArtAug or Step-Video-T2V. They must not replace or be mixed into clean-test metrics.

Each CPU benchmark performs three warmup passes followed by 20 single-image forward timings, saving raw repetitions plus mean/p50/p95. It excludes decoding, preprocessing, queueing, Grad-CAM and network/browser time. The UI reports request roundtrip separately. Training duration includes preparation, first-run downloads when applicable, cached feature extraction and fitting; it is not an isolated optimizer benchmark.

## Explanation and application limitations

Grad-CAM differentiates the predicted-class logit of the selected model with respect to actual spatial activations. Spatially averaged gradients weight feature channels, followed by ReLU, interpolation and normalization. Input gradients allow attribution through frozen backbones. MobileNetV4 uses the final pre-pooling spatial block, avoiding its post-pooling 1×1 convolution. Degenerate/constant maps produce an explicit error. Grad-CAM is a coarse attribution, not hand segmentation or causal evidence.

Only local checkpoints with trained-model provenance, exact class order and matching preprocessing can serve predictions. No random or fixture predictions appear in production. Uploads/camera frames are not retained by the application. The four-way classifier has no reliable out-of-distribution rejection; uncalibrated softmax scores can be high for unrelated objects. Background, lighting, skin tone, hand shape and pose biases remain possible. Broader consented, person-independent evaluation is required before consequential use.

## Reproduction and source citations

Run from the project root after installing the isolated environment:

```powershell
.venv/Scripts/python -m ml.dataset
.venv/Scripts/python -m ml.train --epochs 15 --models all
.venv/Scripts/python -m pytest backend/tests ml/tests -q
.venv/Scripts/python scripts/smoke_test.py
```

Training explicitly downloads pretrained weights when first needed. Serving reconstructs architectures with `pretrained=False` and loads local task checkpoints. See `README.md` for frontend checks and separate Windows/POSIX startup commands. Exact outcomes belong in `docs/RELEASE_STATUS.md`; a written command is not proof it ran.

- Qin et al., [MobileNetV4: Universal Models for the Mobile Ecosystem](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php), ECCV 2024.
- [timm MobileNetV4 Conv Small model card](https://huggingface.co/timm/mobilenetv4_conv_small.e3600_r256_in1k), exact pretrained identifier/provenance.
- Selvaraju et al., [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391), ICCV 2017.

The assignment separately requests a relevant Neurohive citation. The primary MobileNetV4 citation is supplied; no verified MobileNetV4 Neurohive page is asserted. Instructor confirmation of this citation choice is an unresolved submission item. The modern architecture is not evidence that this small experiment achieves state-of-the-art gesture recognition performance.


## November7 real-data expansion and GPU fine-tuning

OfficialHaGRID selective ZIP ranges/CRC and annotation joins produced2800 additional real ROI images. Train/val/test additional counts are500/100/100 perlogicalclass; original 70/class train added gives2280 train/400newval/400newtest. Original60val is retained but not used for new selection; original 60test is a separate historical cohort. Newtrain1866 users includes280originaltrainparticipants; newval218 users andnewtest292 users do not intersect train or eachother. All400 originalusers are excluded from additions. Extra logicalpeace includes75% peace and25% peace_inverted; sourcegestures/IDs and per-cropSHA256 are saved. Matchingbox+15%margin is gray 128letterboxed320; no validation/test augmentation. CustomHaGRIDpubliclicense is explicitly not a CreativeCommonslicense despite itsCCBY-SA-derived text.

New immutable IDs resnet18_finetuned (224/ImageNet) and mobilenetv4_finetuned (256/.5/.5) use fresh real ImageNet initialization. Two headwarmup epochs freezebackbone/BN; up to 8fullbackbone epochs useAdamW1e-5backbone/1e-4head, weightdecay1e-4, batch 16, seed42, patience4. Select only fine-stage minimum newvalidationloss. The default UI model uses newvalidationmacroF1 only, not test. GPU training uses torch 2.6.0cu124 on GTX1660Ti6GB. ResNetAMP passed; MobileFP16 gradient smoke failed, so Mobile explicitly uses verifiedstableGPUfloat32. Temporarysmokes export no taskcheckpoint or performance metric.

The same common fresh400 test compares immutableold3 andnew2 after selection. Originaltest results remain separate. Data/normalization/augmentation/optimization all change, so this is not a causal singlefactor ablation; originalfrozenMobile noaug/moderate pair remains the controlled conventional-augmentation ablation with5xexposureconfound. Native rectangular crop controls and exact prepared-input base for GradCAM reduce train/serving crop mismatch. Personalphoneimages remainfutureholdout and are never added to training automatically.
