# Reproducible experiments

From the project root, with the isolated environment installed:

```powershell
.venv/Scripts/python -m ml.dataset
.venv/Scripts/python -u -m ml.train --epochs 15 --models all
.venv/Scripts/python -m pytest ml/tests -q
```

The first command safely extracts the supplied ZIP into the private `data/` directory and writes `artifacts/data_audit.json`. A byte-hash/source-ID audit and cross-split perceptual dHash screen cannot establish participant independence.

Training explicitly downloads pretrained ResNet18 and MobileNetV4 weights on the first run. Four experiments use seed 42, batch size 32, Adam at 0.001 with weight decay 0.0001, a maximum of 15 epochs, and minimum validation loss selection with patience 5. The A1 architectures and notebook moderate transforms are reproduced; variants remain in memory rather than undergoing the notebook's JPEG recompression. Existing original A1 scores/weights were not provided.

SimpleCNN is learned from scratch. ResNet18 and MobileNetV4 freeze their pretrained backbone, cache training/validation features, then learn the same dropout-plus-linear 4-class head. MobileNetV4 runs without and with the four fixed moderate variants plus each original. Validation and clean test receive no augmentation. The clean held-out test is evaluated after validation selection; deterministic dim-light, blur, and JPEG stress evaluations are separate. These perturbations are ordinary image processing, not ArtAug or synthetic video generation.

The original three model IDs deliberately share RGB, gray-128 letterboxing, 224x224 input and ImageNet mean/std. The actual timm MobileNetV4 pretrained configuration uses 256x256 and mean/std 0.5; the original frozen-feature experiment adapts it to the shared A1 protocol, identically in training and serving. Results must not be presented as the official MobileNetV4 benchmark protocol or a state-of-the-art performance claim.

Each run writes a full local checkpoint, exact metadata, clean/stress classification scores, per-class metrics, confusion matrix, predictions, epoch histories, warm forward timing, and generated PNG plots. Checkpoint reconstruction uses `pretrained=False` and never downloads at inference. Training refuses to replace an existing checkpoint; archive a complete run before retraining. The production API serves only trained checkpoints for the three production IDs; the `mobilenetv4_noaug` checkpoint is the ablation.

The MobileNetV4 ablation fixes epoch budget and optimizer settings, not the number of optimizer steps: the moderate dataset has 1,400 examples and the no-augmentation dataset 280. Its result compares these two training recipes; augmentation and greater optimization exposure cannot be causally separated by this experiment.

The shared preprocessing operates on a selected hand ROI. It does not locate hands or reject unrelated objects. Softmax scores and Grad-CAM are not guarantees of correctness. Source photos remain private and must not be redistributed without verified consent/license.

## Expanded-data GPU fine-tuning

The separate `resnet18_finetuned` and `mobilenetv4_finetuned` IDs use fresh real ImageNet-pretrained backbones, two epochs of head warmup, and up to eight epochs of full-backbone fine-tuning. The backbone learning rate is 0.00001 and the head rate 0.0001 during fine-tuning. The head warmup keeps backbone BatchNorm statistics frozen. CUDA FP16 mixed precision is enabled on compatible hardware; the default batch size is 16. Only checkpoints after actual backbone optimization are eligible for selection, using new HaGRID validation loss.

```powershell
.venv-gpu/Scripts/python -u -m ml.finetune --models all --head-epochs 2 --finetune-epochs 8 --batch-size 16
```

This command requires the audited `data/expanded/manifest.json` to declare `ready: true`, contain official user IDs, and record verified original-participant exclusion. It rejects source/user split overlap, original holdout IDs, unsafe paths, incomplete overlap-review flags and mismatched recorded crop hashes. Original training data is retained; original validation/test images do not enter expanded training or checkpoint selection.

ResNet18 retains 224/ImageNet preprocessing. The new MobileNetV4 uses its native 256 input and mean/std 0.5. Training and serving read the same per-model policy. This correction, additional real data, and full-backbone optimization change together, so the new-versus-original result is not a single-factor ablation. New model metadata includes selected-validation metrics, actual first-convolution parameter change, peak GPU memory, preprocessing and manifest provenance. Original60 and expanded held-out tests are evaluated after selection; their metrics remain separate. Warm CPU inference timings exclude CUDA training, uploads and explanations. Existing original checkpoints/results are never replaced.
