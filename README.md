# GestureLab

Computer Vision Assignment 2 application: four-class gesture recognition, A1 architecture reproductions, MobileNetV4 transfer learning, real Grad-CAM, fixed-split evaluation and augmentation ablation.

## Windows (PowerShell, project root)

```powershell
./scripts/setup.ps1
# Originals are private and ignored. Place source/Assignment2.docx,
# source/notebook.ipynb and data/ready_dataset.zip if preparing a new checkout.
.venv/Scripts/python -m ml.dataset
# Explicitly downloads pretrained weights on first training; CPU default, seed 42.
.venv/Scripts/python -m ml.train --epochs 15 --models all
./scripts/run_all.ps1
```

Open http://127.0.0.1:5173. API documentation: http://127.0.0.1:8000/docs. Keep the startup terminal running; Ctrl+C stops both services. Existing verified checkpoints can be reused; training is not required on each startup. The training CLI refuses to overwrite existing run checkpoints: execute training only for missing runs in a fresh artifact directory or a separate checkout, preserving the current measured release. Setup needs internet for packages; serving uses only local files.

Or start in two VS Code terminals:

```powershell
.venv/Scripts/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev -- --host 127.0.0.1
```

## macOS / Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
npm --prefix frontend ci
.venv/bin/python -m ml.dataset
.venv/bin/python -m ml.train --epochs 15 --models all
sh scripts/run_all.sh
```

The measured environment is Windows / Python 3.10.9 CPU. Other platforms are documented but unverified. PyTorch package resolution may need a platform-specific CPU install from https://pytorch.org/get-started/locally/ before installing remaining requirements.

## Using the app

- Studio: upload a hand image, select a hand crop (avoid distant full scenes), choose an available trained model, enable Grad-CAM and predict. The original and explanation share the submitted 224-square padded crop. Camera requires localhost/HTTPS and browser permission; align a hand inside the guide then capture.
- Compare: send the same crop to all available trained classifiers; compare probabilities and measured timing.
- Experiments: saved clean-test accuracy, macro-F1, curves, confusion, ablation and separate deterministic stress metrics. Missing checkpoints/metrics remain explicitly unavailable.
- About: model provenance, methodology and privacy limitations.

Upload/camera images stay in memory. Do not distribute the supplied dataset or any report screenshots containing private images until consent/rights are verified. Grad-CAM is approximate, softmax scores uncalibrated, and the classifier has no reliable unrelated-object rejection.

## Verification

```powershell
.venv/Scripts/python -m pytest backend/tests ml/tests -q
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run build
.venv/Scripts/python scripts/smoke_test.py
```

In VS Code, Terminal > Run Task > GestureLab: Start app starts both services (stop existing services first).

See docs/RELEASE_STATUS.md for executed outcomes and artifacts/verification/ for logs. For browser checks run `npm --prefix frontend exec playwright install chromium` followed by `npm --prefix frontend run test:e2e`. The independent real-checkpoint smoke is `node scripts/browser_smoke.cjs` with `PLAYWRIGHT_MODULE` set to the absolute frontend/node_modules/playwright path and `PLAYWRIGHT_CHANNEL` set to an installed browser channel (default msedge). The actual local demo is `artifacts/demo/gesturelab-demo.webm` (3:49, silent), with timed sidecar captions `gesturelab-demo.srt`. Review it before submission; it contains a private supplied hand photograph.

## Scientific artifacts

- `artifacts/data_audit.json`: source counts, hashes, perceptual duplicate screening and provenance gaps.
- `artifacts/checkpoints/*.pt`: locally trained state plus exact metadata, ignored by Git.
- `artifacts/metrics/*.json`: per-run actual histories, clean metrics, per-class scores, confusion, timing and stress.
- `artifacts/metrics/comparison.csv`, `ablation.csv`, `artifacts/experiment_manifest.json`: measured comparison/provenance.
- `artifacts/plots/*.png`: plots generated from real evaluation outputs.
- `report/report.pdf` and report source: project report covering the implementation, measured results and required evaluation.
- `docs/demo_script.md`: four-minute narration guide. The actual 229.24-second silent local browser recording and sidecar captions are under `artifacts/demo/`; review, consent and sharing remain student tasks.

Training runs use notebook architectures/seed/optimizer and moderate transforms, reproduced on supplied originals. Variants stay in memory rather than notebook JPEG recompression. MobileNetV4 uses native 256 pretrained weights adapted to a shared 224 ImageNet-normalized crop; this is not the official ImageNet evaluation recipe. The initial transfer-learning runs keep their backbones frozen; the expanded runs below optimize them. Minimum validation cross-entropy selects weights. Test images receive no training augmentation; dim/blur/JPEG stress is separate. The tiny historical test (60), common source dataset and remaining session/identity uncertainty limit generalization claims. Official user-ID separation was checked during expansion, as documented below. The experiment cannot establish state-of-the-art accuracy.

Primary references: [MobileNetV4 ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php), [timm model card](https://huggingface.co/timm/mobilenetv4_conv_small.e3600_r256_in1k), [Grad-CAM](https://arxiv.org/abs/1610.02391). The assignment's specific Neurohive citation choice requires instructor confirmation; no synthetic generators are implemented or claimed.

Source repository: https://github.com/dayosama0/GestureLab (private). The application runs locally; no deployed app or shared video URL is provided. Datasets, trained checkpoints, image-bearing reports and recordings are intentionally excluded from Git. A fresh clone needs the private data and locally trained checkpoints or an explicit training run before prediction is available. Moodle deadline is November 7, 2026 at 23:59 Asia/Almaty; practice-session defense is also required.


## Expanded-data GPU iteration (7 November2026)

NewrealHaGRID sourceaudit:400 original UUIDs/labels matched400 distinct upstreamparticipants; originalsource split intersections0. Added 2000 train/400 val/400 test crops excludesall 400 originalpeople. Newtraining2280 images, selection400 images(218 users), freshholdout400 images(292 users); historicaltest60 is separate. Logicalpeace additionallyincludesback-of-hand peace_inverted. See docs/expansion_plan.md and artifacts/expansion_audit.json.

GPUtraining uses a separate environment, leaving CPUserving requirements unchanged:

```powershell
python -m venv .venv-gpu
.venv-gpu/Scripts/python -m pip install --upgrade pip
.venv-gpu/Scripts/python -m pip install -r requirements-gpu.txt
.venv/Scripts/python -m ml.hagrid_expand --train 500 --val 100 --test 100 --workers 6 --budget-gb 8
.venv-gpu/Scripts/python -m ml.finetune --models all --head-epochs 2 --finetune-epochs 8 --batch-size 16 --device cuda
```

Completed checkpoint IDs are immutable and refuse overwrites; reproduce training in a separate clean experiment checkout instead of replacing the release. Real ImageNet model downloads happen only during explicit preparation. GPUtested:GTX1660Ti6GB, driver560.94, torch 2.6.0cu124; Mobiletraining uses FP32 because fp16gradientcompatibilityfailed, ResNet uses AMP. Both architectures passedfinitebackwardandweightupdate smoke atbatch 16.

After both newruns finish, verify with `python scripts/smoke_expansion.py`, `node scripts/smoke_expansion.cjs` and `python scripts/compare_expanded.py` (the comparison is an immutable final test, run once). NewIDs are resnet18_finetuned224/ImageNet and mobilenetv4_finetuned256/.5mean/std. The appdefaults bysavednew-validationF1, nottest. Tall/wide handcrop controls and exactpreparedinput display match trainingletterboxing; no hand detector is claimed.

Userphoneimages remainheldout for a future final check. The existing exports/gesturelab-laptop.zip and229.24-secondsilentdemo are prior three-model snapshots; refresh afterfinalphonevalidation instead of presenting them as the newrelease. Current training/integration outcomes are in docs/RELEASE_STATUS.md.


## Current report and interface

The application is branded GestureLab. Result panels keep the predicted gesture, four scores, inference time and device visible; optional Technical details contain preprocessing, API roundtrip and checkpoint version. Models, training data and measured results are unchanged.

The current local report is available as report/report.docx and report/report.pdf: seven pages in Times New Roman 14 pt. Both files contain a source hand photograph and remain outside Git. The Markdown source is report/report.md. Rebuild with a Python environment containing python-docx using `python scripts/build_report.py`; the entry point builds the DOCX and exports it through the Windows Word-compatible COM application. The measured export used WPS Writer. The earlier ReportLab builder is retained as scripts/build_work_report.py. It covers the A1 continuation, MobileNetV4 addition, application workflow, real metrics/curves/ablation, latency and ethics. Earlier research-style versions remain archived for reference.

Training used real HaGRID photographs plus conventional training-only augmentation. No text-to-image or text-to-video samples were added. Dataset sources, actual measurements and limitations remain documented.
