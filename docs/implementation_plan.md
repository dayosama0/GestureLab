# Implementation plan

Source assignment and notebook inspected on 2026-10-30. Originals copied from Downloads; no original files moved. Python 3.10.9, Node 26.13.0; isolated environment required.

1. Establish contracts, install dependencies, safely extract and audit private dataset.
2. In parallel implement ml/, backend/, frontend/ with disjoint owners.
3. Run seed-42 SimpleCNN, frozen ResNet18, MobileNetV4 no/moderate augmentation; select checkpoints on validation then evaluate clean test and separate stress tests.
4. Integrate real checkpoints, Grad-CAM, API and browser. Run pytest, frontend checks/build and integration.
5. Generate measured report/figures, local run guide, demo script and release evidence. Recorded video and external submission remain manual.

## Shared contract (authoritative for implementation)
- Classes: fist, open_palm, peace, thumbs_up.
- ml.models: create_model(model_id, pretrained=False), target_layer(model), MODEL_IDS = simple_cnn, resnet18, mobilenetv4. Mobilenet ablation run ID mobilenetv4_noaug, production model ID mobilenetv4.
- ml.preprocessing: prepare_image(PIL image) -> RGB letterboxed 224x224 using gray 128; image_tensor(PIL image) -> normalized CHW tensor. Input is a user selected hand ROI; frontend provides guided crop for camera and crop controls for uploads. No hand detector is claimed.
- Checkpoints artifacts/checkpoints/{model_id}.pt: dict state_dict, metadata. Metadata: model_id, classes, trained=true, input_size=224, mean, std, seed, augmentation, pretrained_source, best_epoch. Local model restore uses pretrained=False.
- Metrics artifacts/metrics/{run_id}.json: run_id, model_id, metadata, history [{epoch,train_loss,val_loss,train_accuracy,val_accuracy}], metrics {accuracy,precision_macro,recall_macro,f1_macro,per_class}, confusion_matrix (4x4), latency {mean_ms,p50_ms,p95_ms,device}, stress (condition -> metrics). Never fabricate unavailable fields.
- GET /api/health: {status,device,ready,models}; GET /api/models: {classes,models:[{id,name,available,reason,version,metadata}]}.
- POST /api/predict file/model_id/explain: {model_id,version,label,probabilities:{class:float},latency_ms,device,preprocessing,explanation: null | {image_base64,mime_type,width,height,method}}. Invalid file 400, >10MB 413, absent model 503; no uploads retained. Explain image corresponds to processed 224-square ROI.
- GET /api/experiments: {available,runs:[full run JSON]}; GET /api/experiments/{run_id}: full JSON or 404. Only validated fixed safe run IDs.
- Frontend Vite proxy /api to localhost:8000, npm scripts typecheck/lint/test/build, localhost dev port 5173.

## Commands / gates
.venv/Scripts/python -m pytest backend/tests ml/tests
.venv/Scripts/python -m ml.train --help (training CLI owned by ML)
npm --prefix frontend run typecheck; npm --prefix frontend run lint; npm --prefix frontend test; npm --prefix frontend run build
Checkpoint-based API integration and Grad-CAM nonconstant check required. Report PDF 6–8 pages; video not claimed unless recorded.
