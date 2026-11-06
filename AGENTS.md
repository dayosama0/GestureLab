# GestureLab AI — project operating rules for Codex

## Scope
Implement Astana IT University Computer Vision Assignment 2 based on `Assignment2.docx`, `notebook.ipynb`, and `ready_dataset.zip`. Read `ARCHITECTURE.md` first. Keep training/testing scientific, auditable and real.

## Nonnegotiable
- Never claim a metric, trained weight, screenshot, synthetic generator, report or video exists without verifying its source artifact.
- Do not substitute random predictions or demo fixtures into production endpoints.
- No augmentation of val/test for clean-test metrics; preserve class mapping `fist`, `open_palm`, `peace`, `thumbs_up` and source split.
- Match train/serving hand-ROI padding, image shape, channel order and per-model normalization.
- Use real model gradients for Grad-CAM; do not fake visual explanations.
- Treat input images as sensitive personal assets; do not retain uploads, push private data or expose them without permission.
- Prefer fully working upload prediction + Grad-CAM + comparative experiments over speculative extras.
- Do not add AI-agent workflow implementation details to the academic report; report scientific methodology and actual software architecture only.

## Working practice
- Root agent owns plans, contracts, integration, final QA. Delegate disjoint `ml/`, `backend/`, `frontend/`, `docs/`; the QA agent reviews and records issues.
- Parallel work only in non-overlapping files; agree API contracts first. Subagents report: files changed, exact commands, outcomes, blockers.
- Use a short durable `docs/progress.md` and `docs/RELEASE_STATUS.md`; resolve real blockers; don't stop at plans.
- No unverified claims of 'SOTA'; modern primary model is real MobileNetV4 Conv Small, paper ECCV 2024. Do not equate PIL augmentations with ArtAug or Step-Video-T2V.

## Verify before completion
- Execute Python tests and relevant ML smoke, frontend typecheck/lint/test/build, API integration and browser E2E if available.
- In final report list exact command outputs and actual paths. Mark unrun tests as unrun; missing checkpoints as unavailable.
- Before release: ensure 6–8-page report, actual figures, 3–5-minute recorded demo by user if not automatable, valid repo/app access, ethical consent statements.
