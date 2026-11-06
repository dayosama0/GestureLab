# Independent QA review

Reviewed 2026-10-30 against source assignment, ARCHITECTURE.md and docs/implementation_plan.md. This record separates code review, executed checks and remaining submission gates.

## Findings and resolution

| Priority | Finding | Resolution / evidence |
|---|---|---|
| High | FastAPI UploadFile normally spools images over 1 MB to disk; size check ran after multipart parsing. UI explicitly promises in-memory handling. | Fixed in backend/app/upload_limit.py under root delegation: receive cap 10 MB + 64 KB multipart allowance before parsing, also for unknown/chunked lengths; spool threshold exceeds bounded body; endpoint retains exact 10 MB file cap. backend/tests/test_upload_limit.py proves rejection before parser and valid >1 MB image remains unrolled and closed. |
| High | Camera permission can resolve after navigation and leave an offscreen stream active. | Frontend added request generation invalidation, stale-stream track cleanup and unmount cleanup in frontend/src/App.tsx. |
| High | Upload/ROI could change during inference and display old prediction alongside new source. | Frontend blocks uploads while requestRef active and disables source/ROI controls while busy. |
| Medium | Camera guide assumed 16:9 instead of actual stream dimensions. | Frontend computes square guide from loaded video metadata, matching capture's 75% shortest-side crop. |
| High | Initial MobileNet head expected960 but backbone output1280. | ML owner corrected head dimension based on actual timm architecture; rerun full Python suite required after training. |
| High | Last MobileNet convolution follows pooling, yielding1x1 attribution. | ML owner selects pre-pooling backbone.blocks[-1]; backend uses real activations and gradients, rejecting degenerate maps. |

## Executed evidence

- `.venv\Scripts\python -m pytest backend/tests/test_upload_limit.py -q`: **2 passed in4.06s**.
- `node --check scripts/browser_smoke.cjs`: exit0.
- `node scripts/browser_smoke.cjs`: exit0, **8 PASS** in initial checkpoint run. Actual SimpleCNN held-out fist upload, real API prediction and Grad-CAM, same-image available-model comparison, factual saved experiment, all four screens at360px with no full-page overflow, no uncaught browser runtime errors. Explicit browser-only fixtures verify missing-checkpoint disablement and denied camera permission. Evidence: artifacts/verification/browser_smoke_initial.txt and browser_smoke.json.
- Browser used installed Microsoft Edge headless via bundled Playwright module. Bundled browser1234 executable was unavailable; no browser installation claimed. `PLAYWRIGHT_MODULE`, `PLAYWRIGHT_CHANNEL`, `GESTURELAB_URL` can override local defaults.
- First browser attempt had exact-text selector mismatch because status container includes LOCAL; script corrected and full rerun passed. Earlier bundled headless launch failure is recorded, not an application defect.
- Screenshots contain only empty Studio and denied-camera state; no source photographs were saved as screenshots. Desktop screenshot visually inspected: no obvious overlap or clipping.

## Scientific / privacy review

- Four fixed labels and source split preserved; clean val/test are never augmented. Moderate training-only transforms are distinct from separately evaluated dim/blur/JPEG stress conditions. No generative synthesis is claimed.
- Model registry refuses absent/untrained/wrong-mapping artifacts and restores local state_dict with strict keys and weights_only loading. No random production prediction fallback found.
- Grad-CAM derives selected-class gradients at spatial model features; the RGB overlay colormap is applied to actual CAM. Nonconstant overlay alone does not prove attribution quality, but code uses real autograd and rejects constant raw maps.
- A1 comparison is an architecture reproduction: virtual image variants omit original notebook's JPEG95 encoding roundtrip. Native MobileNetV4 preprocessing differs from explicit shared224/ImageNet adaptation; report must state this rather than claim official model-card evaluation.
- No participant/session manifest or consent/license evidence supplied; exact/perceptual screening cannot prove person-independent generalization. Source dataset remains private.
- Training ablation uses equal epoch/optimizer settings but augmented training has5x examples and optimizer steps per epoch; do not interpret results as isolating transform content from training exposure.

## Remaining verification gates at this checkpoint

- Rerun browser smoke after all three production checkpoints and four experiment JSONs exist; initial browser run exercised one available model only.
- Restart running API after upload middleware change before final HTTP/browser integration.
- Run final complete Python suite, frontend typecheck/lint/test/build and report artifact consistency checks. Real camera hardware capture is not established by denied-permission test; physical camera walkthrough remains manual unless separately exercised.
- Confirm final PDF6-8 pages, primary references and actual result figures. Recorded3-5 minute video, external repo/app access, consent and requested Neurohive citation/instructor confirmation remain author submission gates unless actual evidence is added.
## Final integrated QA evidence

- API was restarted after privacy middleware changes. Actual HTTP registry shows3 available trained models and experiments endpoint returns4 evaluated runs.
- `.venv\Scripts\python -m pytest backend/tests -q`: **16 passed in7.42s**, zero skipped. Includes actual held-out checkpoint+Grad-CAM integration for SimpleCNN, ResNet18 and MobileNetV4, plus both privacy/body-limit tests. Captured artifacts/verification/backend_qa.txt.
- `node scripts/browser_smoke.cjs`: **exit0;8 PASS** after all models finished. Actual MobileNetV4 held-out fist upload predicted fist; returned version f759b9ea8be49117dcd958db62359b5cc2eac16aeec442bf2265af4054433d2f. Genuine explanation rendered and opacity control adjusted. Comparison rendered actual predictions from all3 trained models. Experiments rendered all4 real saved runs. Four pages at360px had no page-wide overflow or uncaught runtime errors. Missing-weights and denied-camera checks are explicitly browser-only fixtures. Evidence artifacts/verification/browser_smoke_final.txt and browser_smoke.json.
- Visual review of desktop empty Studio and360px camera-denied screenshot shows no obvious clipping/overlap. No private source-image screenshots were retained.
- Independent artifact integrity command checked SHA256 against each physical checkpoint; every confusion matrix totals60 with15 examples in each true-class row; all four saved prediction CSV accuracies exactly match JSON metrics. Output artifacts/verification/artifact_integrity.txt. Clean accuracies: SimpleCNN0.95, ResNet180.9666667, MobileNetV4 noaug0.9166667, MobileNetV4 moderate0.9333333. These are measured small-split results, not population/SOTA guarantees.
- All initially raised implementation findings above are resolved in reviewed source. Remaining author gates: physical-camera walkthrough, recorded3-5 minute video, consent/redistribution confirmation, external submission/repo access and Neurohive citation choice/instructor confirmation. Report final page/visual checks and full ML/frontend suite remain orchestrator-owned verification.

## Final root integration and artifact closure

Root final Python suite:24 passed,0skipped (artifacts/verification/pytest_final.txt). Frontend typecheck/lint/build exit0,9unit tests and6UIbrowser tests passed. Report is7pages with actual results and Grad-CAM, rendered and visually verified. A real silent local browser walkthrough at artifacts/demo/gesturelab-demo.webm is229.24seconds, sidecar captions and full decode verified; private source image appears, no external sharing. Physical camera remained unavailable in the actual browser request. Remaining manual tasks are student identity, consent/rights, citation confirmation and reviewer/submission URLs.
