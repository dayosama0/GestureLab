# Release status - 2026-10-30

Local implementation and scientific experiment gates pass. Submission still requires author identity, consent/rights confirmation, external access and instructor citation confirmation. No public deployment or repository push was performed.

| Gate | Status | Verified evidence |
|---|---|---|
| Source materials | PASS | source/Assignment2.docx and source/notebook.ipynb inspected; data/ready_dataset.zip copied from Downloads |
| Dataset audit | PASS | artifacts/data_audit.json:400 valid 224 RGB,70/15/15 per class; no exact/repeated-ID duplicates; no screened dHash candidates; subject independence unverified |
| Real baseline training | PASS | SimpleCNN and frozen ImageNet ResNet18 artifacts/checkpoints/*.pt, corresponding run JSON and training.log |
| Modern model / ablation | PASS | Real pretrained MobileNetV4 noaug/moderate checkpoints; fixed seed/split/epoch settings,5x exposure confound disclosed |
| Held-out scientific results | PASS | comparison.csv, ablation.csv, classification reports, predictions CSV, confusion and plots; all four checkpoint hashes verified |
| Genuine explanation | PASS | Actual selected-class gradients;3 production models,2 held-out inputs/model in integration and ML smoke JSON |
| Python suite | PASS | `.venv/Scripts/python -m pytest backend/tests ml/tests -q`:24 passed 8.72s,0 skips; artifacts/verification/pytest_final.txt |
| Frontend static/build checks | PASS | typecheck/lint/build exit 0; exact output artifacts/verification/frontend_*.txt |
| Frontend unit tests | PASS | `npm --prefix frontend test`:9 passed; artifacts/verification/frontend_test.txt |
| Browser integration | PASS | `node scripts/browser_smoke.cjs`:8 PASS with 3 real models/4 runs; private image upload,CAM,comparison,360px,missing/denied states |
| Browser UI fixtures | PASS | artifacts/verification/frontend_e2e.txt: `npm --prefix frontend run test:e2e` 6 passed in 5.2s across desktop/mobile; fixture checks do not establish physical-camera operation |
| Real API smoke | PASS | `.venv/Scripts/python scripts/smoke_test.py`:3 models,6 predictions,4 runs; integration.json |
| Upload privacy | PASS | Pre-parser body limit, in-memory allowed multipart,2 tests incl chunked oversize and 2.4 MB PNG; no user uploads retained |
| Dependency checks | PASS | pip check:No broken requirements found; frontend npm audit 0 vulnerabilities; requirements.txt/package-lock.json |
| Scientific report | PASS | report/report.pdf:7 pages, generated from real artifacts; report.md and scripts/build_report.py; all 7 page PNGs/contact sheet inspected, detailed pages 3/7 inspected |
| Model/data card and run guide | PASS | docs/model_data_card.md,methodology.md,README.md,Windows/POSIX scripts,VSCode tasks |
| Local app | PASS | API 127.0.0.1:8000 + UI 127.0.0.1:5173 started and actual browser verified |
| Physical webcam | MANUAL | Permission-denial/lifecycle tested; real browser camera request in recording returned Camera unavailable; successful hardware capture remains unverified |
| Recorded 3-5 minute demo | PASS | artifacts/demo/gesturelab-demo.webm: actual browser recording, 229.24 seconds (3:49), 1280x800, silent with gesturelab-demo.srt; local private only; manifest.json and extracted frame evidence |
| Consent / public data rights | MANUAL | Supplied evidence does not establish consent/license; private images/report/video not published |
| Student details | MANUAL | Report author/student ID must be filled by student |
| Neurohive citation | MANUAL | Primary ECCV/timm/Grad-CAM sources verified; relevant Neurohive citation not verified, seek instructor confirmation |
| External repository/video access | MANUAL | Local Git repository initialized on main, no commits/push/remote URLs supplied; do not invent links |
| Moodle / defense | MANUAL | Deadline 7 November 2026 23:59 Asia/Almaty, practice-session defense required |

## Measured clean test results (60 source images)

| Run | Accuracy | Macro-F1 | Mean warm CPU forward |
|---|---:|---:|---:|
| SimpleCNN moderate |95.00%|0.949981|2.002ms|
| ResNet18 moderate |96.67%|0.966648|48.608ms|
| MobileNetV4 noaug |91.67%|0.916681|21.986ms|
| MobileNetV4 moderate |93.33%|0.930747|20.064ms|

Source: artifacts/metrics/comparison.csv and per-run JSON. Modern method did not beat ResNet18. CPU benchmarks20 warm forward passes,3 warmups,2 threads; API includes Grad-CAM when requested, so timings differ. Hardware Intel i3-8100,16 GB RAM,Windows 10,Python 3.10.9,torch 2.14.1+cpu. Test set tiny, subject/session overlap unknown; no SOTA performance claim.

## Actual commands and outputs

- `.venv/Scripts/python -m ml.dataset`:400 images fixed source counts; audit JSON.
- `.venv/Scripts/python -u -m ml.train --epochs 15 --models all`:exit 0; 4 trained/evaluated runs; artifacts/training.log.
- `.venv/Scripts/python -m ml.smoke`:PASS all 3 production checkpoint gradients and differing inputs; ml_checkpoint_smoke.json.
- `.venv/Scripts/python -m pytest backend/tests ml/tests -q`:24 passed in 8.72s (final).
- `.venv/Scripts/python scripts/smoke_test.py`:PASS 3 models6 predictions4 runs.
- `npm --prefix frontend run typecheck`, `run lint`, `run build`:exit 0.
- `npm --prefix frontend test`:1 file, 9 tests passed.
- `node scripts/browser_smoke.cjs`:8 PASS actual integrated app; browser-only missing/denial fixtures labelled.
- `.venv/Scripts/python scripts/build_report.py`:Report generated7 pages.
- `pdftoppm -r 85 -png report/report.pdf artifacts/verification/report`:exit 0; Poppler reported missing optional Symbol/ArialUnicode display fonts, rendered pages visually verified with no corresponding defects.
- macOS/Linux startup commands: documented, not executed on this Windows host.

## Resolved setup / integration failures

- Initial old-pip CPU-index dependency installation failed metadata/build-dependency resolution. Full PyPI installation succeeded; torch imports as CPU, pip check passed. No environment blocker remains.
- Initial combined suite 18 passed / 3 skipped / 1 failed exposed MobileNet head 960 vs 1280 mismatch. Correct feature head fixed; full final 24 pass0 skip.
- First report layout10 pages rejected; adjusted spacing, rerendered and verified7 pages.
- Early browser executable/selector issues resolved; final actual browser evidence passed. See qa_review.md and frontend/VERIFICATION.md.

- `npm --prefix frontend run test:e2e`: 6 passed in 5.2s; artifacts/verification/frontend_e2e.txt.
- `node scripts/record_demo.cjs`: actual local video verified 229.24 seconds, 12,074,244 bytes, 1280x800, silent with timed sidecar captions. Recordings and sample frames are private under artifacts/demo/. Student should review and optionally narrate, then confirm consent before sharing.
- Local Git initialized on main; no commits, remote publication, or uploads performed.

Final video QA: full WebM decoded successfully using bundled ffmpeg1011 at 1 frame/second (exit 0); frames45/110/175 seconds visually inspected. Actual camera request returned unavailable, honestly shown. Local recording has no narration; sidecar SRT supplied.
