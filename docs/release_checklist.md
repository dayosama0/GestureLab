# Submission checklist

This checklist separates required evidence from manual submission tasks. Check current executable outcomes in `docs/RELEASE_STATUS.md`; do not infer completion from the existence of implementation code or this checklist.

| Requirement | Evidence to verify | Completion condition |
| --- | --- | --- |
| A1 topic/data continuity | `source/Assignment2.docx`, `source/notebook.ipynb`, private source ZIP and `artifacts/data_audit.json` | Source split/class mapping preserved; audit reviewed; baseline reproduction labeled accurately |
| Modern method | MobileNetV4 task checkpoints, metadata, runtime architecture and primary citation | Real pretrained weights loaded during training; actual task optimization; strict checkpoint reload; API prediction succeeds |
| Baseline vs modern comparison | Four per-run JSONs, `comparison.csv`, prediction CSVs, histories and confusion plots | Values recompute from actual held-out labels/predictions; no invented or placeholder metrics |
| Controlled ablation | `mobilenetv4_noaug` / `mobilenetv4` artifacts and `ablation.csv` | Settings and fixed split documented; fivefold optimization-exposure confound disclosed |
| Clean/stress separation | Per-run metrics/stress fields and protocol | Validation/clean test unaugmented; dim/blur/JPEG reported as separate correlated perturbations |
| Upload and visual explanation | Checkpoint-based API and browser evidence | Real image → selected trained model → normalized probabilities → genuine spatial gradient overlay |
| Camera | Permission-denied/cleanup tests and successful actual camera capture if hardware permits | No leaked stream; correct crop guide; physical camera path manually verified or limitation explicitly pending |
| Latency and environment | Raw warm forward timings, mean/p50/p95, named CPU/Python/PyTorch; UI roundtrip | Timing scope/warmup/repeats disclosed; browser overhead not presented as model time |
| Local privacy | Upload size/stream tests, memory spool test, no-retention code | Sensitive uploads not retained; private sources/screenshots absent from public repository |
| Python/frontend QA | Executed pytest, typecheck, lint, unit tests, build and browser smoke logs | Exact pass/fail/skip counts recorded; any remaining skipped acceptance tests called out |
| Scientific PDF | `report/report.pdf`, rendered page inspection, source and real figures | 6–8 pages; actual figures/metrics; readable layout; no unverified success claims |
| Author details | Report author/student identifier | Student fills correct details before submission; no guessed identity |
| Ethics/rights | Owner confirmation of source-photo consent/license and intended sharing | Confirm rights before public data or image-bearing report/video distribution; otherwise keep private |
| Citation choice | Primary MobileNetV4 source plus instructor confirmation | Resolve assignment's relevant Neurohive citation instruction without inventing an article |
| Demo recording | Real playable 3–5-minute screen recording, reviewed duration/audio | `docs/demo_script.md` is only a script; student reviews the actual silent local recording, adds narration if needed and shares only with confirmed consent |
| Repository access | Actual repository URL/access tested by intended reviewer | No repository URL is supplied until created/verified; exclude private source images and large caches |
| App access | Working localhost run guide or actual app URL | Local guide is sufficient if followed successfully; never invent a deployment link |
| Moodle submission | PDF, local run guide/app access, repository and real video link | Student uploads by **7 November 2026, 23:59 Asia/Almaty** |
| Oral defense | Practice-session preparation | Student can explain architecture, split/ablation limits, Grad-CAM, failure cases and reproduce the app |

A genuine local silent video is verified at artifacts/demo/gesturelab-demo.webm (229.24 seconds, sidecar captions). Student review/narration as needed and consent before sharing remain pending. No external repository, public deployment or Moodle submission is established. These remain manual until actual artifacts/access are verified. Do not publish private data as a shortcut to satisfying submission access.
