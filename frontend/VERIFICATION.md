# Frontend verification — 2026-10-30

Executed from project root:

- `npm --prefix frontend install`: installed actual dependencies and package lock.
- `npm --prefix frontend run typecheck`: exit 0.
- `npm --prefix frontend run lint`: exit 0.
- `npm --prefix frontend test`: 1 file passed, 9 tests passed (Vitest 5.0.5).
- `npm --prefix frontend run build`: exit 0; Vite 7.3.7, 2170 transformed modules. Charts split into a separate vendor bundle.
- `npm --prefix frontend audit`: 0 vulnerabilities after upgrading Vitest from initial scaffold version.
- `npm --prefix frontend exec playwright install chromium`: downloaded Chromium 156 and headless shell. First premature browser run failed because the headless shell download was unfinished; rerun after download succeeded.
- `npm --prefix frontend run test:e2e`: 6 passed in 10.2 seconds across desktop and 360px mobile Chromium.

Browser E2E here tests isolated UI fixtures: missing weights, factual empty experiments, navigation, no horizontal overflow, denied camera with upload fallback, late permission cleanup. Fixtures are restricted to test request interception. Actual model/Grad-CAM API verification belongs to root/QA integration and is not claimed by these tests.

Camera uses explicit capture rather than an unbounded live request loop. Only one inference request can be active. Crop/image changes are blocked until the response completes to prevent misleading alignment of stale explanations. Late media streams are immediately stopped after navigation or unmount.

Run backend on 127.0.0.1:8000, then `npm --prefix frontend run dev`. Vite listens at http://127.0.0.1:5173 and proxies /api. No frontend runtime external font/CDN dependency; inference data comes exclusively from the local backend. `npm run build` output is in frontend/dist.

## Expanded training support — 2026-11-07

Changed only frontend files for the expansion integration:

- `src/image.ts`: source-resolution square crop, maximum1536px edge; no upscaling to224; canvas composited onto opaque(128,128,128) before PNG encoding. API applies each checkpoint's own resize/normalization.
- `src/api.ts`: typed preprocessing, optional independent expanded-test metrics, and preferred model selection using only finite[0,1] saved validation F1 on available fine-tuned models; stable ID tie break, first-available fallback.
- `src/App.tsx`: dynamic input dimensions, fine-tuned model tags, original-test labels and separate independent expanded-test panel; actual latency device labels. Revised methodology distinguishes frozen-feature baselines and staged backbone fine-tuning without claiming unavailable weights or future dataset counts.
- `tests/image.test.tsx`, `tests/App.test.tsx`, `e2e/studio.spec.ts`: crop dimensions/cap, gray compositing sequence, validation-only default selection, expanded holdout separation, actual browser PNG gray pixels and native resolution.

Commands executed after final code changes:

- `npm --prefix frontend run typecheck`: exit0.
- `npm --prefix frontend run lint`: exit0.
- `npm --prefix frontend test`: 2 files passed,21 tests passed; duration2.19s.
- `npm --prefix frontend run build`: exit0; Vite7.3.7,2170 modules,4.75s; main242.72kB(76.49kB gzip), charts386.10kB(113.20kB gzip), CSS19.38kB(5.03kB gzip).
- `npm --prefix frontend run test:e2e`:8 passed(10.0s), desktop and360px mobile. Test-only route fixtures exercise unavailable models/metrics and camera failure; actual browser canvas roundtrip verifies360x360PNG and pixel[128,128,128,255]. New checkpoint inference/expanded metrics must be verified separately by root/API integration once training artifacts exist.

No new checkpoint, score, dataset size, consent or successful fine-tuning is asserted by this frontend verification. No PWA, hosting, HTTPS or upload retention change was made.

## Root final verification - rectangular crop edition

Typecheck/lint/build passed. Vitest22passed; Playwright10passed in8.8s across desktop/mobile360, including rectangular234x360PNG preservation and actual crop-shape controls. Backend now supplies the exact prepared image so the native256base matches its GradCAMoverlay. Separate real browser acceptance passed12checks using5trained models and6actual runs; defaultResNet was verified from validationF1 rather than test ranking. Evidence:artifacts/verification/expansion_frontend_checks.json, expansion_smoke.json and expansion_browser_stdout.txt.

## Presentation cleanup — 2026-11-07

UI-only edits in src/App.tsx, src/styles.css and index.html. Branding is GestureLab; removed AI suffix and repeated chips/taglines. Prediction label, four probabilities, measured model latency and device remain prominent. API roundtrip, input dimensions and checkpoint hash moved into a closed Technical details disclosure. About now briefly covers usage/models, audited3200HaGRIDimages(original400+additional2800), dataset/model/license links and privacy/consent limitations. Experiment tables, curves, confusion matrix, expanded holdout, ablation and stress metrics retained; evaluation protocol moved to a disclosure. Inference, ROI, camera and API logic unchanged.

Updated tests/App.test.tsx to verify preprocessing and API roundtrip are initially hidden and available when Technical details opens.

Final verification:
- npm --prefix frontend run typecheck: exit0.
- npm --prefix frontend run lint: exit0.
- npm --prefix frontend test:2files passed,22tests passed(2.31s).
- npm --prefix frontend run build: exit0,Vite7.3.7,2170modules,7.62s. Main241.69kB(gzip75.97kB),charts386.10kB(gzip113.20kB),CSS20.53kB(gzip5.26kB).
- npm --prefix frontend run test:e2e:10passed,desktop+360pxmobile. Existing permission/cleanup/nativecrop/aspectratio/navigation states preserved.

Initial unit run exposed the old expectation that input dimensions appeared directly in the result; updated it to verify the new disclosure behavior, then22/22passed. Root/QA independently verifies real checkpoint inference after presentation changes.
