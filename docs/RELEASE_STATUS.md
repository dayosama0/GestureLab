# Release status - 7 November 2026

Expanded training and local application verification: PASS. The next step is the user's personal iPhone photo check. Final laptop packaging and demo refresh are deliberately deferred until that check. No external publication or transfer was performed.

## Executed gates

| Gate | Result | Evidence |
|---|---|---|
| Original provenance | 400 UUIDs and labels matched 400 distinct official HaGRID users; zero original split user intersections | data/expanded/original_subject_join.json |
| Expanded dataset | 2280 train / 400 new validation / 400 new test records; original val/test preserved | artifacts/expansion_audit.json; data/expanded/manifest.json |
| Participant separation | 1866 training, 218 new validation and 292 new test users, with disjoint sets; all 400 original users excluded from additions | acquisition audit and expansion_data_integrity.txt |
| Duplicate screening | No retained source/hash overlap or unresolved cross-split perceptual candidates | acquisition audit |
| GPU training | Two real runs completed, each 2 head + 8 backbone epochs; selected epoch 9 using new validation loss | artifacts/finetuning.log; per-run JSON |
| Actual backbone updates | Nonzero first-convolution weight changes; strict checkpoint reloads | per-run metadata and smoke JSON |
| Legacy preservation | 13 original checkpoint/metric SHA256 values unchanged | legacy_before_expansion.json; expansion_smoke.json |
| API integration | Five models, ten real predictions, six experiment runs, exact 224/256 input/explanation alignment | expansion_integration.json and expansion_integration_stdout.txt |
| Python tests | 50 passed in 14.13s; zero skipped | expansion_pytest_final.txt |
| ML tests in CUDA environment | 24 passed in 7.93s | finetune_pytest_gpu_env.txt |
| Frontend | Typecheck/lint/build passed; 22 unit tests; 10 UI browser tests passed in 8.8s | expansion_frontend_checks.json; frontend/VERIFICATION.md |
| Real browser integration | 12 checks passed: five real models, six runs, upload/CAM, validation-ranked default, comparison and 360px layout | expansion_browser_stdout.txt; expansion_smoke.json |
| Local service | API ready on127.0.0.1:8000, UI on127.0.0.1:5173; refresh the tab to fetch new weights | actual HTTP and browser checks |
| Scientific report | Seven-page Times New Roman 14 pt Word report with exported PDF, visually checked | report/report.docx; report/report.pdf; word_report_qa.json |

Verification filenames above are under artifacts/verification/ unless a different directory is shown.

## Same fresh holdout: 400 images / 292 participants

| Model | Accuracy | Macro-F1 | Same-host CPU mean |
|---|---:|---:|---:|
| simple_cnn | 90.00% | 0.897637 | 1.481ms |
| resnet18 | 93.75% | 0.937510 | 43.149ms |
| mobilenetv4 | 91.50% | 0.914737 | 18.587ms |
| resnet18_finetuned | 100.00% | 1.000000 | 49.360ms |
| mobilenetv4_finetuned | 100.00% | 1.000000 | 15.092ms |

Both new models also scored 60/60 on the historical test. That test was already examined in the first phase. These are curated HaGRID hand ROIs; personal-camera performance remains unknown. New data, orientation coverage, preprocessing, augmentation and backbone learning change together, so improvement cannot be attributed to a single factor. No state-of-the-art accuracy claim is made.

CPU profile: i3-8100 at3.60GHz, 16GB RAM, Windows10, PyTorch2.14.1+cpu, two threads, three warmups and20 single-image forward timings. Decode, preprocessing, Grad-CAM and network are excluded. Raw repetitions are retained. CUDA training used torch2.6.0+cu124 on GTX1660Ti6GB, driver560.94. ResNet used AMP; Mobile used float32 after an observed fp16 gradient compatibility failure. Both passed finite architecture backward/update checks with batch16. GPU peak allocated memory excludes driver/display reservations.

## Exact commands and actual outcomes

```powershell
.venv/Scripts/python -u -m ml.hagrid_expand --train 500 --val 100 --test 100 --workers 6 --budget-gb 8
# ready:true; 3080 manifest records. Log: data/hagrid_cache/expansion.log
.venv-gpu/Scripts/python -u -m ml.finetune --models all --head-epochs 2 --finetune-epochs 8 --batch-size 16 --patience 4 --device cuda
# exit0; both trained/evaluated runs. Log: artifacts/finetuning.log
.venv/Scripts/python scripts/compare_expanded.py
# PASS: five models on one fresh holdout. expansion_comparison_stdout.txt
.venv/Scripts/python scripts/smoke_expansion.py
# PASS: five models, ten predictions, six runs. expansion_integration_stdout.txt
.venv/Scripts/python -m pytest backend/tests ml/tests -q
# 50 passed in14.13s, zero skipped. expansion_pytest_final.txt
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e
# all successful; 22 unit tests and10 browser tests.
node scripts/smoke_expansion.cjs
# exit0; 12 PASS. expansion_browser_stdout.txt
.venv/Scripts/python scripts/build_expanded_report.py
# Report generated:8 pages. All page PNGs inspected; key pages also full-size.
```

The final acquisition run transferred3,148,976,183 bytes; this excludes earlier cached attempts. HaGRID uses a custom public license, explicitly not a standard Creative Commons license despite its derived wording. Actual terms and source attribution are cached. Identity/consent and sharing rights are not independently re-established.

## Pending user steps

- Supply personal iPhone photos for the reserved final check; they have never entered training or validation.
- After that check, refresh the laptop package. Existing exports/gesturelab-laptop.zip is the old three-model snapshot.
- Review/update the demo. The existing silent3:49 video records the old version, not the new five-model comparison.
- Physical camera capture is unverified; the earlier real request reported camera unavailable. Upload flow is verified.
- Complete student details, confirm source-sharing/citation choices, provide reviewer links, submit to Moodle and prepare the defense. Deadline:7 November2026 at23:59 Asia/Almaty.

## Resolved issues and limits

Mobile fp16 gradients failed compatibility checks, so Mobile training explicitly used stable GPU float32. A UI-test lint issue was fixed; final lint passes. Final new-model verification was run by the root using QA-authored acceptance code after subagent usage limits ended their turns. Independent QA reviewed data/protocol code and recovered original users before training. The academic report contains only scientific methodology and actual software architecture.


## Presentation cleanup - current version

The current brand is GestureLab in the interface, browser title and API metadata. Repeated badges/taglines were removed; preprocessing, API roundtrip and checkpoint version are in optional Technical details. Prediction, four scores, measured inference time/device, model comparison, training curves, ablation, stress scores and Grad-CAM remain functional. Main page headings now describe their function directly.

The current PDF is a six-page completed-work report. Its authoritative generator is scripts/build_work_report.py; scripts/build_report.py delegates to it. The prior eight-page version is archived at report/archive/report_expansion_2026-11-07.pdf. Sources, actual metrics, training augmentation, licensing/consent limits and absence of T2I/T2V training data remain factual. No exclusively manual-authorship assertion is added.

Checks: frontend typecheck/lint/build passed,22unit tests and10browser tests passed; the real five-model browser acceptance passed12checks. Report visual/content QA passed. Evidence: docs/presentation_cleanup_qa.md, artifacts/verification/presentation_expansion_stdout.txt and presentation_report_qa.json. Model weights, measured JSON/CSV results and the13protectedlegacyartifacthashes remain unchanged.

Final cleanup Python rerun: `.venv/Scripts/python -m pytest backend/tests ml/tests -q` ->50 passed in12.20s,0skipped; artifacts/verification/presentation_pytest.txt.
