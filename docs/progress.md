# Progress
- Source materials located and inspected; architecture and assignment requirements read.
- Contracts fixed in implementation_plan.md. No weights/metrics existed at start.
- Dependency installation completed; pip check: no broken requirements. Python 3.10.9, torch 2.14.1+cpu, timm1.0.30.
- Dataset audit:400 valid 224 RGB photos, fixed per-class70/15/15, zero exact/repeated-ID duplicates, zero screened cross-split dHash candidates. Consent/subject independence unverified.
- Initial combined pytest:18 passed,3 skipped,1 failed (MobileNet embedding width). ML fixed1280 feature head; subsequent ML tests 8 passed.
- Actual four-run training started by ML specialist. Independent QA reviewing integration.
- All four training/evaluations finished (real pretrained downloads succeeded); modern model underperformed reproduced ResNet18, disclosed.
- Full Python final 24 passed0 skips; API smoke3 models6held-out predictions4 runs PASS; independent browser8 checks PASS on actual weights plus isolated missing/denied cases.
- Frontend final typecheck/lint/build pass; unit 9 passed; report generated7 pages and all rendered pages visually reviewed.
- VSCode project opened; local API 8000 / UI 5173 running. Added VSCode tasks and safe setup/start scripts.
- Real3-5 minute local browser recording underway; external sharing not performed.

- Genuine local silent browser video finalized:229.24s (3:49),12,074,244bytes,1280x800, sidecar captions; actual trained API outputs, camera-unavailable fallback; never uploaded.
- Report refreshed with verified video note; still7pages. Remaining author tasks are identity/consent/citation confirmation, reviewer links/submission and physical camera if desired.

2026-11-07: User reports poor live peace/thumbs_up and confirms iPhone + Windows PC/laptop; states source dataset came from HaGRID GitHub. Verified GTX1660Ti6GB with nvidia-smi and official assignment alternative-method clause. Prepared local transfer ZIP builder and docs/mobile_and_transfer.md; phone HTTPS/PWA, expanded-data acquisition and CUDA training are not yet implemented. Existing checkpoints/metrics preserved.
- 2026-11-07: Started authorized HaGRID expansion and staged GPU fine-tuning. Legacy weights/results preserved; future personal-phone photos reserved for final holdout, no laptop transfer yet. Contract in docs/expansion_plan.md.

2026-11-07: Acquisition complete and independently checked:3080manifestrecords (2280train/400newval/400newtest), all400original upstream users resolved/excluded from additions; no retainedcrosssplitperceptual candidates. CUDAarchitecture smoke passed forbatch16:ResNetAMP,MobileFP32 afterobservednonfinitefp16gradients. TrueGPUtraining all2newmodels running artifacts/finetuning.log. Rectangularhandcrop+exactpreparedGradCAMbaseadded;22frontendunit and10browser tests pass,lintfixed. Modelcards updated, new8page report generator prepared for actual metrics only.

2026-11-07 FINAL: realGPUtraining2models completed, selectedepoch9/10 each. Originaltest60/60 and newHaGRIDtest400/400 each. FairfreshholdoutlegacyCNN90%,ResNet93.75%,Mobile91.5%; new2each100%. SourceCSV/checkpointhashes independentlychecked. API5modelsready; defaultResNetbyvalidationF1. Finalpytest50passed0skipped; frontend22unit/10browser+typecheck/lint/buildpassed; QA-authoredactualbrowser12PASS runbyroot. Reportupdated8pagesandrendered/visuallyverified; originalreportarchived. Personalphonecheck pendinguserimages; laptopZIP/demorefreshdeferredperuserorder.
- 2026-11-07: User requested cleaner product text/branding and a practical completed-work report. UI/report edits only; keep factual metrics, sources, ethics and model artifacts. No claim of exclusively manual authorship. Current expanded report archived before revision.

2026-11-07 presentation cleanup complete: brandGestureLab, directpageheadings, optionalTechnicaldetails, conciseAbout/privacy, metrics/curves/ablation/GradCAM retained. Practicalcompleted-work report6pages replaces8pagepaperstyle; oldversion archived. Frontendtypecheck/lint/build,22unit/10browser,real12checks pass. No model/data/metric edits; no all-manualauthorship claim. ReportQA presentation_report_qa.json.

2026-11-07 Word formatting: generated editable report/report.docx in Times New Roman 14 pt, including headings and tables, exported that DOCX through native WPS Writer COM to report/report.pdf (7 pages). All 7 PNG pages inspected. Font and output hashes recorded in artifacts/verification/word_report_qa.json. Packaged LibreOffice renderer unavailable; native export and Poppler rendering succeeded. Previous practical PDF archived. No ML/app changes; those test suites were not rerun for document formatting.

2026-11-07 GitHub release: created private dayosama0/GestureLab and pushed main as component-based commits. Source, tests, numeric results and plots included; datasets, checkpoints, photographs, DOCX/PDF reports and recordings excluded. Rechecked Python 50 passed (13.66s), frontend 22 passed, Playwright 10 passed (10.2s), typecheck/lint/build exit 0, five-model integration PASS. Remote main SHA verified with git ls-remote; laptop transfer and personal phone check remain pending.

Report and repository presentation: added Turaliyev Ali, SE-2430; removed the report header date and redundant repository-import date commentary. Metrics and experiment records unchanged.
