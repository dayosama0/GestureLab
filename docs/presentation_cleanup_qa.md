# Presentation cleanup independent QA

Reviewed 7 November2026. Scope: concise GestureLab UI and practical academic work report, preserving existing scientific results, source credit and auditable artifacts. QA owns this review record; production UI/report changes remain with their assigned owners.

## Functional cleanup review

- Brand now reads GestureLab. Main prediction retains the gesture label, all4 probability values, actual model inference time, compute device and a visible uncalibrated-score caveat.
- API roundtrip, prepared input size and full checkpoint version move into a collapsed Technical details disclosure. These values remain available, and model/data audit artifacts have not been deleted.
- Compare continues to run all5 actual checkpoints on the same ROI. Experiment table, train/validation loss and accuracy curves, confusion matrix, conventional augmentation ablation, separate400-image expanded holdout and programmatic stress tests remain present.
- About retains HaGRID dataset/modern model sources,3200-image count, separate original/new test cohorts, privacy, consent uncertainty and license link. Removing branding or internal implementation noise does not establish that the work was entirely manual; no such authorship claim is present in reviewed UI.
- The measured100% fine-tuned results apply only to the saved original60 and curated HaGRID400 cohorts. Personal-phone evaluation remains pending; this must remain explicit in the report and release guidance.

## Executed acceptance

`node scripts/smoke_expansion.cjs` completed exit0 after frontend cleanup with12PASS and no selector modifications. Evidence: artifacts/verification/presentation_expansion_stdout.txt and current expansion_smoke.json.

It verified13 unchanged legacy checkpoint/metric hashes,5 available real trained models,6 saved runs, actual checkpoint probability/Grad-CAM calls for every model, native Mobile256/.5 preprocessing, matching prepared base/overlay dimensions, full5-model same-image comparison,6-run experiment view, validation-ranked default and360px four-page layout without full-page overflow or uncaught browser errors.

## Pending at this review checkpoint

- Compact prediction/details visual review at desktop and360px after root's API title restart.
- New practical6-page report source/PDF still being authored; final content, numeric provenance, scientific-source/synthetic disclosure and rendered-page checks pending.
- Prior3:49 recording demonstrates the earlier3-model application. It must not be represented as a new5-model recording.

## Final cleanup verification

- After the title-only API restart, an additional actual browser prediction checked compact result/details behavior at1440px and360px. Four probability rows and two main timing fields (model latency/device) remain visible. Technical details are closed by default and open to reveal roundtrip, native256 prepared-input description and full checkpoint version. Long hashes wrap without page overflow. Output-panel-only screenshots exclude the hand image. Evidence: artifacts/verification/presentation_layout.json, presentation-result-desktop.png, presentation-result-mobile.png and presentation-details-mobile.png.
- Root's final plain headings were checked directly: Gesture recognition and About the project. Updated empty Studio desktop/mobile and About mobile screenshots were visually inspected with no obvious overlap or clipping. No private hand photographs appear in these screenshots.
- Final report is exactly6 A4 pages,410175bytes, SHA25655a7b46499ff4fc709281242352170e7ff9dfa63e121798969aa353622c7e2b6. PDF title/footers/metadata now consistently say GestureLab. Earlier residual GestureLabAI branding was raised and corrected by the report owner.
- Full report source/text was reviewed; all six final rendered pages were checked together, with detailed page5/6 visual inspection before the final branding-only update. Layout is readable with aligned tables/figures and no observed clipping. Final contact/render evidence is in tmp/pdfs/work_report/.
- Required scientific material remains: original60 and fresh400 comparison metrics, actual training curves, conventional augmentation ablation and its5x exposure confound, actual selected-model Grad-CAM, same-host CPU timing scope, native per-model preprocessing, real-image HaGRID source/custom license, train-only augmentation and no text-to-image/video additions. Original400 users are identified through official metadata; consent remains independently unverified.
- The100% results remain limited to the saved curated cohorts, with personal-phone performance pending and no state-of-the-art/universal accuracy guarantee. ResNet mixed precision versus Mobile float32 is correctly reported. No claim that the work was entirely manual and no agent-orchestration implementation details were found in the academic report.
- The previous3:49 recording is correctly described as the earlier3-model application, requiring refresh for5models. Successful physical-camera capture, personal-phone holdout, student submission details and external sharing remain separate unresolved author steps.

Final assessment: **PASS for presentation cleanup**. Functional acceptance assertions were preserved; no scientific audit, checkpoint metadata or saved metric artifact was deleted or weakened by this cleanup. Root owns the final combined checks and release status.
