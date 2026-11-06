# Expansion / GPU fine-tuning independent QA

Reviewed 2026-11-07 against docs/expansion_plan.md and project scientific-integrity rules. This is an incremental evidence record; code readiness is distinct from completed data acquisition, real trained checkpoints and measured results.

## Review scope and current gates

| Gate | Current evidence / status |
|---|---|
| Legacy preservation | Four physical checkpoint SHA256 values match their original per-run metric JSON. Root also saved13-file pre-expansion checkpoint/metric snapshot at artifacts/verification/legacy_before_expansion.json. Final snapshot comparison pending completion. |
| New model identity | Separate resnet18_finetuned and mobilenetv4_finetuned IDs. Models restore same underlying architectures without serving-time download. New artifacts use exclusive creation and refuse pre-existing IDs. |
| True fine-tuning | ml/finetune.py explicitly disables backbone gradients for head warmup, runs warmup backbone.eval to preserve BN statistics, then unfreezes all backbone parameters and restores training mode for actual gradient optimization. Selected first-convolution L2 change must exceed0. |
| Validation selection | Only finetune-stage checkpoints compete on new-validation cross-entropy. Final original60 and new expansion test evaluated after selection and strict reload. Original60 was previously examined; code discloses that the new subject-disjoint expansion test is the fresh independent evaluation. Personal iPhone images remain reserved for a future final holdout. |
| Preprocessing | Legacy default224/ImageNet preserved. New ResNet224/ImageNet and Mobile256/mean=.5/std=.5 are approved per-ID policies in backend registry and shared dynamic preprocessing. Backend uses same prepared image for Grad-CAM. |
| Timing | New train environment records actual CUDA/GPU/AMP/memory when available; final benchmark runs on CPU strict reload, separate from GPU training. Do not compare GPU and CPU timing without scope labels. |
| Expansion source | Data owner implemented official HaGRID class ZIP bounded HTTP206 reads plus official v2 annotations. ZIP member CRC verified; matching gesture bbox with15%margin; source frames in memory; saved320RGB gray-letterboxed crops with source/user IDs and hashes. Actual final data audit pending. |
| Actual expansion inference | Pending real new checkpoints. No new metric or trained-weight claim is made at this review checkpoint. |

## Findings sent to owners

1. **Participant recovery qualification:** Initial validator could permit unknown original-training users while unconditionally claiming participant-disjoint data. ML owner revised summary to distinguish known-user sets, unknown original IDs and full-independence evidence; data owner reported400/400 source UUID recovery and400 original users, pending persisted audit verification. Requested original train/val/test user intersections to replace prior uncertainty with actual metadata evidence.
2. **Perceptual screening:** New cross-split dHash candidates initially remained flagged without review gating. Requested removal/manual adjudication or explicit unresolved status before clean results; ML loader now rejects metadata with perceptual_overlap_review_complete=False. Actual candidate/rejection counts pending final audit.
3. **Resume sampling:** Initial resume initialized used_users empty despite cached records and could select multiple images for a previously selected user/class. Data owner notified to seed used_users from cached selected rows. No production-file edits performed by QA.
4. **Browser preprocessing statements:** Frontend initially hardcoded224/ImageNet even for Mobile fine-tuned256/.5 and resized every submitted crop224. Frontend owner now implements native ROI up to1536px, opaque gray transparency, dynamic metadata input/norm/heatmap descriptions. Independent browser acceptance pending checkpoints.
5. **Independent test visibility:** Original API experiment schema discarded new top-level original_test/expanded_test fields. Backend now explicitly exposes optional labelled holdouts; frontend owner renders independent expansion results separately when saved.

## QA-owned acceptance script

scripts/smoke_expansion.cjs uses only real artifacts/API results and requires5 trained models/6 experiment runs. It verifies unchanged13-file legacy snapshot, new checkpoint hashes, actual backbone-change evidence, selected-validation metadata, original60 versus expansion test counts/confusion totals, CPU timing labels, all5 model probability/Grad-CAM requests, dynamic224/256 dimensions and native Mobile normalization, browser upload/opacity,5-model same-image comparison,6-run experiment view and360px navigation/no page overflow. It saves no uploaded-image screenshots.

Executed so far: `node --check scripts/smoke_expansion.cjs` exit0. Full integration is deliberately unrun until actual new weights and acquisition audit are ready. It must not be counted as a passing expanded-model test yet.
## Independent upstream participant join (executed)

QA read cached official annotation JSON files and independently joined the400 original image UUIDs, without relying on the acquisition audit's claim. Result:400/400 recovered;400 distinct upstream user IDs. Original split user counts train280 / validation60 / test60. User intersections train-validation0, train-test0, validation-test0. Recovery used the official fist, like, palm and peace annotations across source train/validation/test. This establishes separation under upstream user identifiers, not independently verified human identity or consent.

Data owner subsequently reported fixes for resume user tracking, pre-save rejection of cross-split perceptual candidates, validation/test perceptual screening against all original samples, and explicit original user-overlap audit. ML owner added a hard training gate requiring ready:true and verified full subject separation, plus checks against copied original holdout UUIDs. Final expanded records still require independent verification after acquisition completes.
## Source-license inspection

QA extracted text from the cached actual data/hagrid_cache/HaGRID_license_en_us.pdf. Its footnote explicitly states that this is not a Creative Commons license, while describing the text as a reworking of Attribution-ShareAlike4.0. Data owner was asked to label it as HaGRID's custom public license with attribution/conditions, derived fromCCBY-SA4.0, and retain the actual license artifact. Publicity/privacy rights are addressed separately; upstream user-ID separation does not establish consent for public sharing.

An actual downloaded private fist crop was visually inspected in place: the crop contains the annotated fist and gray square padding. It is an upscaled small hand box, so source-detail limitations remain real. No image-bearing screenshot or report figure was exported during this inspection.

## Final root-executed acceptance (7 November2026)

The original QA agent reviewed the protocol and independently recovered source users before training. After agent usage limits, root executed the QA-authored final acceptance script. Do not attribute the following final executions to an independent agent.

- Data ready:2280train/400val/400fresh-test, all400originalusers recovered/excluded, disjoint user sets, no retained cross-split perceptual candidates. Root checked3080unique records/IDs/hashes and original13artifactSHA256s.
- Real CUDA training finished both runs; each10epochs with selectedepoch9. True firstconv changes are nonzero. MobileFP16 compatibility gradients failed, so explicitGPUFP32 was used; ResNetAMP succeeded. Both trained models scored60/60 original and400/400 fresh; raw predictionCSV counts agree.
- Common fresh holdout legacy accuracy:SimpleCNN90%,ResNet18 93.75%,MobileNetV4 91.5%; bothfine models100%. These curatedHaGRID results are not personal-camera guarantees.
- FullPython:50passed14.13s,0skipped. GPUenvironment MLtests24passed7.93s. Frontend22unit and10UIbrowser tests; typecheck/lint/buildpassed.
- Actual APIintegration:5models,10predictions,6runs; correctly sized224/256 prepared inputs and genuineCAM. Native rectangular crop controls now preserve ROIaspect and exactpreparedinputbase avoids graypadding misalignment.
- `node scripts/smoke_expansion.cjs`:12PASS including all5realmodels,6runs, validation-ranked default, native256base/overlay,5modelCompare and360pxnavigation. Legacy13hashes unchanged. Evidence:artifacts/verification/expansion_smoke.json and expansion_browser_stdout.txt.
- Report8pages, allrenderedpages visually inspected; actualsourcecurves/confusion/CAM and original/newtest cohorts clearly labelled. See expansion_report_qa.json. Personalphonephotos are reserved for nextstep; olddemo/ZIP remain priorsnapshots pendingrefresh afterthatcheck.
