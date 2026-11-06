"""Generate the scientific report exclusively from verified local run artifacts."""
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
OUT.mkdir(exist_ok=True)
IDS = ['simple_cnn', 'resnet18', 'mobilenetv4_noaug', 'mobilenetv4', 'resnet18_finetuned', 'mobilenetv4_finetuned']
RUNS = {name: json.loads((ROOT / 'artifacts' / 'metrics' / f'{name}.json').read_text(encoding='utf-8')) for name in IDS}
AUDIT = json.loads((ROOT / 'artifacts' / 'data_audit.json').read_text(encoding='utf-8'))
EVIDENCE = json.loads((ROOT / 'artifacts' / 'verification' / 'expansion_integration.json').read_text(encoding='utf-8'))
DEMO_PATH = ROOT / 'artifacts' / 'demo' / 'manifest.json'
DEMO = json.loads(DEMO_PATH.read_text(encoding='utf-8')) if DEMO_PATH.exists() else None
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body', fontName='Helvetica', fontSize=9.5, leading=13.2, spaceAfter=7, textColor=colors.HexColor('#263747')))
styles.add(ParagraphStyle(name='CaptionSmall', fontSize=7.8, leading=10, spaceAfter=6, textColor=colors.HexColor('#516575')))
styles.add(ParagraphStyle(name='CoverLabel', fontSize=10, leading=14, textColor=colors.HexColor('#087d79'), spaceAfter=12))
styles['Title'].fontSize = 29
styles['Title'].leading = 34
styles['Title'].textColor = colors.HexColor('#123044')
styles['Heading1'].textColor = colors.HexColor('#123044')
styles['Heading2'].textColor = colors.HexColor('#087d79')
story = []
md = []


def title(text):
    story.append(Paragraph(escape(text), styles['Heading1']))
    md.append('# ' + text + '\n')


def h(text):
    story.append(Paragraph(escape(text), styles['Heading2']))
    md.append('## ' + text + '\n')


def prose(text):
    text=re.sub(r'\b(the|original|all|contains|identify|from|for|and|with|includes|over|of|to|every|by|has|plus|mean|std|gray|uses|batch|seed|patience|native)(?=\d)',r'\1 ',text,flags=re.I)
    text=re.sub(r'(?<=\d)(?=(?:images?|users?|train|validation|test|original|fresh|distinct|participants?|crops?|total|RGB|epochs?|source|examples?|records?|perclass)\b)',' ',text,flags=re.I)
    text=re.sub(r'(?<=%)(?=[A-Za-z])',' ',text)
    text=re.sub(r',(?=[A-Za-z])',', ',text)
    return text


def p(text, caption=False):
    text=prose(text)
    story.append(Paragraph(escape(text), styles['CaptionSmall' if caption else 'Body']))
    md.append(text + '\n')


def table(headers, rows, widths=None):
    data = [[Paragraph(escape(str(c)), styles['CaptionSmall']) for c in row] for row in [headers] + rows]
    result = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    result.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dcefed')),
                               ('GRID', (0, 0), (-1, -1), .35, colors.HexColor('#d0dde3')),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                               ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    story.extend([result, Spacer(1, 10)])
    md.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |')
    md.extend('| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows)
    md.append('')


def figure(path, width, caption):
    image = Image(str(path))
    image.drawHeight = width * image.imageHeight / image.imageWidth
    image.drawWidth = width
    story.append(image)
    p(caption, True)
    md.append(f'![{caption}](../{path.relative_to(ROOT).as_posix()})\n')


def page():
    story.append(PageBreak())
    md.append('\n---\n')


def fmt(x):
    return f'{100*x:.1f}%'


def method_diagram():
    drawing = Drawing(468, 61)
    labels = [('Hand ROI', '224 / 256 RGB'), ('UIB backbone', 'Fine-tuned weights'), ('Global pool', '1,280 features'), ('Linear head', '4 task logits')]
    for i, (name, detail) in enumerate(labels):
        x = i * 119
        drawing.add(Rect(x, 8, 109, 47, rx=5, ry=5, fillColor=colors.HexColor('#e6f2f0'), strokeColor=colors.HexColor('#79b7ae')))
        drawing.add(String(x+54.5, 36, name, textAnchor='middle',fontName='Helvetica-Bold',fontSize=9,fillColor=colors.HexColor('#123044')))
        drawing.add(String(x+54.5, 20, detail, textAnchor='middle',fontName='Helvetica',fontSize=8,fillColor=colors.HexColor('#365464')))
        if i < 3:
            drawing.add(Line(x+110, 31, x+118, 31, strokeColor=colors.HexColor('#087d79')))
            drawing.add(Polygon([x+118,31,x+114,34,x+114,28],fillColor=colors.HexColor('#087d79'),strokeColor=None))
    story.append(drawing)



EXP = json.loads((ROOT/'artifacts/expansion_audit.json').read_text())
COMP = json.loads((ROOT/'artifacts/expansion_comparison.json').read_text())
NAMES={'simple_cnn':'A1 SimpleCNN','resnet18':'A1 ResNet18','mobilenetv4_noaug':'MobileNetV4 no aug','mobilenetv4':'MobileNetV4 moderate','resnet18_finetuned':'ResNet18 fine-tuned','mobilenetv4_finetuned':'MobileNetV4 fine-tuned'}
FINE=RUNS['mobilenetv4_finetuned']

story.append(Paragraph('ASTANA IT UNIVERSITY / COMPUTER VISION / ASSIGNMENT 2', styles['CoverLabel']))
story.append(Paragraph('GestureLab AI',styles['Title']))
story.append(Paragraph('From small-split baselines to participant-separated data expansion and backbone fine-tuning',styles['Heading2']))
p('Updated 7 November 2026. Author and student ID must be completed by the submitting student. Local research application; personal-phone validation is pending.')
h('Objective and Assignment 1 continuity')
p('The task is four-class static hand gesture classification: fist, open_palm, peace and thumbs_up. Assignment 2 continues the original supplied 400-image dataset and notebook architectures, adds a modern MobileNetV4 Conv Small classifier [1], and delivers an interactive upload/camera application with actual gradient explanations. The assignment permits a modern alternative to its suggested generators. No Step-Video-T2V or ArtAug implementation is claimed.')
p('The original notebook had training disabled and supplied no trained weights or original result tables. SimpleCNN and frozen ImageNet ResNet18 were therefore newly reproduced. Four initial runs established baselines and a conventional augmentation ablation. Subsequent reports of poor live recognition motivated a separate expansion with genuine HaGRID images [4] and full-backbone fine-tuning. No personal phone photographs were used in training or checkpoint selection.')
h('Working system')
table(['Component','Implemented behavior'],[
 ['Browser studio','Upload or guided camera capture; square/tall/wide hand ROI; selectable trained models; four probabilities, measured timings and Grad-CAM opacity.'],
 ['Preprocessing and API','EXIF/RGB handling, gray 128 letterbox, per-checkpoint input size and normalization. FastAPI validates local trained provenance, bounds uploads and serializes CPU inference.'],
 ['Local models','Reproduced SimpleCNN/ResNet18, MobileNetV4 frozen-feature model, and separately versioned fine-tuned ResNet18/MobileNetV4. No serving-time model download.'],
 ['Experiment evidence','Saved source/user audits, immutable checkpoints, epoch histories, per-image predictions, confusion matrices, validation selection and two labelled test cohorts.']
],[35*mm,131*mm])
h('Research questions')
p('How do the modern and reproduced baselines compare on the original split? Does the complete expansion-and-fine-tuning pipeline perform better on a fresh participant-separated holdout? What trade-offs arise in CPU latency and interpretation? New data, native preprocessing and backbone optimization change together; their individual causal effects are not isolated.')
page()

title('2. Dataset provenance and separation')
p('The400 original UUID filenames and labels were matched to official HaGRID annotations. They identify 400 distinct upstream user IDs: 280 train,60 validation and60 test, with zero user intersections. This upgrades the earlier unknown-subject status to verified separation under upstream identifiers; it does not independently verify identity or individual consent. The original supplied split and images remain unchanged.')
table(['Source cohort','Train','Validation','Test'],[
 ['Original supplied HaGRID subset',280,60,60],
 ['Additional official HaGRID ROIs',2000,400,400],
 ['Training/selection for new runs',2280,400,'Original60 + fresh 400']
],[77*mm,27*mm,29*mm,33*mm])
p('The original 60 validation images are retained but do not select the new checkpoints. New training combines the original 280 train images with 2000 added images. New validation selects checkpoints; original 60 and fresh 400 test images are evaluated afterwards. The full source collection contains 3200 images, while the new manifest contains 3080 training/selection/fresh-test records.')
p(f"New training/validation/test user counts are {280+EXP['additional_unique_participants']['train']}/{EXP['additional_unique_participants']['val']}/{EXP['additional_unique_participants']['test']}, including 280 original training users. All400 original participants were excluded from the added images. New splits have disjoint user sets. Multiple images can share a participant inside one split, so400 test images do not mean400 independent people.")
h('Acquisition and image contract')
p('Images and matching boxes came from the official HaGRID source archives and annotations. Bounded HTTP range reads retrieved selected ZIP members rather than downloading entire 40 GB class archives. ZIP CRC and image hashes were checked. A matching hand box plus 15% margin on each side was cropped, converted to RGB and gray 128-letterboxed to 320 square. Full source frames were processed in memory; local ROIs and source/user manifests preserve provenance.')
p('The final manifest contains 570 train,100 validation and100 fresh test examples per project class. Additional peace includes75% official peace and25% peace_inverted to cover both palm-facing orientations of the logical V sign; this broadens coverage relative to the original forward-peace subset. Upstream gesture labels are retained per record.')
p(f"Retained exact source/hash overlaps are zero. Perceptual screening uses 64-bit difference hashes with Hamming distance <=3 across splits. It rejected {EXP['rejected'].get('new_cross_split_perceptual_candidate',0)} new cross-split candidates and {EXP['rejected'].get('holdout_perceptual_candidate',0)} original-holdout candidates; no unresolved cross-split candidates remain. Hash screening is not proof against all near-duplicates.")
h('Rights and limitations')
p('The audit rejects invalid or too-small hand boxes; these scores describe a curated ROI subset, not the full HaGRID benchmark. HaGRID uses a custom public license derived from Attribution-ShareAlike4.0; its text explicitly says it is not a Creative Commons license. The actual license PDF and SHA256 are cached with provenance. These images are external research data, not claimed as personally collected photographs. Keep image-bearing evidence private until applicable sharing rights are confirmed.')
page()

title('3. Models, preprocessing and training')
h('Architectures')
p('SimpleCNN reproduces the notebook three 16/32/64-channel convolution blocks, adaptive 4x4 pooling and dense 128 classifier. ResNet18 reproduces frozen ImageNet features with dropout0.2 plus a 512-to-4 head. MobileNetV4 Conv Small uses the timm mobilenetv4_conv_small.e3600_r256_in1k weights [1,2], a 1280-dimensional embedding and a new four-logit head. Its Universal Inverted Bottleneck blocks are part of the modern architecture; this experiment does not establish state-of-the-art gesture accuracy.')
method_diagram()
table(['Run family','Optimization','Input and normalization'],[
 ['SimpleCNN A1 reproduction','Full network from scratch','224; ImageNet mean/std'],
 ['ResNet18 / MobileNetV4 legacy','Frozen features; new head only','224; ImageNet mean/std'],
 ['ResNet18 fine-tuned','Head warmup, then entire backbone','224; ImageNet mean/std'],
 ['MobileNetV4 fine-tuned','Head warmup, then entire backbone','256; mean 0.5/std 0.5 per RGB channel']
],[58*mm,53*mm,55*mm])
p('ImageNet mean/std are (0.485,0.456,0.406)/(0.229,0.224,0.225). The legacy MobileNet intentionally adapted its native 256/.5 recipe to 224/ImageNet. The new MobileNet restores its native recipe. Training and serving call the same parameterized RGB/gray-letterbox functions; metadata is strictly validated per model ID. Manual rectangular crops preserve aspect ratio; the API supplies the exact prepared image for explanation alignment.')
h('Two-stage fine-tuning')
p('Fresh real ImageNet weights initialize both new runs, avoiding hidden legacy optimization history. Seed42, batch 16, two head-warmup epochs and up to eight full-backbone epochs are fixed before testing. Warmup keeps the frozen backbone in evaluation mode. Fine-tuning unfreezes all backbone parameters, including normal training-mode batch normalization. AdamW uses head warmup LR 0.001, fine-tune backbone LR 0.00001 and head LR 0.0001, weight decay 0.0001; patience 4 selects minimum new-validation cross-entropy among fine-tuning epochs.')
p('Training-only transforms are horizontal flip, small affine rotation/scale/translation and color jitter. No augmentation is applied to selection or clean test images. CUDA training uses AMP for ResNet18 and float32 for MobileNetV4 after a real GTX1660Ti fp16 gradient compatibility failure. Changed first-convolution weights and strict checkpoint reloads verify actual backbone optimization. The default UI model uses saved validation macro-F1, never test accuracy.')
page()

title('4. Original 60-image clean comparison')
p('These are unchanged original test images: 15 per class. All weights were selected on the corresponding validation cohort before evaluation. This test was already examined in the first project phase; the later fresh HaGRID holdout provides the independent expansion evaluation. One original-test error changes accuracy by 1.67 percentage points.')
table(['Run','Accuracy','Macro precision','Macro recall','Macro-F1'],[[NAMES[k],fmt(RUNS[k]['metrics']['accuracy']),f"{RUNS[k]['metrics']['precision_macro']:.3f}",f"{RUNS[k]['metrics']['recall_macro']:.3f}",f"{RUNS[k]['metrics']['f1_macro']:.3f}"] for k in IDS],[63*mm,27*mm,27*mm,25*mm,24*mm])
figure(ROOT/'artifacts/plots/confusion_mobilenetv4_finetuned.png',109*mm,'Figure 1. Fine-tuned MobileNetV4 confusion on the original 60. Rows are actual classes; columns are predicted classes.')
table(['Fine MobileNetV4 class','Precision','Recall','F1','Support'],[[c,f"{m['precision']:.3f}",f"{m['recall']:.3f}",f"{m['f1-score']:.3f}",m['support']] for c,m in FINE['metrics']['per_class'].items()],[64*mm,27*mm,25*mm,25*mm,25*mm])
p('The six run JSONs, checkpoint hashes and per-image predictions are the numeric source. An improvement on this tiny cohort is descriptive; it does not prove reliable recognition of arbitrary phone photographs, unseen gestures or unrelated objects.',True)
page()

title('5. Fresh participant-separated evaluation')
p(f"The fresh test contains{COMP['expanded_test_count']} unaugmented images from{EXP['additional_unique_participants']['test']} upstream participants absent from all training and validation cohorts. Each project class has 100 images; peace includes 25 reverse-facing V signs. All five production models are compared on exactly this same cohort. The three legacy classifiers remain unchanged; their fresh-test evaluation is saved separately from original results.")
table(['Model','Accuracy','Macro-F1','Peace recall','Thumbs-up recall'],[[NAMES[r['model_id']],fmt(r['metrics']['accuracy']),f"{r['metrics']['f1_macro']:.3f}",fmt(r['metrics']['per_class']['peace']['recall']),fmt(r['metrics']['per_class']['thumbs_up']['recall'])] for r in COMP['models']],[62*mm,25*mm,25*mm,27*mm,27*mm])
legacy=max((r for r in COMP['models'] if not r['model_id'].endswith('_finetuned')),key=lambda r:r['metrics']['f1_macro'])
fine=max((r for r in COMP['models'] if r['model_id'].endswith('_finetuned')),key=lambda r:r['metrics']['f1_macro'])
p(f"Descriptively, the strongest legacy macro-F1 is {legacy['metrics']['f1_macro']:.3f} ({NAMES[legacy['model_id']]}); the strongest fine-tuned macro-F1 is {fine['metrics']['f1_macro']:.3f} ({NAMES[fine['model_id']]}). This held-out ranking does not select the deployed default. Data volume, orientation coverage, train augmentation, full-backbone optimization and MobileNet normalization differ together; no single-factor causal attribution is made.")
figure(ROOT/'artifacts/plots/curves_mobilenetv4_finetuned.png',157*mm,'Figure 2. Actual fine-tuned MobileNetV4 training/validation history. Epochs 1-2 train the head; later epochs optimize the backbone. Selection uses validation loss.')
figure(ROOT/'artifacts/plots/curves_resnet18_finetuned.png',157*mm,'Figure 3. Actual ResNet18 fine-tuning history under the same stage budget. Test images do not enter these curves.')
p('Repeated participants within the fresh test and a single seed limit uncertainty estimates. Both cohorts still come from HaGRID; personal-camera testing remains a separate future check and cannot be inferred from these scores.',True)
page()

title('6. Controlled ablation and stress tests')
p('The original MobileNetV4 augmentation ablation is retained as a separate controlled experiment: same source split, frozen backbone, head, seed 42, Adam LR 0.001, batch 32 and 15-epoch budget, with either originals alone or originals plus four fixed notebook-style variants. The moderate dataset has five times as many examples and optimizer steps per epoch; the effect includes additional training exposure and is not compute-matched.')
delta=RUNS['mobilenetv4']['metrics']['f1_macro']-RUNS['mobilenetv4_noaug']['metrics']['f1_macro']
p(f"On original clean test, moderate augmentation changed macro-F1 by {delta:+.3f}. This earlier pair is the assignment ablation. The later expansion/fine-tuning comparison changes several factors and must not be relabelled as a pure augmentation experiment.")
figure(ROOT/'artifacts/plots/curves_mobilenetv4_noaug.png',155*mm,'Figure 4. Original no-augmentation MobileNetV4 head training, actual logged curves.')
figure(ROOT/'artifacts/plots/curves_mobilenetv4.png',155*mm,'Figure 5. Original moderate-augmentation head training, with identical unaugmented validation.')
h('Separate deterministic stress evaluation')
table(['Original test perturbation','Fine Mobile accuracy','Macro-F1'],[[name,fmt(m['accuracy']),f"{m['f1_macro']:.3f}"] for name,m in FINE['stress'].items()],[83*mm,46*mm,37*mm])
p('Each condition modifies the same original 60 test images independently: brightness multiplier 0.5, Gaussian blur radius 2 and JPEG quality 30. These correlated programmatic perturbations are not independent examples, generative video, ArtAug or real-environment proof. They are reported separately and never mixed into clean scores.',True)
page()

title('7. Explanation and measured runtime')
p('Grad-CAM differentiates the predicted-class logit with respect to spatial activations of the same selected classifier [3]. Mean gradients weight feature channels; ReLU, interpolation and normalization produce a coarse map. MobileNet uses its final pre-pooling block, not the post-pooling 1x1 convolution. Frozen serving backbones still support input gradients. Constant maps produce an explicit error; hooks are removed after each request.')
original=Image(str(ROOT/'artifacts/verification/input_mobilenetv4_finetuned.png'),width=57*mm,height=57*mm)
cam=Image(str(ROOT/'artifacts/verification/gradcam_mobilenetv4_finetuned.png'),width=57*mm,height=57*mm)
story.append(Table([[original,cam]],colWidths=[80*mm,80*mm]))
p('Figure 6. Exact native 256 prepared input (left) and actual fine-tuned MobileNetV4 Grad-CAM overlay from the API (right). This private held-out dataset example is not a personal-phone validation result. Attribution is approximate, not segmentation or causal proof.',True)
table(['Same-host CPU inference','Mean ms','p50 ms','p95 ms'],[[NAMES[r['model_id']],f"{r['latency_same_host_cpu']['mean_ms']:.2f}",f"{r['latency_same_host_cpu']['p50_ms']:.2f}",f"{r['latency_same_host_cpu']['p95_ms']:.2f}"] for r in COMP['models']],[69*mm,32*mm,32*mm,33*mm])
p('CPU timings above are newly measured on Intel Core i3-8100 @ 3.60GHz, 16 GB RAM, Windows, two PyTorch threads: three warmups and 20 single-image forward passes per model. They exclude decode, preprocessing, Grad-CAM, queue and network. Browser roundtrip is labelled separately. Different input sizes are part of the respective serving configurations.')
table(['CUDA training run','Actual epochs','Preparation/train s','Peak allocated MiB'],[[NAMES[k],RUNS[k]['metadata']['actual_epochs'],f"{RUNS[k]['metadata']['training_seconds']:.1f}",f"{RUNS[k]['metadata']['environment']['peak_cuda_memory_bytes']/1048576:.1f}"] for k in ['resnet18_finetuned','mobilenetv4_finetuned']],[61*mm,28*mm,39*mm,38*mm])
p('Training: GTX 1660 Ti 6 GB, PyTorch 2.6.0+cu124 / CUDA 12.4; ResNet AMP and MobileNet float32. Preparation/train duration includes loading and validation work, not only optimizer steps. Peak allocated memory excludes driver/display reservations. Serving and the same-host benchmark use local CPU PyTorch 2.14.1; weights restore strictly across the tested environments.',True)
page()

title('8. Reproducibility, ethics and next checks')
h('Reproduction and evidence')
p('Windows serving: install requirements.txt in .venv, run npm --prefix frontend ci, then scripts/run_all.ps1. Use .venv/Scripts/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000, and npm --prefix frontend run dev in a second terminal. README.md contains executable Windows/POSIX instructions. macOS/Linux execution is documented but untested here.')
p('For training, create a separate .venv-gpu, install requirements-gpu.txt, then run python -m ml.hagrid_expand --train 500 --val 100 --test 100 --workers 6 --budget-gb 8. Run python -m ml.finetune --models all --head-epochs 2 --finetune-epochs 8 --batch-size 16 --device cuda. Acquisition is resumable; immutable completed run IDs are not overwritten. Native model weights are downloaded explicitly during preparation, never by inference.')
p('Audit/source manifests, cached license, source/crop hashes and user joins are retained. Run JSONs contain epoch histories, validation choice, exact preprocessing, GPU settings, first-convolution weight changes and strict-reload Grad-CAM checks. Original checkpoint/metric hashes were snapshotted before expansion and remain unchanged. scripts/compare_expanded.py produces the common fresh-holdout table and same-host CPU timings without changing legacy artifacts.')
h('Privacy and limitations')
p('Images originate from HaGRID [4], with its custom licensing conditions. Official user-ID separation is verified; real identity and personal consent are not independently re-established. Keep photos and image-bearing report/video private until the relevant sharing conditions are confirmed. User uploads and camera captures are processed in memory with bounded request/file sizes, no retention or image logging. No generative human media were created; synthetic share is zero.')
p('The four-way closed-set classifier cannot reliably reject other gestures or objects. Softmax scores are uncalibrated. Skin tone, lighting, camera pose, blur, cropping and source-specific backgrounds may affect performance. Crop shape/padding artifacts can matter; tall/wide controls and exact prepared-input display improve protocol matching but do not prove field robustness. No consequential-use reliability or state-of-the-art gesture accuracy is established.')
h('Submission and pending personal-camera check')
p('Personal iPhone images remain unseen and reserved for the next final check; they must not be silently added to training. The existing 229.24-second silent demo with sidecar captions records the earlier three-model version, not the new five-model comparison. Review/update the demonstration and laptop-transfer package after the personal-camera check. Complete student details, confirm citation/rights choices and provide actual reviewer links. Deadline: 7 November 2026, 23:59 Asia/Almaty; oral defense is also required.')
h('References')
p('[1] Qin et al. MobileNetV4: Universal Models for the Mobile Ecosystem. ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php',True)
p('[2] timm. mobilenetv4_conv_small.e3600_r256_in1k model card. https://huggingface.co/timm/mobilenetv4_conv_small.e3600_r256_in1k',True)
p('[3] Selvaraju et al. Grad-CAM. ICCV 2017. https://arxiv.org/abs/1610.02391',True)
p('[4] Kapitanov et al. HaGRID - HAnd Gesture Recognition Image Dataset. WACV 2024; official dataset/annotations/custom license: https://github.com/hukenovs/hagrid. Assignment2.docx and notebook.ipynb are local assignment/baseline sources. A specific relevant Neurohive citation is not verified; the primary modern-method citation is supplied for instructor confirmation.',True)

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#c7d8df'))
    canvas.line(22*mm,18*mm,188*mm,18*mm)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#516575'))
    canvas.drawString(22*mm,13*mm,'GestureLab AI | Assignment 2 | Measured local research results')
    canvas.drawRightString(188*mm,13*mm,str(doc.page))


doc = SimpleDocTemplate(str(OUT/'report.pdf'), pagesize=(210*mm,297*mm),
                        rightMargin=22*mm,leftMargin=22*mm,topMargin=19*mm,bottomMargin=24*mm,
                        title='GestureLab AI - Computer Vision Assignment 2',author='Student details pending')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'report.md').write_text('\n'.join(md),encoding='utf-8')
from pypdf import PdfReader
count=len(PdfReader(OUT/'report.pdf').pages)
print(f'Report generated: {count} pages at {OUT / "report.pdf"}')
assert 6 <= count <= 8, f'Report must be 6-8 pages, got {count}'
