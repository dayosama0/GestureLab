"""Build the practical Assignment 2 work report from saved completed runs."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
FIGURES = OUT / "figures"
RUN_IDS = ["simple_cnn", "resnet18", "mobilenetv4_noaug", "mobilenetv4",
           "resnet18_finetuned", "mobilenetv4_finetuned"]
NAMES = {"simple_cnn": "A1 SimpleCNN", "resnet18": "A1 ResNet18",
         "mobilenetv4_noaug": "MobileNetV4, no aug.", "mobilenetv4": "MobileNetV4, moderate",
         "resnet18_finetuned": "ResNet18, fine-tuned",
         "mobilenetv4_finetuned": "MobileNetV4, fine-tuned"}


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


RUNS = {key: read(f"artifacts/metrics/{key}.json") for key in RUN_IDS}
EXP = read("artifacts/expansion_audit.json")
COMP = read("artifacts/expansion_comparison.json")
JOIN = read("data/expanded/original_subject_join.json")
INTEGRATION = read("artifacts/verification/expansion_integration.json")
assert EXP["ready"] and COMP["expanded_test_count"] == 400
assert JOIN["matched"] == 400 and JOIN["unique_users"] == 400
assert COMP["legacy_artifacts_unchanged"] is True
assert INTEGRATION["legacy_artifacts_unchanged"] is True
FRESH = {row["model_id"]: row for row in COMP["models"]}
for key in RUN_IDS:
    assert RUNS[key]["metadata"]["trained"] is True
for key in ("resnet18_finetuned", "mobilenetv4_finetuned"):
    assert RUNS[key]["metadata"]["train_count"] == 2280
    assert RUNS[key]["metadata"]["validation_count"] == 400
    assert RUNS[key]["metadata"]["expanded_test_count"] == 400

OUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="WorkBody", fontName="Helvetica", fontSize=10,
                         leading=14.4, spaceAfter=8, textColor=colors.HexColor("#263747")))
styles.add(ParagraphStyle(name="WorkSmall", fontName="Helvetica", fontSize=8.1,
                         leading=11, spaceAfter=6, textColor=colors.HexColor("#516575")))
styles.add(ParagraphStyle(name="WorkCell", fontName="Helvetica", fontSize=8.3,
                         leading=11.2, textColor=colors.HexColor("#263747")))
styles.add(ParagraphStyle(name="WorkLabel", fontName="Helvetica-Bold", fontSize=9,
                         leading=13, spaceAfter=13, textColor=colors.HexColor("#087d79")))
styles["Title"].fontSize = 29
styles["Title"].leading = 34
styles["Title"].textColor = colors.HexColor("#123044")
styles["Heading1"].fontSize = 18
styles["Heading1"].leading = 23
styles["Heading1"].spaceAfter = 13
styles["Heading1"].textColor = colors.HexColor("#123044")
styles["Heading2"].fontSize = 12
styles["Heading2"].leading = 16
styles["Heading2"].spaceBefore = 9
styles["Heading2"].spaceAfter = 6
styles["Heading2"].textColor = colors.HexColor("#087d79")
story, markdown = [], []


def p(text, small=False):
    story.append(Paragraph(escape(text), styles["WorkSmall" if small else "WorkBody"]))
    markdown.append(text + "\n")


def heading(text, level=1):
    story.append(Paragraph(escape(text), styles[f"Heading{level}"]))
    markdown.append("#" * level + " " + text + "\n")


def table(headers, rows, widths):
    cells = [[Paragraph(escape(str(value)), styles["WorkCell"]) for value in row]
             for row in [headers] + rows]
    element = Table(cells, colWidths=[x * mm for x in widths], repeatRows=1, hAlign="LEFT")
    element.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcefed")),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#ccdae0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([element, Spacer(1, 9)])
    markdown.extend(["| " + " | ".join(headers) + " |",
                     "| " + " | ".join(["---"] * len(headers)) + " |"])
    markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    markdown.append("")


def page():
    story.append(PageBreak())
    markdown.append("\n---\n")


def figure(path, width_mm, caption):
    image = Image(str(path))
    image.drawHeight = width_mm * mm * image.imageHeight / image.imageWidth
    image.drawWidth = width_mm * mm
    story.extend([image, Spacer(1, 4), Paragraph(escape(caption), styles["WorkSmall"])])
    markdown.append(f"![{caption}](../{path.relative_to(ROOT).as_posix()})\n")


def flow(labels, details):
    drawing = Drawing(470, 65)
    for i, (label, detail) in enumerate(zip(labels, details)):
        x = i * 119
        drawing.add(Rect(x, 8, 109, 49, rx=5, ry=5, fillColor=colors.HexColor("#e6f2f0"),
                         strokeColor=colors.HexColor("#79b7ae")))
        drawing.add(String(x+54.5, 37, label, textAnchor="middle", fontName="Helvetica-Bold",
                           fontSize=9, fillColor=colors.HexColor("#123044")))
        drawing.add(String(x+54.5, 21, detail, textAnchor="middle", fontName="Helvetica",
                           fontSize=8, fillColor=colors.HexColor("#365464")))
        if i < 3:
            drawing.add(Line(x+110, 33, x+118, 33, strokeColor=colors.HexColor("#087d79")))
            drawing.add(Polygon([x+118,33,x+114,36,x+114,30],
                                fillColor=colors.HexColor("#087d79"), strokeColor=None))
    story.append(drawing)
    markdown.append(" -> ".join(f"{a} ({b})" for a,b in zip(labels,details)) + "\n")


def percent(value):
    return f"{100*value:.2f}%"


def make_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for key, color, label in [("mobilenetv4_noaug", "#8980b8", "No augmentation"),
                              ("mobilenetv4", "#087d79", "Moderate augmentation")]:
        history = RUNS[key]["history"]
        axes[0].plot([x["epoch"] for x in history], [x["val_accuracy"] for x in history],
                     label=label, color=color, linewidth=2)
    axes[0].set(title="Initial MobileNetV4: validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    history = RUNS["mobilenetv4_finetuned"]["history"]
    for field, color, label in [("train_accuracy", "#087d79", "Train"),
                                 ("val_accuracy", "#8980b8", "Validation")]:
        axes[1].plot([x["epoch"] for x in history], [x[field] for x in history],
                     label=label, color=color, linewidth=2)
    axes[1].axvline(2.5, color="#89979c", linestyle="--", linewidth=1)
    axes[1].set(title="Expanded MobileNetV4: training accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.legend(loc="lower right", fontsize=8)
        axis.grid(alpha=.2)
        axis.set_ylim(0, 1.04)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.5)
    path = FIGURES / "work_training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


story.append(Paragraph("ASTANA IT UNIVERSITY / COMPUTER VISION / ASSIGNMENT 2", styles["WorkLabel"]))
story.append(Paragraph("GestureLab", styles["Title"]))
story.append(Paragraph("Hand gesture recognition: completed work report", styles["Heading2"]))
p("Turaliyev Ali | SE-2430 | Four classes: fist, open palm, peace and thumbs up")
heading("1. What was built")
p("GestureLab continues the Assignment 1 hand gesture task and turns it into a local web application. The work includes reproduced SimpleCNN and ResNet18 baselines, a modern MobileNetV4 classifier, a larger real-image dataset, two fine-tuned models and Grad-CAM explanations. Five trained models are available in the app; the initial MobileNetV4 no-augmentation run is retained as an experiment.")
p("Assignment 1 supplied 400 prepared hand photographs and a notebook defining the baseline architectures. It did not supply trained checkpoints or result tables. The baselines were therefore trained again and are reported as reproductions. The original images, labels and split were kept, allowing the new models to be evaluated on the same 60-image test set.")
heading("Application workflow", 2)
flow(["Input", "Hand crop", "Local model", "Result"],
     ["Upload / camera", "Preserve aspect ratio", "Selected checkpoint", "Label + Grad-CAM"])
table(["Screen", "Completed functionality"], [
    ["Studio", "Image upload, hand crop controls, model selection, four class scores, inference/roundtrip timing and a Grad-CAM opacity slider."],
    ["Compare", "Runs the same image through available models and displays their predictions, scores and timings side by side."],
    ["Experiments", "Displays saved results, training curves, confusion matrices, augmentation comparison and separate stress-condition scores."],
    ["About", "Explains model sources, preprocessing, privacy and practical limitations."],
], [30,136])
p("The camera interface includes browser permission handling, a positioning guide and image capture for the same prediction workflow. Permission-denied and unavailable-camera states are handled. Image upload and prediction using the saved checkpoints have been verified.")
heading("Completed result", 2)
p("Both fine-tuned models classified all 60 original test images and all 400 fresh test images correctly. The fresh cohort contains users excluded from training and validation. These are results on curated HaGRID hand crops; performance on the user's personal phone photographs has not yet been measured.")
page()

heading("2. Dataset preparation")
p("The project uses real HaGRID images. The original 400 UUID filenames and labels were matched to official annotations, identifying 400 distinct upstream user IDs. The original train, validation and test sets have no user intersections under those identifiers. All 400 original users were excluded when selecting the additional images [4].")
table(["Dataset group", "Train", "Validation", "Test"], [
    ["Original supplied subset", "280", "60", "60"],
    ["Additional HaGRID crops", "2,000", "400", "400"],
    ["New training / selection split", "2,280", "400", "400 fresh"],
], [77,29,29,31])
p("The complete source collection contains 3,200 images. The expanded training manifest contains 3,080 records: original training images plus added training, validation and fresh test images. The original 60 validation and 60 test images remain outside that manifest and were not changed. Each project class has 570 training images, 100 new validation images and 100 fresh test images.")
heading("Hand crop preparation", 2)
p("Matching official hand boxes were used to crop the additional images, with a 15% margin on each side. Crops were converted to RGB, padded with gray (128,128,128) while preserving aspect ratio, and stored at 320 square before model-specific resizing. Original prepared images remain unchanged. The application uses the corresponding crop, padding and normalization policy when receiving a new input.")
p("The logical peace class includes both front- and reverse-facing V signs: added peace images are 75% official peace and 25% peace_inverted. The upstream label is retained in each record. Fist, open palm and thumbs up retain their corresponding upstream classes. This adds orientation coverage while keeping the same four project labels.")
heading("Separation and checks", 2)
users = EXP["additional_unique_participants"]
p(f"The expanded train, validation and fresh test contain {280+users['train']:,}, {users['val']} and {users['test']} distinct users respectively, including the 280 original training users. Their user sets are disjoint. There are no retained duplicate source IDs or identical crop hashes. Difference-hash screening rejected {EXP['rejected'].get('new_cross_split_perceptual_candidate',0)} cross-split candidates and {EXP['rejected'].get('holdout_perceptual_candidate',0)} candidate near the original holdouts.")
p("User separation is verified through official annotation IDs, not independent identity checks. Several images may belong to one user within a split, and perceptual hashing cannot detect every visually similar image. The fresh 400 images represent 292 users, not 400 independent people.")
heading("Augmentation and generated media", 2)
p("The source images are real HaGRID photographs. Training-only augmentation uses flips, small affine transforms and color changes. No text-to-image or text-to-video samples were added. Validation and clean test images are not augmented.")
page()

heading("3. Models and training")
heading("Baseline models from Assignment 1", 2)
p("SimpleCNN follows the notebook: three convolution blocks with 16, 32 and 64 channels, batch normalization and ReLU, adaptive 4 x 4 pooling, a dense 128 layer, dropout 0.3 and four outputs. It is trained from scratch. ResNet18 uses ImageNet pretrained features with a new dropout 0.2 and linear 512-to-4 classification head.")
heading("Modern addition: MobileNetV4 Conv Small", 2)
p("MobileNetV4 is an efficient model family published at ECCV 2024 [1]. The Conv Small version uses Universal Inverted Bottleneck blocks to build compact visual features. It was selected as a modern, practical classifier that can be trained on the available GPU and run locally on a CPU. The exact pretrained model is timm/mobilenetv4_conv_small.e3600_r256_in1k [2].")
flow(["RGB hand ROI", "UIB backbone", "Pooled features", "New classifier"],
     ["256 x 256", "Fine-tuned weights", "1,280 values", "4 gesture logits"])
table(["Model group", "Training approach", "Input / normalization"], [
    ["Initial SimpleCNN", "Full network, from scratch", "224; ImageNet mean/std"],
    ["Initial ResNet18 / MobileNetV4", "Frozen features; train new head", "224; ImageNet mean/std"],
    ["Fine-tuned ResNet18", "Head warmup, then full backbone", "224; ImageNet mean/std"],
    ["Fine-tuned MobileNetV4", "Head warmup, then full backbone", "256; mean 0.5, std 0.5"],
], [55,60,51])
p("The initial MobileNetV4 used the shared 224/ImageNet recipe. The fine-tuned MobileNetV4 uses its native 256-pixel, 0.5-normalized recipe. ImageNet RGB mean/std are (0.485,0.456,0.406)/(0.229,0.224,0.225). Checkpoint metadata controls preprocessing in both training and serving.")
heading("Fine-tuning completed", 2)
p("Both new models started from fresh ImageNet weights and used seed 42, batch size 16, two head-warmup epochs and eight full-backbone epochs. AdamW used learning rate 0.001 during warmup, then 0.00001 for the backbone and 0.0001 for the head, with weight decay 0.0001. Minimum loss on the new validation set selected each checkpoint; the selected epoch was 9 for both runs.")
rn = RUNS["resnet18_finetuned"]["metadata"]
mn = RUNS["mobilenetv4_finetuned"]["metadata"]
p(f"Training ran on an NVIDIA GeForce GTX 1660 Ti with PyTorch 2.6.0+cu124. ResNet18 used mixed precision and took {rn['training_seconds']/60:.1f} minutes. MobileNetV4 used float32 after an fp16 gradient compatibility issue and took {mn['training_seconds']/60:.1f} minutes. Backbone weights changed during optimization; these are full fine-tuned models, not only newly trained heads.")
page()

heading("4. Model comparison and training curves")
p("The original test contains 60 images, 15 per class. It was already evaluated during the first project phase. The separate fresh test contains 400 unaugmented images, 100 per class, from users absent from training and validation. Results below keep the cohorts separate; accuracy and macro-F1 come from saved predictions and labels.")
table(["Model / run", "Original accuracy", "Original F1", "Fresh accuracy", "Fresh F1"], [
    [NAMES[key], percent(RUNS[key]["metrics"]["accuracy"]), f"{RUNS[key]['metrics']['f1_macro']:.3f}",
     percent(FRESH[key]["metrics"]["accuracy"]) if key in FRESH else "Not evaluated",
     f"{FRESH[key]['metrics']['f1_macro']:.3f}" if key in FRESH else "-"] for key in RUN_IDS
], [62,28,24,28,24])
p("The unchanged A1 ResNet18 is the strongest initial classifier on the fresh test at 93.75% accuracy. Both fine-tuned models score 100% on this curated fresh cohort. The update combines more data, added peace orientations, training augmentation and backbone learning; MobileNetV4 preprocessing also changes. The gain cannot be assigned to one change alone.")
figure(make_curves(), 166, "Figure 1. Saved epoch histories: initial MobileNetV4 validation accuracy with/without moderate augmentation (left); expanded MobileNetV4 training and validation accuracy (right). The dashed line marks the end of head warmup.")
heading("How the results were obtained", 2)
p("A checkpoint was selected by validation loss, saved and strictly reloaded before final testing. The new original-test results are 15 correct predictions in every class. The fresh confusion matrices are 100 correct predictions in every class for both fine-tuned models. Per-class scores, confusion matrices and prediction CSVs are retained in artifacts/metrics/.")
p("The 100% result describes these held-out hand crops, not all possible camera inputs. The images were screened for valid hand boxes and come from the same dataset source. Personal phone photographs, unusual backgrounds and unseen gestures may behave differently. One error on the original test changes accuracy by 1.67 percentage points.")
page()

heading("5. Ablation, explanation and latency")
heading("Initial augmentation ablation", 2)
table(["Initial MobileNetV4 run", "Test accuracy", "Macro-F1"], [
    ["No augmentation", percent(RUNS['mobilenetv4_noaug']['metrics']['accuracy']), f"{RUNS['mobilenetv4_noaug']['metrics']['f1_macro']:.3f}"],
    ["Moderate augmentation", percent(RUNS['mobilenetv4']['metrics']['accuracy']), f"{RUNS['mobilenetv4']['metrics']['f1_macro']:.3f}"],
], [85,41,40])
p("This pair keeps the seed, architecture, optimizer, validation rule and 15-epoch budget fixed. Moderate augmentation adds four fixed variants per training image: 1,400 tensors instead of 280. It also increases optimizer steps, so the result includes both transform diversity and extra exposure. This ablation is separate from the later expansion and fine-tuning update.", True)
heading("Visual explanation", 2)
input_path = ROOT / "artifacts/verification/input_mobilenetv4_finetuned.png"
cam_path = ROOT / "artifacts/verification/gradcam_mobilenetv4_finetuned.png"
pair = Table([[Image(str(input_path), width=54*mm, height=54*mm),
               Image(str(cam_path), width=54*mm, height=54*mm)]], colWidths=[83*mm,83*mm])
story.append(pair)
p("Figure 2. Actual prepared input (left) and fine-tuned MobileNetV4 Grad-CAM overlay (right), returned by the selected classifier. Gradients weight its spatial features before pooling [3]. The map is approximate evidence, not segmentation or a guarantee of correctness. Image-bearing report content requires appropriate sharing rights.", True)
markdown.extend([f"![Prepared input](../{input_path.relative_to(ROOT).as_posix()})\n",
                 f"![Actual Grad-CAM](../{cam_path.relative_to(ROOT).as_posix()})\n"])
heading("Measured CPU inference", 2)
table(["Production model", "Mean ms", "p50 ms", "p95 ms"], [
    [NAMES[key], f"{row['latency_same_host_cpu']['mean_ms']:.2f}",
     f"{row['latency_same_host_cpu']['p50_ms']:.2f}", f"{row['latency_same_host_cpu']['p95_ms']:.2f}"]
    for key,row in FRESH.items()
], [76,30,30,30])
p("Same host: Intel Core i3-8100 at 3.60 GHz, Windows 10, PyTorch 2.14.1+cpu, two threads. Three warmups precede 20 single-image forward timings. These exclude decoding, preprocessing, Grad-CAM and browser/network time; the app shows request roundtrip separately. CUDA training time is a different measure.", True)
stress = RUNS['mobilenetv4_finetuned']['stress']
p("Separate stress checks on the original 60 images gave fine-tuned MobileNetV4 accuracy of "
  f"{percent(stress['dim_brightness_0.5']['accuracy'])} with brightness x0.5, "
  f"{percent(stress['gaussian_blur_radius_2']['accuracy'])} with Gaussian blur radius 2 and "
  f"{percent(stress['jpeg_quality_30']['accuracy'])} with JPEG quality 30. These are image perturbations, not new real-environment samples.", True)
page()

heading("6. Running the application and practical limits")
heading("Local implementation", 2)
p("The interface uses React, TypeScript and Vite. A FastAPI backend loads local PyTorch checkpoints, validates the model ID and preprocessing metadata, then returns the predicted label, four scores, timings and the requested gradient explanation. Only trained local checkpoints can serve predictions. The backend does not fetch model weights during inference.")
p("The local app provides five trained model choices and six saved experiment runs, including the initial augmentation ablation. Upload, comparison and experiment views are working. Physical-camera capture and final evaluation on personal phone images remain unverified.")
heading("Start in two project terminals", 2)
table(["Terminal", "Windows PowerShell command"], [
    ["API", ".venv/Scripts/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"],
    ["Interface", "npm --prefix frontend run dev -- --host 127.0.0.1"],
    ["Browser", "Open http://127.0.0.1:5173; the camera requires localhost/HTTPS and browser permission."],
], [25,141])
p("Install the saved Python requirements and frontend package lock before first use. The existing checkpoints can be reused. Expanded GPU training is reproducible with .venv-gpu/Scripts/python -m ml.finetune --models all --head-epochs 2 --finetune-epochs 8 --batch-size 16 --patience 4 --device cuda. The source/user manifest, preprocessing and selected epochs are saved with the run artifacts. Full setup commands are in README.md.", True)
heading("Privacy, rights and limitations", 2)
p("Images uploaded or captured by the app are processed without application retention. HaGRID uses a custom public license, whose text explicitly says it is not a Creative Commons license; the official terms and attribution must be followed [4]. Matching annotation user IDs establishes split separation, but individual consent and sharing rights were not independently verified. Source photographs and image-bearing report/video evidence should remain private until those rights are confirmed.")
p("Softmax scores are uncalibrated, and the classifier cannot reliably reject unrelated objects. Lighting, crop quality, background, pose and skin-tone coverage can affect results. The work demonstrates a completed local classification application and measured improvement on its specified cohorts; it does not establish state-of-the-art gesture recognition or guaranteed personal-camera accuracy. The existing 3:49 demo records the earlier three-model version and needs updating for the five-model app.")
heading("References", 2)
p("[1] Qin et al. MobileNetV4: Universal Models for the Mobile Ecosystem. ECCV 2024. www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php", True)
p("[2] timm model and pretrained weights: huggingface.co/timm/mobilenetv4_conv_small.e3600_r256_in1k", True)
p("[3] Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017. arxiv.org/abs/1610.02391", True)
p("[4] Official HaGRID dataset, annotations and custom license: github.com/hukenovs/hagrid and github.com/hukenovs/hagrid/tree/master/license. The relevant Neurohive citation choice in the assignment still requires instructor confirmation; the modern method is documented by its primary paper [1].", True)


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#c7d8df"))
    canvas.line(22*mm,18*mm,188*mm,18*mm)
    canvas.setFont("Helvetica",8)
    canvas.setFillColor(colors.HexColor("#516575"))
    canvas.drawString(22*mm,13*mm,"GestureLab | Assignment 2 | Completed work report")
    canvas.drawRightString(188*mm,13*mm,str(doc.page))


doc = SimpleDocTemplate(str(OUT / "report.pdf"), pagesize=(210*mm,297*mm),
                        leftMargin=22*mm,rightMargin=22*mm,topMargin=19*mm,bottomMargin=24*mm,
                        title="GestureLab - Assignment 2 Work Report")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT / "report.md").write_text("\n".join(markdown), encoding="utf-8")
from pypdf import PdfReader
count = len(PdfReader(OUT / "report.pdf").pages)
print(f"Practical work report generated: {count} pages at {OUT / 'report.pdf'}")
assert 6 <= count <= 8, f"Expected 6-8 pages, got {count}"
