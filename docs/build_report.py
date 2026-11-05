"""Build the assignment report (docs/assignment.md + PDF) from outputs/metrics.json.

Usage:  python docs/build_report.py
Requires: reportlab
"""
import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PDF = ROOT / "Assignment_Handwritten_Digit_Recognition.pdf"
MD = ROOT / "docs" / "assignment.md"
REPO = "https://github.com/luffy-code244/digit-recognition"
AUTHOR = "Saket"
TITLE = "Handwritten Digit Recognition using Machine Learning"
DATE = datetime.now().strftime("%d %B %Y")

m = json.loads((OUT / "metrics.json").read_text())
ACC = m["test_accuracy"] * 100
TRAIN_ACC = m["train_accuracy"] * 100
F1 = m["macro_f1"]
cm_ = m["confusion_matrix"]
ERRORS = m["test_samples"] - sum(cm_[i][i] for i in range(10))
pairs = sorted(((cm_[i][j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:4]
CONF_TXT = ", ".join(f"{i} predicted as {j} ({c})" for c, i, j in pairs)
pc = m["per_class"]
best = max(pc, key=lambda k: pc[k]["f1-score"])
worst = min(pc, key=lambda k: pc[k]["f1-score"])

# ---------------------------------------------------------------- content
# Each section: (heading, [blocks]); block = ("p", text) | ("ul", [items]) | ("table", rows, widths)
#                                     | ("img", path, width_cm, caption) | ("code", text) | ("h3", text)
BRIEF = [
    ("h3", "Assignment 1 – Machine Learning Internship Programme"),
    ("table", [["Field", "Details"],
               ["Assignment title", "Handwritten Digit Recognition"],
               ["Domain", "Machine Learning / Computer Vision"],
               ["Submission", "Assignment report (PDF) + public GitHub repository link"],
               ["Expected effort", "1 week"]], [4.5, 11]),
    ("p", "<b>Objective.</b> Build a machine learning system that can recognise handwritten digits (0–9) from "
          "images. The intern should demonstrate the full ML workflow: loading a dataset, preprocessing, "
          "training a model, evaluating it with appropriate metrics, and using the trained model to make "
          "predictions on new images."),
    ("p", "<b>Requirements.</b>"),
    ("ul", ["Use the MNIST handwritten digit dataset (or an equivalent public dataset).",
            "Preprocess the data (normalisation, reshaping) appropriately for the chosen model.",
            "Train at least one classification model (e.g. Logistic Regression, SVM, MLP or CNN).",
            "Achieve a test accuracy of at least 95 % on held-out data.",
            "Evaluate the model with accuracy, a classification report and a confusion matrix.",
            "Provide a script or function that predicts the digit in a new image supplied by the user.",
            "Write clean, modular, documented code that someone else can run from the instructions."]),
    ("p", "<b>Deliverables.</b>"),
    ("ul", ["Public GitHub repository containing source code, requirements file and a README with setup "
            "and run instructions.",
            "Saved trained model and result plots (confusion matrix, sample predictions).",
            "This assignment report in PDF format covering the problem, dataset, methodology, results, "
            "how to reproduce, and conclusion, with the GitHub link included."]),
    ("p", "<b>Evaluation criteria.</b> Correctness and accuracy of the model (30 %), code quality and "
          "structure (25 %), quality of the analysis and report (25 %), reproducibility and documentation (20 %)."),
]

SECTIONS = [
    ("1. Assignment Brief", BRIEF),
    ("2. Introduction and Problem Statement", [
        ("p", "Handwritten digit recognition is the task of automatically identifying which digit (0–9) is "
              "written in an image. It is a classic problem in computer vision and pattern recognition, with "
              "real-world uses such as reading postal codes on envelopes, processing bank cheques, digitising "
              "forms and grading exam sheets."),
        ("p", "The difficulty comes from the large variation in human handwriting: the same digit can be "
              "written with different stroke thickness, slant, size and style, and some digits look alike "
              "(for example 4 and 9, or 3 and 5). A good recogniser must learn the general shape of each digit "
              "rather than memorising specific examples."),
        ("p", "<b>Problem statement:</b> given a 28×28 grayscale image of a single handwritten digit, predict "
              "the correct digit class (one of 10 classes). This is a supervised multi-class classification "
              "problem. The goal of this project is to build, evaluate and package such a classifier so that "
              "it reaches at least 95 % accuracy on unseen test images and can be used on new images."),
    ]),
    ("3. Dataset: MNIST", [
        ("p", "The project uses the <b>MNIST</b> (Modified National Institute of Standards and Technology) "
              "database, the standard benchmark for handwritten digit recognition. It is downloaded "
              "automatically from OpenML (<i>mnist_784</i>) using <font face='Courier'>sklearn.datasets.fetch_openml</font>."),
        ("table", [["Property", "Value"],
                   ["Total images", "70,000"],
                   ["Training set", "60,000 images (standard split)"],
                   ["Test set", "10,000 images (standard split, never used in training)"],
                   ["Image size", "28 × 28 pixels, grayscale (784 features)"],
                   ["Pixel range", "0 (background) to 255 (ink)"],
                   ["Classes", "10 (digits 0–9), roughly balanced (~7,000 per class)"]], [4.5, 11]),
        ("p", "Each digit was size-normalised to fit a 20×20 box and centred by centre of mass on a 28×28 "
              "canvas. The test set was written by different people from the training set, so test accuracy "
              "is a fair measure of how well the model generalises to new writers."),
    ]),
    ("4. Methodology", [
        ("h3", "4.1 Preprocessing"),
        ("ul", ["<b>Flattening:</b> each 28×28 image is represented as a vector of 784 pixel values.",
                "<b>Normalisation:</b> pixel values are divided by 255 to lie in the range [0, 1]. This keeps "
                "the inputs on a similar scale and helps gradient-based training converge faster.",
                "<b>Data type:</b> features are stored as float32 to reduce memory usage.",
                "<b>Train/test split:</b> the standard 60,000 / 10,000 split is used. In addition, 10 % of the "
                "training set is held out internally as a validation set for early stopping."]),
        ("h3", "4.2 Model choice"),
        ("p", "A <b>Multi-Layer Perceptron (MLP)</b>, a fully connected feed-forward neural network, was chosen "
              "using scikit-learn's <font face='Courier'>MLPClassifier</font>. Compared with linear models "
              "(e.g. Logistic Regression, ~92 % on MNIST) an MLP can learn non-linear combinations of pixels and "
              "reaches much higher accuracy, while still training in seconds on a CPU without needing a deep "
              "learning framework or GPU. An SVM with an RBF kernel can be similarly accurate but is far slower "
              "to train on 60,000 samples."),
        ("table", [["Hyper-parameter", "Value"],
                   ["Architecture", "784 → 256 (ReLU) → 128 (ReLU) → 10 (softmax)"],
                   ["Optimiser", "Adam, learning rate 0.001"],
                   ["Batch size", "128"],
                   ["L2 regularisation (alpha)", "1e-4"],
                   ["Max epochs", "30, with early stopping (patience 5 on validation accuracy)"],
                   ["Validation split", "10 % of training data"],
                   ["Random seed", "42 (reproducible)"]], [4.5, 11]),
        ("h3", "4.3 Training"),
        ("p", f"The model was trained on 60,000 training images. Early stopping monitored the validation "
              f"accuracy and stopped training after <b>{m['epochs_run']} epochs</b>, when it no longer improved, "
              f"which prevents overfitting. Training took about <b>{m['train_time_sec']:.0f} seconds</b> on an "
              f"8-core CPU. The trained model is saved with joblib to "
              f"<font face='Courier'>models/digit_model.joblib</font>."),
        ("img", OUT / "training_curve.png", 12.5, "Figure 1. Training loss and validation accuracy per epoch."),
        ("h3", "4.4 Prediction on new images"),
        ("p", "To use the model on images that are not from MNIST (e.g. a photo or a digit drawn in Paint), "
              "<font face='Courier'>src/predict.py</font> applies the same preparation MNIST uses: convert to "
              "grayscale, invert if the background is light, crop to the digit, resize it into a 20×20 box "
              "keeping its aspect ratio, centre it on a 28×28 canvas by centre of mass, and scale to [0, 1]. "
              "It then prints the predicted digit, its confidence and the top-3 candidates."),
    ]),
    ("5. Results", [
        ("p", f"The final model was evaluated once on the 10,000 held-out test images."),
        ("table", [["Metric", "Value"],
                   ["Test accuracy", f"{ACC:.2f} %"],
                   ["Training accuracy", f"{TRAIN_ACC:.2f} %"],
                   ["Macro-averaged F1-score", f"{F1:.4f}"],
                   ["Misclassified test images", f"{ERRORS} of {m['test_samples']:,}"],
                   ["Epochs trained", str(m["epochs_run"])],
                   ["Target from brief", "≥ 95 %  –  achieved"]], [7, 8.5]),
        ("p", f"The model achieves <b>{ACC:.2f} % test accuracy</b>, comfortably above the 95 % target. "
              f"Training accuracy ({TRAIN_ACC:.2f} %) is only slightly higher than test accuracy, which shows the "
              f"model generalises well and is not heavily overfitting."),
        ("h3", "5.1 Per-class performance"),
        ("table", [["Digit", "Precision", "Recall", "F1-score", "Test images"]] +
                  [[k, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1-score']:.4f}", f"{int(v['support'])}"]
                   for k, v in pc.items()], [2.5, 3.2, 3.2, 3.2, 3.4]),
        ("p", f"Every digit is recognised with an F1-score above 0.96. Digit <b>{best}</b> is the easiest "
              f"(F1 {pc[best]['f1-score']:.4f}) and digit <b>{worst}</b> the hardest (F1 {pc[worst]['f1-score']:.4f}), "
              f"mainly because other digits, especially 4, are sometimes predicted as 9."),
        ("h3", "5.2 Confusion matrix"),
        ("img", OUT / "confusion_matrix.png", 11.5, "Figure 2. Confusion matrix on the MNIST test set (rows = true digit, columns = predicted digit)."),
        ("p", f"The confusion matrix has a strong diagonal: almost all images are assigned to their correct "
              f"class. The largest off-diagonal values are: {CONF_TXT}. These errors make sense visually: a 4 "
              f"with a closed top looks like a 9, the curves of 3 and 9 or 5 overlap, and a 2 with a loop at the "
              f"bottom resembles an 8."),
        ("h3", "5.3 Sample predictions"),
        ("img", OUT / "sample_predictions.png", 12, "Figure 3. Random test images with predicted and true labels (green = correct, red = wrong)."),
        ("img", OUT / "misclassified.png", 13, "Figure 4. Examples of misclassified test images."),
        ("p", "The misclassified examples are mostly unusually written, very slanted, broken or ambiguous digits "
              "that a human could also find hard to read. A custom black-on-white drawing of a 7 "
              "(<font face='Courier'>samples/handdrawn_7.png</font>) was also correctly predicted as 7 with "
              "high confidence, which confirms that the preprocessing pipeline works on non-MNIST images."),
    ]),
    ("6. How to Run / Reproduce", [
        ("p", f"The complete code is available at <link href='{REPO}' color='blue'><u>{REPO}</u></link>. "
              "The trained model is included, so predictions work immediately; retraining is optional."),
        ("h3", "Setup"),
        ("code", "git clone https://github.com/luffy-code244/digit-recognition.git\n"
                 "cd digit-recognition\n"
                 "python -m venv .venv\n"
                 "source .venv/bin/activate        # Windows: .venv\\Scripts\\activate\n"
                 "pip install -r requirements.txt"),
        ("h3", "Predict"),
        ("code", "python -m src.predict --image samples/digit_7.png   # your own image\n"
                 "python -m src.predict --demo -n 10                  # random MNIST test images"),
        ("h3", "Retrain (optional)"),
        ("code", "python -m src.train          # full training, writes models/ and outputs/\n"
                 "python -m src.train --quick  # fast smoke test on 10k images"),
        ("h3", "Repository structure"),
        ("table", [["Path", "Purpose"],
                   ["src/common.py", "Dataset download/caching and shared paths"],
                   ["src/train.py", "Training, evaluation, saving the model, plots and metrics"],
                   ["src/predict.py", "Image preprocessing and prediction (file or demo mode)"],
                   ["models/digit_model.joblib", "Trained MLP model"],
                   ["outputs/", "Confusion matrix, sample predictions, training curve, metrics.json"],
                   ["samples/", "Example digit images for testing predict.py"],
                   ["requirements.txt", "Python dependencies"],
                   ["README.md", "Project overview, setup and usage"]], [5.5, 10]),
    ]),
    ("7. Conclusion", [
        ("p", f"This project built a complete handwritten digit recognition system using a Multi-Layer "
              f"Perceptron trained on MNIST. With simple preprocessing (pixel normalisation) and a two-hidden-layer "
              f"network, the model reached <b>{ACC:.2f} % accuracy</b> on 10,000 unseen test images, exceeding the "
              f"95 % requirement, and trains in about {m['train_time_sec']:.0f} seconds on a CPU. The evaluation shows "
              f"consistent performance across all ten digits, with the remaining errors concentrated on visually "
              f"similar pairs such as 4/9 and 3/9."),
        ("p", "All deliverables of the brief are met: a public GitHub repository with modular, documented code, "
              "a saved model, result plots, a prediction script for new images, and this report."),
        ("p", "<b>Learning outcomes:</b> the end-to-end ML workflow (data loading, preprocessing, model selection, "
              "training with early stopping, evaluation with multiple metrics), interpreting a confusion matrix, "
              "and making a model usable on real-world inputs through consistent preprocessing."),
        ("p", "<b>Future work:</b>"),
        ("ul", ["Train a Convolutional Neural Network (CNN), which typically reaches 99 %+ on MNIST by exploiting the 2-D structure of images.",
                "Use data augmentation (rotation, shifting, scaling) to improve robustness to real handwriting.",
                "Build a simple web app or drawing canvas (e.g. Streamlit or Gradio) for live predictions.",
                "Extend to multi-digit recognition, such as reading full numbers."]),
    ]),
    ("Appendix A. GitHub Repository", [
        ("repo",),
        ("p", "The repository contains all source code, the trained model "
              "(<font face='Courier'>models/digit_model.joblib</font>), the result figures in "
              "<font face='Courier'>outputs/</font>, the README with setup instructions, and this report."),
        ("h3", "Appendix B. Software environment"),
        ("table", [["Component", "Version / details"],
                   ["Language", "Python 3"],
                   ["Libraries", "scikit-learn, NumPy, Matplotlib, Pillow, joblib, pandas"],
                   ["Hardware", "8-core CPU, no GPU required"],
                   ["Report generation", "ReportLab (docs/build_report.py)"]], [4.5, 11]),
    ]),
]

# ---------------------------------------------------------------- markdown
def strip(t):
    import re
    t = re.sub(r"<link href='([^']+)'[^>]*><u>[^<]+</u></link>", r"\1", t)
    t = t.replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*")
    t = re.sub(r"<font face='Courier'>(.*?)</font>", r"`\1`", t)
    return re.sub(r"<[^>]+>", "", t)


def build_md():
    L = [f"# {TITLE}", "", f"**Internship Assignment Submission**  ", f"**Submitted by:** {AUTHOR}  ",
         f"**Date:** {DATE}  ", f"**GitHub repository:** {REPO}", ""]
    for head, blocks in SECTIONS:
        L += [f"## {head}", ""]
        for b in blocks:
            k = b[0]
            if k == "p": L += [strip(b[1]), ""]
            elif k == "h3": L += [f"### {b[1]}", ""]
            elif k == "ul": L += [f"- {strip(i)}" for i in b[1]] + [""]
            elif k == "code": L += ["```bash", b[1], "```", ""]
            elif k == "repo": L += [f"**{REPO}**", ""]
            elif k == "img": L += [f"![{b[3]}](../outputs/{b[1].name})", "", f"*{b[3]}*", ""]
            elif k == "table":
                rows = b[1]
                L += ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                L += ["| " + " | ".join(r) + " |" for r in rows[1:]] + [""]
    MD.write_text("\n".join(L))


# ---------------------------------------------------------------- PDF
NAVY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#2E86C1")
LIGHT = colors.HexColor("#EAF2F8")

ss = getSampleStyleSheet()
S = {
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=7, alignment=4),
    "h2": ParagraphStyle("h2", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16, textColor=NAVY, spaceBefore=6, spaceAfter=10),
    "h3": ParagraphStyle("h3", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=ACCENT, spaceBefore=8, spaceAfter=5, keepWithNext=1),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=9, textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9.5, leading=12),
    "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.8, leading=12, backColor=colors.HexColor("#F4F6F7"),
                           borderColor=colors.HexColor("#D5D8DC"), borderWidth=0.5, borderPadding=7, spaceBefore=4, spaceAfter=12, leftIndent=7, rightIndent=7),
}


def table(rows, widths):
    data = [[Paragraph(c, S["cellh"]) for c in rows[0]]] + [[Paragraph(c, S["cell"]) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFC9CA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [t, Spacer(1, 10)]


def img(path, wcm, cap):
    from PIL import Image as PI
    w, h = PI.open(path).size
    W = wcm * cm
    return [KeepTogether([Image(str(path), width=W, height=W * h / w), Spacer(1, 4), Paragraph(cap, S["cap"])])]


def repo_box():
    p = Paragraph(f"<font size=10 color='#555555'>GitHub Repository</font><br/><br/>"
                  f"<font size=14><b><link href='{REPO}' color='#1F3A5F'>{REPO}</link></b></font>",
                  ParagraphStyle("rb", alignment=TA_CENTER, leading=18))
    t = Table([[p]], colWidths=[15.5 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.5, ACCENT), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                           ("TOPPADDING", (0, 0), (-1, -1), 16), ("BOTTOMPADDING", (0, 0), (-1, -1), 16)]))
    return [t, Spacer(1, 14)]


def cover():
    st = lambda n, **k: ParagraphStyle(n, alignment=TA_CENTER, **k)
    f = [Spacer(1, 3 * cm),
         Paragraph("INTERNSHIP ASSIGNMENT SUBMISSION", st("a", fontName="Helvetica-Bold", fontSize=12, textColor=ACCENT, leading=16)),
         Spacer(1, 0.6 * cm),
         Paragraph("Handwritten Digit Recognition<br/>using Machine Learning", st("t", fontName="Helvetica-Bold", fontSize=26, leading=34, textColor=NAVY)),
         Spacer(1, 0.5 * cm),
         Paragraph("Classifying MNIST digits with a Multi-Layer Perceptron neural network", st("s", fontName="Helvetica-Oblique", fontSize=12.5, textColor=colors.HexColor("#555555"))),
         Spacer(1, 1.2 * cm),
         Image(str(OUT / "sample_predictions.png"), width=9 * cm, height=9 * cm * 0.86),
         Spacer(1, 1.2 * cm)]
    rows = [["Submitted by", AUTHOR], ["Assignment", "Machine Learning Internship – Assignment 1"],
            ["Date of submission", DATE], ["Test accuracy", f"{ACC:.2f} %"],
            ["GitHub repository", f"<link href='{REPO}' color='#2E86C1'><u>{REPO}</u></link>"]]
    data = [[Paragraph(f"<b>{a}</b>", S["cell"]), Paragraph(b, S["cell"])] for a, b in rows]
    t = Table(data, colWidths=[4.5 * cm, 10.5 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#BFC9CA")),
                           ("BOX", (0, 0), (-1, -1), 1, NAVY), ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return f + [t, PageBreak()]


def toc():
    f = [Paragraph("Contents", S["h2"])]
    for head, _ in SECTIONS:
        f.append(Paragraph(head, ParagraphStyle("toc", fontName="Helvetica", fontSize=11.5, leading=22, leftIndent=10)))
    f += [Spacer(1, 1 * cm), Paragraph("Summary", S["h3"]),
          Paragraph(f"A handwritten digit classifier was trained on the 60,000-image MNIST training set using a "
                    f"two-hidden-layer MLP (scikit-learn). It achieves <b>{ACC:.2f} % accuracy</b> on the 10,000-image "
                    f"test set (target ≥ 95 %), with a macro F1-score of {F1:.4f}. The code, trained model and results "
                    f"are available at <link href='{REPO}' color='blue'><u>{REPO}</u></link>.", S["body"]),
          PageBreak()]
    return f


def on_page(c, doc):
    if doc.page == 1:
        c.setFillColor(NAVY); c.rect(0, A4[1] - 1.2 * cm, A4[0], 1.2 * cm, fill=1, stroke=0)
        c.setFillColor(ACCENT); c.rect(0, 0, A4[0], 0.6 * cm, fill=1, stroke=0)
        return
    c.setStrokeColor(ACCENT); c.setLineWidth(0.8)
    c.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)
    c.setFont("Helvetica", 8.5); c.setFillColor(colors.HexColor("#555555"))
    c.drawString(2 * cm, A4[1] - 1.3 * cm, "Handwritten Digit Recognition – Internship Assignment")
    c.drawRightString(A4[0] - 2 * cm, A4[1] - 1.3 * cm, AUTHOR)
    c.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    c.drawString(2 * cm, 1.0 * cm, REPO)
    c.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Page {doc.page}")


def build_pdf():
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.1 * cm,
                            bottomMargin=2 * cm, title=TITLE, author=AUTHOR, subject="Internship assignment submission")
    story = cover() + toc()
    for n, (head, blocks) in enumerate(SECTIONS):
        if head.startswith(("2.", "4.", "5.", "6.", "Appendix")):
            if n: story.append(PageBreak())
        story.append(Paragraph(head, S["h2"]))
        skip = set()
        for bi, b in enumerate(blocks):
            if bi in skip:
                continue
            k = b[0]
            nxt = blocks[bi + 1] if bi + 1 < len(blocks) else ("",)
            if k == "p": story.append(Paragraph(b[1], S["body"]))
            elif k == "h3" and nxt[0] == "img":
                story.append(KeepTogether([Paragraph(b[1], S["h3"])] + img(nxt[1], nxt[2], nxt[3])))
                skip.add(bi + 1)
            elif k == "h3": story += [CondPageBreak(5 * cm), Paragraph(b[1], S["h3"])]
            elif k == "ul":
                story.append(ListFlowable([ListItem(Paragraph(i, S["body"]), leftIndent=14) for i in b[1]],
                                          bulletType="bullet", start="•", leftIndent=14))
            elif k == "table": story += table(b[1], b[2])
            elif k == "img": story += img(b[1], b[2], b[3])
            elif k == "code": story.append(Preformatted(b[1], S["code"]))
            elif k == "repo": story += repo_box()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    build_md()
    build_pdf()
    print("Wrote", MD, "and", PDF)
