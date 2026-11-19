#!/usr/bin/env python3
"""Build the DATA 255 LEVIR-CD report in IEEE two-column Word format.

Same pattern as data266-lab2-team05/report/build_ieee_report.py:
create DOCX first (Times New Roman, two-column), then export PDF with Word.
"""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Twips

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures_compressed"
OUT = ROOT / "DATA255_BuildSafeVision_IEEE.docx"
COL_W = Inches(3.45)


def set_run_font(run, size=10, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), "Times New Roman")


def add_page_field(paragraph):
    run = paragraph.add_run()
    set_run_font(run, size=8)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def set_columns(section, n, space_twips=288):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    if n > 1:
        cols.set(qn("w:space"), str(space_twips))
    else:
        cols.attrib.pop(qn("w:space"), None)


def configure_page(section):
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.63)
    section.right_margin = Inches(0.63)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.6)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    add_page_field(p)


def P(doc, text="", size=10, bold=False, italic=False, align="justify",
      before=0, after=0, first=0, center=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if first:
        pf.first_line_indent = Inches(first)
    if center or align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if text:
        run = p.add_run(text)
        set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def heading(doc, text):
    return P(doc, text, size=10, bold=True, align="center", before=11, after=3, center=True)


def subhead(doc, text):
    return P(doc, text, size=10, italic=True, align="left", before=8, after=2)


def body(doc, text, indent=True):
    return P(doc, text, size=10, align="justify", before=0, after=2, first=0.15 if indent else 0)


def caption(doc, text):
    return P(doc, text, size=8, align="center", before=1, after=6, center=True)


def figure(doc, path, cap, width=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run()
    run.add_picture(str(path), width=width or COL_W)
    caption(doc, cap)


def shade_header(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F2F2F2")
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "666666")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def set_table_fixed(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    total = sum(widths)
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(total))
    tblW.set(qn("w:type"), "dxa")
    layout = tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblPr.append(layout)
    layout.set(qn("w:type"), "fixed")
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for child in list(grid):
            grid.remove(child)
    else:
        grid = OxmlElement("w:tblGrid")
        tblPr.addnext(grid)
    for w in widths:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w))
        grid.append(gc)
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Twips(w)
            tcPr = cell._tc.get_or_add_tcPr()
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW")
                tcPr.append(tcW)
            tcW.set(qn("w:w"), str(w))
            tcW.set(qn("w:type"), "dxa")


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    set_table_fixed(table, widths)
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(h)
        set_run_font(run, size=8, bold=True)
        shade_header(cell)
        set_cell_border(cell)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            set_run_font(run, size=8)
            set_cell_border(cell)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)
    return table


def table_title(doc, text):
    P(doc, text, size=8, bold=True, align="center", before=8, after=1, center=True)


def build():
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), "Times New Roman")

    sec0 = doc.sections[0]
    configure_page(sec0)
    set_columns(sec0, 1)

    P(doc, "Deep Learning for Building Change Detection in Multi-Temporal Satellite Imagery",
      size=16, bold=True, align="center", before=0, after=6, center=True)
    P(doc, "Sneha Singh, Ritika Mukesh Neema, Danish Waseem, and Rajesh Paruchuri",
      size=11, align="center", before=2, after=0, center=True)
    P(doc, "Department of Applied Data Science, San Jose State University, San Jose, CA, USA",
      size=10, italic=True, align="center", before=0, after=0, center=True)
    P(doc, "DATA 255 Deep Learning Vision, Fall 2026  ·  Team BuildSafe Vision",
      size=10, align="center", before=0, after=2, center=True)
    P(doc, "GitHub: https://github.com/danwaseem/levir-change-detection",
      size=9, align="center", before=0, after=8, center=True)

    abs_p = P(doc, "", size=9, align="justify", before=2, after=2)
    r1 = abs_p.add_run("Abstract—")
    set_run_font(r1, size=9, bold=True, italic=True)
    r2 = abs_p.add_run(
        "We study building change detection on LEVIR-CD: mark construction or demolition "
        "between two aligned satellite views. Simple RGB difference is weak under lighting "
        "and vegetation shift. We implement the Daudt triad (FC-EF, FC-Siam-diff, "
        "FC-Siam-conc) in one PyTorch pipeline, keep absolute difference as a reference, "
        "and run loss and augmentation ablations. Change pixels are rare (~4–5%), so the "
        "primary score is change-class F1 at threshold 0.5. Under matched BCE+Dice training, "
        "FC-EF mean validation F1 is 0.718 (3 seeds) and FC-Siam-diff is 0.543; FC-Siam-conc "
        "reaches 0.773 on one seed. BCE-only FC-EF hits the best validation F1 (0.778) and "
        "a locked test F1 of 0.780. At width base=16, H1 (Siam-diff better) and H2 (BCE+Dice "
        "better) are not supported. Absolute difference stays far lower (F1 0.129)."
    )
    set_run_font(r2, size=9, bold=True)

    idx = P(doc, "", size=9, align="justify", before=2, after=4)
    a = idx.add_run("Index Terms—")
    set_run_font(a, size=9, bold=True, italic=True)
    b = idx.add_run(
        "change detection, remote sensing, U-Net, Siamese networks, LEVIR-CD, semantic segmentation"
    )
    set_run_font(b, size=9, bold=True)

    sec1 = doc.add_section(WD_SECTION.CONTINUOUS)
    configure_page(sec1)
    set_columns(sec1, 2, 288)

    heading(doc, "I.  INTRODUCTION")
    body(
        doc,
        "Cities need maps of new or removed buildings, not every brightness change between "
        "two satellite dates. Shadows, plants, and sensor conditions can look like change "
        "even when the buildings did not move. We treat the problem as dense binary "
        "segmentation on a before/after pair.",
        indent=False,
    )
    body(
        doc,
        "Research question. Under the same splits, tiles, and metrics, does a shared-encoder "
        "Siamese U-Net beat a six-channel early-fusion U-Net (FC-EF) on building-change F1, "
        "and how do loss and paired augmentation change accuracy and compute?",
    )
    body(
        doc,
        "Hypotheses. (H1) FC-Siam-diff improves change-class F1/IoU over FC-EF by comparing "
        "dates in feature space, maybe with higher inference cost. (H2) BCE+Dice finds rare "
        "change pixels better than BCE alone. Success means honest evidence for or against "
        "these claims. This is a course prototype, not an operational system.",
    )
    body(
        doc,
        "What we deliver: a shared PyTorch pipeline for the Daudt triad on LEVIR-CD, a "
        "nine-run GPU matrix with three seeds for the main comparison, and one locked test "
        "after freeze with failure analysis.",
    )

    heading(doc, "II.  RELATED WORK")
    body(
        doc,
        "U-Net is the usual encoder–decoder for dense labels [1]. Daudt et al. compared "
        "FC-EF, FC-Siam-diff, and FC-Siam-conc for change detection [2]; we reimplemented "
        "all three under one training loop. Chen and Shi released LEVIR-CD and STANet and "
        "pushed change-class F1 [3]. We use their data and metric style. We do not rebuild "
        "STANet attention. Public FC-style scores near 0.81–0.86 often use heavier schedules "
        "or ImageNet backbones [4], [5]. Those numbers are context only. Our nets use width "
        "base=16 (~0.48–0.56M parameters) and train from scratch.",
        indent=False,
    )

    heading(doc, "III.  DATASET AND EVALUATION PROTOCOL")
    subhead(doc, "A.  Dataset")
    body(
        doc,
        "LEVIR-CD [3] has 637 RGB pairs (1024×1024, ~0.5 m/pixel, Texas Google Earth) with "
        "binary building construction/demolition masks. Academic use only; Google Earth terms "
        "apply. Official split: 445 train / 64 val / 128 test. We cut each pair into 16 "
        "non-overlapping 256×256 tiles after the pair is assigned to a split (7,120 / 1,024 / "
        "2,048 tiles). Masks are binarized at intensity ≥128. Changed pixels are only about "
        "4–5%, so we do not use overall accuracy for selection.",
        indent=False,
    )
    subhead(doc, "B.  Leakage audit")
    body(
        doc,
        "Filename-stem overlap across official splits is 0 "
        "(results/levir_audit/stem_overlap_report.json). That blocks trivial identity "
        "leakage. It does not prove nearby scenes never correlate; a lat/long distance "
        "check is future work.",
        indent=False,
    )
    subhead(doc, "C.  Metrics and test policy")
    body(
        doc,
        "Primary metric for checkpointing is change-class F1 at threshold 0.5. Secondary "
        "metrics are IoU, precision, recall, and boundary F1. Learned models keep threshold "
        "0.5. Absolute difference uses a validation-chosen threshold (0.4). We select on "
        "validation only. Freeze date 2026-11-08; test is touched once with no retuning.",
        indent=False,
    )

    heading(doc, "IV.  METHODS")
    body(
        doc,
        "Fig. 1 shows the pipeline. Absolute RGB difference is the non-learned reference. "
        "FC-EF stacks A and B as six channels into one U-Net (~483k params). FC-Siam-diff "
        "and FC-Siam-conc share an encoder and fuse with |fA−fB| or concat (~483k / ~564k).",
        indent=False,
    )
    figure(
        doc,
        FIG / "architecture.jpg",
        "Fig. 1. Bi-temporal inputs are tiled to 256×256. Absolute difference is the reference. "
        "Learned models are FC-EF, FC-Siam-diff, and FC-Siam-conc, scored at threshold 0.5.",
        width=Inches(3.45),
    )
    body(
        doc,
        "Training uses Adam (1e-3, weight decay 1e-4), batch 4, up to 30 epochs with early "
        "stopping, and best validation F1. We compare BCE vs BCE+Dice. Augmentation is "
        "synced flips and 90° rotations on A, B, and the mask. Principal seeds: 1511, 42, "
        "123. Full matrix ran on the university CUDA GPU lab; Mac MPS was only for smoke "
        "and early feasibility.",
    )

    heading(doc, "V.  EXPERIMENTAL SETUP")
    body(
        doc,
        "Nine full runs (scripts/run_plan.py --mode full): three-seed FC-EF and FC-Siam-diff "
        "with BCE+Dice+aug; one FC-Siam-conc; FC-EF BCE ablation; FC-EF no-aug ablation. "
        "Configs are under configs/. Same splits, tile size, Adam settings, epoch budget, "
        "eval code, and threshold 0.5 for the principal comparison.",
        indent=False,
    )
    body(
        doc,
        "Latency protocol: after training, batch-1 inference does 3 warm-up forwards, then "
        "10 timed forwards with device sync (src/levir_cd/train.py). Preprocessing time is "
        "not inside that number.",
    )

    heading(doc, "VI.  RESULTS")
    subhead(doc, "A.  Reference and overview")
    body(
        doc,
        "Absolute RGB difference reaches validation F1 0.129 (IoU 0.069). An early Mac "
        "feasibility FC-EF run reached 0.732 and is not part of the locked comparison. "
        "Fig. 2 summarizes the main validation F1 numbers.",
        indent=False,
    )
    figure(
        doc,
        FIG / "comparison_f1_bar.jpg",
        "Fig. 2. Validation change-class F1 for the reference, principal means, triad "
        "completion, and locked FC-EF+BCE (0.778).",
    )

    subhead(doc, "B.  Principal comparison")
    body(
        doc,
        "Table I is the three-seed BCE+Dice comparison. FC-EF mean±std validation F1 is "
        "0.718±0.011. FC-Siam-diff is 0.543±0.045. FC-Siam-conc is 0.773 on seed 1511 only. "
        "Seed spread is tighter for FC-EF (Fig. A2). At seed 1511, FC-EF+BCE climbs fastest "
        "on the validation F1 curve (Fig. A4).",
        indent=False,
    )
    table_title(doc, "TABLE I.  PRINCIPAL COMPARISON (BCE+DICE, AUG ON)")
    add_table(
        doc,
        ["Model", "Seed", "Val F1", "IoU", "P", "R", "Lat. (s)"],
        [
            ["FC-EF", "1511", "0.702", "0.541", "0.734", "0.673", "0.0015"],
            ["FC-EF", "42", "0.727", "0.571", "0.707", "0.749", "0.0014"],
            ["FC-EF", "123", "0.726", "0.570", "0.753", "0.701", "0.0014"],
            ["FC-Siam-diff", "1511", "0.503", "0.336", "0.674", "0.402", "0.0021"],
            ["FC-Siam-diff", "42", "0.606", "0.435", "0.608", "0.604", "0.0021"],
            ["FC-Siam-diff", "123", "0.521", "0.352", "0.941", "0.360", "0.0021"],
            ["FC-Siam-conc", "1511", "0.773", "0.630", "0.836", "0.719", "0.0028"],
        ],
        [1100, 650, 700, 650, 650, 650, 750],
    )

    subhead(doc, "C.  Ablations")
    body(
        doc,
        "Table II and Fig. 3 hold architecture and seed fixed (FC-EF, 1511). BCE alone beats "
        "BCE+Dice. Turning off geometric aug also helps versus BCE+Dice+aug.",
        indent=False,
    )
    table_title(doc, "TABLE II.  FC-EF ABLATIONS (SEED 1511)")
    add_table(
        doc,
        ["Setting", "Val F1", "IoU"],
        [
            ["BCE+Dice, aug on", "0.702", "0.541"],
            ["BCE only, aug on", "0.778", "0.636"],
            ["BCE+Dice, aug off", "0.745", "0.594"],
        ],
        [2800, 900, 900],
    )
    figure(
        doc,
        FIG / "ablation_f1_bar.jpg",
        "Fig. 3. FC-EF ablations at seed 1511. BCE-only is best.",
    )

    subhead(doc, "D.  Compute and locked test")
    body(
        doc,
        "Table III is the cost view at seed 1511. FC-EF is strongest on F1 and lowest on "
        "batch-1 latency. Full matrix: university CUDA lab (workdir /app/dl in configs), "
        "batch 4, 256×256 tiles. Verification host (2026-11-18): Darwin arm64, Python 3.13.5, "
        "PyTorch 2.10.0 (results/environment.json). Lab GPU SKU was not kept in committed "
        "summaries because checkpoints are gitignored.",
        indent=False,
    )
    table_title(doc, "TABLE III.  COMPUTE TRADE-OFF (SEED 1511)")
    add_table(
        doc,
        ["Model", "Params", "Train", "Batch-1", "Val F1"],
        [
            ["FC-EF (BCE)", "483k", "~51 min", "0.0014 s", "0.778"],
            ["FC-Siam-diff", "483k", "~47 min", "0.0021 s", "0.503"],
            ["FC-Siam-conc", "564k", "~48 min", "0.0028 s", "0.773"],
        ],
        [1400, 800, 900, 900, 800],
    )
    body(
        doc,
        "Locked setup: FC-EF + BCE + aug, seed 1511 (best val F1 0.778 at epoch 13). "
        "Test F1 / IoU / P / R = 0.780 / 0.639 / 0.770 / 0.790; boundary F1 0.356. No "
        "retuning after test. Test tracks validation closely, so the freeze looks stable.",
    )

    heading(doc, "VII.  ERROR ANALYSIS, LIMITATIONS, AND RESPONSIBLE USE")
    body(
        doc,
        "Fig. 4 shows locked-model validation examples. Large new buildings often come back. "
        "Small structures and boundaries stay weak.",
        indent=False,
    )
    figure(
        doc,
        FIG / "validation_gallery.jpg",
        "Fig. 4. Locked FC-EF+BCE validation examples (before, after, GT, prediction @ 0.5).",
        width=COL_W,
    )
    body(
        doc,
        "On the locked test, empty tiles are 1,113/2,048. Small-change tiles (≤256 labeled "
        "pixels) miss often (56/61; mean recall 0.083 vs 0.702 on large change). There are "
        "212 false alarms on empty tiles. Pooled test F1 is 0.780 while mean per-tile F1 is "
        "only 0.297—expected when many tiles are empty or tiny.",
    )
    body(
        doc,
        "Hypothesis outcomes. H1 is not supported for FC-Siam-diff (0.543 vs 0.718). "
        "FC-Siam-conc is competitive on one seed (0.773), so the negative result is about "
        "difference fusion at this capacity, not every Siamese design. H2 is not supported: "
        "BCE (0.778) beats BCE+Dice (0.702) on the ablation seed.",
    )
    body(
        doc,
        "Limits: base=16 caps peak F1 versus heavier public setups; stem overlap is only a "
        "partial geo check; Siam-conc and ablations are single-seed; data is Texas Google "
        "Earth only; small objects remain hard.",
    )
    body(
        doc,
        "Responsible use. LEVIR-CD is academic under Google Earth terms. We make no medical, "
        "safety-critical, surveillance, or deployment claims. Texas-only data may not "
        "transfer. The model can miss small buildings and false-alarm on empty tiles, so it "
        "needs human review if anyone reused it outside class.",
    )

    body(
        doc,
        "What worked for us: one shared dataloader and metric code made the comparison fair, "
        "and freezing before test stopped us from chasing the test set. Absolute difference "
        "was useful as a sanity floor. What did not work: difference fusion at this width, "
        "and BCE+Dice as a free lunch for imbalance. If we had more GPU time, the next honest "
        "step would be wider models under the same splits, not another uncontrolled search.",
    )

    heading(doc, "VIII.  CONCLUSION")
    body(
        doc,
        "Under a matched LEVIR-CD protocol, learned models beat absolute difference "
        "(0.129 vs ~0.70–0.78 F1). FC-EF beats FC-Siam-diff; BCE beats BCE+Dice in our "
        "ablation; the frozen FC-EF+BCE model reaches test F1 0.780. The research question "
        "gets a clear, partly negative answer instead of a forced win.",
        indent=False,
    )

    heading(doc, "ACKNOWLEDGMENT")
    body(
        doc,
        "Sneha Singh: data pipeline, leakage audit, abs-diff reference, and protocol. "
        "Danish Waseem: Siamese models, GPU matrix, locked test, and qualitative gallery. "
        "Ritika Mukesh Neema: FC-EF/aug config packaging and the Week 4 progress report. "
        "Rajesh Paruchuri: metrics and secondary scores, comparison table, latency/compute "
        "rows, environment.json, report figure packaging, failure-count tables, and the "
        "signed reproduction check (docs/REPRODUCTION.md).",
        indent=False,
    )
    body(
        doc,
        "Assistance disclosure. Claude was used only to clarify hard wording around ablation "
        "interpretation when seed and architecture results disagreed. Design, training, "
        "metrics, figures, and claims were produced and checked by the team against results/.",
    )

    heading(doc, "REFERENCES")
    refs = [
        "[1]  O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015, pp. 234–241.",
        "[2]  R. C. Daudt, B. Le Saux, and A. Boulch, “Fully convolutional Siamese networks for change detection,” in IEEE ICIP, 2018, pp. 4063–4067.",
        "[3]  H. Chen and Z. Shi, “A spatial-temporal attention-based method and a new dataset for remote sensing image change detection,” Remote Sensing, vol. 12, no. 10, art. 1662, 2020.",
        "[4]  Open-CD LEVIR-CD benchmark tables (context only). https://github.com/likyoo/open-cd",
        "[5]  I. Corley et al., “A change detection reality check,” arXiv:2402.06994, 2024 (context only).",
    ]
    for ref in refs:
        p = P(doc, ref, size=8, align="justify", before=0, after=2)
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.first_line_indent = Inches(-0.2)

    heading(doc, "APPENDIX")
    body(
        doc,
        "Extra training curves used in diagnosis are below. Artifact index: proposal "
        "docs/Data255_ProjectProposal.pdf; reproduction docs/REPRODUCTION.md; configs/; "
        "locked test results/locked_test/full_unet_bce_aug_seed1511_tile256/.",
        indent=False,
    )
    figure(
        doc,
        FIG / "locked_learning_curves.jpg",
        "Fig. A1. Locked FC-EF+BCE train/val loss and validation F1/IoU (best epoch 13).",
    )
    figure(
        doc,
        FIG / "seed_variability_boxplot.jpg",
        "Fig. A2. Seed variability for the principal BCE+Dice comparison.",
    )
    figure(
        doc,
        FIG / "locked_precision_recall.jpg",
        "Fig. A3. Validation precision/recall for the locked FC-EF+BCE run.",
    )
    figure(
        doc,
        FIG / "val_f1_trajectories.jpg",
        "Fig. A4. Validation F1 trajectories at seed 1511.",
    )

    doc.save(OUT)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    build()
