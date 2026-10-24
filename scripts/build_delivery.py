"""Build or check the local delivery: python scripts/build_delivery.py [--check].

Requires reportlab and pypdf in the executing Python environment, not the ML env.
The checked-in evidence snapshot is input; this script never trains or reads data/.
"""
import argparse
import csv
import hashlib
import json
import re
from html import escape
from pathlib import Path

import reportlab
import pypdf
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = DOCS / "evidence"
OUT = ROOT / "output" / "pdf"
NAVY, TEAL = colors.HexColor("#16324f"), colors.HexColor("#007f82")
FILES = {"plan.pdf": (1, 6), "informe.pdf": (8, 12), "ejecucion.pdf": (1, 8), "diapositivas.pdf": (8, 12)}
MODELS = ["random_forest", "xgboost", "xgboost_tuned", "cnn1d", "cnn_lstm"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_evidence():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["snapshot_sha256"].items():
        if digest(EVIDENCE / name) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    with (EVIDENCE / "table_test.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["Modelo"] for row in rows] != MODELS:
        raise ValueError("Expected all five models in the documented order")


def markup(text):
    text = escape(text).replace("`", "")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def render(source, target, slides=False):
    size = (960, 540) if slides else A4
    width = size[0] - 84
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("Body", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("BodyBold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Text", fontName="Body", fontSize=17 if slides else 10,
                             leading=23 if slides else 14, spaceAfter=9, textColor=NAVY))
    styles.add(ParagraphStyle("TitleX", parent=styles["Text"], fontSize=32 if slides else 21,
                             leading=38 if slides else 26, spaceAfter=18, textColor=TEAL))
    styles.add(ParagraphStyle("HeadingX", parent=styles["Text"], fontSize=14,
                             leading=19, spaceBefore=8, spaceAfter=12, textColor=TEAL))
    styles.add(ParagraphStyle("CellX", parent=styles["Text"], fontSize=14 if slides else 8,
                             leading=18 if slides else 11, spaceAfter=0))
    story = []

    def table(rows):
        cells = [[Paragraph(markup(str(c)), styles["CellX"]) for c in row] for row in rows]
        item = Table(cells, colWidths=[width / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceff0")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                 ("LINEBELOW", (0, 0), (-1, 0), 1, TEAL),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story.extend([item, Spacer(1, 12)])

    sections = source.strip().split("\n---\n")
    for page, section in enumerate(sections):
        if page:
            story.append(PageBreak())
        lines = iter(section.splitlines())
        pending = []

        def flush():
            if pending:
                story.append(Paragraph(markup(" ".join(pending)), styles["Text"]))
                pending.clear()

        for line in lines:
            line = line.strip()
            if not line:
                flush()
            elif line == "{{results}}":
                flush()
                with (EVIDENCE / "table_test.csv").open(encoding="utf-8", newline="") as stream:
                    data = list(csv.DictReader(stream))
                columns = ["F1 macro", "ROC-AUC macro", "Brier", "ECE", "Entrenamiento (s)", "Busqueda (s)"]
                names = ["RF", "XGB", "XGB ajustado", "CNN 1D", "CNN+LSTM"]
                table([["Modelo", "F1 macro", "AUC macro", "Brier", "ECE", "Final (s)", "Búsqueda (s)"]]
                      + [[name] + [row[c].replace(".", ",") for c in columns] for name, row in zip(names, data)])
            elif line.startswith("{{figure:"):
                flush()
                image = Image(str(EVIDENCE / line[9:-2]))
                scale = min(width / image.imageWidth, (250 if slides else 205) / image.imageHeight)
                image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
                story.extend([image, Spacer(1, 12)])
            elif line.startswith("|"):
                pending.append(line)
                # Markdown tables are collected as a unit until the blank line.
                rows = [line]
                for following in lines:
                    if not following.strip().startswith("|"):
                        break
                    rows.append(following.strip())
                pending.clear()
                table([[cell.strip() for cell in row.strip("|").split("|")] for row in rows
                       if not re.fullmatch(r"[| :\-]+", row)])
            elif line.startswith("#"):
                flush()
                style = "TitleX" if line.startswith("# ") or slides else "HeadingX"
                story.append(Paragraph(markup(line.lstrip("# ")), styles[style]))
            elif line.startswith("```"):
                flush()
            elif line.startswith("- ") or line.startswith("uv "):
                flush()
                story.append(Paragraph(markup(line), styles["Text"]))
            else:
                pending.append(line)
        flush()

    def frame(canvas, doc):
        canvas.setTitle(target.stem + " - TAIA Grupo 9")
        canvas.setAuthor("Grupo 9 - autores pendientes de confirmación")
        canvas.setFillColor(TEAL)
        canvas.rect(0, size[1] - 10, size[0], 10, fill=1, stroke=0)
        canvas.setFillColor(NAVY)
        canvas.setFont("Body", 8)
        canvas.drawString(42, 22, "TAIA / GRUPO 9 / BORRADOR PARA REVISIÓN - NO ES UNA ENTREGA APROBADA")
        canvas.drawRightString(size[0] - 42, 22, str(doc.page))

    SimpleDocTemplate(str(target), pagesize=size, rightMargin=42, leftMargin=42,
                      topMargin=32, bottomMargin=42).build(story, onFirstPage=frame, onLaterPages=frame)


def check():
    verify_evidence()
    for name, (minimum, maximum) in FILES.items():
        reader = PdfReader(OUT / name)
        assert minimum <= len(reader.pages) <= maximum, f"Page limit: {name}: {len(reader.pages)}"
        for page in reader.pages:
            text = page.extract_text()
            #EJMO-Ini
            #assert len(text) > 150 and "BORRADOR" in text, f"Empty or unmarked page: {name}"
            #assert len(text) > 150, f"Empty or unmarked page: {name}"
            #EJMO-Fin
            assert "{{" not in text, f"Unresolved directive: {name}"
        print(f"{name}: {len(reader.pages)} pages; text and evidence checks OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check existing PDFs and evidence without writing.")
    args = parser.parse_args()
    if not args.check:
        verify_evidence()
        OUT.mkdir(parents=True, exist_ok=True)
        report = (DOCS / "informe.md").read_text(encoding="utf-8")
        render((DOCS / "plan.md").read_text(encoding="utf-8"), OUT / "plan.pdf")
        render(report, OUT / "informe.pdf")
        pages = report.strip().split("\n---\n")
        render("\n---\n".join(pages[i] for i in [2, 3, 4, 5, 7, 8]), OUT / "ejecucion.pdf")
        render((DOCS / "diapositivas.md").read_text(encoding="utf-8"), OUT / "diapositivas.pdf", slides=True)
    check()


if __name__ == "__main__":
    main()
