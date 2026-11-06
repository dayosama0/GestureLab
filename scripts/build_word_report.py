"""Format the completed-work Markdown report as an editable 14 pt Word file.

Export the resulting DOCX with scripts/export_word_report.ps1, using Word.
"""
import re
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Cm(21.59), Cm(27.94)
section.top_margin = section.bottom_margin = Cm(1.6)
section.left_margin = section.right_margin = Cm(1.7)
section.header_distance = section.footer_distance = Cm(0.7)
for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption']:
    style = doc.styles[name]
    style.font.name = 'Times New Roman'
    style.font.size = Pt(14)
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    fonts = style.element.get_or_add_rPr().rFonts
    for attr in list(fonts.attrib):
        if 'Theme' in attr:
            del fonts.attrib[attr]
    for key in ['ascii', 'hAnsi', 'cs', 'eastAsia']:
        fonts.set(qn('w:' + key), 'Times New Roman')
    for border in list(style.element.xpath('.//w:pBdr')):
        border.getparent().remove(border)
    fmt = style.paragraph_format
    fmt.line_spacing = 1.0
    fmt.space_after = Pt(4)
    fmt.space_before = Pt(0)
for name in ['Heading 1', 'Heading 2']:
    doc.styles[name].font.bold = True
    doc.styles[name].paragraph_format.space_before = Pt(8)
    doc.styles[name].paragraph_format.keep_with_next = True
doc.styles['Title'].font.bold = True
doc.styles['Caption'].font.bold = False
doc.styles['Caption'].font.italic = False
doc.styles['Title'].paragraph_format.space_after = Pt(6)
doc.core_properties.title = 'GestureLab Assignment 2 Work Report'
doc.core_properties.subject = 'Hand gesture classification application and measured experiments'
doc.core_properties.author = 'Turaliyev Ali'
doc.core_properties.keywords = 'GestureLab, Assignment 2, computer vision'
doc.add_paragraph('GestureLab Assignment 2 Work Report', 'Title')

def paragraph(text, style=None):
    p = doc.add_paragraph(text, style)
    p.paragraph_format.widow_control = True
    return p

def table(rows):
    n = len(rows[0])
    widths = {2: [3.0, 15.19], 3: [6.6, 5.3, 6.29],
              4: [7.0, 3.73, 3.73, 3.73],
              5: [6.2, 3.0, 2.7, 3.3, 2.99]}[n]
    # Preserve room for model names and use short, explicit numeric headers.
    if n == 5:
        rows[0] = ['Model / run', 'Original accuracy', 'Original F1', 'Fresh accuracy', 'Fresh F1']
    t = doc.add_table(rows=0, cols=n)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for c, width in zip(t.columns, widths):
        c.width = Cm(width)
    for ri, values in enumerate(rows):
        cells = t.add_row().cells
        for j, (cell, value) in enumerate(zip(cells, values)):
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = value
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1
            if j > 0 and n >= 4:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(14)
                run.bold = ri == 0
            props = cell._tc.get_or_add_tcPr()
            borders = OxmlElement('w:tcBorders')
            for edge in ['top', 'left', 'bottom', 'right']:
                e = OxmlElement('w:' + edge)
                for k, v in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                    e.set(qn('w:' + k), v)
                borders.append(e)
            props.append(borders)
            if ri == 0:
                fill = OxmlElement('w:shd')
                fill.set(qn('w:fill'), 'E8E8E8')
                props.append(fill)
        trpr = t.rows[-1]._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            trpr.append(OxmlElement('w:tblHeader'))
    paragraph('').paragraph_format.space_after = Pt(0)

lines = (ROOT / 'report/report.md').read_text(encoding='utf-8').splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line or line == '---':
        i += 1
        continue
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            vals = [s.strip() for s in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+', v) for v in vals):
                rows.append(vals)
            i += 1
        table(rows)
        continue
    if line.startswith('#'):
        level = len(line) - len(line.lstrip('#'))
        text = line.lstrip('# ').replace('1. ', '1 ', 1) if level == 1 else line.lstrip('# ')
        text = re.sub(r'^(\d+)\. ', r'\1 ', text)
        text = text.replace('Modern addition: ', '').replace(',', '')
        paragraph(text, 'Heading 1' if level == 1 else 'Heading 2')
    elif line.startswith('![Prepared input]'):
        p = paragraph('')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for filename in ['input_mobilenetv4_finetuned.png', 'gradcam_mobilenetv4_finetuned.png']:
            p.add_run().add_picture(str(ROOT / 'artifacts/verification' / filename), width=Cm(5.0))
            p.add_run('  ')
    elif line.startswith('![Actual Grad-CAM]'):
        pass
    elif line.startswith('!['):
        alt, rel = re.match(r'!\[(.*?)\]\((.*?)\)', line).groups()
        p = paragraph('')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str((ROOT / 'report' / rel).resolve()), width=Cm(17.9))
        paragraph(alt, 'Caption')
    else:
        paragraph(line)
    i += 1

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = footer.add_run()
r.font.name, r.font.size = 'Times New Roman', Pt(14)
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
r._r.addnext(field)
out = ROOT / 'report/report.docx'
# Explicit run formatting also keeps alternative Windows Word renderers from
# resolving a heading theme font instead of the requested family.
for p in doc.paragraphs:
    for run in p.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        fonts = run._element.get_or_add_rPr().rFonts
        for attr in list(fonts.attrib):
            if 'Theme' in attr:
                del fonts.attrib[attr]
        for key in ['ascii', 'hAnsi', 'cs', 'eastAsia']:
            fonts.set(qn('w:' + key), 'Times New Roman')
doc.save(out)
print(out)
