"""Шаблон стилей Word для report.docx: стандартный reference.docx pandoc со шрифтами PT (кириллица),
страницей A4 и тонкими границами таблиц. Только стандартная библиотека.

Запуск: python docs/report/make_reference_docx.py → docs/report/reference.docx
"""
import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "reference.docx"
BORDERS = ('<w:tblBorders>' + "".join(
    f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
    for s in ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>')


def patch(name: str, xml: str) -> str:
    if name == "word/theme/theme1.xml":   # шрифты темы: заголовки — PT Sans, текст — PT Serif (латиница и кириллица)
        xml = re.sub(r'(<a:majorFont>\s*<a:latin typeface=")[^"]*"[^/]*/>', r'\1PT Sans"/>', xml)
        xml = re.sub(r'(<a:minorFont>\s*<a:latin typeface=")[^"]*"[^/]*/>', r'\1PT Serif"/>', xml)
        xml = re.sub(r'(<a:majorFont>.*?<a:font script="Cyrl" typeface=")[^"]*"', r'\1PT Sans"', xml, flags=re.S)
        xml = re.sub(r'(<a:minorFont>.*?<a:font script="Cyrl" typeface=")[^"]*"', r'\1PT Serif"', xml, flags=re.S)
    if name == "word/styles.xml":
        xml = xml.replace('w:ascii="Consolas" w:hAnsi="Consolas"', 'w:ascii="PT Mono" w:hAnsi="PT Mono"')
        m = re.search(r'(<w:style w:type="table"[^>]*w:styleId="Table".*?<w:tblPr>)', xml, flags=re.S)
        if m and "tblBorders" not in xml[m.end():m.end() + 600]:
            xml = xml[:m.end()] + BORDERS + xml[m.end():]
    if name == "word/document.xml":     # A4, поля 2 см: pandoc берёт страницу из sectPr шаблона
        page = ('<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1247" '
                'w:left="1247" w:header="567" w:footer="567" w:gutter="0"/>')
        xml = re.sub(r'<w:pgSz[^>]*/>', '', xml)
        xml = re.sub(r'<w:pgMar[^>]*/>', '', xml)
        xml = re.sub(r'<w:sectPr( [^>]*)?>', lambda m: m.group(0) + page, xml, count=1)
        xml = re.sub(r'<w:sectPr( [^>]*)?/>', lambda m: m.group(0)[:-2] + '>' + page + '</w:sectPr>', xml, count=1)
    return xml


def main() -> None:
    src = HERE / "_default_reference.docx"
    with open(src, "wb") as f:
        subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], stdout=f, check=True)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".xml"):
                data = patch(item.filename, data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)
    src.unlink()
    print(f"{OUT.relative_to(HERE.parents[1])}: шрифты PT, A4, границы таблиц")


if __name__ == "__main__":
    main()
