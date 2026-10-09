"""Проверка вёрстки PDF: ни одно слово не выходит за поля страницы (правое и левое, допуск 3 pt — выступ глифов).

Запуск: python docs/report/check_margins.py docs/report/report.pdf  — код возврата 1 при нарушении.
Поля — из metadata.yaml (geometry): левое 2,2 см, правое 2 см.
"""
import re
import subprocess
import sys

CM = 72 / 2.54
LEFT, RIGHT, TOL = 2.2 * CM, 2.0 * CM, 3.0   # допуск: у выключенных строк глиф выступает на ~2,3 pt


def main(path: str) -> int:
    html = subprocess.run(["pdftotext", "-bbox", path, "-"], capture_output=True, text=True, check=True).stdout
    bad = []
    for n, page in enumerate(re.findall(r"<page (.*?)</page>", html, flags=re.S), 1):
        width = float(re.search(r'width="([\d.]+)"', page).group(1))
        for x0, x1, word in re.findall(r'<word xMin="([\d.]+)" yMin="[\d.]+" xMax="([\d.]+)" yMax="[\d.]+">(.*?)</word>', page):
            if float(x1) > width - RIGHT + TOL or float(x0) < LEFT - TOL:
                bad.append((n, word, float(x0), float(x1)))
    for n, word, x0, x1 in bad[:20]:
        print(f"стр. {n}: «{word}» за полем (x {x0:.1f}–{x1:.1f})")
    print(f"проверка полей: {'OK' if not bad else f'{len(bad)} слов за полями'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "report.pdf"))
