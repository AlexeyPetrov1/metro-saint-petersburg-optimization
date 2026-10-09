#!/usr/bin/env bash
# Сборка отчёта: рисунки → сверка чисел → report.pdf и report.docx → страницы PDF в PNG (pages/, для просмотра).
# Запуск из корня репозитория или отсюда: bash docs/report/build.sh [--no-pages]
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
PY="${ROOT}/.venv/bin/python"

echo "== рисунки отчёта"
(cd "$ROOT" && "$PY" docs/report/make_figures.py)

echo "== сверка чисел с источниками"
(cd "$ROOT" && "$PY" docs/report/check_numbers.py)

if command -v xelatex >/dev/null; then ENGINE=xelatex; else ENGINE=tectonic; fi
COMMON=(report.md --metadata-file=metadata.yaml --lua-filter=nbsp.lua --resource-path=.:"$ROOT" --toc)

# PDF: разделы нумерует LaTeX (приложения — буквами через \startappendix), crossref даёт только ссылки
echo "== PDF ($ENGINE)"
pandoc "${COMMON[@]}" --filter pandoc-crossref --lua-filter=code_path.lua --lua-filter=keep.lua --number-sections --pdf-engine="$ENGINE" -H preamble.tex \
  -o report.pdf 2> >(grep -v "accessing absolute path" >&2)

# docx: разделы и подписи нумерует pandoc-crossref (приложения — атрибутом label="А")
echo "== проверка полей PDF"
"$PY" check_margins.py report.pdf

echo "== docx"
[ -f reference.docx ] || "$PY" make_reference_docx.py
pandoc "${COMMON[@]}" -M numberSections=true --filter pandoc-crossref --reference-doc=reference.docx -o report.docx

if [ "${1:-}" != "--no-pages" ]; then
  echo "== страницы PDF → pages/"
  rm -rf pages && mkdir -p pages
  pdftoppm -r 70 -png report.pdf pages/p
  "$PY" - <<'PYEOF'
from pathlib import Path
from PIL import Image
pages = sorted(Path("pages").glob("p-*.png"))
for i in range(0, len(pages), 4):
    ims = [Image.open(p) for p in pages[i:i + 4]]
    w = max(max(im.size) for im in ims)          # ячейка — квадрат по большей стороне: альбомные страницы не обрезаются
    sheet = Image.new("RGB", (2 * w, 2 * w), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % 2) * w + (w - im.size[0]) // 2, (k // 2) * w + (w - im.size[1]) // 2))
    sheet.save(f"pages/sheet-{i // 4 + 1:02d}.png")
print(f"страниц: {len(pages)}")
PYEOF
fi
echo "готово: docs/report/report.pdf, docs/report/report.docx"
