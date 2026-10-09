from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pandas as pd
from PIL import Image
from pypdf import PdfReader

from .pipeline import sniff_format


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def decode_text(raw):
    for encoding in ['utf-8-sig', 'cp1251']:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError('Unknown text encoding')


def inventory(root, raw_dir, ref_dir):
    old_path = ref_dir / 'manifest_sources.csv'
    old = pd.read_csv(old_path).to_dict('records') if old_path.exists() else []
    by_name = {r['file_name']: r for r in old}
    files = sorted(p for p in raw_dir.rglob('*') if p.is_file())
    references = sorted(p for p in ref_dir.rglob('*') if p.is_file())
    raw_hashes = {file_hash(p): p.relative_to(root).as_posix() for p in files}
    rows, texts = [], {}
    for p in files + references:
        data = p.read_bytes()
        digest, fmt = sha256(data).hexdigest(), sniff_format(data)
        rel = p.relative_to(root).as_posix()
        previous = by_name.get(p.name, {}) if p in files else {}
        source_id = previous.get('source_id', ('REF_' if p in references else 'RAW_') + digest[:12])
        if fmt == 'pptx' and p in files:
            source_id = 'SOURCE_PRESENTATION'
        url = previous.get('url', '')
        if source_id == 'SOURCE_PRESENTATION':
            url = 'https://docs.google.com/presentation/d/1oOxBkuXpDgqjKwI8-DZb_f50HZTN_wwg/edit'
        base = dict(source_id=source_id, relative_path=rel, source_kind='reference' if p in references else 'file',
                    actual_format=fmt, size_bytes=len(data), sha256=digest, source_url=url,
                    previous_sha256=previous.get('sha256', ''),
                    version_status='unchanged' if previous.get('sha256') == digest else
                    'changed' if previous else 'not_in_previous_manifest',
                    purpose='reference registry' if p in references else
                    'passenger entries' if fmt == 'xlsx' else 'transport work' if fmt == 'html' else
                    'original archive' if fmt == 'zip' else 'line and rolling stock reference',
                    processing_method='', read_status='', read_result='')
        try:
            if fmt in ('docx', 'pptx', 'xlsx', 'zip'):
                with ZipFile(BytesIO(data)) as z:
                    bad = z.testzip()
                    if bad: raise ValueError(f'CRC error in {bad}')
                    if fmt == 'zip':
                        base.update(processing_method='ZIP CRC and member SHA-256', read_status='archive_verified',
                                    read_result=f'{len(z.infolist())} members; duplicates excluded from extraction')
                        for member in sorted(z.infolist(), key=lambda m: m.filename):
                            if member.is_dir(): continue
                            content = z.read(member)
                            h = sha256(content).hexdigest()
                            rows.append(dict(source_id=source_id+'#'+member.filename,
                                             relative_path=rel+'::'+member.filename, source_kind='archive_member',
                                             actual_format=sniff_format(content), size_bytes=len(content), sha256=h,
                                             source_url=url, purpose='archived original', processing_method='SHA-256 comparison',
                                             read_status='duplicate_in_archive' if h in raw_hashes else 'not_extracted',
                                             read_result=raw_hashes.get(h, 'No identical extracted file; review required')))
                    elif fmt == 'xlsx':
                        e = ET.fromstring(z.read('xl/workbook.xml'))
                        names = [n.attrib['name'] for n in e.iter() if n.tag.endswith('}sheet')]
                        base.update(processing_method='openpyxl; validated schema', read_status='pending_extraction',
                                    read_result=json.dumps({'sheets': names}, ensure_ascii=False))
                    else:
                        members = ['word/document.xml'] if fmt == 'docx' else sorted(
                            n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml'))
                        blocks = []
                        for n in members:
                            e = ET.fromstring(z.read(n))
                            blocks.append(dict(locator=n, text=' '.join(t.text or '' for t in e.iter() if t.tag.endswith('}t'))))
                        media = sorted(n for n in z.namelist() if n.startswith(('word/media/', 'ppt/media/')))
                        texts[source_id] = dict(source_path=rel, source_url=url,
                                                text='\n'.join(b['text'] for b in blocks), blocks=blocks)
                        base.update(processing_method='OOXML text and embedded image metadata; no OCR',
                                    read_status='text_read_images_unreviewed' if media else 'text_read',
                                    read_result=f'{len(texts[source_id]["text"])} text characters; {len(media)} embedded images')
                        for n in media:
                            content = z.read(n)
                            item = dict(source_id=source_id+'#'+n, relative_path=rel+'::'+n,
                                        source_kind='embedded_image', actual_format=sniff_format(content),
                                        size_bytes=len(content), sha256=sha256(content).hexdigest(), source_url=url,
                                        purpose='image within document', processing_method='Pillow metadata and integrity; no OCR')
                            try:
                                with Image.open(BytesIO(content)) as im:
                                    info = f'{im.format} {im.width}x{im.height}; numerical content unverified'
                                    im.verify()
                                item.update(read_status='image_verified_content_unreviewed', read_result=info)
                            except Exception as e:
                                item.update(read_status='needs_visual_review', read_result=str(e))
                            rows.append(item)
            elif fmt == 'pdf':
                pdf = PdfReader(BytesIO(data))
                blocks = [dict(locator=f'page {i}', text=page.extract_text() or '')
                          for i, page in enumerate(pdf.pages, 1)]
                text = '\n'.join(b['text'] for b in blocks)
                texts[source_id] = dict(source_path=rel, source_url=url, text=text, blocks=blocks)
                base.update(processing_method='pypdf page and text read; no OCR',
                            read_status='text_read' if text.strip() else 'scanned_pdf_needs_visual_review',
                            read_result=f'{len(pdf.pages)} pages; {len(text.strip())} text characters')
            elif fmt in ('png', 'jpeg'):
                with Image.open(BytesIO(data)) as im:
                    result = f'{im.format} {im.width}x{im.height}; numerical content unverified'
                    im.verify()
                base.update(processing_method='Pillow integrity; no OCR', read_status='image_verified_content_unreviewed',
                            read_result=result)
            elif fmt == 'html':
                base.update(processing_method='HTMLParser with declared charset; line 1 only', read_status='pending_extraction')
            elif fmt == 'text':
                text, encoding = decode_text(data)
                texts[source_id] = dict(source_path=rel, source_url=url, text=text)
                base.update(processing_method=f'text decode: {encoding}', read_status='text_read',
                            read_result=f'{len(text)} characters')
            else:
                base.update(processing_method='signature detection', read_status='unsupported',
                            read_result='Unsupported actual format; no numerical extraction')
        except Exception as e:
            base.update(read_status='read_error', read_result=f'{type(e).__name__}: {e}')
        rows.append(base)
    attached = ref_dir / 'attachment_sources.csv'
    review_path = ref_dir / 'maintenance_photo_review.json'
    reviewed = json.loads(review_path.read_text(encoding='utf-8')).get('sources', {}) if review_path.exists() else {}
    if attached.exists():
        for item in pd.read_csv(attached).to_dict('records'):
            if item['sha256'] in raw_hashes: continue
            photo = reviewed.get(item['source_id'])
            rows.append(dict(source_id=item['source_id'], relative_path=item['file_name'],
                             source_kind='chat_image' if photo else 'missing_attachment',
                             actual_format='unavailable', size_bytes=None, sha256=None,
                             previous_sha256=item['sha256'], purpose=item['description'],
                             processing_method='manual visual review of chat image' if photo else 'prepared registry only; original unavailable',
                             read_status='visually_reviewed_chat_image' if photo else 'missing',
                             read_result=photo['source_locator'] + '; original bytes/size/SHA-256 unavailable' if photo else
                             'Not present locally or in the linked Drive folders'))
    table = pd.DataFrame(rows).sort_values('relative_path').reset_index(drop=True)
    table['size_bytes'] = table.size_bytes.astype('Int64')
    return table, texts
