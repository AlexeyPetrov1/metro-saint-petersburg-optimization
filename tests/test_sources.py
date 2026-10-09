from hashlib import sha256
import json
from zipfile import ZipFile

import pandas as pd

from metro_data.sources import inventory


def test_archive_is_not_counted_as_an_independent_copy(tmp_path):
    raw = tmp_path / 'data/raw/metro'
    raw.mkdir(parents=True)
    (raw / 'example.txt').write_text('source', encoding='utf-8')
    with ZipFile(raw / 'bundle.zip', 'w') as z:
        z.write(raw / 'example.txt', 'example.txt')
    refs = tmp_path / 'docs/reference'
    refs.mkdir(parents=True)
    table, texts = inventory(tmp_path, raw, refs)
    member = table[table.relative_path == 'data/raw/metro/bundle.zip::example.txt'].iloc[0]
    assert member.read_status == 'duplicate_in_archive'
    assert member.sha256 == sha256(b'source').hexdigest()
    assert (raw / 'example.txt').read_bytes() == b'source'


def test_chat_photo_has_review_but_no_invented_binary_hash(tmp_path):
    raw = tmp_path / 'data/raw/metro'
    raw.mkdir(parents=True)
    refs = tmp_path / 'docs/reference'
    refs.mkdir(parents=True)
    (refs / 'attachment_sources.csv').write_text(
        'source_id,file_name,description,sha256\nA01,image.png,Table 3,legacy_digest\n')
    (refs / 'maintenance_photo_review.json').write_text(json.dumps({'sources': {
        'A01': {'source_locator': 'current chat image 2, Table 3', 'binary_available': False}
    }}))
    table, _ = inventory(tmp_path, raw, refs)
    row = table[table.source_id == 'A01'].iloc[0]
    assert row.source_kind == 'chat_image' and row.read_status == 'visually_reviewed_chat_image'
    assert pd.isna(row.sha256) and row.previous_sha256 == 'legacy_digest'
