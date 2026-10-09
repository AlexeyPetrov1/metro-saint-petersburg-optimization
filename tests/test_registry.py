from pathlib import Path
import shutil

import pandas as pd
import pytest

from metro_data.build import build_registry
from metro_data.pipeline import SourceError

ROOT = Path(__file__).resolve().parents[1]


TEXTS = {'D13': {'text': 'Время полного оборота по Л1 (туда обратно) = 99 мин\n'
                    'Резервные составы находятся в депо "Автово" и "Северное" '
                    'в идеале их 4 состава по два в каждом депо время выхода на Линию 15-20 мин'}}


def copy_refs(target, with_review=False):
    for name in ['parameters_draft.json', 'maintenance_rules.csv']:
        shutil.copyfile(ROOT / 'docs/reference' / name, target / name)
    if with_review:
        shutil.copyfile(ROOT / 'docs/reference/maintenance_photo_review.json',
                        target / 'maintenance_photo_review.json')


def test_registry_preserves_unknowns_and_unverified_maintenance(tmp_path):
    copy_refs(tmp_path)
    registry, rules = build_registry(tmp_path, TEXTS)
    p = registry['parameters']
    assert p['reserve_actual_available']['value'] is None
    assert p['energy_price']['status'] == 'unknown'
    assert p['reference_roundtrip']['value'] is None
    assert p['reference_roundtrip']['status'] == 'withdrawn'
    assert p['reserve_file_release_range']['value'] == [900, 1200]
    assert p['release_delay']['value'] is None and p['release_delay']['status'] == 'conflict'
    assert len(rules) == 14
    assert rules.source_verification.eq('reference_only').all()
    assert not rules.upper_is_operating_target.any()


def test_withdrawn_roundtrip_is_not_required_in_source_text(tmp_path):
    copy_refs(tmp_path)
    texts = {'D13': {'text': TEXTS['D13']['text'].split('\n', 1)[1]}}
    registry, _ = build_registry(tmp_path, texts)
    assert registry['parameters']['reference_roundtrip']['value'] is None


def test_photo_review_preserves_asymmetric_tolerance_and_scoped_notes():
    registry, rules = build_registry(ROOT / 'docs/reference', TEXTS)
    assert rules.source_verification.eq('visual_review_chat_attachment').all()
    assert rules.status.eq('source_observed').all()
    balti = rules[rules.series == '81-725/726/727'].set_index('maintenance_type')
    assert tuple(balti.loc['ТО-2', ['lower', 'nominal', 'upper']]) == (30000, 35000, 43000)
    assert 'салонных воздушных фильтров' in balti.loc['ТО-1', 'note']
    assert 'салонных воздушных фильтров' not in balti.loc['ТО-2', 'note']
    assert 'осмотру и наладке' in balti.loc['ПДР-1', 'note']
    assert 'осмотру и наладке' not in balti.loc['ПР-1', 'note']
    eo = registry['parameters']['eo_inspection_validity']
    assert (eo['value'], eo['lower'], eo['upper'], eo['unit']) == (20, 16, 24, 'h')
    assert registry['parameters']['eo_frequency']['value'] == 1
    assert not eo['upper_is_operating_target']


def test_changed_csv_cannot_keep_photo_verified_status(tmp_path):
    copy_refs(tmp_path, with_review=True)
    rules = pd.read_csv(tmp_path / 'maintenance_rules.csv')
    rules.loc[(rules.series == '81-725/726/727') & (rules.maintenance_type == 'ТО-2'), 'upper'] = 42000
    rules.to_csv(tmp_path / 'maintenance_rules.csv', index=False)
    with pytest.raises(SourceError, match='Photo review disagrees'):
        build_registry(tmp_path, TEXTS)
