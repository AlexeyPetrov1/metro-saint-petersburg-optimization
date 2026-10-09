import json

import pytest

from metro48_dashboard import render_dashboard, write_dashboard


def comparison():
    return dict(reference={'to1_resource_equivalent_rub': 1000}, arms=[dict(arm='GRU',
        train_km=100, mean_km_per_fixed_train=100 / 68, to1_resource_equivalent_rub=980,
        saving_percent=2)])


def test_viewer_is_interactive_and_preserves_evidence_limits():
    html = render_dashboard(comparison(), {'scores': []})
    assert 'id="arm-select"' in html
    assert 'addEventListener' in html
    assert 'конечный эпизод' in html
    assert '30 минут' in html and '15 минут' in html
    assert 'Полный день: FAIL' in html
    assert 'Денежная экономия неизвестна' in html
    assert '980' in html


def test_payload_cannot_close_script_element():
    source = comparison()
    source['arms'][0]['arm'] = '</script><img src=x onerror=alert(1)>'
    html = render_dashboard(source, {'scores': []})
    assert '</script><img' not in html
    assert '\\u003c/script\\u003e' in html


def test_unknown_money_is_kept_unknown():
    html = render_dashboard({'reference': {}, 'arms': [{'arm': 'GRU'}]}, {'scores': []})
    assert 'null' in html
    assert 'неизвестно' in html


def test_common_profile_ml_is_separate_from_original_profile():
    source = comparison()
    source['forecast_comparison'] = {'OLD_ML_SHARED_PROFILE': {'wape_percent': 7.713}}
    html = render_dashboard(source, {'scores': [{'scope': 'all', 'model': 'OLD_ML',
        'stage': 'development', 'wape_percent': 7.308}]})
    assert 'OLD_ML_SHARED_PROFILE' in html and '7.713' in html and '7.308' in html
    assert 'исходный профиль' in html and 'общий профиль' in html


def test_existing_viewer_is_never_overwritten(tmp_path):
    path = write_dashboard(tmp_path, comparison(), {'scores': []})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_dashboard(tmp_path, comparison(), {'scores': []})
    assert path.read_bytes() == original
