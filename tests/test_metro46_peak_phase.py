import copy

import pytest

from metro46_peak_phase import replace_peak_phases, rebuild_with_peak_phase


@pytest.fixture
def calendars():
    return {terminal: [dict(start_second=7200, end_second=14400,
        headway_seconds=115, phase_seconds=phase), dict(start_second=39600,
        end_second=46800, headway_seconds=115, phase_seconds=phase)]
        for terminal, phase in [('north', 0), ('south', 57)]}


def test_changes_only_the_selected_peak_without_mutating_input(calendars):
    saved = copy.deepcopy(calendars)
    changed = replace_peak_phases(calendars, 7200, {'north': 30, 'south': 87})
    assert calendars == saved
    assert changed['north'][0]['phase_seconds'] == 30
    assert changed['south'][0]['phase_seconds'] == 87
    assert changed['north'][1] == saved['north'][1]
    assert changed['south'][1] == saved['south'][1]


@pytest.mark.parametrize('phase', [True, -1, 115, 30.5])
def test_phase_must_fit_the_actual_headway(calendars, phase):
    with pytest.raises(ValueError):
        replace_peak_phases(calendars, 7200, {'north': phase, 'south': 57})


def test_unknown_terminal_is_rejected(calendars):
    with pytest.raises(ValueError, match='terminal'):
        replace_peak_phases(calendars, 7200, {'missing': 0})


def test_missing_peak_is_rejected(calendars):
    with pytest.raises(ValueError, match='peak'):
        replace_peak_phases(calendars, 0, {'north': 0, 'south': 57})


def _peak_calendars(*args):
    return {'north': [dict(start_second=7200, end_second=14400,
        headway_seconds=115, phase_seconds=0)]}


def _builder(data, as_of, end, *, headway=None, peak_calendar=True, deadline=None):
    return _peak_calendars(data, as_of, end), (headway, peak_calendar, deadline)


def test_adapter_preserves_builder_globals_and_keyword_defaults():
    original = _builder.__globals__['_peak_calendars']
    changed, arguments = rebuild_with_peak_phase({}, 3600, 65400, {'north': 30},
        peak_start_second=7200, headway=115, deadline=50, builder=_builder)
    assert changed['north'][0]['phase_seconds'] == 30
    assert arguments == (115, True, 50)
    assert _builder.__globals__['_peak_calendars'] is original
    assert _builder({}, 3600, 65400)[0]['north'][0]['phase_seconds'] == 0
