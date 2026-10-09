from datetime import date

import pandas as pd
import pytest

from metro_calendar import DEFAULT_CALENDAR_SOURCE, ProductionCalendar, calendar_day, load_calendar


def days():
    return pd.DataFrame(dict(date=['2026-09-09', '2026-09-12', '2026-09-13', '2026-09-14'],
        is_workday=[False, True, False, False], holiday_name=['Holiday', '', '', ''],
        is_transfer_dayoff=[False, False, False, True], is_shortened=[False, True, False, False]))


def test_absent_or_uncovered_calendar_never_infers_holidays_or_school_terms():
    assert calendar_day('2026-09-09')['day_kind'] == 'unknown'
    assert calendar_day('2026-09-09')['school_period'] == 'unknown'
    assert calendar_day('2027-01-01', ProductionCalendar(days()))['day_kind'] == 'unknown'
    supplied = ProductionCalendar(days().drop(columns='is_shortened'))
    assert calendar_day('2026-09-12', supplied)['is_shortened'] is None


def test_calendar_distinguishes_working_weekend_holiday_and_regular_weekend():
    calendar = ProductionCalendar(days())
    assert calendar_day('2026-09-09', calendar)['day_kind'] == 'holiday'
    shortened = calendar_day('2026-09-12', calendar)
    assert shortened['day_kind'] == 'workday' and shortened['is_shortened']
    assert calendar_day('2026-09-13', calendar)['day_kind'] == 'weekend'
    assert calendar_day('2026-09-14', calendar)['day_kind'] == 'holiday'


def test_supplied_calendar_has_provenance_and_explicit_availability_assumption():
    calendar = load_calendar()
    assert calendar.source_path == str(DEFAULT_CALENDAR_SOURCE.resolve())
    assert len(calendar.days) == 365 and len(calendar.sha256) == 64
    assert calendar_day('2026-01-01', calendar)['day_kind'] == 'holiday'
    assert calendar_day('2026-03-09', calendar)['day_kind'] == 'holiday'
    assert calendar_day('2026-09-30', calendar)['day_kind'] == 'workday'
    assert 'assumed available in advance' in calendar.manifest()['availability_assumption']
    assert 'ingested_at' not in calendar.manifest()


def test_school_period_requires_explicit_supplied_period_table():
    periods = pd.DataFrame([dict(date_start='2026-09-01', date_end='2026-12-31', school_period='term')])
    calendar = ProductionCalendar(days(), school_periods=periods)
    assert calendar_day('2026-09-13', calendar)['school_period'] == 'term'
    assert calendar_day('2027-01-01', calendar)['school_period'] == 'unknown'


def test_lookup_uses_moscow_civil_date_for_timezone_timestamps():
    value = calendar_day(pd.Timestamp('2026-09-11T22:00:00Z'), ProductionCalendar(days()))
    assert value['date'] == '2026-09-12' and value['day_kind'] == 'workday'


@pytest.mark.parametrize('damage', ['duplicates', 'missing_flag', 'string_flag', 'overlapping_school'])
def test_calendar_rejects_ambiguous_or_malformed_inputs(damage):
    frame, periods = days(), None
    if damage == 'duplicates':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif damage == 'missing_flag':
        frame['is_workday'] = frame.is_workday.astype('boolean')
        frame.loc[0, 'is_workday'] = pd.NA
    elif damage == 'string_flag':
        frame['is_workday'] = frame.is_workday.astype(str)
    else:
        periods = pd.DataFrame([dict(date_start='2026-09-01', date_end='2026-09-30', school_period='term'),
                                dict(date_start='2026-09-15', date_end='2026-10-01', school_period='break')])
    with pytest.raises(ValueError):
        ProductionCalendar(frame, school_periods=periods)
