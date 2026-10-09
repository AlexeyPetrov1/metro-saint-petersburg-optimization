"""Supplied production calendar, with explicit prior-availability assumptions."""
from datetime import date
import hashlib
from pathlib import Path

import pandas as pd


DEFAULT_CALENDAR_SOURCE = Path(__file__).resolve().parent / 'data/processed/metro_ml_inputs/data/interim/calendar_ru_2026.parquet'
AVAILABILITY_ASSUMPTION = 'Public production calendar is assumed available in advance; historical ingestion time is not known.'


def _day(value):
    if pd.isna(value):
        raise ValueError('calendar dates cannot be missing')
    if type(value) is date:
        return value
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('Europe/Moscow')
    return stamp.date()


def _hash(frame):
    return hashlib.sha256(frame.to_json(orient='records', date_format='iso').encode()).hexdigest()


class ProductionCalendar:
    def __init__(self, days, *, source_path=None, school_periods=None, school_source_path=None):
        if not {'date', 'is_workday'} <= set(days):
            raise ValueError('calendar requires date and is_workday')
        self.days = days.copy()
        self.days['date'] = self.days.date.map(_day)
        if self.days.date.duplicated().any():
            raise ValueError('duplicate calendar dates')
        for field in ('is_workday', 'is_shortened', 'is_transfer_dayoff', 'is_working_weekend'):
            if field in self.days and (not pd.api.types.is_bool_dtype(self.days[field]) or self.days[field].isna().any()):
                raise ValueError(f'calendar {field} requires known boolean flags')
        self.source_path = str(Path(source_path).resolve()) if source_path else None
        self.sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path else _hash(self.days)
        self.school_source_path = str(Path(school_source_path).resolve()) if school_source_path else None
        self.school_sha256 = None
        self.school_periods = {}
        if school_periods is not None:
            if 'school_period' not in school_periods or not school_periods.school_period.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
                raise ValueError('school calendar requires explicit nonempty school_period labels')
            self.school_sha256 = (hashlib.sha256(Path(school_source_path).read_bytes()).hexdigest()
                                  if school_source_path else _hash(school_periods))
            for row in school_periods.to_dict('records'):
                if 'date' in row:
                    start = end = _day(row['date'])
                elif {'date_start', 'date_end'} <= set(row):
                    start, end = _day(row['date_start']), _day(row['date_end'])
                else:
                    raise ValueError('school calendar requires a date or explicit inclusive date bounds')
                if end < start:
                    raise ValueError('school calendar end precedes start')
                for day in pd.date_range(start, end).date:
                    if day in self.school_periods:
                        raise ValueError('overlapping school calendar periods')
                    self.school_periods[day] = row['school_period']
        self._lookup = self.days.set_index('date').to_dict('index')

    def manifest(self):
        return dict(source_path=self.source_path, sha256=self.sha256,
            availability_assumption=AVAILABILITY_ASSUMPTION,
            school_period='unknown unless explicitly supplied', school_source_path=self.school_source_path,
            school_sha256=self.school_sha256,
            school_availability_assumption=('Supplied period calendar is assumed known before forecasting; ingestion time is not known.'
                                            if self.school_sha256 is not None else None))


def load_calendar(path=DEFAULT_CALENDAR_SOURCE, *, school_calendar=None):
    path = Path(path)
    school_path = None if school_calendar is None or isinstance(school_calendar, pd.DataFrame) else Path(school_calendar)
    periods = (pd.read_parquet(school_path) if school_path and school_path.suffix == '.parquet' else
               pd.read_csv(school_path) if school_path else school_calendar)
    return ProductionCalendar(pd.read_parquet(path), source_path=path, school_periods=periods, school_source_path=school_path)


def calendar_day(value, calendar=None):
    """Return civil-day kind; never infer school terms or fill absent calendar dates."""
    day = _day(value)
    if isinstance(calendar, pd.DataFrame):
        calendar = ProductionCalendar(calendar)
    if calendar is not None and not isinstance(calendar, ProductionCalendar):
        raise ValueError('calendar must be a ProductionCalendar or DataFrame')
    row = calendar._lookup.get(day) if calendar is not None else None
    result = dict(date=day.isoformat(), day_kind='unknown', school_period='unknown', is_shortened=None,
                  holiday_name=None, availability_assumption=AVAILABILITY_ASSUMPTION if calendar is not None else None)
    if calendar is not None:
        result['school_period'] = calendar.school_periods.get(day, 'unknown')
    if row is None:
        return result
    name = row.get('holiday_name')
    named = isinstance(name, str) and bool(name.strip())
    result['day_kind'] = ('workday' if row['is_workday'] else 'holiday' if
                          named or row.get('is_transfer_dayoff', False) or day.weekday() < 5 else 'weekend')
    result['holiday_name'] = name if named else None
    result['is_shortened'] = bool(row['is_shortened']) if 'is_shortened' in row else None
    return result
