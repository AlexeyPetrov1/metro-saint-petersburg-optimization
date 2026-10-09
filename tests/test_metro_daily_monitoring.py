import metro_daily_run


def test_daily_runner_passes_prepared_monitoring_to_real_controller(tmp_path, monkeypatch):
    context = dict(observations='counts', event_registry='events', knowledge_mode='strict')
    monkeypatch.setattr(metro_daily_run, 'load_monitoring_context', lambda output: context)
    monkeypatch.setattr(metro_daily_run, 'bootstrap', lambda output: dict(data=dict(stations=[dict(station_id='L1_01',kind='line')],
        plans=[dict(plan_id='P')],period_start='2026-09-30T06:00:00+03:00',headway_policy={}),
        profiles={},groups=[],meta=dict(service_end_second=7200),roster={},checkpoint={}))
    class Runner:
        def __init__(self, *args):
            pass
    monkeypatch.setattr(metro_daily_run, 'GroupedPassengerSimulation', Runner)
    monkeypatch.setattr(metro_daily_run, 'read_forecast', lambda *args: {})
    data = metro_daily_run.bootstrap(tmp_path)['data']
    monkeypatch.setattr(metro_daily_run, 'make_candidates', lambda *args: ([dict(candidate_id='KEEP',_data=data)],0))
    seen = []
    def decide(*args, **kwargs):
        seen.append(kwargs['monitoring_context'])
        # Stop after observing the real runner's handoff.
        raise RuntimeError('checked controller handoff')
    monkeypatch.setattr(metro_daily_run, 'decide_operational', decide)
    import pytest
    with pytest.raises(RuntimeError, match='handoff'):
        metro_daily_run.run_strategy(tmp_path, 'MEDIAN', max_steps=1)
    assert seen == [context]
