import copy
import json
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace
from datetime import datetime

import pytest

import metro44_catalog as catalog
from goal43_policy import load_policy


def fleet(km):
    return [dict(train_id=f'T{i}', train_type='Baltiets' if i < 37 else 'Yubileyny', km=km)
            for i in range(68)]


def evaluated(name, mean=70, p95=138):
    exposure = {'admissible': True, 'global': dict(max_load=100, exceptional_time_share=0,
        exceptional_car_seconds=0, passenger_car_seconds=1000, hard_limit_car_seconds=0), 'windows': []}
    return dict(candidate_id=name, status='evaluated', physical_passed=True,
        passengers_checked=True, service_passed=True, policy_passed=True,
        metrics=dict(max_car_load=100, waiting_passenger_seconds=mean * 100),
        trials=[dict(future_exposure=exposure, headways=dict(passed=True),
            mature_cohort=dict(complete=True, arrived=100, unboarded=0,
                              mean_wait_seconds=mean, p95_wait_seconds=p95),
            initial_obligations=dict(complete=True, pending_at_end=0))])


@pytest.fixture
def context(monkeypatch):
    bundle = json.loads((Path(__file__).resolve().parents[1] /
        'data/processed/metro_daily_experiments_20261009/forecasts/ML_10.json').read_text(encoding='utf-8'))
    ids = sorted({r['station_id'] for r in bundle['rows']})
    data = dict(period_start='2026-09-30T08:15:00+03:00', marker='KEEP',
                stations=[dict(station_id=s, kind='line') for s in ids],
                headway_policy=load_policy(), plans=[dict(plan_id='P', trips=[])])
    events = []

    def gate(changed, *args, **kwargs):
        events.append(('physical', changed['marker']))
        valid = changed['marker'] != 'BAD'
        return dict(movement_passed=valid, terminal_resources=dict(passed=valid),
                    _data_digest=catalog.digest(changed))

    def passengers(data, sources, snapshots, profiles, registry, bundle, policy, *, candidates, **kwargs):
        names = [r['candidate_id'] for r in candidates if r['status'] == 'movement_checked']
        events.append(('passengers', names))
        return dict(candidate_table=[evaluated(n, 70 if n == 'KEEP' else 79,
                                              138 if n == 'KEEP' else 149) for n in names],
                    search_complete=True)

    def builder(data, second, end, **kwargs):
        events.append(('builder', kwargs))
        changed = copy.deepcopy(data)
        changed['marker'] = kwargs.get('name', 'CHEAP')
        return changed, dict(movement_passed=True)

    monkeypatch.setattr(catalog, 'physical_gate', gate)
    monkeypatch.setattr(catalog, 'decide_operational', passengers)
    monkeypatch.setattr(catalog, 'closed_per_train', lambda d, r, **kw: fleet(10 if d['marker'] == 'KEEP' else 8))
    return SimpleNamespace(data=data, ids=ids, bundle=bundle, events=events,
        builder=builder, runner=SimpleNamespace(motion={}), state=dict(second=6300))


def calculate(context, **kwargs):
    return catalog.calculate_catalog(context.data, context.runner, context.state, {}, [], 'unused',
        'MPC_ML', context.ids, 6300, 22200, 3, monotonic() + 30, Event(),
        bundle_override=context.bundle, profile_specs=[dict(candidate_id='CHEAP', builder='stable',
                                                          headway=160, wear_priority=True)],
        builders={'stable': context.builder}, **kwargs)


def test_strict_never_relaxes_keep_wait_budgets(context):
    result = calculate(context)
    assert result['decision']['selection']['selected_candidate_id'] == 'KEEP'
    assert result['decision']['selection']['waiting_budgets'] == dict(mean_seconds=70, p95_seconds=138)
    assert result['decision']['quality_policy']['mode'] == 'strict'


def test_explicit_pareto_is_development_and_never_claims_cash(context):
    result = calculate(context, quality_policy='pareto')
    decision = result['decision']
    assert decision['selection']['selected_candidate_id'] == 'CHEAP'
    assert decision['selection']['waiting_budgets'] == dict(mean_seconds=90, p95_seconds=180)
    assert decision['quality_policy']['user_requirement'] is False
    assert decision['quality_policy']['posthoc_development'] is True
    assert decision['selection']['economic_improvement_claimed'] is False
    assert decision['cash_saving_rub'] is None
    assert decision['forecast_digest'] == catalog.digest(context.bundle)


def test_terminal_rejection_precedes_passenger_validation(context):
    def bad(data, *args, **kwargs):
        return dict(data, marker='BAD'), dict(movement_passed=True)
    context.builder = bad
    result = calculate(context)
    tested = next(v for event, v in context.events if event == 'passengers')
    assert tested == ['KEEP']
    row = next(r for r in result['decision']['candidate_table'] if r['candidate_id'] == 'CHEAP')
    assert row['status'] == 'rejected' and not row.get('passengers_checked', False)


def test_hook_does_not_mutate_data_snapshot_or_forecast(context):
    before = copy.deepcopy((context.data, context.state, context.bundle))
    calculate(context)
    assert (context.data, context.state, context.bundle) == before


@pytest.mark.parametrize('mode', ['unknown', None, 'STRICT'])
def test_unknown_quality_policy_fails_before_work(context, mode):
    with pytest.raises(ValueError, match='quality policy'):
        calculate(context, quality_policy=mode)
    assert not context.events


@pytest.mark.parametrize('change', [dict(future_truth_intentionally_used=True), dict(operationally_available=False)])
def test_causal_model_rejects_future_truth_before_building(context, change):
    context.bundle['source_versions'].update(change)
    with pytest.raises(ValueError, match='causal'):
        calculate(context)
    assert not context.events


def test_baseline_is_one_public_forecast_arm(context, monkeypatch):
    calls = []
    def read(source, mode, ids, at):
        calls.append((mode, at))
        return context.bundle
    monkeypatch.setattr(catalog, 'read_forecast', read)
    catalog.calculate_catalog(context.data, context.runner, context.state, {}, [], 'cache',
        'MPC_BASELINE_FORECAST', context.ids, 6300, 22200, 0, monotonic() + 30, Event())
    assert calls == [('MEDIAN', '2026-09-30T10:00:00+03:00')]
    with pytest.raises(ValueError, match='causal'):
        catalog.calculate_catalog(context.data, context.runner, context.state, {}, [], 'cache',
            'MPC_ORACLE', context.ids, 6300, 22200, 0, monotonic() + 30, Event())


def test_deadline_and_cancellation_prevent_generation(context):
    for deadline, event in [(monotonic() - 1, Event()), (monotonic() + 30, Event())]:
        if deadline > monotonic(): event.set()
        with pytest.raises(TimeoutError):
            catalog.calculate_catalog(context.data, context.runner, context.state, {}, [], 'unused',
                'MPC_ML', context.ids, 6300, 22200, 3, deadline, event,
                bundle_override=context.bundle)
    assert not context.events


def test_profile_catalog_uses_pareto_budget_and_bridge_only_offpeak(context, monkeypatch):
    calls = []
    slots = [dict(start_second=6300, end_second=7200, minimum_headway_seconds=150)]
    monkeypatch.setattr(catalog, '_slots', lambda *args: slots)
    monkeypatch.setattr(catalog, 'estimated_keep_wait_budget', lambda *args: dict(mean_wait_budget_seconds=75))
    def solve(rows, **kw):
        calls.append(kw['mean_wait_budget_seconds'])
        return dict(status='optimal', slots=[dict(start_second=6300, end_second=7200,
            up_headway_seconds=160, down_headway_seconds=170)])
    monkeypatch.setattr(catalog, 'solve_joint_profile', solve)
    specs, search = catalog.catalog_profiles(context.data, context.bundle, context.state, 6300,
        quality_policy='pareto', deadline=monotonic() + 30)
    assert calls == [90]
    assert [s['candidate_id'] for s in specs] == ['CP_DIRECTIONAL', 'H125_OFF160_WEAR', 'INTERVAL160_BRIDGE']
    assert specs[-1]['wear_priority'] is False
    assert specs[0]['headway'][0]['headway_seconds'] == 160
    assert specs[0]['headway'][1]['headway_seconds'] == 170
    specs, _ = catalog.catalog_profiles(context.data, context.bundle, context.state, 3600,
        quality_policy='strict', deadline=monotonic() + 30)
    assert calls[-1] == 75
    assert all(s['builder'] != 'bridge' for s in specs)


def test_clock_profile_keeps_both_weekday_peaks_and_offpeak_minimum(context):
    profile = catalog.peak_offpeak_profile(context.data, 3600, 39600, peak_seconds=125, offpeak_seconds=160)
    origin = datetime.fromisoformat(context.data['period_start'])
    for at, expected in [('2026-09-30T09:59:59+03:00', 125),
                         ('2026-09-30T10:00:00+03:00', 160),
                         ('2026-09-30T17:30:00+03:00', 125)]:
        assert catalog.headway_seconds(profile, datetime.fromisoformat(at), 'L1_01',
                                        context.data['headway_policy'], origin) == expected


def test_shared_terminal_gate_cannot_accept_three_visits_or_two_standby():
    for total, standby in [(3, 1), (2, 2)]:
        assert not catalog.terminal_gate(dict(passed=True,
            maximum_visits_by_terminal={'L1_01': total}, maximum_standby_by_terminal={'L1_01': standby}))
    assert catalog.terminal_gate(dict(passed=True, maximum_visits_by_terminal={'L1_01': 2},
                                      maximum_standby_by_terminal={'L1_01': 1}))
    assert not catalog.terminal_gate({})


def test_duplicate_physical_plans_are_not_evaluated_twice(context):
    specs = [dict(candidate_id=name, builder='stable', headway=160, wear_priority=True) for name in ['A', 'B']]
    result = catalog.calculate_catalog(context.data, context.runner, context.state, {}, [], 'unused',
        'MPC_ML', context.ids, 6300, 22200, 3, monotonic() + 30, Event(),
        bundle_override=context.bundle, profile_specs=specs, builders={'stable': context.builder})
    tested = next(v for event, v in context.events if event == 'passengers')
    assert tested == ['KEEP', 'A']
    assert next(r for r in result['decision']['candidate_table'] if r['candidate_id'] == 'B')['status'] == 'duplicate'


@pytest.mark.parametrize('depot,lead', [('AVTOVO', 720), ('NORTHERN', 780)])
def test_new_depot_release_uses_inclusive_type_lead_not_global_freeze(depot, lead):
    entry = dict(trip_id='ENTRY', train_id='T', route=[f'ENTRY_{depot}'], passenger=False)
    service = dict(trip_id='SERVICE', train_id='T', route=['P'], passenger=True)
    source = dict(plans=[dict(trips=[])], leg_timing=[])
    data = dict(plans=[dict(trips=[entry, service])], stations=[dict(station_id=f'DEPOT_{depot}', kind='depot')],
        tracks=[dict(track_id=f'ENTRY_{depot}',
        from_station=f'DEPOT_{depot}')], headway_policy=load_policy(),
        leg_timing=[dict(trip_id='ENTRY', leg_index=0, departure_second=100),
                    dict(trip_id='SERVICE', leg_index=0, departure_second=100 + lead)])
    assert catalog._preparation_ok(data, source, 100)
    data['leg_timing'][1]['departure_second'] -= 1
    assert not catalog._preparation_ok(data, source, 100)
    # A previously committed release is preserved rather than frozen for 13 minutes again.
    assert catalog._preparation_ok(data, copy.deepcopy(data), 100)


def test_builder_returning_after_cancellation_cannot_produce_selection(context):
    cancelled = Event()
    def builder(data, *args, **kwargs):
        cancelled.set()
        return dict(data, marker='CHEAP'), dict(movement_passed=True)
    with pytest.raises(TimeoutError):
        catalog.calculate_catalog(context.data, context.runner, context.state, {}, [], 'unused',
            'MPC_ML', context.ids, 6300, 22200, 3, monotonic() + 30, cancelled,
            bundle_override=context.bundle, profile_specs=[dict(candidate_id='A', builder='stable')],
            builders={'stable': builder})
    assert not any(event == 'passengers' for event, value in context.events)


def test_no_passenger_evaluation_when_even_keep_fails_physical_gate(context, monkeypatch):
    monkeypatch.setattr(catalog, 'physical_gate', lambda *args, **kw: dict(movement_passed=False))
    result = calculate(context)
    assert not any(event == 'passengers' for event, value in context.events)
    assert result['decision']['selection']['selected_candidate_id'] is None
    assert result['decision']['selection']['mitigation_candidate_id'] is None


def test_complete_closed_keep_passes_independent_physical_gate():
    data = json.loads((Path(__file__).resolve().parents[1] /
        'data/processed/metro44_continuation_6h_closed_20261009/runs/CONT6H_CLOSED_KEEP_S1_7/movement_input.json')
        .read_text(encoding='utf-8'))
    before = copy.deepcopy(data)
    audit = catalog.physical_gate(data, 6300, 22200, reference=data)
    assert audit['movement_passed'] and audit['obligations_closed']
    assert audit['terminal_resources']['maximum_total_by_terminal'] == {'L1_01': 2, 'L1_19': 2}
    assert audit['minimum_headway'] >= 150 and audit['maximum_headway'] <= 480
    assert audit['peak_trains_on_line'] == 53 and data == before


def test_injected_bridge_cannot_bypass_current_peak_guard(context):
    bundle = json.loads((Path(__file__).resolve().parents[1] /
        'data/processed/metro_daily_experiments_20261009/forecasts/ML_09.json').read_text(encoding='utf-8'))
    result = catalog.calculate_catalog(context.data, context.runner, dict(second=3600), {}, [], 'unused',
        'MPC_ML', context.ids, 3600, 22200, 3, monotonic() + 30, Event(),
        bundle_override=bundle, profile_specs=[dict(candidate_id='BRIDGE', builder='bridge', headway=160)],
        builders={'bridge': context.builder})
    assert not any(event == 'builder' for event, value in context.events)
    row = next(r for r in result['decision']['candidate_table'] if r['candidate_id'] == 'BRIDGE')
    assert row['status'] == 'rejected' and 'offpeak' in row['reason']
