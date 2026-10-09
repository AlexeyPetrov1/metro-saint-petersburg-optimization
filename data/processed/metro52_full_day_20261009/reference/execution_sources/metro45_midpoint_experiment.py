"""One predeclared155-second bridge trial; existing driver and inputs stay intact."""
import argparse
from pathlib import Path
import sys

import metro45_experiment as experiment
from metro44_daily_resource_audit import mileage_ledger


def run_midpoint(output,*,trial=None):
    return (trial or experiment.run)(output,scenario='S1',mode='ML',policy='155',bridge_seconds=480)


def assess(candidate,reference,bounds,*,physical):
    safe=(physical is True and candidate.get('actual_service_admissible') is True
          and candidate.get('all_service_arrivals_completed') is True
          and candidate.get('movement_obligations_closed') is True
          and candidate.get('cohort',{}).get('complete') is True
          and candidate.get('mature_cohort',{}).get('complete') is True)
    no_worse=all(candidate[k]<=reference[k] for k in ('mean_wait_seconds','p95_wait_seconds'))
    budget=candidate['mean_wait_seconds']<=90 and candidate['p95_wait_seconds']<=180
    return dict(candidate_complete_and_service_safe=safe,strict_no_wait_regression=no_worse,
        strict_window_resource_improvement=safe and no_worse and bounds['saving_train_km_lower_bound']>0,
        research_90_180_budget_passed=budget,research_wait_budget_is_user_requirement=False,
        research_tradeoff_eligible=safe and budget and bounds['saving_train_km_lower_bound']>0,
        release_authorized=False,enabled_by_default=False,cash_savings_rub=None)


def load_arm(base,result):
    folder=base/result['run']['folder']
    return (experiment.read(folder/'movement_input.json'),experiment.read(folder/'motion.json'))


def main(output):
    output=Path(output).resolve()
    modules={Path(m.__file__).resolve() for m in list(sys.modules.values())
             if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'
             and (Path(m.__file__).resolve().parent==experiment.ROOT or
                  experiment.SNAPSHOT in Path(m.__file__).resolve().parents)}
    modules.add(Path(__file__).resolve())
    sources_before={str(p):experiment.sha(p) for p in sorted(modules)}
    candidate=run_midpoint(output)
    experiment.write(output/'wrapper_sources_before.json',sources_before)
    if candidate['status']!='evaluated':
        experiment.write(output/'comparison.json',dict(status='rejected',reason=candidate.get('reason'),
            preset_headway_seconds=155,release_authorized=False,enabled_by_default=False,cash_savings_rub=None))
        return candidate
    keep_base=experiment.ROOT/'data/processed/metro44_continuation_6h_closed_20261009'
    bridge_base=experiment.ROOT/'data/processed/metro44_bridge_off160_actual_6h_20261009_v2'
    reference_paths=[keep_base/'keep_result.json',bridge_base/'s1_result.json']
    reference_hashes={str(p):experiment.sha(p) for p in reference_paths}
    keep,bridge=(experiment.read(p) for p in reference_paths)
    assert candidate['contract']['groups_digest']==keep['groups_digest']==bridge['groups_digest']
    for reference in (keep,bridge):
        assert candidate['metrics']['mean_wait_scope']==reference['metrics']['mean_wait_scope']
        assert candidate['metrics']['cohort']['arrived']==reference['metrics']['cohort']['arrived']
        assert candidate['metrics']['fixed_fleet_size']==reference['metrics']['fixed_fleet_size']==68
    cand_data,cand_motion=load_arm(output,candidate)
    ref_data,ref_motion=load_arm(keep_base,keep)
    bridge_data,bridge_motion=load_arm(bridge_base,bridge)
    bridge_bounds=experiment.paired_window_bounds(bridge_data,bridge_motion,cand_data,cand_motion,3600,21600)
    ledgers={name:mileage_ledger(data['trains'],motion['transport_work']) for name,data,motion in
             [('KEEP',ref_data,ref_motion),('BRIDGE155',cand_data,cand_motion),('BRIDGE160',bridge_data,bridge_motion)]}
    sources_after={str(p):experiment.sha(p) for p in sorted(modules)}
    assert sources_before==sources_after
    assert reference_hashes=={str(p):experiment.sha(p) for p in reference_paths}
    result=dict(status='evaluated',preset_headway_seconds=155,bridge_seconds=480,
        preset_chosen_before_actual_run=True,forecast_used_for_policy_choice=False,
        development_only=True,independent_holdout=False,release_authorized=False,enabled_by_default=False,
        groups_digest=candidate['contract']['groups_digest'],common_wait_scope_verified=True,
        vs_KEEP=assess(candidate['metrics'],keep['metrics'],candidate['window_bounds'],physical=candidate['physical_passed']),
        vs_BRIDGE160=assess(candidate['metrics'],bridge['metrics'],bridge_bounds,physical=candidate['physical_passed']),
        window_bounds_vs_KEEP=candidate['window_bounds'],window_bounds_vs_BRIDGE160=bridge_bounds,
        full_closed_ledgers=ledgers,
        waiting={name:{k:r['metrics'][k] for k in ('mean_wait_seconds','p95_wait_seconds','max_car_load',
                 'actual_service_admissible','all_service_arrivals_completed','movement_obligations_closed')}
                 for name,r in [('KEEP',keep),('BRIDGE155',candidate),('BRIDGE160',bridge)]},
        sources_preserved=True,reference_hashes_preserved=True,cash_savings_rub=None,
        lost_revenue_rub=None,net_profit_rub=None,to1_resource_is_cash=False,
        scope='paired S1 finite6hdevelopment with common continuation; full closed tail reported separately')
    experiment.write(output/'wrapper_sources_after.json',sources_after)
    experiment.write(output/'comparison.json',result)
    print({name:result['waiting'][name] for name in result['waiting']},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
