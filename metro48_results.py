"""Publish only completed paired episodes that passed independent verification."""

from metro44_selection import economy_delta, resource_value


def paired_summary(reference, candidates, audits):
    base = reference['metrics']
    checks = ('accepted', 'exact_current_kernel_journal_parity',
        'all_20_quarter_prefixes_current_movements_and_begun_services_preserved',
        'local_recommendation_within_480', 'sources_and_inputs_preserved')
    rows = []
    for arm, candidate in candidates.items():
        audit = audits[arm]; current = candidate['metrics']
        if reference['full_day'] or candidate['full_day'] or audit['full_day']:
            raise ValueError('this comparison is a finite episode')
        if not all(audit.get(k) is True for k in checks):
            raise ValueError('independent verification did not pass')
        if (current['entered'] != base['entered'] or current['completed'] != base['completed']
                or current['completed'] != current['entered']
                or current['cohort']['arrived'] != base['cohort']['arrived']
                or not current['cohort']['complete'] or current['cohort']['unboarded']
                or any(current[k] for k in ('queued_at_closure','approaching_at_closure','onboard_at_closure'))
                or not current['movement_obligations_closed'] or not current['actual_service_admissible']):
            raise ValueError('the complete original cohort and obligations are required')
        resource = resource_value(current['per_train']); baseline = resource_value(base['per_train'])
        delta = economy_delta(base['per_train'],current['per_train'],service_admissible=True)
        rows.append(dict(arm=arm,train_km=resource['train_km'],
            mean_km_per_fixed_train=resource['mean_km_per_fixed_train'],
            to1_resource_equivalent_rub=resource['to1_resource_equivalent_rub'],
            saving_to1_resource_rub=delta['to1_resource_equivalent_rub'],
            saving_percent=100*delta['to1_resource_equivalent_rub']/baseline['to1_resource_equivalent_rub'],
            saving_km_per_fixed_train=delta['observed_reduction_km_per_fixed_train'],
            to1_resource_saving_per_fixed_train_rub=delta['to1_resource_equivalent_rub']/68,
            mean_wait_seconds=current['cohort']['mean_wait_seconds'],
            p95_wait_seconds=current['cohort']['p95_wait_seconds'],
            maximum_pipeline_seconds=candidate['maximum_pipeline_seconds'],
            all_entered=current['entered'], controlled_entered=current['cohort']['arrived'],
            independently_accepted=True, full_day=False, cash_saving_rub=None))
    return dict(reference=resource_value(base['per_train']),arms=rows,full_day=False,
        line_id=1,cash_saving_rub=None,lost_revenue_rub=None,net_profit_effect_rub=None,
        objective='TO1 resource equivalent on one complete preserved finite episode',
        savings_do_not_include_common_source_normalization=True)
