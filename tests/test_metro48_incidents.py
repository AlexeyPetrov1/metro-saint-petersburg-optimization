from metro48_incidents import process_incident


def test_material_incident_and_revision_recompute_immediately_without_future_truth():
    calls = []
    incident = dict(incident_id='stop',train_id='B001',station_id='L1_12',severity='stop',
        estimated_end_at='2026-09-30T09:42:13+03:00',actual_end_at='2026-09-30T09:52:13+03:00')
    def calculate(known,at,deadline,cancelled):
        calls.append((known,at)); return dict(validated=True,action='protect_blocked_platform')
    ledger,result = process_incident([],incident,'2026-09-30T09:37:13+03:00',calculate)
    assert result['recomputed'] is True
    assert calls[0][1] == '2026-09-30T09:37:13+03:00'
    assert 'actual_end_at' not in calls[0][0]
    ledger,result = process_incident(ledger,incident,'2026-09-30T09:37:20+03:00',calculate)
    assert result['recomputed'] is False and len(calls)==1
    incident['estimated_end_at']='2026-09-30T09:52:13+03:00'
    ledger,result = process_incident(ledger,incident,'2026-09-30T09:38:00+03:00',calculate)
    assert result['recomputed'] is True and len(calls)==2
    assert ledger[-1]['revision']==2


def test_late_result_cannot_be_delivered_as_recommendation():
    incident=dict(incident_id='stop',severity='stop')
    bound=lambda callback,budget: dict(completed=False,elapsed_seconds=480.1,result=None)
    ledger,result=process_incident([],incident,'2026-09-30T09:37:13+03:00',None,bounded=bound)
    assert result['status']=='evaluation_timeout'
    assert ledger[-1]['recommendation'] is None and ledger[-1]['within_budget'] is False


def test_invalid_candidate_stays_explicit_and_does_not_claim_savings():
    incident=dict(incident_id='stop',severity='stop')
    ledger,result=process_incident([],incident,'2026-09-30T09:37:13+03:00',
        lambda *args:dict(validated=False,status='no_admissible_schedule',unavoidable_violation=True))
    assert result['status']=='evaluation_incomplete'
    assert ledger[-1]['recommendation'] is None
