"""Opt-in joint producer for metro45_controller.calculate(joint_builder=...)."""

from time import monotonic

from metro45_joint_dispatch import propose_joint_dispatch


def make_joint_builder(selected_train_ids, *, solver_seconds=60, window_seconds=60,
                       clock_stability=False, allow_future_exit_retiming=False):
    fleet = tuple(selected_train_ids)
    if not fleet or len(set(fleet)) != len(fleet):
        raise ValueError('explicit distinct trains required')
    if not 0 < solver_seconds <= 120 or not 0 <= window_seconds <= 120:
        raise ValueError('bounded solve and departure windows required')

    def build(data, second, service_end, *, deadline, phase_seconds=0):
        if monotonic() >= deadline:
            raise TimeoutError('joint catalogue deadline')
        if phase_seconds != 0:
            raise ValueError('joint producer requires its explicit departure windows')
        changed, audit = propose_joint_dispatch(data, second, fleet,
            window_seconds=window_seconds, time_limit_seconds=solver_seconds,
            service_end_second=service_end, deadline=deadline,
            clock_stability=clock_stability,
            allow_future_exit_retiming=allow_future_exit_retiming)
        if changed is None or not audit.get('movement_passed'):
            message = audit.get('reason', 'joint plan not physically verified')
            error = TimeoutError if audit.get('status') == 'timeout' else ValueError
            raise error(message)
        return changed, {**audit, 'passenger_validation_required': True,
                         'cash_saving_rub': None, 'release_authorized': False}

    return build
