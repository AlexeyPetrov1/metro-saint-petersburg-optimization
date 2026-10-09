from metro44_bridge_independent import failed_windows, prefix_equal, scoped_windows


def test_prefix_is_half_open_and_keeps_content():
    first = [{'second': 4, 'load': 3}, {'second': 10, 'load': 9}]
    second = [{'second': 4, 'load': 3}, {'second': 10, 'load': 1}]
    assert prefix_equal(first, second, 10)
    second[0]['load'] = 2
    assert not prefix_equal(first, second, 10)


def test_window_failure_in_shared_past_remains_outside_control_scope():
    windows = [{'start': 0, 'end': 900, 'status': 'fail'},
               {'start': 900, 'end': 1800, 'status': 'pass'}]
    assert scoped_windows(windows, 900, 1800) == [windows[1]]
    assert scoped_windows(windows, 0, 1800) == windows


def test_empty_tail_is_unknown_and_not_an_overload_failure():
    windows = [{'status': 'fail'}, {'status': 'unknown'}, {'status': 'pass'}]
    assert failed_windows(windows) == [windows[0]]
