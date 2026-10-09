import pytest

from metro44_noidle_terminal import bind_terminal_adapter


def test_adapter_replaces_only_existing_compile_import():
    source = 'from dwell import compile_operational_timing, timing_key\ncompile_operational_timing(data)\n'
    changed = bind_terminal_adapter(source)
    assert 'from dwell import timing_key' in changed
    assert 'compile_terminal_timing as compile_operational_timing' in changed
    assert changed.endswith('compile_operational_timing(data)\n')


def test_unexpected_compiler_import_is_rejected():
    with pytest.raises(ValueError, match='one compiler import'):
        bind_terminal_adapter('different')
