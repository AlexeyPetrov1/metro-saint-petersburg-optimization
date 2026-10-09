from datetime import time

import pytest

from goal43_source_refresh import QuarterSheet, refresh, semantic_comparison
import pandas as pd


class Sheet:
    title = "test"

    def iter_rows(self, values_only=True):
        yield ["header"]
        yield [None, None, "03:00:00", "03:15:00"]
        yield ["station", 10, "03:00:00", 7]


def test_adapter_changes_only_the_time_header():
    rows = list(QuarterSheet(Sheet()).iter_rows())
    assert rows[1] == [None, None, time(3), time(3, 15)]
    assert rows[2] == ["station", 10, "03:00:00", 7]


def test_semantic_comparison_joins_by_identity_not_order_or_source_label():
    old = pd.DataFrame({"vestibule_id": ["A", "B"], "interval_start": [1, 1],
                        "entry_count": pd.array([3, 4], dtype="Int64"), "source_id": ["old", "old"]})
    new = old.iloc[::-1].copy()
    new["source_id"] = "new"
    result = semantic_comparison(old, new)
    assert result["value_rows_differing"] == 0
    assert result["semantic_equal"] is True
    new.loc[new.vestibule_id == "A", "entry_count"] = 10
    result = semantic_comparison(old, new)
    assert result["value_rows_differing"] == 1
    assert result["new_total_entries"] == 14
    assert result["semantic_equal"] is False


def test_duplicate_input_keys_do_not_silently_merge():
    row = pd.DataFrame({"vestibule_id": ["A", "A"], "interval_start": [1, 1], "entry_count": [1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        semantic_comparison(row, row)


def test_refresh_rejects_legacy_output_paths_before_any_read(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        refresh(tmp_path, tmp_path / "raw", tmp_path / "data/processed/metro")
