import numpy as np
import pandas as pd
import pytest

from src.stack.data import blocks_of


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_blocks_follow_elapsed_quarters_for_each_timestamp_unit(unit):
    first = pd.date_range("2026-02-27", periods=40, freq="15min", tz="UTC")
    second = pd.date_range("2026-05-01", periods=40, freq="15min", tz="UTC")
    timestamps = first.append(second).as_unit(unit)
    np.testing.assert_array_equal(blocks_of(timestamps), [0] * 40 + [1] * 40)
