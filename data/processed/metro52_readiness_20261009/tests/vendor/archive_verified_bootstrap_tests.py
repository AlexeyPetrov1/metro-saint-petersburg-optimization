from readiness_vendor_inputs import ADAPTATION, canonical_source_hash


def test_original_source_hash_survives_windows_line_endings():
    original = b"x = 1\ny = 2\n"
    assert canonical_source_hash(original) == canonical_source_hash(original.replace(b"\n", b"\r\n"))


def test_source_change_is_not_hidden_by_line_ending_normalization():
    assert canonical_source_hash(b"x = 1\n") != canonical_source_hash(b"x = 2\r\n")


def test_organizer_workbook_alias_preserves_archive_bytes():
    alias = ADAPTATION["raw_workbook_alias"]
    assert alias["unique_decoded_name"]
    assert alias["input_sha256"] == alias["archive_member_sha256"]
