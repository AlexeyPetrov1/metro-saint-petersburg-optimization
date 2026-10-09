import hashlib
import zipfile

import pytest

import readiness_vendor_inputs as bootstrap
from readiness_vendor_inputs import ADAPTATION, canonical_source_hash


def test_original_source_hash_survives_windows_line_endings():
    original = b"x = 1\ny = 2\n"
    assert canonical_source_hash(original) == canonical_source_hash(original.replace(b"\n", b"\r\n"))


def test_source_change_is_not_hidden_by_line_ending_normalization():
    assert canonical_source_hash(b"x = 1\n") != canonical_source_hash(b"x = 2\r\n")


def test_organizer_workbook_alias_preserves_archive_bytes():
    alias = ADAPTATION["raw_workbook_alias"]
    assert alias["unique_decoded_name"]
    assert alias["input_sha256"] == alias["published_sha256"]
    if alias["archive_available"]:
        assert alias["input_sha256"] == alias["archive_member_sha256"]


def workbook_case(tmp_path):
    workbook = tmp_path / "data/raw/entries.xlsx"
    workbook.parent.mkdir(parents=True)
    workbook.write_bytes(b"unchanged organizer workbook")
    member = workbook.relative_to(tmp_path).as_posix()
    return workbook, {"files": {member: hashlib.sha256(workbook.read_bytes()).hexdigest()},
                      "archives": {"data": {"path": str(tmp_path / "source.zip")}}}


def test_portable_workbook_uses_published_hash_when_archive_absent(tmp_path):
    workbook, manifest = workbook_case(tmp_path)
    proof = bootstrap.verify_workbook(workbook, manifest, tmp_path)
    assert proof["verification_source"] == "published_input_manifest_sha256"
    assert proof["archive_member_sha256"] is None
    assert proof["input_sha256"] == proof["published_sha256"]


def test_portable_workbook_rejects_modified_bytes(tmp_path):
    workbook, manifest = workbook_case(tmp_path)
    workbook.write_bytes(b"changed workbook")
    with pytest.raises(RuntimeError, match="published manifest"):
        bootstrap.verify_workbook(workbook, manifest, tmp_path)


def test_present_archive_mismatch_is_not_hidden_by_manifest(tmp_path):
    workbook, manifest = workbook_case(tmp_path)
    with zipfile.ZipFile(manifest["archives"]["data"]["path"], "w") as archive:
        archive.writestr(workbook.relative_to(tmp_path).as_posix(), b"different archive bytes")
    with pytest.raises(RuntimeError, match="original archive"):
        bootstrap.verify_workbook(workbook, manifest, tmp_path)
