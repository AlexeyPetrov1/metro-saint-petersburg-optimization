"""Test-only input paths and verified Windows newline equivalence."""

import hashlib
import json
import unicodedata
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
VENDOR = PROJECT / "vendor/metro_petersburg"
INPUTS = PROJECT / "data/processed/metro51_inputs_20261009"
ADAPTATION = {}


def canonical_source_hash(raw):
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def verify_workbook(workbook, manifest, input_root):
    member = workbook.relative_to(input_root).as_posix()
    actual = hashlib.sha256(workbook.read_bytes()).hexdigest()
    expected = manifest["files"].get(member)
    if actual != expected:
        raise RuntimeError("Organizer workbook differs from published manifest")
    archive = Path(manifest["archives"]["data"]["path"])
    original = None
    source = "published_input_manifest_sha256"
    if archive.is_file():
        with zipfile.ZipFile(archive) as stream:
            original = hashlib.sha256(stream.read(member)).hexdigest()
        if actual != original:
            raise RuntimeError("Organizer workbook differs from original archive")
        source = "original_archive_and_published_manifest_sha256"
    return dict(input=str(workbook), archive=str(archive), member=member,
                input_sha256=actual, published_sha256=expected,
                archive_member_sha256=original, verification_source=source,
                archive_available=archive.is_file(), published_bytes_identical=True)


def pytest_configure(config):
    from src import config as paths

    aliases = {}
    for name, value in list(vars(paths).items()):
        if not isinstance(value, Path) or not value.is_relative_to(VENDOR / "data"):
            continue
        replacement = INPUTS / value.relative_to(VENDOR)
        if replacement.exists():
            setattr(paths, name, replacement)
            aliases[name] = dict(original=str(value), input=str(replacement))
    expected_name = unicodedata.normalize("NFC", paths.SPB_FLOW_FILE)
    raw_matches = []
    for path in paths.SPB_RAW.rglob("*.xlsx"):
        try:
            decoded = path.name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            decoded = path.name
        if unicodedata.normalize("NFC", decoded) == expected_name:
            raw_matches.append(path)
    if len(raw_matches) != 1:
        raise RuntimeError("Hourly organizer workbook alias must be unique")
    workbook = raw_matches[0]
    manifest = json.loads((INPUTS / "input_manifest.json").read_text(encoding="utf-8"))
    workbook_proof = verify_workbook(workbook, manifest, INPUTS)
    paths.SPB_FLOW_FILE = workbook.name
    from src import model

    model.MODELS = INPUTS / "models"
    frozen = model.load_config()
    sources = {}
    for relative in model.FROZEN_CODE:
        raw = (VENDOR / relative).read_bytes()
        normalized = canonical_source_hash(raw)
        expected = frozen["code_sha256"][relative]
        if normalized != expected:
            raise RuntimeError(f"Source differs beyond CRLF/LF: {relative}")
        sources[relative] = dict(
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            canonical_lf_sha256=normalized,
            frozen_sha256=expected,
            only_line_endings_differ=hashlib.sha256(raw).hexdigest() != expected,
        )
    model.code_hashes = lambda: {
        relative: canonical_source_hash((VENDOR / relative).read_bytes())
        for relative in model.FROZEN_CODE
    }
    ADAPTATION.update(
        test_only=True, production_code_changed=False, models_refitted=False,
        frozen_config_changed=False, b2_cache_changed=False,
        input_root=str(INPUTS), input_aliases=aliases,
        model_root=str(model.MODELS), canonical_source_checks=sources,
        raw_workbook_alias=dict(workbook_proof, expected_name=expected_name,
                                unique_decoded_name=True),
    )


def pytest_sessionfinish(session, exitstatus):
    import numpy
    import pandas

    ADAPTATION.update(exit_code=int(exitstatus), pandas=pandas.__version__, numpy=numpy.__version__)
    (HERE / "bootstrap_evidence.json").write_text(
        json.dumps(ADAPTATION, ensure_ascii=False, indent=2), encoding="utf-8"
    )
