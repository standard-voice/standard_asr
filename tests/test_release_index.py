# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The release workflow's index checks: guard, verify, and attestations."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_index.py"

Document = dict[str, object]
Fetch = Callable[[str], "Document | None"]

PYPI = "https://pypi.org"
TESTPYPI = "https://test.pypi.org"
WHEEL = "standard_asr-1.0.0-py3-none-any.whl"
SDIST = "standard_asr-1.0.0.tar.gz"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_index", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_index = _load()


def _dist(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    dist = tmp_path / "dist"
    dist.mkdir()
    files: dict[str, str] = {}
    for name, body in ((WHEEL, b"wheel"), (SDIST, b"sdist")):
        (dist / name).write_bytes(body)
        files[name] = hashlib.sha256(body).hexdigest()
    return dist, files


def _document(files: dict[str, str]) -> Document:
    return {"urls": [{"filename": name, "digests": {"sha256": sha}} for name, sha in files.items()]}


def _answers(pypi: Document | None, testpypi: Document | None) -> Fetch:
    # The two hosts share a suffix, so match on the full origin.
    def fetch(url: str) -> Document | None:
        if url.startswith(TESTPYPI):
            return testpypi
        if url.startswith(PYPI):
            return pypi
        raise AssertionError(f"unexpected url {url}")

    return fetch


def _sequence(*documents: Document | None) -> Fetch:
    answers: Iterator[Document | None] = iter(documents)

    def fetch(url: str) -> Document | None:
        return next(answers)

    return fetch


def _clock(*ticks: float) -> Callable[[], float]:
    values = iter(ticks)
    return lambda: next(values)


def _no_sleep(_: float) -> None:
    return None


def test_guard_passes_when_both_indexes_are_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist, _ = _dist(tmp_path)
    assert release_index.guard("1.0.0", dist, _answers(None, None)) == 0
    out = capsys.readouterr().out
    assert "pypi: standard-asr 1.0.0 is absent" in out
    assert "testpypi: standard-asr 1.0.0 is absent" in out


def test_guard_accepts_a_partial_upload_of_the_same_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The resume case: the index holds one of the two files, with the digest
    # of this build. The publish step fills the other one.
    dist, files = _dist(tmp_path)
    fetch = _answers(pypi=_document({WHEEL: files[WHEEL]}), testpypi=None)
    assert release_index.guard("1.0.0", dist, fetch) == 0
    assert "pypi: holds 1 of 2 files; the publish fills the rest" in capsys.readouterr().out


def test_guard_reports_a_complete_upload_as_a_no_op(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist, files = _dist(tmp_path)
    fetch = _answers(pypi=_document(files), testpypi=_document(files))
    assert release_index.guard("1.0.0", dist, fetch) == 0
    assert capsys.readouterr().out.count("the publish is a no-op") == 2


def test_guard_fails_when_an_index_holds_another_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The burned case: the index holds a file of this version with other
    # content, so no publish can make the two agree.
    dist, _ = _dist(tmp_path)
    fetch = _answers(pypi=None, testpypi=_document({WHEEL: "0" * 64}))
    assert release_index.guard("1.0.0", dist, fetch) == 1
    out = capsys.readouterr().out
    assert "::error::testpypi already holds standard-asr 1.0.0 with other content" in out
    assert "Bump the version" in out


def test_guard_fails_when_an_index_holds_a_file_this_build_did_not_produce(tmp_path: Path) -> None:
    dist, files = _dist(tmp_path)
    extra = dict(files)
    extra["standard_asr-1.0.0-cp313-cp313-linux_x86_64.whl"] = "1" * 64
    assert release_index.guard("1.0.0", dist, _answers(pypi=_document(extra), testpypi=None)) == 1


def test_verify_waits_for_the_index_then_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist, files = _dist(tmp_path)
    fetch = _sequence(None, _document({WHEEL: files[WHEEL]}), _document(files))
    slept: list[float] = []
    code = release_index.verify(
        "1.0.0", dist, "pypi", 600.0, fetch, slept.append, _clock(0.0, 0.0, 10.0, 20.0, 30.0)
    )
    assert code == 0
    assert slept == [15, 15]
    out = capsys.readouterr().out
    assert f"pypi: waiting for {WHEEL}, {SDIST}" in out
    assert f"pypi: waiting for {SDIST}" in out
    assert f"pypi: {SDIST} sha256 {files[SDIST]} matches" in out


def test_verify_fails_at_once_on_another_digest(tmp_path: Path) -> None:
    dist, files = _dist(tmp_path)
    wrong = dict.fromkeys(files, "f" * 64)
    slept: list[float] = []
    fetch = _answers(pypi=None, testpypi=_document(wrong))
    assert release_index.verify("1.0.0", dist, "testpypi", 600.0, fetch, slept.append) == 1
    assert slept == []


def test_verify_fails_when_the_deadline_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist, _ = _dist(tmp_path)
    fetch = _answers(pypi=None, testpypi=None)
    code = release_index.verify("1.0.0", dist, "pypi", 600.0, fetch, _no_sleep, _clock(0.0, 601.0))
    assert code == 1
    assert "still lacks" in capsys.readouterr().out


def test_attestations_waits_then_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist, _ = _dist(tmp_path)
    polls: dict[str, int] = {}

    def fetch(url: str) -> Document | None:
        assert url.startswith(f"{PYPI}/integrity/standard-asr/1.0.0/")
        name = url.rsplit("/", 2)[-2]
        polls[name] = polls.get(name, 0) + 1
        # The wheel's record appears on the second poll; the sdist's at once.
        if name == WHEEL and polls[name] == 1:
            return None
        return {"attestation_bundles": [{"publisher": {}}], "version": 1}

    slept: list[float] = []
    clock = _clock(0.0, 0.0, 10.0)
    code = release_index.attestations("1.0.0", dist, "pypi", 600.0, fetch, slept.append, clock)
    assert code == 0
    assert slept == [15]
    out = capsys.readouterr().out
    assert f"pypi: {WHEEL} has 1 attestation bundle(s)" in out
    assert f"pypi: {SDIST} has 1 attestation bundle(s)" in out
    assert polls == {WHEEL: 2, SDIST: 1}


def test_attestations_fails_when_the_deadline_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist, _ = _dist(tmp_path)

    def fetch(url: str) -> Document | None:
        assert url.startswith(f"{TESTPYPI}/integrity/")
        return {"attestation_bundles": []}

    clock = _clock(0.0, 601.0)
    code = release_index.attestations("1.0.0", dist, "testpypi", 600.0, fetch, _no_sleep, clock)
    assert code == 1
    out = capsys.readouterr().out
    assert "::error::testpypi has no attestation for" in out
    assert "bump the version" in out


def test_fetch_json_retries_server_errors_and_rate_limits_and_returns_none_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Response:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return self._body

    errors: list[urllib.error.HTTPError] = []

    def http_error(url: str, code: int, reason: str) -> urllib.error.HTTPError:
        # A real HTTPError holds the response body; the reader must close it.
        error = urllib.error.HTTPError(url, code, reason, {}, io.BytesIO(b""))  # type: ignore[arg-type]
        errors.append(error)
        return error

    def urlopen(request: object, timeout: float) -> _Response:
        url = str(getattr(request, "full_url", ""))
        calls.append(url)
        if url.endswith("/missing/json"):
            raise http_error(url, 404, "not found")
        if url.endswith("/flaky/json") and len(calls) < 3:
            raise http_error(url, 503, "busy")
        if url.endswith("/broken/json"):
            raise http_error(url, 400, "bad")
        if url.endswith("/limited/json") and calls.count(url) < 2:
            raise http_error(url, 429, "slow down")
        return _Response(b'{"urls": []}')

    monkeypatch.setattr(release_index.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(release_index.time, "sleep", _no_sleep)
    assert release_index.fetch_json("https://pypi.org/pypi/x/missing/json") is None
    assert release_index.fetch_json("https://pypi.org/pypi/x/flaky/json") == {"urls": []}
    assert len(calls) == 3
    with pytest.raises(urllib.error.HTTPError):
        release_index.fetch_json("https://pypi.org/pypi/x/broken/json")
    # A rate limit is an answer to wait for, not a verdict on the version.
    assert release_index.fetch_json("https://pypi.org/pypi/x/limited/json") == {"urls": []}
    # Every error path, swallowed, retried, or raised, released its response.
    assert [error.code for error in errors] == [404, 503, 400, 429]
    assert all(error.fp.closed for error in errors)


def test_index_files_rejects_a_malformed_answer() -> None:
    with pytest.raises(ValueError, match="without a 'urls' list"):
        release_index.index_files("pypi", "1.0.0", _sequence({"info": {}}))
    with pytest.raises(ValueError, match="malformed 'urls' entry"):
        release_index.index_files("pypi", "1.0.0", _sequence({"urls": ["x"]}))
    with pytest.raises(ValueError, match="without a name or sha256"):
        release_index.index_files("pypi", "1.0.0", _sequence({"urls": [{"filename": "a"}]}))


def test_local_files_requires_exactly_the_wheel_and_the_sdist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holds nothing; a release of 1.0.0 is exactly"):
        release_index.local_files(tmp_path, "1.0.0")
    dist, _ = _dist(tmp_path)
    with pytest.raises(ValueError, match="a release of 2.0.0 is exactly"):
        release_index.local_files(dist, "2.0.0")
    (dist / "standard_asr-1.0.0-cp313-cp313-linux_x86_64.whl").write_bytes(b"stray")
    with pytest.raises(ValueError, match="a release of 1.0.0 is exactly"):
        release_index.local_files(dist, "1.0.0")


def test_main_dispatches_each_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist, files = _dist(tmp_path)
    common = ["--version", "1.0.0", "--dist", str(dist)]
    monkeypatch.setattr(release_index, "fetch_json", _answers(pypi=_document(files), testpypi=None))
    assert release_index.main([*common, "guard"]) == 0
    assert release_index.main([*common, "verify", "--index", "pypi", "--timeout", "1"]) == 0

    def attested(url: str) -> Document | None:
        return {"attestation_bundles": [{}]}

    monkeypatch.setattr(release_index, "fetch_json", attested)
    attest = [*common, "attestations", "--index", "pypi", "--timeout", "1"]
    assert release_index.main(attest) == 0
