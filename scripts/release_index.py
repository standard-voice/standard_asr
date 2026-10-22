# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a built ``dist/`` with what PyPI and TestPyPI hold for one version.

The release workflow (``.github/workflows/release.yml``) calls this script at
three points, and the design case ``docs/internal/feat_plan/release-pipeline-redesign.md``
says why each exists:

* ``guard``: before anything is published. For each index, the version is
  either absent, or every file the index already holds has the digest of the
  freshly built file with the same name. A file the index holds with another
  digest, or a file the build did not produce, means the version is burned on
  that index: the remedy is a new version. A file the index lacks is fine, the
  publish step fills it. This is what makes a second dispatch of the same
  version a resume instead of a failure.
* ``verify``: after a publish. The index must hold every built file with the
  same digest. The index can lag a publish by a little, so the command polls
  until the deadline.
* ``attestations``: after a publish. Every built file must have a PEP 740
  provenance record on the index's integrity API. A file that was uploaded
  without one never gets one, so a record missing at the deadline means the
  file name is burned on that index.

Only the standard library is used, so the workflow runs it with the runner's
``python3`` and nothing to install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

#: The two indexes the pipeline publishes to, by the name the workflow uses.
INDEXES: Mapping[str, str] = {"pypi": "https://pypi.org", "testpypi": "https://test.pypi.org"}

PROJECT = "standard-asr"

#: What ``fetch_json`` returns for a 404: the version is absent.
Fetch = Callable[[str], "dict[str, object] | None"]


def fetch_json(url: str, attempts: int = 3) -> dict[str, object] | None:
    """Return the JSON document at ``url``, or ``None`` on a 404.

    A server error, a rate limit (429), or a failed connection says nothing
    about the version, so the read is tried again a few times. A 404 is an
    answer, not a failure.

    Args:
        url: The address to read.
        attempts: How many times to try before giving up.

    Returns:
        The decoded JSON object, or ``None`` when the server answers 404.

    Raises:
        urllib.error.HTTPError: For an HTTP error other than 404 on the last attempt.
        urllib.error.URLError: When the request cannot be made on the last attempt.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https hosts
                document: object = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if (error.code < 500 and error.code != 429) or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
        else:
            if not isinstance(document, dict):
                raise ValueError(f"{url} answered with something other than a JSON object")
            return cast("dict[str, object]", document)
        time.sleep(5)
    raise AssertionError("unreachable")  # pragma: no cover


def index_files(index: str, version: str, fetch: Fetch = fetch_json) -> dict[str, str] | None:
    """Return the files an index holds for ``version``, by name, with their sha256.

    Args:
        index: ``"pypi"`` or ``"testpypi"``.
        version: The project version to look up.
        fetch: The JSON reader, replaceable in tests.

    Returns:
        A mapping from file name to sha256, or ``None`` when the index holds
        no release of that version.
    """
    document = fetch(f"{INDEXES[index]}/pypi/{PROJECT}/{version}/json")
    if document is None:
        return None
    urls = document.get("urls")
    if not isinstance(urls, list):
        raise ValueError(f"{index} answered without a 'urls' list for {PROJECT} {version}")
    files: dict[str, str] = {}
    for raw in cast("list[object]", urls):
        if not isinstance(raw, dict):
            raise ValueError(f"{index} answered with a malformed 'urls' entry: {raw!r}")
        entry = cast("dict[str, object]", raw)
        name = entry.get("filename")
        digests = entry.get("digests")
        sha256 = (
            cast("dict[str, object]", digests).get("sha256") if isinstance(digests, dict) else None
        )
        if not isinstance(name, str) or not isinstance(sha256, str):
            raise ValueError(f"{index} answered with a file without a name or sha256: {entry!r}")
        files[name] = sha256
    return files


def expected_files(version: str) -> set[str]:
    """Return the two file names a release of ``version`` consists of.

    Args:
        version: The version being released.

    Returns:
        The names of the universal wheel and the sdist.
    """
    return {f"standard_asr-{version}-py3-none-any.whl", f"standard_asr-{version}.tar.gz"}


def local_files(dist: Path, version: str) -> dict[str, str]:
    """Return the sha256 of every file in ``dist``, by file name.

    Args:
        dist: The directory ``uv build`` wrote.
        version: The version being released.

    Returns:
        A mapping from file name to sha256.

    Raises:
        ValueError: If ``dist`` does not hold exactly the universal wheel and
            the sdist of ``version``.
    """
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dist.iterdir())
        if path.is_file()
    }
    expected = expected_files(version)
    if set(files) != expected:
        raise ValueError(
            f"{dist} holds {sorted(files) or 'nothing'}; a release of {version} is exactly "
            f"{sorted(expected)}"
        )
    return files


def compare(local: Mapping[str, str], remote: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Compare the built files with the files an index holds.

    Args:
        local: File name to sha256 for the build.
        remote: File name to sha256 for the index.

    Returns:
        Two lists of messages: the files that make the version burned (an
        index file with another digest, or an index file the build did not
        produce), and the files the index does not hold yet.
    """
    burned: list[str] = []
    missing: list[str] = []
    for name, digest in local.items():
        held = remote.get(name)
        if held is None:
            missing.append(name)
        elif held != digest:
            burned.append(f"{name}: index sha256 {held}, built sha256 {digest}")
    for name in remote:
        if name not in local:
            burned.append(f"{name}: the index holds it, but this build did not produce it")
    return burned, missing


def guard(version: str, dist: Path, fetch: Fetch = fetch_json) -> int:
    """Check both indexes before publishing; see the module docstring.

    Args:
        version: The version being released.
        dist: The directory ``uv build`` wrote.
        fetch: The JSON reader, replaceable in tests.

    Returns:
        The process exit code: 0 when every index is absent or consistent,
        1 when the version is burned on an index.
    """
    local = local_files(dist, version)
    exit_code = 0
    for index in INDEXES:
        remote = index_files(index, version, fetch)
        if remote is None:
            print(f"{index}: {PROJECT} {version} is absent; a fresh publish")
            continue
        burned, missing = compare(local, remote)
        if burned:
            exit_code = 1
            print(f"::error::{index} already holds {PROJECT} {version} with other content:")
            for line in burned:
                print(f"::error::  {line}")
            print(
                f"::error::{index} keeps a file name forever, so this version cannot be "
                "released from this build. Bump the version and release that."
            )
        elif missing:
            held = len(local) - len(missing)
            print(f"{index}: holds {held} of {len(local)} files; the publish fills the rest")
        else:
            print(f"{index}: holds every file of this build; the publish is a no-op")
    return exit_code


def verify(
    version: str,
    dist: Path,
    index: str,
    timeout: float,
    fetch: Fetch = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> int:
    """Wait until ``index`` holds every built file with the same digest.

    Args:
        version: The version being released.
        dist: The directory ``uv build`` wrote.
        index: ``"pypi"`` or ``"testpypi"``.
        timeout: Seconds to keep polling before giving up.
        fetch: The JSON reader, replaceable in tests.
        sleep: The pause between polls, replaceable in tests.
        now: The clock, replaceable in tests.

    Returns:
        The process exit code: 0 when the index holds the build, 1 when a
        file differs or the deadline passes.
    """
    local = local_files(dist, version)
    deadline = now() + timeout
    while True:
        remote = index_files(index, version, fetch)
        burned, missing = compare(local, remote or {})
        if burned:
            print(f"::error::{index} holds {PROJECT} {version} with other content:")
            for line in burned:
                print(f"::error::  {line}")
            return 1
        if not missing:
            for name, digest in local.items():
                print(f"{index}: {name} sha256 {digest} matches")
            return 0
        if now() >= deadline:
            print(f"::error::{index} still lacks {', '.join(missing)} after {timeout:.0f} s")
            return 1
        print(f"{index}: waiting for {', '.join(missing)}")
        sleep(15)


def attestations(
    version: str,
    dist: Path,
    index: str,
    timeout: float,
    fetch: Fetch = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> int:
    """Wait until the index's integrity API has a provenance record for every built file.

    A file that was uploaded without an attestation never gets one: the
    index attaches attestations only to a new upload, and a re-upload of an
    existing file is skipped. So a record that is still missing at the
    deadline is a burned file, not a slow index.

    Args:
        version: The version being released.
        dist: The directory ``uv build`` wrote.
        index: ``"pypi"`` or ``"testpypi"``.
        timeout: Seconds to keep polling before giving up.
        fetch: The JSON reader, replaceable in tests.
        sleep: The pause between polls, replaceable in tests.
        now: The clock, replaceable in tests.

    Returns:
        The process exit code: 0 when every file has at least one attestation
        bundle, 1 when the deadline passes first.
    """
    pending = sorted(local_files(dist, version))
    deadline = now() + timeout
    while True:
        still_pending: list[str] = []
        for name in pending:
            document = fetch(f"{INDEXES[index]}/integrity/{PROJECT}/{version}/{name}/provenance")
            bundles = document.get("attestation_bundles") if document else None
            if isinstance(bundles, list) and bundles:
                count = len(cast("list[object]", bundles))
                print(f"{index}: {name} has {count} attestation bundle(s)")
            else:
                still_pending.append(name)
        if not still_pending:
            return 0
        if now() >= deadline:
            names = ", ".join(still_pending)
            print(f"::error::{index} has no attestation for {names} after {timeout:.0f} s")
            print(
                f"::error::A file uploaded without an attestation never gets one; "
                f"{index} keeps the file name, so bump the version."
            )
            return 1
        print(f"{index}: waiting for attestations of {', '.join(still_pending)}")
        pending = still_pending
        sleep(15)


def main(argv: list[str] | None = None) -> int:
    """Run one of the three commands; see the module docstring.

    Args:
        argv: The command line, without the program name.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Compare a built dist/ with what PyPI and TestPyPI hold for one version."
    )
    parser.add_argument("--version", required=True, help="the version being released")
    parser.add_argument("--dist", type=Path, default=Path("dist"), help="the built distributions")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("guard", help="before publishing: absent, or consistent with this build")
    verify_parser = commands.add_parser(
        "verify", help="after a publish: the index holds this build"
    )
    verify_parser.add_argument("--index", choices=sorted(INDEXES), required=True)
    verify_parser.add_argument("--timeout", type=float, default=600.0, help="seconds to poll")
    attest_parser = commands.add_parser("attestations", help="after a publish: provenance records")
    attest_parser.add_argument("--index", choices=sorted(INDEXES), required=True)
    attest_parser.add_argument("--timeout", type=float, default=600.0, help="seconds to poll")
    args = parser.parse_args(argv)
    # Pass the reader at call time, so a test can replace ``fetch_json``.
    if args.command == "guard":
        return guard(args.version, args.dist, fetch_json)
    if args.command == "verify":
        return verify(args.version, args.dist, args.index, args.timeout, fetch_json)
    return attestations(args.version, args.dist, args.index, args.timeout, fetch_json)


if __name__ == "__main__":
    sys.exit(main())
