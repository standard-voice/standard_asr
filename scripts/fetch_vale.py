# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch the pinned Vale binary and print its path.

``scripts/vale.sh`` runs one exact Vale version on every machine, and this
script owns that pin. It downloads the release archive for
the current platform, checks its SHA256 against the table below, extracts the
binary into ``.tools/vale/<version>/``, and prints the binary's path. A cached
binary is reused without touching the network. Set ``VALE=/path/to/vale`` to
skip the fetch, for an air-gapped machine or a distribution package.

Usage::

    python scripts/fetch_vale.py
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

#: The one Vale version the gate runs, locally and in CI.
VALE_VERSION = "3.17.1"

#: SHA256 of every release archive, keyed by Vale's platform tag, copied from
#: the release's checksums file:
#: https://github.com/vale-cli/vale/releases/download/v3.17.1/vale_3.17.1_checksums.txt
ARCHIVE_SHA256: dict[str, str] = {
    "Linux_64-bit": "db947f89f2292e6a0381a61de155f6a5f5cb4cb460ca178ea412ef605559cefd",
    "Linux_arm64": "92d91ebf9ee69ec077379be95cd09e6710ab33d3d5bab66bb482e66ebc80dc23",
    "Windows_64-bit": "0be3fead4e845fc7e740ad8a7e744eee3041ef8c874748a947b7adb63250a642",
    "Windows_arm64": "6fe10e873b09cf31feab2780ac9738ba42d69fb00fac491faadd3cb97c040587",
    "macOS_64-bit": "b37ab999dfd1414d041bd2e94ced103292d634da76f954c385bf789dc7f5f939",
    "macOS_arm64": "80cacf85ef23f53cfdd77355ec41a6ef99aec136f15dfb3517723482f35593f9",
}

_RELEASES = "https://github.com/vale-cli/vale/releases/download"
_ROOT = Path(__file__).resolve().parents[1]
#: The version is part of the path, so a changed pin is a cache miss by construction.
_CACHE_DIR = _ROOT / ".tools" / "vale" / VALE_VERSION


class FetchError(Exception):
    """The binary could not be obtained or verified; the message says why."""


def platform_tag() -> str:
    """Return Vale's platform tag for this operating system and CPU.

    Returns:
        The tag, for example ``macOS_arm64``; it is a key of ``ARCHIVE_SHA256``.

    Raises:
        FetchError: If Vale publishes no archive for this platform.
    """
    system = {"Linux": "Linux", "Darwin": "macOS", "Windows": "Windows"}.get(platform.system())
    machine = platform.machine().lower()
    arch = {"x86_64": "64-bit", "amd64": "64-bit", "arm64": "arm64", "aarch64": "arm64"}
    tag = f"{system}_{arch.get(machine)}"
    if tag not in ARCHIVE_SHA256:
        raise FetchError(
            f"no Vale {VALE_VERSION} archive for {platform.system()} on {machine}; "
            f"supported: {', '.join(sorted(ARCHIVE_SHA256))}"
        )
    return tag


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        raise FetchError(f"download of {url} failed: {exc}") from exc


def _extract_member(archive: Path, member: str, dest: Path) -> None:
    """Copy one member out of a ``.zip`` or ``.tar.gz`` archive, nothing else."""
    try:
        if archive.suffix == ".zip":
            with (
                zipfile.ZipFile(archive) as bundle,
                bundle.open(member) as src,
                dest.open("wb") as out,
            ):
                shutil.copyfileobj(src, out)
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                src = bundle.extractfile(bundle.getmember(member))
                if src is None:
                    raise FetchError(f"{member} in {archive.name} is not a regular file")
                with src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
    except (KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise FetchError(f"{archive.name} does not contain {member}: {exc}") from exc
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _check_version(binary: Path) -> None:
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or VALE_VERSION not in result.stdout:
        raise FetchError(
            f"{binary} reports {result.stdout.strip() or result.stderr.strip()!r}, "
            f"not Vale {VALE_VERSION}; the pin table is wrong"
        )


def fetch() -> Path:
    """Return the path of the pinned Vale binary, downloading it once per version.

    Returns:
        The binary named by ``VALE`` when that variable is set; otherwise the
        verified binary under ``.tools/vale/<version>/``.

    Raises:
        FetchError: If the platform is unsupported, the download fails, the
            archive's SHA256 does not match the pin, or the binary reports a
            different version.
    """
    override = os.environ.get("VALE")
    if override:
        return Path(override)
    tag = platform_tag()
    windows = tag.startswith("Windows")
    binary_name = "vale.exe" if windows else "vale"
    target = _CACHE_DIR / binary_name
    if target.is_file():
        return target
    name = f"vale_{VALE_VERSION}_{tag}.{'zip' if windows else 'tar.gz'}"
    url = f"{_RELEASES}/v{VALE_VERSION}/{name}"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # The temporary directory sits beside the target so the final rename is
    # atomic, and two concurrent runs both end with a byte-identical binary.
    with tempfile.TemporaryDirectory(dir=_CACHE_DIR) as tmp:
        archive = Path(tmp) / name
        _download(url, archive)
        actual = _sha256(archive)
        if actual != ARCHIVE_SHA256[tag]:
            raise FetchError(
                f"SHA256 mismatch for {url}: expected {ARCHIVE_SHA256[tag]}, got {actual}"
            )
        extracted = Path(tmp) / binary_name
        _extract_member(archive, binary_name, extracted)
        _check_version(extracted)
        os.replace(extracted, target)
    return target


def main() -> int:
    """Print the binary's path, or the reason it could not be obtained.

    Returns:
        Zero on success, one on failure.
    """
    try:
        path = fetch()
    except FetchError as exc:
        print(f"fetch_vale: {exc}", file=sys.stderr)
        print(
            "fetch_vale: set VALE=/path/to/vale to use an installed binary instead.",
            file=sys.stderr,
        )
        return 1
    # scripts/vale.sh captures this line with `$(...)`, which strips a
    # trailing "\n" but not the "\r" that Windows text mode would write
    # before it, so the path goes out as bytes with a bare "\n".
    sys.stdout.buffer.write(os.fsencode(path) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
