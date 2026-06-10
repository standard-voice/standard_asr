# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the audio input negotiation matrix."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from standard_asr.audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioPath,
    AudioStorageUri,
    AudioUrl,
    InputKind,
)
from standard_asr.audio_negotiation import (
    ConversionOp,
    ConversionPlan,
    NoViablePath,
    UnsafeAudioUrlError,
    can_accept,
    negotiate,
    negotiate_or_raise,
    validate_fetchable_url,
)
from standard_asr.exceptions import IncompatibleAudioInputError

ARR = InputKind.ARRAY
FILE = InputKind.ENCODED_FILE
BYTES = InputKind.ENCODED_BYTES
URL = InputKind.FETCHABLE_URL
STORAGE = InputKind.STORAGE_URI


def _arr() -> AudioArray:
    return AudioArray(np.zeros(8, dtype=np.float32), 16000)


def test_array_passthrough_to_array_engine() -> None:
    plan = negotiate(_arr(), {ARR})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough
    assert plan.target_kind is ARR


def test_array_encodes_to_wav_for_file_engine() -> None:
    plan = negotiate(_arr(), {FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    # The ENCODE_WAV op is the (lossy) float32->int16 step; the lossy diagnostic
    # is emitted by that op at execution time, not flagged on the plan.
    assert plan.operations == (ConversionOp.ENCODE_WAV,)
    assert plan.target_kind is BYTES


def test_array_to_url_only_engine_fails() -> None:
    result = negotiate(_arr(), {URL})
    assert isinstance(result, NoViablePath)


def test_path_passthrough_file() -> None:
    plan = negotiate(AudioPath("a.wav"), {FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough
    assert plan.target_kind is FILE


def test_path_read_to_bytes_when_only_bytes() -> None:
    plan = negotiate(AudioPath("a.wav"), {BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.READ_FILE,)


def test_path_decode_to_array_needs_extra() -> None:
    plan = negotiate(AudioPath("a.mp3"), {ARR})
    assert isinstance(plan, ConversionPlan)
    # The DECODE op is the step that needs the [audio] extra; the plan no longer
    # carries a duplicate flag (execute_plan never read it).
    assert plan.operations == (ConversionOp.DECODE,)


def test_bytes_passthrough() -> None:
    plan = negotiate(AudioBytes(b"x"), {BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough


def test_bytes_decode_to_array() -> None:
    plan = negotiate(AudioBytes(b"x"), {ARR})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.DECODE,)


def test_base64_decode_to_bytes() -> None:
    plan = negotiate(AudioBase64("AAAA"), {BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.B64_DECODE,)


def test_base64_to_array_two_steps() -> None:
    plan = negotiate(AudioBase64("AAAA"), {ARR})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.B64_DECODE, ConversionOp.DECODE)


def test_url_passthrough_when_accepted() -> None:
    plan = negotiate(AudioUrl("https://x/a.wav"), {URL, FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough
    assert plan.target_kind is URL


def test_url_to_array_engine_v1_fails() -> None:
    result = negotiate(AudioUrl("https://x/a.wav"), {ARR})
    assert isinstance(result, NoViablePath)


def test_can_accept() -> None:
    assert can_accept(_arr(), {ARR}) is True
    assert can_accept(_arr(), {URL}) is False


def test_negotiate_or_raise_ok() -> None:
    assert isinstance(negotiate_or_raise(_arr(), {ARR}), ConversionPlan)


def test_negotiate_or_raise_raises() -> None:
    with pytest.raises(IncompatibleAudioInputError) as exc:
        negotiate_or_raise(_arr(), {URL})
    assert "AudioArray" in str(exc.value)


# --- File-only engine rejects in-memory payloads (R3/R4 correctness) ---


def test_bytes_to_file_only_engine_no_viable_path() -> None:
    # In-memory bytes cannot be delivered to an engine that accepts only files on
    # disk (the standard will not write a temp file). Must be NoViablePath, not a
    # wrong-shape ENCODED_BYTES payload.
    result = negotiate(AudioBytes(b"x"), {FILE})
    assert isinstance(result, NoViablePath)
    assert "encoded_file" in result.hint


def test_base64_to_file_only_engine_no_viable_path() -> None:
    result = negotiate(AudioBase64("AAAA"), {FILE})
    assert isinstance(result, NoViablePath)
    assert "encoded_file" in result.hint


def test_array_to_file_only_engine_no_viable_path() -> None:
    # The encoder produces in-memory bytes (BytesIO, R4); a file-only engine
    # cannot receive them.
    result = negotiate(_arr(), {FILE})
    assert isinstance(result, NoViablePath)


def test_path_to_url_only_engine_no_viable_path() -> None:
    # A local file cannot be turned into a fetchable URL; the standard does not
    # synthesize one in v1. The hint points at the remote shape the engine
    # accepts (uploading is the caller's job).
    result = negotiate(AudioPath("a.wav"), {URL})
    assert isinstance(result, NoViablePath)
    assert "AudioUrl" in result.hint


def test_bytes_to_url_only_engine_no_viable_path() -> None:
    # Bytes to a URL-only engine: neither encoded_file nor array is accepted, so
    # the hint falls through to the upload-it-yourself explanation.
    result = negotiate(AudioBytes(b"x"), {URL})
    assert isinstance(result, NoViablePath)
    assert "AudioUrl" in result.hint


def test_bytes_to_file_and_bytes_engine_passthrough() -> None:
    # When both file and bytes are accepted, bytes still pass through.
    plan = negotiate(AudioBytes(b"x"), {FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.target_kind is BYTES
    assert plan.is_passthrough


# --- Dead-end hints: local data to a remote-only (url/storage) engine ---


def _assert_remote_only_hint(hint: str, *, url: bool, storage: bool) -> None:
    """Assert a hint recommends exactly the engine's accepted remote shapes."""
    # Recommending AudioPath would be a dead end: this engine class rejects
    # every local shape too.
    assert "AudioPath" not in hint
    assert "upload" in hint.lower()
    assert ("AudioUrl" in hint) is url
    assert ("AudioStorageUri" in hint) is storage


def test_array_to_url_only_engine_dead_end_hint() -> None:
    result = negotiate(_arr(), {URL})
    assert isinstance(result, NoViablePath)
    _assert_remote_only_hint(result.hint, url=True, storage=False)


def test_array_to_storage_only_engine_dead_end_hint() -> None:
    result = negotiate(_arr(), {STORAGE})
    assert isinstance(result, NoViablePath)
    _assert_remote_only_hint(result.hint, url=False, storage=True)
    assert "s3://" in result.hint


def test_path_to_storage_only_engine_dead_end_hint() -> None:
    result = negotiate(AudioPath("a.wav"), {STORAGE})
    assert isinstance(result, NoViablePath)
    _assert_remote_only_hint(result.hint, url=False, storage=True)


def test_bytes_to_url_and_storage_engine_hint_lists_both() -> None:
    result = negotiate(AudioBytes(b"x"), {URL, STORAGE})
    assert isinstance(result, NoViablePath)
    _assert_remote_only_hint(result.hint, url=True, storage=True)


def test_base64_to_storage_only_engine_dead_end_hint() -> None:
    result = negotiate(AudioBase64("AAAA"), {STORAGE})
    assert isinstance(result, NoViablePath)
    _assert_remote_only_hint(result.hint, url=False, storage=True)


def test_url_to_storage_only_engine_dead_end_hint() -> None:
    # AudioUrl to a storage-only engine: suggesting AudioPath would be the same
    # dead end, so the hint points at AudioStorageUri instead.
    result = negotiate(AudioUrl("https://x/a.wav"), {STORAGE})
    assert isinstance(result, NoViablePath)
    _assert_remote_only_hint(result.hint, url=False, storage=True)


def test_url_to_array_engine_keeps_audiopath_hint() -> None:
    # The engine accepts a local shape, so suggesting AudioPath stays correct.
    result = negotiate(AudioUrl("https://x/a.wav"), {ARR})
    assert isinstance(result, NoViablePath)
    assert "AudioPath" in result.hint


def test_array_to_empty_accepted_set_has_explicit_hint() -> None:
    # Degenerate declaration: no accepted kinds at all still fails explicitly
    # with a hint, never a misleading AudioPath/AudioUrl recommendation.
    result = negotiate(_arr(), set())
    assert isinstance(result, NoViablePath)
    assert "no accepted input kinds" in result.hint


def test_dead_end_hint_reaches_incompatible_error() -> None:
    # negotiate_or_raise carries the direction-aware hint into the raised error.
    with pytest.raises(IncompatibleAudioInputError) as exc:
        negotiate_or_raise(AudioPath("a.wav"), {URL, STORAGE})
    assert "AudioUrl" in str(exc.value)
    assert "AudioStorageUri" in str(exc.value)


# --- AudioStorageUri ---


def _su() -> AudioStorageUri:
    return AudioStorageUri("s3://bucket/key.wav")


def test_storage_uri_passthrough_to_storage_engine() -> None:
    plan = negotiate(_su(), {STORAGE})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough
    assert plan.target_kind is STORAGE


def test_storage_uri_round_trips_via_can_accept() -> None:
    assert can_accept(_su(), {STORAGE}) is True


@pytest.mark.parametrize("accepted", [{ARR}, {FILE}, {BYTES}, {URL}, {FILE, BYTES}])
def test_storage_uri_fails_to_non_storage_engines(accepted: set[InputKind]) -> None:
    result = negotiate(_su(), accepted)
    assert isinstance(result, NoViablePath)
    assert "storage_uri" in result.hint


def test_storage_uri_negotiate_or_raise_includes_hint() -> None:
    with pytest.raises(IncompatibleAudioInputError) as exc:
        negotiate_or_raise(_su(), {ARR})
    assert "AudioStorageUri" in str(exc.value)
    assert "storage_uri" in str(exc.value)


# --- R5 SSRF validation ---


def test_validate_url_rejects_non_https() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("http://example.com/a.wav")


def test_validate_url_rejects_missing_host() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("https:///a.wav")


def test_validate_url_rejects_loopback_literal() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("https://127.0.0.1/a.wav")


def test_validate_url_rejects_link_local_metadata() -> None:
    # The classic cloud-metadata SSRF target.
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("https://169.254.169.254/latest/meta-data/")


def test_validate_url_rejects_private_rfc1918() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("https://10.0.0.5/a.wav")


def test_validate_url_rejects_ipv6_loopback() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("https://[::1]/a.wav")


def test_validate_url_rejects_ipv4_mapped_ipv6_private() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("https://[::ffff:10.0.0.1]/a.wav")


def test_validate_url_allows_public_ip_literal() -> None:
    # A public IP literal needs no DNS; this must pass.
    validate_fetchable_url("https://93.184.216.34/a.wav")


def test_validate_url_rejects_cgnat_literal() -> None:
    # CGNAT 100.64.0.0/10 (RFC 6598) is neither is_private nor is_loopback etc.,
    # so only the not-is_global hardening catches it. Its classification is
    # stable (non-global) across Python 3.10-3.13.
    with pytest.raises(UnsafeAudioUrlError, match="private/reserved"):
        validate_fetchable_url("https://100.64.0.1/a.wav")


def test_validate_url_rejects_test_net_literal() -> None:
    # TEST-NET-1 (192.0.2.0/24) is consistently non-forwardable: is_private on
    # older Pythons, non-global on 3.12+; rejected either way.
    with pytest.raises(UnsafeAudioUrlError, match="private/reserved"):
        validate_fetchable_url("https://192.0.2.1/a.wav")


def test_validate_url_allows_global_dns_literal() -> None:
    # 8.8.8.8 is is_global on every supported Python; the not-is_global
    # hardening must not over-reject genuinely public addresses.
    validate_fetchable_url("https://8.8.8.8/a.wav")


def test_validate_url_rejects_resolved_cgnat_address(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hardening also applies to resolved (named-host) addresses.
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("100.64.0.1"))
    with pytest.raises(UnsafeAudioUrlError, match="private/reserved"):
        validate_fetchable_url("https://cgnat.example.com/a.wav")


def test_validate_url_malformed_url_raises_unsafe() -> None:
    # urlsplit raises a bare ValueError for an unbalanced IPv6 bracket; the
    # validator must re-raise it as the contracted UnsafeAudioUrlError (its
    # documented single exception type), never leak the ValueError.
    with pytest.raises(UnsafeAudioUrlError, match="malformed"):
        validate_fetchable_url("https://[::1/a.wav")


def test_validate_url_opt_in_allows_private() -> None:
    # The opt-in relaxes the private-address rejection (still HTTPS-only).
    validate_fetchable_url("https://127.0.0.1/a.wav", allow_private_addresses=True)


def test_validate_url_opt_in_still_requires_https() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("http://127.0.0.1/a.wav", allow_private_addresses=True)


def test_validate_url_unresolvable_host_rejected() -> None:
    with pytest.raises(UnsafeAudioUrlError):
        validate_fetchable_url("https://nonexistent.invalid./a.wav")


_AddrInfo = list[tuple[object, object, object, str, tuple[str, int]]]


def _fake_getaddrinfo(*addrs: str) -> Callable[..., _AddrInfo]:
    """Return a typed ``getaddrinfo`` stub resolving to the given addresses."""
    import socket

    infos: _AddrInfo = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (addr, 443)) for addr in addrs
    ]

    def _getaddrinfo(*_a: object, **_k: object) -> _AddrInfo:
        return infos

    return _getaddrinfo


def test_validate_url_resolved_public_host_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    # A named host that resolves to a public address is accepted (the DNS
    # resolution + per-address public check is the path under test).
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
    validate_fetchable_url("https://audio.example.com/a.wav")


def test_validate_url_resolved_unparseable_address_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Defense in depth: if the system resolver returns a malformed address string,
    # re-parsing it raises ValueError and the URL is rejected, not trusted.
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("not-an-ip-address"))
    with pytest.raises(UnsafeAudioUrlError, match="unparseable resolved address"):
        validate_fetchable_url("https://weird.example.com/a.wav")


def test_validate_url_resolved_private_host_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # DNS rebinding defense: a name that resolves to a private address is rejected
    # even though the name itself is not an IP literal.
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.0.5"))
    with pytest.raises(UnsafeAudioUrlError, match="private/reserved"):
        validate_fetchable_url("https://internal.example.com/a.wav")


def test_validate_url_resolves_to_no_addresses_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo())
    with pytest.raises(UnsafeAudioUrlError, match="no addresses"):
        validate_fetchable_url("https://empty.example.com/a.wav")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com:99999/a.wav",  # port out of range
        "https://example.com:notaport/a.wav",  # non-numeric port
    ],
)
def test_validate_url_malformed_port_raises_unsafe(url: str) -> None:
    # A malformed port makes urlsplit raise a bare ValueError when parts.port is
    # accessed; it must be re-raised as the contracted UnsafeAudioUrlError, not
    # escape as an unexpected 500 in the server path.
    with pytest.raises(UnsafeAudioUrlError, match="malformed port"):
        validate_fetchable_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://8.8.8.8:99999/a.wav",  # IP-literal host, port out of range
        "https://8.8.8.8:notaport/a.wav",  # IP-literal host, non-numeric port
    ],
)
def test_validate_url_malformed_port_raises_for_ip_literal_host(url: str) -> None:
    # The malformed-port guard runs BEFORE the IP-vs-name branch, so a public IP
    # literal with an invalid port is rejected symmetrically with the name-host
    # case (it must not be silently accepted just because the IP needs no DNS).
    with pytest.raises(UnsafeAudioUrlError, match="malformed port"):
        validate_fetchable_url(url)


def test_validate_url_allows_public_ip_literal_with_port() -> None:
    # A public IP literal with a normal explicit port still passes (the symmetric
    # port validation only rejects malformed ports, not legal ones).
    validate_fetchable_url("https://93.184.216.34:8443/a.wav")
