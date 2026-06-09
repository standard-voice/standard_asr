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
    # synthesize one in v1.
    result = negotiate(AudioPath("a.wav"), {URL})
    assert isinstance(result, NoViablePath)
    assert "fetchable URL" in result.hint


def test_bytes_to_url_only_engine_no_viable_path() -> None:
    # Bytes to a URL-only engine: neither encoded_file nor array is accepted, so
    # the hint falls through to the no-URL-synthesis explanation.
    result = negotiate(AudioBytes(b"x"), {URL})
    assert isinstance(result, NoViablePath)
    assert "fetchable URL" in result.hint


def test_bytes_to_file_and_bytes_engine_passthrough() -> None:
    # When both file and bytes are accepted, bytes still pass through.
    plan = negotiate(AudioBytes(b"x"), {FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.target_kind is BYTES
    assert plan.is_passthrough


# --- AudioStorageUri (H6) ---


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
