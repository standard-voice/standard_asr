# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio input negotiation between application-provided and engine-accepted shapes.

The standard layer negotiates a deterministic, lowest-cost conversion path
between the :data:`~standard_asr.audio_input.AudioInput` variant an application
provides and the set of :class:`~standard_asr.audio_input.InputKind` shapes an
engine accepts. When the provided shape is already accepted, the path is a
zero-cost passthrough. When no path exists, negotiation reports a
:class:`NoViablePath`; the engine layer turns that into an
:class:`~standard_asr.exceptions.IncompatibleAudioInputError` at call time.

This module implements the normative conversion matrix (spec, section
"Audio Input & Sample Rate", rule R3). Sample-rate decisions (R6--R8) are
layered on top by the engine base class and are not part of *kind* negotiation.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from .audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioInput,
    AudioPath,
    AudioStorageUri,
    InputKind,
)
from .exceptions import IncompatibleAudioInputError


class UnsafeAudioUrlError(IncompatibleAudioInputError):
    """An ``AudioUrl`` failed the R5 security policy and MUST NOT be forwarded.

    Raised before a URL is handed to an engine when the URL is not HTTPS, or
    resolves (in whole or in part) to a private / loopback / link-local address
    -- the classic SSRF target set (spec R5.1). Subclasses
    :class:`~standard_asr.exceptions.IncompatibleAudioInputError` so existing
    audio-input error handling catches it, while remaining distinguishable.

    Args:
        url: The offending URL.
        reason: A human-readable explanation of why it was rejected.
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(
            provided="AudioUrl",
            accepted=["fetchable_url"],
            hint=(
                f"Refusing to forward {url!r}: {reason}. URLs MUST be HTTPS and "
                "MUST NOT target private/loopback/link-local addresses (SSRF "
                "defense, spec R5). Pass allow_private_addresses=True only for a "
                "trusted internal endpoint."
            ),
        )


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether an IP is in a private/loopback/link-local/reserved range.

    Covers the SSRF target set from spec R5.1: RFC1918, 127/8, 169.254/16, ::1,
    fc00::/7 (and their relatives, plus reserved/unspecified) -- including
    IPv4-mapped IPv6 addresses, which are unwrapped first.

    Args:
        ip: The parsed address to classify.

    Returns:
        ``True`` if the address MUST be rejected by default.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    )


def validate_fetchable_url(url: str, *, allow_private_addresses: bool = False) -> None:
    """Validate an ``AudioUrl`` against the R5 SSRF policy before forwarding.

    The standard never fetches the URL itself in v1 (spec R5); this only
    validates the literal that will be passed to the engine. The check is:
    HTTPS-only, a parseable host, and -- unless opted out -- every address the
    host resolves to must be public.

    Args:
        url: The URL to validate.
        allow_private_addresses: If ``True``, skip the private/loopback/
            link-local rejection (opt-in for trusted internal endpoints). HTTPS
            is still required.

    Raises:
        UnsafeAudioUrlError: If the URL is not HTTPS, has no host, has a
            malformed port, fails to resolve, or resolves to a disallowed
            address. This is the only exception type the validator raises, so a
            caller (e.g. the server) can map it to a single contracted response.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() != "https":
        raise UnsafeAudioUrlError(url, f"scheme {parts.scheme!r} is not HTTPS")
    host = parts.hostname
    if not host:
        raise UnsafeAudioUrlError(url, "the URL has no host component")

    if allow_private_addresses:
        return

    # An IP literal host is checked directly; a name is resolved and ALL
    # returned addresses must be public (defends against DNS that returns a mix).
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_disallowed_ip(literal):
            raise UnsafeAudioUrlError(url, f"host {host} is a private/reserved address")
        return

    # Parsing parts.port is lazy in urlsplit and raises a bare ValueError for a
    # malformed port (":99999" out of range, ":notaport" non-numeric). Without
    # this guard that ValueError escapes the validator (which only catches
    # gaierror), surfacing as an unexpected 500 in the server path instead of the
    # contracted UnsafeAudioUrlError.
    try:
        port = parts.port or 443
    except ValueError as exc:
        raise UnsafeAudioUrlError(url, f"the URL has a malformed port ({exc})") from exc
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeAudioUrlError(url, f"host {host!r} did not resolve ({exc})") from exc
    if not infos:
        raise UnsafeAudioUrlError(url, f"host {host!r} resolved to no addresses")
    for info in infos:
        addr = info[4][0]
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError as exc:
            # Defense in depth: never trust a malformed address string from the
            # system resolver -- reject rather than letting it through unparsed.
            raise UnsafeAudioUrlError(url, f"unparseable resolved address {addr!r}") from exc
        if _is_disallowed_ip(resolved):
            raise UnsafeAudioUrlError(
                url, f"host {host} resolves to private/reserved address {addr}"
            )


class ConversionOp(str, Enum):
    """A single deterministic step in an audio conversion plan.

    Attributes:
        PASSTHROUGH: No transformation -- the provided shape is delivered as-is.
        READ_FILE: Read a local file into encoded bytes.
        DECODE: Decode encoded audio into a waveform array (needs ``[audio]``).
        ENCODE_WAV: Encode a waveform array into WAV/16-bit PCM bytes (lossy).
        B64_DECODE: Decode a base64 / data-URI payload into encoded bytes.
    """

    PASSTHROUGH = "passthrough"
    READ_FILE = "read_file"
    DECODE = "decode"
    ENCODE_WAV = "encode_wav"
    B64_DECODE = "b64_decode"


@dataclass(frozen=True)
class ConversionPlan:
    """A viable plan for delivering provided audio to an engine.

    Args:
        source_type: Name of the provided variant (e.g. ``"AudioPath"``).
        target_kind: The :class:`InputKind` the engine receives.
        operations: Ordered conversion steps to apply.
        lossy: Whether the conversion loses information (e.g. float32->int16).
        requires_audio_extra: Whether the conversion needs the ``[audio]`` extra
            (decoding compressed formats).
    """

    source_type: str
    target_kind: InputKind
    operations: tuple[ConversionOp, ...]
    lossy: bool
    requires_audio_extra: bool

    @property
    def is_passthrough(self) -> bool:
        """Return whether the plan applies no transformation.

        Returns:
            ``True`` if the provided shape is delivered without conversion.
        """
        return self.operations == (ConversionOp.PASSTHROUGH,)


@dataclass(frozen=True)
class NoViablePath:
    """The result of negotiation when no conversion path exists.

    Args:
        source_type: Name of the provided variant.
        accepted: The engine's accepted input kinds.
        hint: Actionable guidance for resolving the mismatch.
    """

    source_type: str
    accepted: frozenset[InputKind]
    hint: str


def negotiate(
    provided: AudioInput, accepted: set[InputKind] | frozenset[InputKind]
) -> ConversionPlan | NoViablePath:
    """Negotiate a conversion path from a provided shape to an accepted one.

    Implements the normative conversion matrix. Preference order within a
    variant is: zero-cost passthrough, then non-lossy conversion, then lossy
    conversion. Sample-rate handling is applied separately by the engine layer.

    Args:
        provided: The :data:`AudioInput` variant the application provided.
        accepted: The :class:`InputKind` shapes the engine accepts.

    Returns:
        A :class:`ConversionPlan` if a path exists, otherwise a
        :class:`NoViablePath`.
    """
    accepted = frozenset(accepted)
    source = type(provided).__name__

    if isinstance(provided, AudioArray):
        return _negotiate_array(source, accepted)
    if isinstance(provided, AudioPath):
        return _negotiate_path(source, accepted)
    if isinstance(provided, AudioBytes):
        return _negotiate_bytes(source, accepted)
    if isinstance(provided, AudioBase64):
        return _negotiate_base64(source, accepted)
    if isinstance(provided, AudioStorageUri):
        return _negotiate_storage_uri(source, accepted)
    # The only remaining variant is AudioUrl.
    return _negotiate_url(source, accepted)


def can_accept(provided: AudioInput, accepted: set[InputKind] | frozenset[InputKind]) -> bool:
    """Return whether the provided audio can be delivered to the engine.

    A pre-call determination helper: ``True`` iff :func:`negotiate` yields a
    :class:`ConversionPlan`.

    Args:
        provided: The provided :data:`AudioInput` variant.
        accepted: The engine's accepted input kinds.

    Returns:
        ``True`` if a viable conversion path exists.
    """
    return isinstance(negotiate(provided, accepted), ConversionPlan)


def negotiate_or_raise(
    provided: AudioInput, accepted: set[InputKind] | frozenset[InputKind]
) -> ConversionPlan:
    """Negotiate a conversion path or raise on failure.

    Args:
        provided: The provided :data:`AudioInput` variant.
        accepted: The engine's accepted input kinds.

    Returns:
        The viable :class:`ConversionPlan`.

    Raises:
        IncompatibleAudioInputError: If no viable path exists.
    """
    result = negotiate(provided, accepted)
    if isinstance(result, NoViablePath):
        raise IncompatibleAudioInputError(
            provided=result.source_type,
            accepted=sorted(k.value for k in result.accepted),
            hint=result.hint,
        )
    return result


def _negotiate_array(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioArray` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (passthrough to array, or lossy WAV encode), or no viable path.
    """
    if InputKind.ARRAY in accepted:
        return ConversionPlan(source, InputKind.ARRAY, (ConversionOp.PASSTHROUGH,), False, False)
    # The encoder writes to an in-memory BytesIO (R4: MUST NOT touch disk), so the
    # encoded result is ENCODED_BYTES. A file-only engine cannot receive it, so
    # this path is viable only when ENCODED_BYTES is accepted (otherwise a
    # wrong-shape silent result -- R3).
    if InputKind.ENCODED_BYTES in accepted:
        # Encode to WAV/16-bit PCM bytes (float32 -> int16 is lossy).
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.ENCODE_WAV,), True, False
        )
    if InputKind.ENCODED_FILE in accepted:
        return NoViablePath(
            source,
            accepted,
            "This engine accepts only files on disk (encoded_file); the standard "
            "encodes arrays to in-memory bytes and will not write a temp file. "
            "Use an engine that accepts arrays or encoded_bytes.",
        )
    return NoViablePath(
        source,
        accepted,
        "Provide a local file via AudioPath, or use an engine that accepts arrays.",
    )


def _negotiate_path(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioPath` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (passthrough file, read-to-bytes, or decode), or no viable path.
    """
    if InputKind.ENCODED_FILE in accepted:
        return ConversionPlan(
            source, InputKind.ENCODED_FILE, (ConversionOp.PASSTHROUGH,), False, False
        )
    if InputKind.ENCODED_BYTES in accepted:
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.READ_FILE,), False, False
        )
    if InputKind.ARRAY in accepted:
        return ConversionPlan(source, InputKind.ARRAY, (ConversionOp.DECODE,), False, True)
    return NoViablePath(source, accepted, "The standard does not synthesize a fetchable URL in v1.")


def _negotiate_bytes(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioBytes` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (passthrough bytes or decode), or no viable path.
    """
    # In-memory bytes can only be delivered as ENCODED_BYTES. A file-only engine
    # (accepts ENCODED_FILE but not ENCODED_BYTES) cannot consume bytes: the
    # standard MUST NOT write a temp file (R4/D1, BytesIO-only), so producing an
    # ENCODED_BYTES payload would be a silent wrong-shape result (R3). Require
    # ENCODED_BYTES to be accepted before passing through.
    if InputKind.ENCODED_BYTES in accepted:
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.PASSTHROUGH,), False, False
        )
    if InputKind.ARRAY in accepted:
        return ConversionPlan(source, InputKind.ARRAY, (ConversionOp.DECODE,), False, True)
    return NoViablePath(source, accepted, _bytes_only_file_hint(accepted))


def _negotiate_base64(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioBase64` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (b64-decode to bytes, or b64-decode then decode), or no path.
    """
    # As with AudioBytes: base64 decodes to in-memory bytes, which a file-only
    # engine cannot accept without the standard writing a temp file (forbidden).
    if InputKind.ENCODED_BYTES in accepted:
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.B64_DECODE,), False, False
        )
    if InputKind.ARRAY in accepted:
        return ConversionPlan(
            source,
            InputKind.ARRAY,
            (ConversionOp.B64_DECODE, ConversionOp.DECODE),
            False,
            True,
        )
    return NoViablePath(source, accepted, _bytes_only_file_hint(accepted))


def _bytes_only_file_hint(accepted: frozenset[InputKind]) -> str:
    """Build a hint for in-memory audio that no accepted shape can receive.

    Args:
        accepted: Engine-accepted input kinds.

    Returns:
        An actionable hint string.
    """
    if InputKind.ENCODED_FILE in accepted:
        return (
            "This engine accepts only files on disk (encoded_file), not in-memory "
            "bytes; the standard will not write a temporary file (SSRF/TOCTOU "
            "safety). Pass the audio as AudioPath to a real local file."
        )
    return "The standard does not synthesize a fetchable URL in v1."


def _negotiate_url(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioUrl` source.

    In v1 the standard never fetches URLs (SSRF risk); a URL is only viable if
    the engine fetches it server-side. Negotiation is a pure, I/O-free structural
    match; the R5 SSRF *security* validation (HTTPS + non-private-address, which
    needs DNS resolution) runs at execution time via
    :func:`validate_fetchable_url` before the URL is forwarded.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A passthrough plan if the engine accepts URLs, otherwise no viable path.
    """
    if InputKind.FETCHABLE_URL in accepted:
        return ConversionPlan(
            source, InputKind.FETCHABLE_URL, (ConversionOp.PASSTHROUGH,), False, False
        )
    return NoViablePath(
        source,
        accepted,
        "This engine does not fetch URLs; in v1 the standard does not fetch them "
        "either. Provide a local file via AudioPath.",
    )


def _negotiate_storage_uri(
    source: str, accepted: frozenset[InputKind]
) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioStorageUri` source.

    A provider storage URI (``s3://``, ``gs://``, ...) is resolvable only by an
    engine that authenticates it with its own cloud-SDK credentials, so it is
    viable solely as a zero-conversion passthrough to a ``storage_uri`` engine.
    The standard is not an upload-broker and cannot fetch from cloud storage
    without engine credentials, so every other accepted shape FAILs (R3): there
    is no way to turn a credentialed storage URI into an array, encoded bytes, a
    local file, or a public fetchable URL.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A passthrough plan if the engine accepts storage URIs, otherwise no
        viable path.
    """
    if InputKind.STORAGE_URI in accepted:
        return ConversionPlan(
            source, InputKind.STORAGE_URI, (ConversionOp.PASSTHROUGH,), False, False
        )
    return NoViablePath(
        source,
        accepted,
        "This engine does not accept provider storage URIs (storage_uri); the "
        "standard cannot fetch cloud storage without the engine's credentials and "
        "is not an upload-broker. Use an engine that accepts storage_uri, or "
        "download the object yourself and pass it as AudioPath/AudioBytes.",
    )


__all__ = [
    "ConversionOp",
    "ConversionPlan",
    "NoViablePath",
    "UnsafeAudioUrlError",
    "can_accept",
    "negotiate",
    "negotiate_or_raise",
    "validate_fetchable_url",
]
