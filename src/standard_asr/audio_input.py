# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio input types for Standard ASR.

This module defines the discriminated union that applications pass to
``transcribe`` and ``start_transcription`` as the ``audio`` argument, together
with the closed convenience coercion from bare Python types.

The discriminated union has six variants -- :class:`AudioPath`,
:class:`AudioBytes`, :class:`AudioArray`, :class:`AudioUrl`,
:class:`AudioBase64` and :class:`AudioStorageUri`. Discrimination is based on
the *explicit type tag*, never
on sniffing the content of a string (see the normative spec, section
"Audio Input & Sample Rate", rule R1). A bare ``str`` is **always** treated as a
local file path; a URL or base64 payload MUST be wrapped explicitly in
:class:`AudioUrl` / :class:`AudioBase64`. This removes all ambiguity and is a
deliberate security boundary against SSRF via attacker-controlled strings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


class InputKind(str, Enum):
    """Closed enumeration of the audio shapes an engine can accept.

    Engines declare the set of shapes they accept via
    ``Properties.accepted_input``. The negotiation layer matches the variant an
    application provides against this set.

    Attributes:
        ARRAY: An already-decoded waveform (NumPy array).
        ENCODED_BYTES: Encoded audio held in memory (e.g. MP3/WAV bytes).
        ENCODED_FILE: An encoded audio file on disk.
        FETCHABLE_URL: A URL the engine/cloud service fetches server-side.
        STORAGE_URI: A provider cloud-storage URI (e.g. ``s3://``, ``gs://``)
            the engine resolves with its own cloud-SDK credentials. Distinct
            from :data:`FETCHABLE_URL`: it is not an HTTPS-fetchable public URL
            and never passes through the standard's SSRF validator.
    """

    ARRAY = "array"
    ENCODED_BYTES = "encoded_bytes"
    ENCODED_FILE = "encoded_file"
    FETCHABLE_URL = "fetchable_url"
    STORAGE_URI = "storage_uri"


@dataclass(frozen=True)
class AudioPath:
    """A local audio file on disk.

    The sample rate is self-describing via the file header.

    Args:
        value: Path to the audio file.
    """

    value: str | os.PathLike[str]


@dataclass(frozen=True)
class AudioBytes:
    """Encoded audio held in memory.

    The sample rate is self-describing via the file header.

    Args:
        data: Encoded audio bytes (e.g. the contents of an MP3/WAV file).
        container: Optional container/format hint (e.g. ``"wav"``, ``"mp3"``).
    """

    data: bytes
    container: str | None = None


@dataclass(frozen=True, eq=False)
class AudioArray:
    """An already-decoded raw waveform.

    Unlike the other variants, an array does not self-describe its sample rate.
    When ``sample_rate`` is ``None`` the global strict / best_effort policy
    decides whether to raise or assume the canonical 16 kHz (see spec R6).

    ``eq`` is disabled because NumPy arrays do not support scalar equality;
    instances therefore compare by identity.

    Args:
        samples: Waveform samples. Canonical form is ``float32`` mono in
            ``[-1, 1]``; multi-channel is ``(n_samples, n_channels)``.
        sample_rate: Sample rate in Hz, or ``None`` if unknown.
    """

    samples: NDArray[np.floating]
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        """Reject a non-floating dtype or a non-positive sample rate at construction.

        Both downstream paths assume floating samples in ``[-1, 1]``: array
        passthrough delivers them unscaled, and WAV encoding scales by full
        int16 range. An integer array (e.g. ``int16`` PCM) would be silently
        mis-scaled by either path -- a wrong-audio result, the cardinal sin --
        so it is rejected at construction with an actionable message.

        A ``sample_rate`` of ``0`` or a negative value is likewise rejected here.
        Otherwise an engine declaring ``accepted_sample_rates="any"`` would be
        handed the bogus rate verbatim (silently, the cardinal sin), while an
        engine declaring a concrete list would crash deep in resampling
        (``ZeroDivisionError`` on the duration check, or a bare ``ValueError``)
        -- the failure mode drifting with engine metadata. Failing at
        construction makes the application bug (a unit/shape mix-up) loud and
        engine-independent, matching ``AudioFormat.sample_rate``'s ``gt=0`` and
        the dtype check above. ``None`` (rate unknown) stays valid: it is
        resolved by the R6 strict/best_effort policy downstream. The number of
        samples is **not** constrained here -- an empty array can be a legitimate
        passthrough boundary input, so emptiness is handled (where it actually
        matters) at resample time, not rejected at construction.

        Raises:
            TypeError: If ``samples`` does not have a floating dtype.
            ValueError: If ``sample_rate`` is not ``None`` and not strictly
                positive.
        """
        if not np.issubdtype(self.samples.dtype, np.floating):
            raise TypeError(
                "AudioArray.samples must have a floating dtype (canonical is "
                f"float32 mono in [-1, 1]); got dtype {self.samples.dtype}. "
                "Convert integer PCM with samples.astype(np.float32) / 32768.0 "
                "(scale to [-1, 1]) before wrapping it in AudioArray."
            )
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError(
                "AudioArray.sample_rate must be a positive number of Hz or None "
                f"(unknown); got {self.sample_rate}. A sample rate cannot be zero "
                "or negative -- check for a unit/shape mix-up (e.g. passing a "
                "sample count or a difference instead of the rate)."
            )


@dataclass(frozen=True)
class AudioUrl:
    """A remote URL the engine or cloud service fetches server-side.

    The semantics are "the server can fetch this". Security constraints
    (HTTPS-only, private/loopback/link-local-address rejection) are enforced
    before the URL is forwarded to an engine, by
    :func:`standard_asr.audio_negotiation.validate_fetchable_url` at plan
    execution (spec R5). In v1 the standard never fetches the URL itself.

    Args:
        value: The remote URL.
    """

    value: str


@dataclass(frozen=True)
class AudioBase64:
    """Base64-encoded (or data-URI) encoded audio.

    Args:
        value: Base64 string or ``data:`` URI.
    """

    value: str


#: Cloud-storage URI schemes a provider engine can resolve with its own SDK
#: credentials. Kept small and extensible-by-constant (no runtime registry):
#: AWS S3 (``s3``), Google Cloud Storage (``gs``/``gcs``), Alibaba OSS (``oss``)
#: and Azure ADLS Gen2 / Blob (``abfs``/``abfss``/``az``/``wasb``/``wasbs``).
STORAGE_URI_SCHEMES: frozenset[str] = frozenset(
    {"s3", "gs", "gcs", "oss", "abfs", "abfss", "az", "wasb", "wasbs"}
)


@dataclass(frozen=True)
class AudioStorageUri:
    """A provider cloud-storage URI the engine resolves with its own credentials.

    Whole engine classes are addressable only by a provider-native storage URI:
    AWS Transcribe batch requires an S3 URI (``Media.MediaFileUri``) and Google
    STT v2 requires a ``gs://`` URI. These are **not** HTTPS-fetchable public
    URLs: the engine resolves them with its own cloud-SDK credentials, so --
    unlike :class:`AudioUrl` -- the standard MUST NOT run the HTTPS public-IP
    SSRF validator over them. The standard never fetches a storage URI itself;
    it only forwards it to an engine that declares ``"storage_uri"`` support.

    Like :class:`AudioUrl` / :class:`AudioBase64`, this variant requires explicit
    construction: a bare ``str`` always coerces to :class:`AudioPath`, never to a
    storage URI (the same SSRF safety stance against attacker-controlled
    strings). The scheme is validated against :data:`STORAGE_URI_SCHEMES` at
    construction; ``file://``, ``http(s)://``, an empty value, or an unknown
    scheme is rejected with a clear error.

    Args:
        value: The storage URI, e.g. ``"s3://bucket/key.wav"`` or
            ``"gs://bucket/key.flac"``. The sample rate is self-describing at
            the remote/server side once the engine resolves it.

    Raises:
        ValueError: If the URI is empty, has no scheme, or uses a scheme outside
            :data:`STORAGE_URI_SCHEMES`.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate the URI scheme against the storage-scheme allowlist.

        Raises:
            ValueError: If the URI is empty, schemeless, or uses a scheme that
                is not an allowlisted provider storage scheme.
        """
        scheme, sep, _ = self.value.partition("://")
        if not sep or not scheme:
            raise ValueError(
                f"AudioStorageUri requires a 'scheme://...' provider storage URI; "
                f"got {self.value!r}. Use one of "
                f"{sorted(STORAGE_URI_SCHEMES)} (e.g. 's3://bucket/key.wav')."
            )
        normalized = scheme.lower()
        if normalized not in STORAGE_URI_SCHEMES:
            raise ValueError(
                f"Unsupported storage URI scheme {scheme!r} in {self.value!r}. "
                f"AudioStorageUri accepts only provider cloud-storage schemes "
                f"{sorted(STORAGE_URI_SCHEMES)}; pass an HTTPS URL as AudioUrl or a "
                "local file as AudioPath instead."
            )


#: The discriminated union accepted by ``transcribe`` / ``start_transcription``.
AudioInput = Union[AudioPath, AudioBytes, AudioArray, AudioUrl, AudioBase64, AudioStorageUri]

#: Bare Python types accepted as a convenience and coerced to :data:`AudioInput`.
AudioInputLike = Union[
    AudioInput,
    str,
    "os.PathLike[str]",
    bytes,
    "NDArray[np.floating]",
    "tuple[NDArray[np.floating], int]",
]


def coerce_audio_input(value: AudioInputLike) -> AudioInput:
    """Coerce a bare Python value into an explicit :data:`AudioInput` variant.

    The coercion is a *closed* convenience mapping (spec, section AI section 3.1):

    =========================  ==========================================
    Bare type                  Coerced to
    =========================  ==========================================
    ``str`` / ``os.PathLike``  :class:`AudioPath` (**always** a local path)
    ``bytes``                  :class:`AudioBytes`
    ``(ndarray, int)``         :class:`AudioArray` with the given sample rate
    ``ndarray``                :class:`AudioArray` with ``sample_rate=None``
    =========================  ==========================================

    A bare ``str`` is never interpreted as a URL, base64 payload, or cloud
    storage URI -- wrap those in :class:`AudioUrl` / :class:`AudioBase64` /
    :class:`AudioStorageUri` explicitly.

    Args:
        value: Either an :data:`AudioInput` variant (returned unchanged) or a
            bare Python value to wrap.

    Returns:
        The corresponding :data:`AudioInput` variant.

    Raises:
        TypeError: If ``value`` is not a recognised audio input type.
    """
    if isinstance(
        value, (AudioPath, AudioBytes, AudioArray, AudioUrl, AudioBase64, AudioStorageUri)
    ):
        return value
    if isinstance(value, (str, os.PathLike)):
        return AudioPath(value)
    if isinstance(value, bytes):
        return AudioBytes(value)
    if isinstance(value, np.ndarray):
        return AudioArray(value)
    # Defensive: the public type narrows to a tuple here, but coercion is a
    # boundary that must reject mistyped runtime input gracefully.
    if isinstance(value, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        return _coerce_array_tuple(value)
    raise TypeError(
        "Unsupported audio input type: "
        f"{type(value).__name__}. Pass an AudioInput variant "
        "(AudioPath/AudioBytes/AudioArray/AudioUrl/AudioBase64/AudioStorageUri) or one of "
        "str/PathLike/bytes/ndarray/(ndarray, sample_rate)."
    )


def _coerce_array_tuple(value: Sequence[object]) -> AudioArray:
    """Coerce a ``(ndarray, sample_rate)`` tuple into :class:`AudioArray`.

    Args:
        value: A two-element sequence of ``(samples, sample_rate)``.

    Returns:
        The corresponding :class:`AudioArray`.

    Raises:
        TypeError: If the tuple shape or element types are wrong.
    """
    if len(value) != 2:
        raise TypeError(
            f"Audio tuple must be (ndarray, sample_rate); got a {len(value)}-element tuple."
        )
    samples, sample_rate = value
    if not isinstance(samples, np.ndarray):
        raise TypeError("First element of the audio tuple must be a NumPy ndarray.")
    # Accept both a builtin ``int`` and a NumPy integer scalar (e.g. the
    # ``np.int64`` a caller gets from ``array.shape`` or ``soundfile.read``);
    # ``bool`` is an ``int`` subclass and is excluded. The rate is normalized to
    # a builtin ``int`` so downstream code never sees a NumPy scalar.
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError("Second element of the audio tuple must be an int sample rate.")
    rate = int(cast("int | np.integer[Any]", sample_rate))
    return AudioArray(cast("NDArray[np.floating]", samples), rate)


__all__ = [
    "STORAGE_URI_SCHEMES",
    "AudioArray",
    "AudioBase64",
    "AudioBytes",
    "AudioInput",
    "AudioInputLike",
    "AudioPath",
    "AudioStorageUri",
    "AudioUrl",
    "InputKind",
    "coerce_audio_input",
]
