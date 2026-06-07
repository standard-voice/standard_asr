# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio input types for Standard ASR.

This module defines the discriminated union that applications pass to
``transcribe`` and ``start_transcription`` as the ``audio`` argument, together
with the closed convenience coercion from bare Python types.

The discriminated union has five variants -- :class:`AudioPath`,
:class:`AudioBytes`, :class:`AudioArray`, :class:`AudioUrl` and
:class:`AudioBase64`. Discrimination is based on the *explicit type tag*, never
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
from typing import TYPE_CHECKING, Union, cast

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
    """

    ARRAY = "array"
    ENCODED_BYTES = "encoded_bytes"
    ENCODED_FILE = "encoded_file"
    FETCHABLE_URL = "fetchable_url"


@dataclass(frozen=True)
class AudioPath:
    """A local audio file on disk.

    The sample rate is self-describing via the file header.

    Args:
        value: Path to the audio file.
    """

    value: str | os.PathLike[str]

    #: The :class:`InputKind` this variant natively provides.
    provided_kind = InputKind.ENCODED_FILE


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

    provided_kind = InputKind.ENCODED_BYTES


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

    provided_kind = InputKind.ARRAY


@dataclass(frozen=True)
class AudioUrl:
    """A remote URL the engine or cloud service fetches server-side.

    The semantics are "the server can fetch this". Security constraints
    (HTTPS-only, private-network rejection) are enforced at negotiation time
    (spec R5). In v1 the standard never fetches the URL itself.

    Args:
        value: The remote URL.
    """

    value: str

    provided_kind = InputKind.FETCHABLE_URL


@dataclass(frozen=True)
class AudioBase64:
    """Base64-encoded (or data-URI) encoded audio.

    Args:
        value: Base64 string or ``data:`` URI.
    """

    value: str

    provided_kind = InputKind.ENCODED_BYTES


#: The discriminated union accepted by ``transcribe`` / ``start_transcription``.
AudioInput = Union[AudioPath, AudioBytes, AudioArray, AudioUrl, AudioBase64]

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

    A bare ``str`` is never interpreted as a URL or base64 payload -- wrap those
    in :class:`AudioUrl` / :class:`AudioBase64` explicitly.

    Args:
        value: Either an :data:`AudioInput` variant (returned unchanged) or a
            bare Python value to wrap.

    Returns:
        The corresponding :data:`AudioInput` variant.

    Raises:
        TypeError: If ``value`` is not a recognised audio input type.
    """
    if isinstance(value, (AudioPath, AudioBytes, AudioArray, AudioUrl, AudioBase64)):
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
        "(AudioPath/AudioBytes/AudioArray/AudioUrl/AudioBase64) or one of "
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
            "Audio tuple must be (ndarray, sample_rate); "
            f"got a {len(value)}-element tuple."
        )
    samples, sample_rate = value
    if not isinstance(samples, np.ndarray):
        raise TypeError("First element of the audio tuple must be a NumPy ndarray.")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("Second element of the audio tuple must be an int sample rate.")
    return AudioArray(cast("NDArray[np.floating]", samples), sample_rate)


__all__ = [
    "AudioArray",
    "AudioBase64",
    "AudioBytes",
    "AudioInput",
    "AudioInputLike",
    "AudioPath",
    "AudioUrl",
    "InputKind",
    "coerce_audio_input",
]
