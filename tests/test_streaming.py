# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming protocol: events, reduce, session, sync bridge."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel, ValidationError

from standard_asr.contract.artifacts import (
    ArtifactAction,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.capabilities import (
    DeclaredCapabilities,
    DiarizationCap,
    FinalityCap,
    FlagCap,
    StreamingCapabilities,
    StreamTimestampsCap,
    WordTimestampsCap,
)
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
    InvalidSessionUseError,
    StreamClosedError,
    StreamFailedError,
)
from standard_asr.contract.results import Word
from standard_asr.runtime import streaming as streaming_module
from standard_asr.runtime._text import (
    is_combining_mark,
    splits_combining_sequence,
)
from standard_asr.runtime.interface import StandardASR, bind_session_capabilities
from standard_asr.runtime.streaming import (
    ARTIFACT_ACQUISITION_FAILED_CODE,
    ARTIFACT_UNAVAILABLE_CODE,
    DEFAULT_DONE_TIMEOUT,
    DIAG_AUDIO_CURSOR_DECREASED,
    DIAG_LIFECYCLE_AFTER_TERMINAL,
    DIAG_LIFECYCLE_CLOSED_SUPERSEDED,
    DIAG_LIFECYCLE_FINAL_AFTER_FINAL,
    DIAG_LIFECYCLE_PARTIAL_AFTER_FINAL,
    DIAG_LIFECYCLE_RETIRED_RESUPERSEDED,
    DIAG_LOCKED_SPEAKER_REWRITTEN,
    DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE,
    DIAG_STABLE_TEXT_ABANDONED,
    DIAG_STABLE_TEXT_CLAMPED,
    DIAG_STABLE_TEXT_REWRITTEN,
    DIAG_SUPERSEDE_CROSS_SPEAKER_MERGE,
    DIAG_SUPERSEDE_REINTRODUCES_SEGMENT,
    DIAG_SUPERSEDE_UNKNOWN_OLD_ID,
    EventBufferOverflowError,
    StreamDeadlines,
    StreamReducer,
    SyncSession,
    TranscriptionEvent,
    TranscriptionSession,
    _cancel_all_tasks,  # pyright: ignore[reportPrivateUsage]
    _CoalescingBuffer,  # pyright: ignore[reportPrivateUsage]
    _LifecycleGuard,  # pyright: ignore[reportPrivateUsage]
    compose_reduced_text,
    reduce_event,
    validate_stable_text,
)

# --------------------------------------------------------------------------- #
# Stable text: the prefix check and the combining-sequence boundary
# --------------------------------------------------------------------------- #
_ZWNJ = "\u200c"
_ZWJ = "\u200d"
_ACUTE = "\u0301"
#: WOMAN, ZERO WIDTH JOINER, PERSONAL COMPUTER: one emoji built from three
#: code points.
_TECHNOLOGIST = "\U0001f469" + _ZWJ + "\U0001f4bb"


def test_validate_stable_text_requires_a_prefix() -> None:
    assert validate_stable_text("hello", "") is True
    assert validate_stable_text("hello", "hel") is True
    assert validate_stable_text("hello", "hello") is True
    # The comparison is character for character: no case folding, no
    # trimming, no normalization.
    assert validate_stable_text("hello", "Hel") is False
    assert validate_stable_text("hello", "hello!") is False
    assert validate_stable_text("hello", " hel") is False
    assert validate_stable_text("e" + _ACUTE, "\u00e9") is False


@pytest.mark.parametrize(
    ("text", "stable_text", "valid"),
    [
        # Characters above U+FFFF are one character each in a Python string,
        # so a cut between them is an ordinary boundary.
        ("\U00020bb7野家", "\U00020bb7", True),
        ("\U00020bb7野家", "\U00020bb7野", True),
        ("\U0001f600 ok", "\U0001f600", True),
        # A combining acute accent belongs to the "e" before it.
        ("e" + _ACUTE + "x", "e", False),
        ("e" + _ACUTE + "x", "e" + _ACUTE, True),
        # Devanagari KA followed by the vowel sign AA (U+093E, category `Mc`,
        # canonical combining class 0).
        ("\u0915\u093e", "\u0915", False),
        ("\u0915\u093e", "\u0915\u093e", True),
        # Thai KO KAI followed by the vowel sign MAI HAN-AKAT (U+0E31,
        # category Mn, canonical combining class 0).
        ("\u0e01\u0e31\u0e19", "\u0e01", False),
        ("\u0e01\u0e31\u0e19", "\u0e01\u0e31", True),
        # A zero width joiner glues both of its neighbors: a cut before it
        # and a cut right after it both split the emoji.
        (_TECHNOLOGIST, "\U0001f469", False),
        (_TECHNOLOGIST, "\U0001f469" + _ZWJ, False),
        (_TECHNOLOGIST, _TECHNOLOGIST, True),
        # A zero width non-joiner follows its base like a combining mark.
        ("a" + _ZWNJ + "b", "a", False),
        ("a" + _ZWNJ + "b", "a" + _ZWNJ, True),
        # The empty stable text always passes. The whole text passes too,
        # unless it ends with a zero width joiner.
        ("e" + _ACUTE, "", True),
        ("e" + _ACUTE, "e" + _ACUTE, True),
    ],
)
def test_validate_stable_text_boundary_rule(text: str, stable_text: str, valid: bool) -> None:
    assert validate_stable_text(text, stable_text) is valid
    assert splits_combining_sequence(text, len(stable_text)) is not valid


#: WAVING BLACK FLAG, five tag letters that spell the region code of
#: Scotland, and CANCEL TAG: the flag of Scotland, one emoji built from seven
#: code points.
_SCOTLAND = "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f"


@pytest.mark.parametrize(
    ("text", "stable_text"),
    [
        pytest.param("\u0e01\u0e33", "\u0e01", id="before-thai-sara-am"),
        pytest.param("\u0e19\u0e49\u0e33", "\u0e19\u0e49", id="thai-sara-am-after-tone-mark"),
        pytest.param("\u0e81\u0eb3", "\u0e81", id="before-lao-am"),
        pytest.param("\u0915\u094d\u0937", "\u0915\u094d", id="after-virama-in-conjunct"),
        pytest.param("\u0995\u09cd\u09b7", "\u0995\u09cd", id="bengali-conjunct"),
        pytest.param("\u0a95\u0acd\u0ab7", "\u0a95\u0acd", id="gujarati-conjunct"),
        pytest.param("\u0b15\u0b4d\u0b37", "\u0b15\u0b4d", id="odia-conjunct"),
        pytest.param("\u0c15\u0c4d\u0c37", "\u0c15\u0c4d", id="telugu-conjunct"),
        pytest.param("\u0d15\u0d4d\u0d37", "\u0d15\u0d4d", id="malayalam-conjunct"),
        pytest.param("\u1100\u1161", "\u1100", id="between-hangul-jamo"),
        pytest.param("\u1112\u1161\u11ab", "\u1112\u1161", id="before-hangul-final-jamo"),
        pytest.param("\U0001f1ef\U0001f1f5", "\U0001f1ef", id="inside-a-flag"),
        pytest.param("\U0001f44d\U0001f3fd", "\U0001f44d", id="before-skin-tone-modifier"),
        pytest.param(_SCOTLAND, _SCOTLAND[:2], id="inside-emoji-tag-sequence"),
        pytest.param("\uff76\uff9e", "\uff76", id="before-halfwidth-voiced-mark"),
        pytest.param("a\r\nb", "a\r", id="between-cr-and-lf"),
    ],
)
def test_validate_stable_text_does_not_catch_every_cut_inside_a_character(
    text: str, stable_text: str
) -> None:
    # Each cut falls inside one user-perceived character, which the protocol
    # forbids, yet the check passes it: the character after the cut is not a
    # combining mark or a joiner. The rows pin the gaps that the docstring of
    # validate_stable_text documents. If the standard layer gains a full
    # grapheme cluster check, a row that starts failing here belongs in the
    # boundary-rule test above instead.
    assert validate_stable_text(text, stable_text) is True
    assert splits_combining_sequence(text, len(stable_text)) is False


def test_splits_combining_sequence_edges() -> None:
    # A cut at position 0 never splits; a cut at the end splits only when the
    # text ends with a zero width joiner that still waits for its right side.
    assert splits_combining_sequence(_ACUTE + "x", 0) is False
    assert splits_combining_sequence("ab", 2) is False
    assert splits_combining_sequence("a" + _ZWJ, 2) is True


def test_is_combining_mark_uses_the_general_category() -> None:
    # Nonspacing (`Mn`), spacing (`Mc`), and enclosing (`Me`) marks all count,
    # including marks whose canonical combining class is 0.
    assert is_combining_mark(_ACUTE) is True
    assert is_combining_mark("\u093e") is True
    assert is_combining_mark("\u0e31") is True
    assert is_combining_mark("\u20dd") is True
    assert is_combining_mark("a") is False
    assert is_combining_mark(_ZWJ) is False


# --------------------------------------------------------------------------- #
# Stable text on the event model
# --------------------------------------------------------------------------- #
def test_stable_text_defaults_per_event_type() -> None:
    # A partial that names no stable text has none; a final that names none
    # is settled as a whole, and so is a closed final.
    assert TranscriptionEvent.partial("s0", "hello").stable_text == ""
    assert TranscriptionEvent.final("s0", "hello").stable_text == "hello"
    assert TranscriptionEvent.closed("s0", "Hello.").stable_text == "Hello."
    assert TranscriptionEvent(type="partial", segment_id="s0", text="hi").stable_text == ""
    assert TranscriptionEvent(type="final", segment_id="s0", text="hi").stable_text == "hi"
    # By default, every other event type carries no stable text.
    assert TranscriptionEvent.done().stable_text is None
    assert TranscriptionEvent.progress(audio_processed_until=1.0).stable_text is None
    assert TranscriptionEvent.supersede(["a"], ["b"]).stable_text is None
    assert TranscriptionEvent.make_error("boom", recoverable=False).stable_text is None


def test_stable_text_explicit_values_are_kept() -> None:
    partial = TranscriptionEvent.partial("s0", "hello world", stable_text="hello ")
    assert partial.stable_text == "hello "
    settled = TranscriptionEvent.final("s0", "hello", stable_text="hello")
    assert settled.stable_text == "hello"


def test_stable_text_must_be_a_prefix_of_text() -> None:
    with pytest.raises(ValidationError, match="not a prefix"):
        TranscriptionEvent.partial("s0", "hello", stable_text="help")
    with pytest.raises(ValidationError, match="not a prefix"):
        TranscriptionEvent.partial("s0", "hi", stable_text="hi there")
    with pytest.raises(ValidationError, match="not a prefix"):
        TranscriptionEvent.final("s0", "hello", stable_text="Hello")


def test_explicit_none_stable_text_is_rejected_on_content_events() -> None:
    # The defaults apply only to an omitted field: an explicit None on a
    # partial or final is malformed, not a request for the default.
    with pytest.raises(ValidationError, match="MUST carry stable_text as a string"):
        TranscriptionEvent.partial("s0", "hello", stable_text=None)
    with pytest.raises(ValidationError, match="MUST carry stable_text as a string"):
        TranscriptionEvent.final("s0", "hello", stable_text=None)
    with pytest.raises(ValidationError, match="MUST carry stable_text as a string"):
        TranscriptionEvent.model_validate(
            {"type": "partial", "segment_id": "s0", "text": "hi", "stable_text": None}
        )


def test_stable_text_default_applies_to_every_input_form() -> None:
    # The default is resolved from the validated text, so it does not depend
    # on the input being a plain dict or the text already being a str.
    partial = TranscriptionEvent.model_validate(
        MappingProxyType({"type": "partial", "segment_id": "s0", "text": "abc"})
    )
    assert partial.stable_text == ""
    final = TranscriptionEvent.model_validate(
        MappingProxyType({"type": "final", "segment_id": "s0", "text": "abc"})
    )
    assert final.stable_text == "abc"
    from_json = TranscriptionEvent.model_validate_json(
        '{"type": "final", "segment_id": "s0", "text": "abc"}'
    )
    assert from_json.stable_text == "abc"
    coerced = TranscriptionEvent(type="final", segment_id="s0", text=cast("str", b"abc"))
    assert coerced.text == "abc"
    assert coerced.stable_text == "abc"


def test_resolved_stable_text_counts_as_set() -> None:
    # A dump that leaves out unset fields still carries the resolved value:
    # a consumer never applies the default itself.
    partial = TranscriptionEvent.partial("s0", "hello").model_dump(exclude_unset=True)
    assert partial["stable_text"] == ""
    final = TranscriptionEvent.final("s0", "hello").model_dump(exclude_unset=True)
    assert final["stable_text"] == "hello"
    assert "stable_text" not in TranscriptionEvent.done().model_dump(exclude_unset=True)


@pytest.mark.parametrize(
    "event",
    [
        TranscriptionEvent.partial("s0", "hello wor", stable_text="hello "),
        TranscriptionEvent.partial("s0", "hello"),
        TranscriptionEvent.final("s0", "hello world"),
        TranscriptionEvent.closed("s0", "Hello, world."),
        TranscriptionEvent.supersede(["a", "b"], ["c"]),
        TranscriptionEvent.supersede(["a"], []),
        TranscriptionEvent.progress(audio_processed_until=1.5),
        TranscriptionEvent.done(),
        TranscriptionEvent.make_error("boom", recoverable=False),
    ],
    ids=[
        "partial-with-stable-text",
        "partial-default",
        "final-default",
        "closed",
        "supersede",
        "supersede-deletion",
        "progress",
        "done",
        "error",
    ],
)
def test_event_json_round_trip_keeps_stable_text(event: TranscriptionEvent) -> None:
    assert TranscriptionEvent.model_validate(event.model_dump(mode="json")) == event
    assert TranscriptionEvent.model_validate_json(event.model_dump_json()) == event
    wire = event.model_dump(mode="json", exclude_none=True)
    assert TranscriptionEvent.model_validate(wire) == event


def test_serialized_content_frame_always_carries_the_resolved_stable_text() -> None:
    # A client in another language reads stable_text from the frame and never
    # applies a default itself, so the resolved string is always serialized,
    # the empty string included.
    partial = TranscriptionEvent.partial("s0", "hello").model_dump(mode="json", exclude_none=True)
    assert partial["stable_text"] == ""
    final = TranscriptionEvent.final("s0", "hello").model_dump(mode="json", exclude_none=True)
    assert final["stable_text"] == "hello"
    done = TranscriptionEvent.done().model_dump(mode="json", exclude_none=True)
    assert "stable_text" not in done


# --------------------------------------------------------------------------- #
# event model
# --------------------------------------------------------------------------- #
def test_supersede_disjoint_enforced() -> None:
    with pytest.raises(ValueError):
        TranscriptionEvent.supersede(["a"], ["a"])


def test_event_model_rejects_structurally_illegal_events() -> None:
    from pydantic import ValidationError

    # partial/final MUST carry segment_id and text.
    with pytest.raises(ValidationError, match="segment_id and text"):
        TranscriptionEvent(type="partial", segment_id="s0")
    with pytest.raises(ValidationError, match="segment_id and text"):
        TranscriptionEvent(type="final", text="hi")
    # error MUST carry a code.
    with pytest.raises(ValidationError, match="MUST carry a code"):
        TranscriptionEvent(type="error")
    # supersede MUST retire something and keep old/new disjoint.
    with pytest.raises(ValidationError, match="retire at least one"):
        TranscriptionEvent(type="supersede", new_ids=["s1"])
    with pytest.raises(ValidationError, match="disjoint"):
        TranscriptionEvent(type="supersede", old_ids=["s1"], new_ids=["s1"])


def test_event_model_rejects_invalid_time_fields() -> None:
    from pydantic import ValidationError

    # Time-frame fields share Word/Segment's invariant: non-negative, finite.
    # A bad time is rejected AT THE EVENT, not deferred to result reduction (or
    # emitted silently over the wire on a partial/progress event).
    with pytest.raises(ValidationError):
        TranscriptionEvent.final("s0", "hi", start=-0.5)
    with pytest.raises(ValidationError):
        TranscriptionEvent.final("s0", "hi", end=-1.0)
    with pytest.raises(ValidationError):
        TranscriptionEvent.partial("s0", "hi", audio_processed_until=-0.1)
    with pytest.raises(ValidationError):
        TranscriptionEvent.final("s0", "hi", start=float("nan"))
    with pytest.raises(ValidationError):
        TranscriptionEvent.final("s0", "hi", end=float("inf"))
    with pytest.raises(ValidationError):
        TranscriptionEvent(type="progress", gap_start=-1.0)
    with pytest.raises(ValidationError):
        TranscriptionEvent(type="error", code="x", retriable_after=-2.0)
    # A valid non-negative finite span still constructs.
    ev = TranscriptionEvent.final("s0", "hi", start=0.0, end=1.5)
    assert ev.start == 0.0 and ev.end == 1.5
    # progress / done need no segment fields.
    assert TranscriptionEvent(type="progress").type == "progress"


def test_is_terminal() -> None:
    assert TranscriptionEvent.done().is_terminal is True
    assert TranscriptionEvent.make_error("x", recoverable=False).is_terminal is True
    assert TranscriptionEvent.make_error("x", recoverable=True).is_terminal is False
    assert TranscriptionEvent.partial("s", "t").is_terminal is False
    assert TranscriptionEvent.final("s", "t").is_terminal is False


def test_closed_finality() -> None:
    ev = TranscriptionEvent.closed("s0", "Hello.")
    assert ev.type == "final"
    assert ev.finality == "closed"


def test_event_speaker_field_and_validator() -> None:
    # speaker is validated ON THE EVENT (shared rule with Segment/Word): a
    # malformed label must fail here, not defer the crash to
    # the reducer's ``Segment(...)`` or flow silently over the WS wire.
    assert TranscriptionEvent.partial("s0", "hi", speaker="A").speaker == "A"
    assert TranscriptionEvent.final("s0", "hi", speaker="speaker_1").speaker == "speaker_1"
    assert TranscriptionEvent.partial("s0", "hi").speaker is None  # default
    for bad in ("", "  ", "A ", " A"):
        with pytest.raises(ValidationError, match="speaker label"):
            TranscriptionEvent.partial("s0", "hi", speaker=bad)


def test_event_speaker_serializes_on_wire_dump() -> None:
    # The server forwards event.model_dump(mode="json") over the WS wire; pin
    # that speaker rides along.
    ev = TranscriptionEvent.final("s0", "hi", speaker="A")
    assert ev.model_dump(mode="json")["speaker"] == "A"


# --------------------------------------------------------------------------- #
# reduce
# --------------------------------------------------------------------------- #
def test_reduce_event_partial_final_supersede() -> None:
    order: list[str] = []
    texts: dict[str, str] = {}
    separators: dict[str, str] = {}
    reduce_event(
        order,
        texts,
        TranscriptionEvent.partial("s1", "hel", text_separator=""),
        separators=separators,
    )
    reduce_event(
        order,
        texts,
        TranscriptionEvent.final("s1", "hello", text_separator=""),
        separators=separators,
    )
    assert (order, texts) == (["s1"], {"s1": "hello"})
    reduce_event(
        order,
        texts,
        TranscriptionEvent.final("s2", "world", text_separator="\n"),
        separators=separators,
    )
    assert compose_reduced_text(order, texts, separators) == "hello\nworld"
    reduce_event(
        order,
        texts,
        TranscriptionEvent.supersede(["s1", "s2"], ["s3"]),
        separators=separators,
    )
    # The replacement takes over the retired block's position; its text
    # arrives with its own final.
    assert (order, texts) == (["s3"], {})
    assert separators == {}
    reduce_event(
        order,
        texts,
        TranscriptionEvent.final("s3", "hello world"),
        separators=separators,
    )
    assert (order, texts) == (["s3"], {"s3": "hello world"})


def test_stream_reducer_result() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.final("s2", "world", start=1.0, end=2.0))
    result = reducer.result()
    assert result.text == "hello world"
    assert result.segments is not None and len(result.segments) == 2


def test_stream_reducer_supersede_removes() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "wrong", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.supersede(["s1"], ["s2"]))
    reducer.add(TranscriptionEvent.final("s2", "right", start=0.0, end=1.0))
    assert reducer.result().text == "right"


def test_stream_reducer_preserves_exact_text_and_explicit_separators() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", " 你好", text_separator="", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.final("s2", "世界 ", text_separator="", start=1.0, end=2.0))
    reducer.add(TranscriptionEvent.final("s3", "\t", text_separator="\n", start=2.0, end=3.0))
    result = reducer.result()
    assert result.text == " 你好世界 \n\t"
    assert result.segments is not None
    assert [segment.text for segment in result.segments] == [" 你好", "世界 ", "\t"]
    assert [segment.text_separator for segment in result.segments] == ["", "", "\n"]


def test_stream_reducer_revisions_replace_all_segment_projection_fields() -> None:
    first_words = _speaker_words("A")
    closed_words = _speaker_words("B", "B")
    reducer = StreamReducer()
    reducer.add(
        TranscriptionEvent.final(
            "s1",
            "draft",
            text_separator=" ",
            words=first_words,
            speaker="A",
            extra={"revision": 1, "nested": {"state": "draft"}},
        )
    )
    reducer.add(
        TranscriptionEvent.closed(
            "s1",
            "定稿 ",
            text_separator="",
            words=closed_words,
            speaker="B",
            extra={"revision": 2},
        )
    )

    result = reducer.result()
    assert result.text == "定稿 "
    assert result.words == closed_words
    assert result.extra == {}
    assert result.segments is not None
    assert result.segments[0].model_dump() == {
        "start": None,
        "end": None,
        "text": "定稿 ",
        "text_separator": "",
        "words": [word.model_dump() for word in closed_words],
        "speaker": "B",
        "channel": None,
        "avg_logprob": None,
        "no_speech_prob": None,
        "temperature": None,
        "compression_ratio": None,
        "extra": {"revision": 2},
    }


def test_stream_reducer_words_follow_live_segment_order_and_null_rule() -> None:
    old_words = _speaker_words("old")
    kept_words = _speaker_words("kept")
    new_words = _speaker_words("new", "new")
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("old", "old", words=old_words))
    reducer.add(TranscriptionEvent.final("kept", "kept", words=kept_words))
    reducer.add(TranscriptionEvent.supersede(["old"], ["new"]))
    reducer.add(TranscriptionEvent.final("new", "新", text_separator="", words=new_words))

    result = reducer.result()
    assert result.text == "新 kept"
    assert result.words == [*new_words, *kept_words]

    unavailable = StreamReducer()
    unavailable.add(TranscriptionEvent.final("s1", "one"))
    assert unavailable.result().words is None

    requested_but_empty = StreamReducer()
    requested_but_empty.add(TranscriptionEvent.final("s1", "silence", words=[]))
    assert requested_but_empty.result().words == []


def test_stable_text_protects_exact_segment_separators() -> None:
    guard = _LifecycleGuard()
    assert (
        guard.admit(
            TranscriptionEvent.partial("s1", "hello", text_separator=" ", stable_text="hello")
        )
        is not None
    )
    assert (
        guard.admit(
            TranscriptionEvent.final("s1", "hello", text_separator="\n", stable_text="hello")
        )
        is None
    )
    assert [diagnostic.code for diagnostic in guard.diagnostics] == [DIAG_STABLE_TEXT_REWRITTEN]


def _speaker_words(*labels: str | None) -> list[Word]:
    """Build one word per label (0.1 s apart), speaker=None for None labels."""
    return [
        Word(start=i * 0.1, end=i * 0.1 + 0.1, text=f"w{i}", speaker=label)
        for i, label in enumerate(labels)
    ]


def test_stream_reducer_propagates_event_speaker() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello", start=0.0, end=1.0, speaker="A"))
    result = reducer.result()
    assert result.segments is not None
    assert result.segments[0].speaker == "A"


def test_stream_reducer_synthesizes_speaker_from_words() -> None:
    # event.speaker None + speaker-bearing words -> THE pinned segment-speaker rule runs
    # in the reducer (it bypasses the batch post-processing): majority wins.
    reducer = StreamReducer()
    reducer.add(
        TranscriptionEvent.final(
            "s1", "a b c", start=0.0, end=1.0, words=_speaker_words("A", "A", "B")
        )
    )
    result = reducer.result()
    assert result.segments is not None
    assert result.segments[0].speaker == "A"


def test_stream_reducer_event_speaker_wins_over_words() -> None:
    # The event-level (segment-level) speaker is authoritative; word speakers
    # refine, they never overrule it (the inheritance direction).
    reducer = StreamReducer()
    reducer.add(
        TranscriptionEvent.final(
            "s1", "a b", start=0.0, end=1.0, speaker="B", words=_speaker_words("A", "A")
        )
    )
    result = reducer.result()
    assert result.segments is not None
    assert result.segments[0].speaker == "B"


def test_stream_reducer_no_speakers_stays_none() -> None:
    reducer = StreamReducer()
    reducer.add(
        TranscriptionEvent.final("s1", "a b", start=0.0, end=1.0, words=_speaker_words(None, None))
    )
    reducer.add(TranscriptionEvent.final("s2", "c", start=1.0, end=2.0))
    result = reducer.result()
    assert result.segments is not None
    assert [segment.speaker for segment in result.segments] == [None, None]


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #
class _EchoSession(TranscriptionSession):
    """Emits a partial then final per fed chunk; supports backpressure tests."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            sid = f"seg-{index}"
            text = chunk.decode()
            yield TranscriptionEvent.partial(sid, text[:1])
            yield TranscriptionEvent.final(sid, text, start=float(index), end=float(index + 1))
            index += 1


class _YieldingEchoSession(TranscriptionSession):
    """Emits an observable partial then final per fed chunk."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            sid = f"seg-{index}"
            text = chunk.decode()
            yield TranscriptionEvent.partial(sid, text[:1])
            await asyncio.sleep(0)
            yield TranscriptionEvent.final(sid, text, start=float(index), end=float(index + 1))
            index += 1


class _ScriptedSession(TranscriptionSession):
    """Replays a fixed event sequence through the session guard and reducer."""

    def __init__(self, events: Iterable[TranscriptionEvent], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._events = list(events)

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        for event in self._events:
            yield event


class _FloodThenEventSession(TranscriptionSession):
    """Fills the event buffer with partials before yielding one scripted event."""

    def __init__(
        self,
        tail_event: TranscriptionEvent,
        *,
        event_buffer_capacity: int = 2,
        leading_partial_count: int = 2,
    ) -> None:
        super().__init__(event_buffer_capacity=event_buffer_capacity)
        self._tail_event = tail_event
        self._leading_partial_count = leading_partial_count

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        for index in range(self._leading_partial_count):
            yield TranscriptionEvent.partial(f"s{index}", "x")
        yield self._tail_event


async def _collect(session: TranscriptionSession) -> list[TranscriptionEvent]:
    events: list[TranscriptionEvent] = []
    async with session:
        async for event in session:
            events.append(event)
    return events


async def _collect_after_producer_runs(session: TranscriptionSession) -> list[TranscriptionEvent]:
    session.feed([])
    async with session:
        await asyncio.sleep(0.05)
        return [event async for event in session]


def test_attach_initial_diagnostics_surface_through_diagnostics() -> None:
    # The base start_transcription template attaches gating / language
    # diagnostics to the session; they MUST surface through diagnostics(),
    # ordered before the runtime's lifecycle-suppression diagnostics.
    from standard_asr.contract.results import Diagnostic

    session = _EchoSession()
    assert session.diagnostics() == []
    injected = [
        Diagnostic(level="warning", code="unsupported_parameter_ignored", message="dropped"),
    ]
    session._attach_initial_diagnostics(injected)  # pyright: ignore[reportPrivateUsage]
    diags = session.diagnostics()
    assert [d.code for d in diags] == ["unsupported_parameter_ignored"]
    # A second attach replaces (does not accumulate) the initial set.
    session._attach_initial_diagnostics([])  # pyright: ignore[reportPrivateUsage]
    assert session.diagnostics() == []


def test_emit_diagnostic_surfaces_in_session_diagnostics() -> None:
    # emit_diagnostic gives _produce a streaming diagnostics channel: the
    # note surfaces through diagnostics(), the streaming counterpart of batch's
    # result.diagnostics.
    from standard_asr.contract.results import Diagnostic

    class _DiagSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            self.emit_diagnostic(
                code="vad_fallback",
                message="used energy VAD",
                level="warning",
                param="vad",
                provided="webrtc",
                effective="energy",
            )
            return
            yield  # pragma: no cover - makes this an async generator

    async def run() -> list[Diagnostic]:
        async with _DiagSession() as session:
            async for _event in session:
                pass
        return session.diagnostics()

    diags = asyncio.run(run())
    note = next(d for d in diags if d.code == "vad_fallback")
    assert note.level == "warning"
    assert note.param == "vad"
    assert note.provided == "webrtc"
    assert note.effective == "energy"


def test_emit_diagnostic_projects_a_pydantic_model_instead_of_raising() -> None:
    """A structured ``provided``/``effective`` is projected, not a session killer.

    ``Diagnostic.provided`` is a wire-JSON slot, so passing a pydantic
    submodel raised ``ValidationError`` out of a live ``_produce`` --
    ``_run_producer`` caught it and force-terminated the session with
    ``engine_error``: the client lost the rest of the transcript for what
    used to be a working non-fatal note. ``emit_diagnostic`` now applies the
    same :func:`to_json_value` projection the standard's own diagnostics
    (gating, language) already use, so the author intent just works; a value
    with genuinely no JSON form still fails loudly (documented ``Raises:``).
    """
    from standard_asr.engine import to_json_value  # the documented import path

    class _Sub(BaseModel):
        beam: int
        hints: list[str]

    class _ModelDiagSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            self.emit_diagnostic(
                code="request_degraded",
                message="request adjusted",
                provided=_Sub(beam=5, hints=["a", "b"]),
                effective=to_json_value(["a"]),
            )
            yield TranscriptionEvent.final("s1", "hello")

    async def run() -> tuple[list[TranscriptionEvent], list[streaming_module.Diagnostic]]:
        async with _ModelDiagSession() as session:
            events = [event async for event in session]
        return events, session.diagnostics()

    events, diags = asyncio.run(run())
    # The stream survived: no engine_error terminal, the final arrived.
    assert [e.type for e in events if e.type == "final"] == ["final"]
    assert all(e.code != "engine_error" for e in events if e.type == "error")
    note = next(d for d in diags if d.code == "request_degraded")
    assert note.provided == {"beam": 5, "hints": ["a", "b"]}
    assert note.effective == ["a"]


def test_session_feed_mode() -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession()
        session.feed([b"abc", b"de"])
        return await _collect(session)

    events = asyncio.run(run())
    types = [e.type for e in events]
    assert types[-1] == "done"
    finals = [e for e in events if e.type == "final"]
    assert {f.text for f in finals} == {"abc", "de"}


def test_session_feed_bytes_is_a_single_chunk() -> None:
    # A bare bytes-like is ONE chunk, not an iterable of int byte values:
    # feed(b"abc") must yield the chunk b"abc", not 97/98/99.
    captured: dict[str, bool] = {}

    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession()
        session.feed(b"abc")
        captured["replayable"] = session.replayable
        return await _collect(session)

    events = asyncio.run(run())
    finals = [e for e in events if e.type == "final"]
    assert {f.text for f in finals} == {"abc"}
    assert events[-1].type == "done"
    # a wrapped bytes-like is a re-iterable collection -> replayable
    assert captured["replayable"] is True


def test_session_manual_mode_and_result() -> None:
    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"hello")
            await session.send_audio(b"world")
            await session.end_audio()
            events = [e async for e in session]
        return events, session.result().text

    events, text = asyncio.run(run())
    assert events[-1].type == "done"
    assert text == "hello world"


def test_session_status_keeps_failure_details_and_requires_explicit_partial_result() -> None:
    class _FailsAfterPartial(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            self.emit_diagnostic(code="model_note", message="The model returned partial text.")
            yield TranscriptionEvent.partial("s0", "partial")
            yield TranscriptionEvent.make_error("native_failed")

    async def run() -> None:
        session = _FailsAfterPartial()
        assert session.status().state == "running"
        with pytest.raises(InvalidSessionUseError, match="partial_result"):
            session.result()
        session.set_input_duration(1.25)
        session.set_input_duration(1.25)
        events = await _collect(session)
        assert events[-1].code == "native_failed"
        status = session.status()
        assert status.state == "failed"
        assert status.terminal_event is events[-1]
        snapshot = session.partial_result()
        assert snapshot.text == ""
        assert snapshot.duration == 1.25
        assert [diagnostic.code for diagnostic in snapshot.diagnostics] == ["model_note"]
        with pytest.raises(StreamFailedError, match="native_failed") as caught:
            session.result()
        assert caught.value.code == "native_failed"
        with pytest.raises(StreamClosedError, match="input duration"):
            session.set_input_duration(1.25)

    asyncio.run(run())


def test_session_status_marks_context_exit_before_terminal_closed() -> None:
    async def run() -> None:
        session = _EchoSession()
        async with session:
            pass
        assert session.status().state == "closed"
        assert session.status().terminal_event is None
        assert session.partial_result().text == ""
        with pytest.raises(StreamClosedError, match="before it delivered a terminal"):
            session.result()

    asyncio.run(run())


def test_session_status_rejects_inconsistent_terminal_shapes() -> None:
    done = TranscriptionEvent.done()
    error = TranscriptionEvent.make_error("failed")
    partial = TranscriptionEvent.partial("s1", "text")

    for state in ("running", "closed"):
        with pytest.raises(ValidationError, match="cannot have a terminal event"):
            streaming_module.SessionStatus.model_validate({"state": state, "terminal_event": done})
    with pytest.raises(ValidationError, match="requires a terminal event"):
        streaming_module.SessionStatus(state="succeeded")
    with pytest.raises(ValidationError, match="requires a terminal event"):
        streaming_module.SessionStatus(state="failed", terminal_event=partial)
    with pytest.raises(ValidationError, match="requires a done event"):
        streaming_module.SessionStatus(state="succeeded", terminal_event=error)
    with pytest.raises(ValidationError, match="requires a terminal error event"):
        streaming_module.SessionStatus(state="failed", terminal_event=done)


def test_session_metadata_rejects_invalid_or_late_updates() -> None:
    session = _EchoSession()
    for value in ("1.0", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite number"):
            session.set_input_duration(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        session.set_input_duration(-0.1)

    session.set_input_duration(1.0)
    with pytest.raises(ValueError, match="already set"):
        session.set_input_duration(2.0)

    first = session._terminate(TranscriptionEvent.done())  # pyright: ignore[reportPrivateUsage]
    second = session._terminate(  # pyright: ignore[reportPrivateUsage]
        TranscriptionEvent.make_error("too_late")
    )
    assert session.status().terminal_event is first
    assert second.code == "too_late"


def test_session_feed_then_manual_raises() -> None:
    # Mixing feed with manual input is a usage error against a
    # still-live session -> InvalidSessionUseError, not StreamClosedError.
    async def run() -> None:
        session = _EchoSession()
        session.feed([b"x"])
        async with session:
            await session.send_audio(b"y")

    with pytest.raises(InvalidSessionUseError):
        asyncio.run(run())


def test_session_feed_then_manual_raises_mixing_error_even_after_feed_done() -> None:
    # With feed active, send_audio MUST deterministically raise the feed/manual
    # mixing error -- not the "after end_audio" message -- regardless of whether
    # the feed task has already exhausted and set _ended.
    async def run() -> str:
        session = _EchoSession()
        session.feed([b"x"])
        # Let the feed task drain to exhaustion so it sets _ended (the race).
        assert session._feed_task is not None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        await session._feed_task  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert session._ended is True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(InvalidSessionUseError) as exc:
            await session.send_audio(b"y")
        return str(exc.value)

    message = asyncio.run(run())
    assert "cannot mix" in message
    assert "end_audio" not in message


def test_session_manual_then_feed_raises() -> None:
    # Manual-then-feed mixing -> InvalidSessionUseError.
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"y")
            session.feed([b"x"])

    with pytest.raises(InvalidSessionUseError):
        asyncio.run(run())


def test_session_send_after_end_raises() -> None:
    # Genuine lifecycle close (input ended) stays StreamClosedError -- the
    # session really is over, distinct from the usage errors above.
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"y")
            await session.end_audio()
            await session.send_audio(b"z")

    with pytest.raises(StreamClosedError):
        asyncio.run(run())


def test_feed_rejects_str_with_actionable_typeerror() -> None:
    # A str satisfies Iterable[str]; passing a file path to feed is
    # a common slip that would otherwise be consumed character by character (or
    # fail deep inside an engine as a confusing engine_error). feed() MUST
    # fail loudly at the call site with a TypeError pointing to the right API.
    session = _EchoSession()
    with pytest.raises(TypeError, match=r"start_transcription\(audio="):
        session.feed("meeting.wav")  # type: ignore[arg-type]


def test_feed_str_rejection_does_not_claim_feed_mode() -> None:
    # Rejecting a str is a pure argument-type error and MUST NOT
    # mutate session state -- the caller can still drive the session correctly
    # afterward (for example, via manual send_audio).
    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession()
        with pytest.raises(TypeError):
            session.feed("oops.wav")  # type: ignore[arg-type]
        # The session never claimed feed mode, so manual input still works.
        async with session:
            await session.send_audio(b"hi")
            await session.end_audio()
            return [event async for event in session]

    events = asyncio.run(run())
    finals = [e for e in events if e.type == "final"]
    assert finals and finals[0].text == "hi"


def test_feed_accepts_async_iterable_only_aiter() -> None:
    # Feed MUST accept any AsyncIterable (the __aiter__ protocol),
    # not just the stricter AsyncIterator. A custom async source that implements
    # only __aiter__ (the most Pythonic shape) is consumed via `async for`, not
    # mis-routed into the sync branch and failed as input_source_error.
    class _AsyncOnlyIterable:
        """An AsyncIterable that is NOT itself an AsyncIterator (no __anext__)."""

        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        def __aiter__(self) -> AsyncIterator[bytes]:
            async def _gen() -> AsyncIterator[bytes]:
                for chunk in self._chunks:
                    yield chunk

            return _gen()

    # Sanity: the source is an AsyncIterable but NOT an AsyncIterator.
    from collections.abc import AsyncIterator as _ABCAsyncIterator

    src = _AsyncOnlyIterable([b"alpha", b"beta"])
    assert not isinstance(src, _ABCAsyncIterator)

    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession()
        session.feed(src)
        return await _collect(session)

    events = asyncio.run(run())
    finals = [e.text for e in events if e.type == "final"]
    assert finals == ["alpha", "beta"]
    assert events[-1].type == "done"


def test_invalid_session_use_error_taxonomy() -> None:
    # The new usage-error type is a StandardASRError and a ValueError
    # (like ConfigError), but explicitly NOT a StreamClosedError -- so an
    # application can distinguish "I drove the session wrong" from "the session
    # ended" by exception type.
    from standard_asr import InvalidSessionUseError as ExportedInvalidSessionUse
    from standard_asr.contract.exceptions import StandardASRError

    assert ExportedInvalidSessionUse is InvalidSessionUseError
    assert issubclass(InvalidSessionUseError, StandardASRError)
    assert issubclass(InvalidSessionUseError, ValueError)
    assert not issubclass(InvalidSessionUseError, StreamClosedError)
    assert not issubclass(StreamClosedError, InvalidSessionUseError)


def test_session_end_audio_idempotent_manual() -> None:
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"y")
            await session.end_audio()
            await session.end_audio()  # idempotent
            _ = [e async for e in session]

    asyncio.run(run())


def test_session_done_timeout() -> None:
    class _HangSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _HangSession(done_timeout=0.05)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "done_timeout"


def test_session_producer_error_surfaced() -> None:
    class _BoomSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            raise RuntimeError("boom")
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _BoomSession()
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "engine_error"
    # Generic exception text is the deliberate operator channel: the summary
    # carries the real message (type-prefixed) for server-side logging.
    assert events[-1].extra["detail"] == "RuntimeError: boom"


def test_session_producer_validation_error_detail_is_scrubbed() -> None:
    """The engine_error detail is scrubbed AT THE SOURCE, while the chain exists.

    A producer fault wrapping a pydantic ``ValidationError`` (here with the
    input echo copied into the wrapper's own message) used to freeze
    ``str(exc)`` -- echo included -- into ``extra["detail"]``, a plain string
    the server then wrote to operator logs where no later redaction layer
    could recognize it.
    """
    secret = "sk-PRODUCER-SECRET"  # noqa: S105 - test fixture, not a real credential

    class _EngineParams(BaseModel):
        beam: int

    class _LeakySession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            try:
                _EngineParams.model_validate({"beam": secret})
            except ValidationError as exc:
                raise RuntimeError(f"engine rejected params: {exc}") from exc
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _LeakySession()
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    terminal = events[-1]
    assert terminal.type == "error"
    assert terminal.code == "engine_error"
    detail = terminal.extra["detail"]
    assert isinstance(detail, str)
    assert secret not in detail
    assert secret not in terminal.model_dump_json()
    # The fault structure survives for the operator: wrapper type + sanitized
    # ValidationError summary.
    assert "RuntimeError" in detail
    assert "ValidationError" in detail
    # The failing field is NAMED (accident model: field-name-shaped loc
    # components are kept), the value never; the wrapper's own message is
    # withheld when it interpolated the chained error's text.
    assert "beam" in detail


def _leaky_artifact_report(artifact_path: str, action_url: str, secret: str) -> ArtifactReport:
    """Build a valid report whose contents must stay out of the projection.

    The exception constructors validate their payloads, so the leak-bait path,
    URL, and secret ride a real report and action -- the same shapes an honest
    engine attaches.

    Args:
        artifact_path: Absolute local path that must never be projected.
        action_url: HTTPS action URL that must never be projected.
        secret: Marker string that must never be projected.

    Returns:
        The artifact report.
    """
    action = ArtifactAction.model_validate(
        {"kind": "accept_terms", "message": f"Accept; {secret}", "url": action_url}
    )
    requirement = ArtifactRequirement(
        artifact_id="checkpoint",
        label="Checkpoint",
        state="missing",
        required_for_inference=True,
        can_acquire_now=False,
        may_acquire_during_inference=False,
        source_is_mutable=False,
        acquisition_blocker="action_required",
        required_actions=(action,),
        location=Path(artifact_path),
    )
    return ArtifactReport.from_requirements(
        mode="streaming", applicable=True, requirements=(requirement,)
    )


def test_session_artifact_unavailable_error_has_safe_terminal_projection() -> None:
    """Artifact unavailability exposes only its safe standard projection."""
    secret = "artifact-unavailable-secret"
    artifact_path = str(Path(Path.cwd().anchor, "private", "models", "account", "checkpoint"))
    action_url = "https://models.example.test/accept?token=secret"

    class _UnavailableSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            try:
                raise RuntimeError(f"native loader leaked {secret}")
            except RuntimeError as exc:
                raise ArtifactUnavailableError(
                    f"Missing {artifact_path}; open {action_url}; {secret}",
                    reason="missing",
                    report=_leaky_artifact_report(artifact_path, action_url, secret),
                ) from exc
            yield TranscriptionEvent.done()  # pragma: no cover

    events = asyncio.run(_collect(_UnavailableSession()))
    terminal = events[-1]

    assert terminal.type == "error"
    assert terminal.code == ARTIFACT_UNAVAILABLE_CODE == "artifact_unavailable"
    assert terminal.recoverable is False
    assert terminal.retriable_after is None
    assert terminal.extra == {"detail": "Required inference artifacts are unavailable."}
    serialized = terminal.model_dump_json()
    assert secret not in serialized
    assert artifact_path not in serialized
    assert action_url not in serialized
    assert all(event.type != "done" for event in events)


def test_session_artifact_acquisition_error_projects_only_retry_delay() -> None:
    """Artifact acquisition projects retry timing without sensitive context."""
    secret = "artifact-acquisition-secret"
    artifact_path = str(Path(Path.cwd().anchor, "private", "cache", "customer", "checkpoint"))
    action_url = "https://models.example.test/request?credential=secret"

    class _AcquisitionSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            try:
                raise RuntimeError(f"native downloader leaked {secret}")
            except RuntimeError as exc:
                raise ArtifactAcquisitionError(
                    f"Download failed at {artifact_path}; open {action_url}; {secret}",
                    reason="failed",
                    report=_leaky_artifact_report(artifact_path, action_url, secret),
                    required_actions=(
                        ArtifactAction.model_validate(
                            {
                                "kind": "accept_terms",
                                "message": f"Accept; {secret}",
                                "url": action_url,
                            }
                        ),
                    ),
                    retriable_after=2.5,
                ) from exc
            yield TranscriptionEvent.done()  # pragma: no cover

    events = asyncio.run(_collect(_AcquisitionSession()))
    terminal = events[-1]

    assert terminal.type == "error"
    assert terminal.code == ARTIFACT_ACQUISITION_FAILED_CODE == "artifact_acquisition_failed"
    assert terminal.recoverable is False
    assert terminal.retriable_after == 2.5
    assert terminal.extra == {"detail": "Inference-artifact acquisition failed."}
    serialized = terminal.model_dump_json()
    assert secret not in serialized
    assert artifact_path not in serialized
    assert action_url not in serialized
    assert all(event.type != "done" for event in events)


def _assert_input_source_error(events: list[TranscriptionEvent], raw_detail: str) -> None:
    terminal = events[-1]
    assert terminal.type == "error"
    assert terminal.code == "input_source_error"
    assert terminal.recoverable is False
    assert all(event.type != "done" for event in events)
    assert raw_detail not in terminal.model_dump_json()


def test_session_sync_source_error_is_terminal_without_done() -> None:
    secret = "sync-source-secret-token"

    def source() -> Iterator[bytes]:
        yield b"alpha"
        raise RuntimeError(secret)

    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _YieldingEchoSession()
        session.feed(source())
        events = await _collect(session)
        return events, session.partial_result().text

    events, text = asyncio.run(run())
    _assert_input_source_error(events, secret)
    assert any(event.type == "partial" and event.text == "a" for event in events[:-1])
    assert text == "alpha"


def test_session_async_source_error_is_terminal_without_done() -> None:
    secret = "async-source-secret-token"

    async def source() -> AsyncIterator[bytes]:
        yield b"bravo"
        await asyncio.sleep(0)
        raise RuntimeError(secret)

    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _YieldingEchoSession()
        session.feed(source())
        events = await _collect(session)
        assert session.status().state == "failed"
        with pytest.raises(StreamFailedError, match="input_source_error"):
            session.result()
        return events, session.partial_result().text

    events, text = asyncio.run(run())
    _assert_input_source_error(events, secret)
    assert any(event.type == "partial" and event.text == "b" for event in events[:-1])
    assert text == "bravo"


def test_session_source_error_on_first_item_is_terminal_without_done() -> None:
    secret = "first-source-secret-token"

    def source() -> Iterator[bytes]:
        raise RuntimeError(secret)
        yield b"never"  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession()
        session.feed(source())
        return await _collect(session)

    events = asyncio.run(run())
    _assert_input_source_error(events, secret)
    assert [event.type for event in events] == ["error"]


def test_session_exit_cancels_feed_when_audio_queue_is_full() -> None:
    class _NonConsumingSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.Event().wait()
            yield TranscriptionEvent.done()  # pragma: no cover

    async def body() -> None:
        session = _NonConsumingSession(audio_queue_maxsize=1)
        session.feed([b"first", b"second"])
        async with session:
            for _ in range(100):
                if session._audio_queue.full():  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    break
                await asyncio.sleep(0)
            else:  # pragma: no cover - would indicate the regression was not exercised
                raise AssertionError("audio queue did not fill")
            assert session._feed_task is not None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            assert not session._feed_task.done()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    async def run() -> None:
        await asyncio.wait_for(body(), timeout=0.5)

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# sync bridge
# --------------------------------------------------------------------------- #
def test_sync_bridge_feed() -> None:
    with SyncSession(_EchoSession()) as sync:
        sync.feed([b"abc", b"de"])
        events = list(sync)
        assert sync.status().state == "succeeded"
        live_result = sync.result()
    assert events[-1].type == "done"
    assert sync.status().state == "succeeded"
    assert live_result == sync.result()
    assert sync.result().text in ("abc de", "de abc") or "abc" in sync.result().text


def test_sync_bridge_manual() -> None:
    with SyncSession(_EchoSession()) as sync:
        sync.send_audio(b"hi")
        sync.end_audio()
        events = list(sync)
    finals = [e for e in events if e.type == "final"]
    assert finals[0].text == "hi"


def test_sync_bridge_forwards_diagnostics() -> None:
    # The sync bridge must mirror the async session's diagnostics() surface (a
    # first-class, compliance-checked method), not just feed/result -- otherwise a
    # synchronously driven session silently loses the parameter-gating / language
    # diagnostics the async session exposes.
    from standard_asr.contract.results import Diagnostic

    session = _EchoSession()
    session._attach_initial_diagnostics(  # pyright: ignore[reportPrivateUsage]
        [Diagnostic(level="warning", code="unsupported_parameter_ignored", message="dropped")]
    )
    with SyncSession(session) as sync:
        diags = sync.diagnostics()
    assert [d.code for d in diags] == ["unsupported_parameter_ignored"]


def test_sync_bridge_serializes_partial_result_and_diagnostics_with_the_loop() -> None:
    """``partial_result()``/``diagnostics()`` run ON the owned loop while live.

    Both reduce/snapshot state the producer task mutates (a supersede pops
    segments mid-walk of the reading order), and they were the ONLY bridge
    members running on the caller's thread: a rendering loop
    (``for ev in sync: render(sync.partial_result())``) raced the producer and
    crashed with a spurious ``KeyError`` or returned a torn result mixing
    pre- and post-supersede segments. Submission to the owned loop is the
    same mutual exclusion every other member already uses. After teardown
    the producer is gone: the direct call is safe, and the
    result-after-the-``with``-block pattern keeps working.
    """
    from standard_asr.runtime.streaming import (
        _SYNC_BRIDGE_LOOP_THREAD_NAME,  # pyright: ignore[reportPrivateUsage]
    )

    reduce_threads: list[str] = []

    class _ThreadRecordingSession(_EchoSession):
        def partial_result(self) -> streaming_module.TranscriptionResult:
            reduce_threads.append(threading.current_thread().name)
            return super().partial_result()

        def diagnostics(self) -> list[streaming_module.Diagnostic]:
            reduce_threads.append(threading.current_thread().name)
            return super().diagnostics()

    with SyncSession(_ThreadRecordingSession()) as sync:
        sync.feed([b"abc"])
        sync.partial_result()
        sync.diagnostics()
        # partial_result() includes the current diagnostics, so it calls both
        # methods on the owned loop before the explicit diagnostics() query.
        assert reduce_threads == [_SYNC_BRIDGE_LOOP_THREAD_NAME] * 3
        list(sync)
    reduce_threads.clear()
    assert sync.partial_result().text  # still functional after teardown...
    sync.diagnostics()
    # ...via the direct call (no loop thread exists anymore to submit to).
    assert _SYNC_BRIDGE_LOOP_THREAD_NAME not in reduce_threads
    assert len(reduce_threads) == 3


# --------------------------------------------------------------------------- #
# Coalescing buffer: stale partial dropped by terminal-for-segment event
# --------------------------------------------------------------------------- #
async def _drain_buffer(buf: _CoalescingBuffer) -> list[TranscriptionEvent]:
    out: list[TranscriptionEvent] = []
    while True:
        ev = await buf.get()
        if ev is None:
            return out
        out.append(ev)


def test_coalescing_sole_mention_partial_is_kept_and_precedes_final() -> None:
    # The pending partial is the consumer's ONLY mention of s0 -- its
    # reading-order declaration. It is kept and delivered BEFORE the final
    # (never after: no revival), so the delivered stream declares s0 at the
    # same relative position the session's reducer did.
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "hel"))  # pending, not delivered
        buf.put(TranscriptionEvent.final("s0", "hello"))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [e.type for e in events] == ["partial", "final"]
    assert [e.text for e in events] == ["hel", "hello"]


def test_coalescing_partial_dropped_by_same_segment_final_once_declared() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "h"))
        first = await buf.get()  # s0 is now declared to the consumer
        buf.put(TranscriptionEvent.partial("s0", "hel"))  # pending again
        buf.put(TranscriptionEvent.final("s0", "hello"))  # invalidates the partial
        buf.close()
        return [first, *(await _drain_buffer(buf))] if first else []

    events = asyncio.run(run())
    # Once the segment was delivered, the stale pending partial MUST be
    # dropped; only the final follows. No partial may be delivered AFTER the
    # final (would revive a dead segment).
    assert [e.type for e in events] == ["partial", "final"]
    assert [e.text for e in events] == ["h", "hello"]


def test_coalescing_sole_mention_partials_kept_before_supersede() -> None:
    # A supersede retiring never-delivered segments would otherwise reach the
    # consumer's reduce with never-declared old_ids (suppressed there while
    # the session spliced it -- textual drift). The declarations are
    # delivered first; the supersede still retires them, in order.
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s1", "aaa"))
        buf.put(TranscriptionEvent.partial("s2", "bbb"))
        buf.put(TranscriptionEvent.supersede(["s1", "s2"], ["s3"]))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [e.type for e in events] == ["partial", "partial", "supersede"]
    assert [e.segment_id for e in events[:2]] == ["s1", "s2"]


def test_coalescing_partial_dropped_by_supersede_once_declared() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s1", "aa"))
        buf.put(TranscriptionEvent.partial("s2", "bb"))
        for _ in range(2):
            await buf.get()  # both segments now declared to the consumer
        buf.put(TranscriptionEvent.partial("s1", "aaa"))
        buf.put(TranscriptionEvent.partial("s2", "bbb"))
        # supersede retires s1 and s2 -> both pending partials MUST be dropped.
        buf.put(TranscriptionEvent.supersede(["s1", "s2"], ["s3"]))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [e.type for e in events] == ["supersede"]


def test_coalescing_partial_dropped_by_closed_once_declared() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "d"))
        first = await buf.get()  # s0 declared
        buf.put(TranscriptionEvent.partial("s0", "draft"))
        buf.put(TranscriptionEvent.closed("s0", "Final."))  # closed = final variant
        buf.close()
        rest = await _drain_buffer(buf)
        return [first, *rest] if first else rest

    events = asyncio.run(run())
    assert [e.type for e in events] == ["partial", "final"]
    assert events[-1].finality == "closed"


def test_coalescing_latest_partial_wins() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "a"))
        buf.put(TranscriptionEvent.partial("s0", "ab"))
        buf.put(TranscriptionEvent.partial("s0", "abc"))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [e.text for e in events] == ["abc"]


def test_coalescing_partial_after_delivery_starts_fresh_slot() -> None:
    async def run() -> list[str | None]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "a"))
        first = await buf.get()  # delivers and frees the slot
        buf.put(TranscriptionEvent.partial("s0", "b"))  # new pending slot
        buf.put(TranscriptionEvent.final("s0", "final"))  # drops the "b" partial
        buf.close()
        rest = await _drain_buffer(buf)
        return [first.text if first else None, *[e.text for e in rest]]

    texts = asyncio.run(run())
    assert texts == ["a", "final"]


def test_coalescing_carries_forward_speaker() -> None:
    # A blind coalesce replace would silently drop the only event that ever
    # carried the speaker (semantic carry-forward). Neither partial here has
    # stable text, so the protocol permits X->None, and the second event's
    # None could in principle be a deliberate withdrawal -- yet carry-forward
    # still re-presents "A". That is deliberate and safe: the speaker of a
    # partial without stable text is non-actionable, so re-presenting a stale
    # provisional speaker cannot drive a wrong irreversible action, and it keeps
    # the buffer self-consistent with the guard, which -- should s0 later gain
    # stable text -- locks the last-accepted non-None speaker, that is, exactly
    # this "A".
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "hel", speaker="A"))
        buf.put(TranscriptionEvent.partial("s0", "hello"))  # speaker not restated
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [(e.text, e.speaker) for e in events] == [("hello", "A")]


def test_coalescing_newer_speaker_wins() -> None:
    # The newer non-None value always wins; carry-forward never resurrects an
    # old value over a restated one.
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "hel", speaker="A"))
        buf.put(TranscriptionEvent.partial("s0", "hello", speaker="B"))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [(e.text, e.speaker) for e in events] == [("hello", "B")]


def test_coalescing_carries_forward_detected_language() -> None:
    # detected_language had the same latent blind-replace bug; same fix.
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "hel", detected_language="en"))
        buf.put(TranscriptionEvent.partial("s0", "hello"))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [(e.text, e.detected_language) for e in events] == [("hello", "en")]


def test_coalescing_carry_forward_only_within_segment() -> None:
    # Carry-forward scope is one segment_id (falls out of the per-segment slot
    # keying): another segment's partial never inherits s0's speaker.
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "hel", speaker="A"))
        buf.put(TranscriptionEvent.partial("s1", "wor"))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [(e.segment_id, e.speaker) for e in events] == [("s0", "A"), ("s1", None)]


# --------------------------------------------------------------------------- #
# Bounded buffers
# --------------------------------------------------------------------------- #
def test_event_buffer_overflow_raises() -> None:
    # Only NEW distinct-segment partials grow the buffer and can overflow;
    # final / supersede bypass the bound (drop-proof).
    buf = _CoalescingBuffer(capacity=2)
    buf.put(TranscriptionEvent.partial("s0", "a"))
    buf.put(TranscriptionEvent.partial("s1", "b"))
    with pytest.raises(EventBufferOverflowError):
        buf.put(TranscriptionEvent.partial("s2", "c"))


def test_event_buffer_coalesced_partial_does_not_grow() -> None:
    buf = _CoalescingBuffer(capacity=1)
    buf.put(TranscriptionEvent.partial("s0", "a"))
    # Re-coalescing the same segment reuses the slot, never overflows.
    buf.put(TranscriptionEvent.partial("s0", "ab"))
    buf.put(TranscriptionEvent.partial("s0", "abc"))


def test_put_forced_bypasses_capacity() -> None:
    buf = _CoalescingBuffer(capacity=1)
    buf.put(TranscriptionEvent.final("s0", "a"))
    # Terminal events must always land even at capacity.
    buf.put_forced(TranscriptionEvent.done())


def test_drop_proof_slots_consume_the_shared_budget() -> None:
    """Drop-proof events alone can spend the budget and starve the next partial.

    Pins the semantics the class docstring states: ``capacity`` is a budget
    over ALL pending events, and drop-proof finals -- never refused
    themselves -- still consume it. Five pending finals in a capacity-4
    buffer mean the FIRST backpressure-eligible put overflows, even though
    zero backpressure-eligible events are pending. An engine author sizing
    ``event_buffer_capacity`` for a final-heavy stream must budget for the
    finals, not only for the partials.
    """
    buf = _CoalescingBuffer(capacity=4)
    for i in range(5):
        buf.put(TranscriptionEvent.final(f"s{i}", "x"))
    with pytest.raises(EventBufferOverflowError):
        buf.put(TranscriptionEvent.partial("p0", "hi"))


def test_undelivered_segment_holds_two_slots() -> None:
    """A never-delivered segment costs a partial slot AND a final slot.

    The sole-mention carve-out keeps the declaration partial alive when its
    final arrives before the consumer ever saw the segment, so under a slow
    consumer each in-flight segment holds two budget slots. Pins the sizing
    rule the class docstring states: capacity 4 fits TWO undelivered
    segments, and the third segment's partial -- not some later event -- is
    the one refused.
    """
    buf = _CoalescingBuffer(capacity=4)
    buf.put(TranscriptionEvent.partial("s1", "a"))
    buf.put(TranscriptionEvent.final("s1", "aa"))  # partial kept: s1 undeclared
    buf.put(TranscriptionEvent.partial("s2", "b"))
    buf.put(TranscriptionEvent.final("s2", "bb"))  # budget now fully spent
    with pytest.raises(EventBufferOverflowError):
        buf.put(TranscriptionEvent.partial("s3", "c"))


def test_final_supersede_never_dropped_at_capacity() -> None:
    # Fill the buffer to capacity with distinct-segment partials, then assert a
    # final and a supersede still land (drop-proof, not converted to overflow).
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer(capacity=2)
        buf.put(TranscriptionEvent.partial("s0", "a"))
        buf.put(TranscriptionEvent.partial("s1", "b"))  # at capacity now
        # A NEW distinct-segment partial would overflow ...
        with pytest.raises(EventBufferOverflowError):
            buf.put(TranscriptionEvent.partial("s2", "c"))
        # ... but final / supersede MUST bypass the bound.
        buf.put(TranscriptionEvent.final("s3", "f"))
        buf.put(TranscriptionEvent.supersede(["s0"], ["s4"]))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    types = [e.type for e in events]
    assert "final" in types and "supersede" in types
    assert "error" not in types
    # The pending s0 partial is its segment's sole mention, so it is kept and
    # delivered BEFORE the supersede that retires it; s1 is unaffected.
    assert any(e.type == "partial" and e.segment_id == "s1" for e in events)
    s0_index = next(i for i, e in enumerate(events) if e.segment_id == "s0")
    supersede_index = next(i for i, e in enumerate(events) if e.type == "supersede")
    assert s0_index < supersede_index


def test_session_backpressure_overflow_emits_error() -> None:
    class _FloodSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            # Many distinct-segment partials (never coalesced, each grows the
            # buffer) overflow a tiny buffer. Finals/supersedes bypass the bound.
            for i in range(50):
                yield TranscriptionEvent.partial(f"s{i}", "x")

    async def run() -> list[TranscriptionEvent]:
        # Consumer never reads until producer is done -> buffer fills.
        session = _FloodSession(event_buffer_capacity=4)
        session.feed([])
        async with session:
            await asyncio.sleep(0.05)  # let producer run ahead and overflow
            return [e async for e in session]

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "backpressure"
    assert events[-1].recoverable is False


def test_progress_heartbeats_also_trigger_backpressure() -> None:
    # A progress heartbeat takes the bounded path, exactly like a new partial
    # slot: put() special-cases partial (coalesce) and final/supersede/error/done
    # (drop-proof), then falls through to _reserve() for everything else. The
    # spec lets (and during long silent work encourages) engines emit periodic
    # progress, so an engine that emits ONLY progress can still overflow a slow
    # consumer's buffer -- the class docstring once claimed only partials could.
    class _ProgressFloodSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            for _ in range(50):
                yield TranscriptionEvent.progress()

    async def run() -> list[TranscriptionEvent]:
        session = _ProgressFloodSession(event_buffer_capacity=4)
        session.feed([])
        async with session:
            await asyncio.sleep(0.05)  # let the producer run ahead and overflow
            return [e async for e in session]

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "backpressure"
    assert events[-1].recoverable is False


def test_adapter_terminal_error_bypasses_backpressure_overflow() -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _FloodThenEventSession(
            TranscriptionEvent.make_error("provider_engine_error", recoverable=False),
            event_buffer_capacity=2,
        )
        return await _collect_after_producer_runs(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "provider_engine_error"
    assert events[-1].recoverable is False
    assert all(event.code != "backpressure" for event in events)


def test_adapter_done_bypasses_backpressure_overflow_without_duplicate() -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _FloodThenEventSession(
            TranscriptionEvent.done(),
            event_buffer_capacity=2,
        )
        return await _collect_after_producer_runs(session)

    events = asyncio.run(run())
    assert events[-1].type == "done"
    assert [event.type for event in events].count("done") == 1
    assert all(event.type != "error" for event in events)


def test_terminal_event_carries_session_detected_language() -> None:
    # The terminal is the one event always delivered, always last, never
    # dropped -- stamping the session's sticky language onto it makes
    # reduce(delivered).detected_language == result().detected_language hold
    # by construction, in every consumer-speed interleaving.
    events = [
        TranscriptionEvent.partial("s0", "h", detected_language="en"),
        TranscriptionEvent.final("s0", "hi"),
    ]

    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _ScriptedSession(events)
        delivered = await _collect(session)
        return delivered, session.partial_result()

    delivered, result = asyncio.run(run())
    assert delivered[-1].type == "done"
    assert delivered[-1].detected_language == "en"
    assert result.detected_language == "en"


def test_terminal_stamp_closes_the_invalidated_language_drift() -> None:
    # The review counterexample: the only language-carrying event is a
    # pending partial invalidated by a language-less final under a slow
    # consumer. The declaration keep-rule already delivers that partial; the
    # terminal stamp additionally guarantees convergence even for streams
    # where the language rode an event that legitimately coalesced away.
    class _Trace(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.partial("s0", "hel", detected_language="en")
            yield TranscriptionEvent.final("s0", "hello")

    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _Trace()
        delivered = await _collect_after_producer_runs(session)
        return delivered, session.result()

    delivered, result = asyncio.run(run())
    assert result.detected_language == "en"
    replay = StreamReducer()
    for event in delivered:
        replay.add(event)
    assert replay.result().detected_language == "en"


def test_coalesce_reordered_language_converges_via_terminal() -> None:
    # The sticky language is order-sensitive (last non-None wins) but
    # coalescing is not: an in-place coalesce moves a NEWER assignment (fr,
    # coalesced into s0's earlier slot) ahead of an OLDER one (de, pending
    # in s1's later slot), so the delivered running language ends at de
    # while the session's is fr. No per-event carry-forward can fix an
    # ordering problem; the terminal stamp makes the end state converge.
    class _Reorder(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.partial("s0", "a", detected_language="en")
            yield TranscriptionEvent.partial("s1", "b", detected_language="de")
            yield TranscriptionEvent.partial("s0", "ab", detected_language="fr")

    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _Reorder()
        delivered = await _collect_after_producer_runs(session)
        return delivered, session.result()

    delivered, result = asyncio.run(run())
    assert result.detected_language == "fr"
    # Delivered content order really is s0(fr) then s1(de) -- the drift the
    # stamp exists to close.
    content = [e for e in delivered if e.is_content]
    assert [e.detected_language for e in content] == ["fr", "de"]
    replay = StreamReducer()
    for event in delivered:
        replay.add(event)
    assert replay.result().detected_language == "fr"


def test_engine_authored_terminal_language_is_never_overwritten() -> None:
    # An engine-emitted terminal carrying its own language IS the newest
    # assignment: the stamp only fills a None, and the reducer commits the
    # engine's value, so both reads agree on the engine's tag.
    events = [
        TranscriptionEvent.partial("s0", "h", detected_language="en"),
        TranscriptionEvent.final("s0", "hi"),
        TranscriptionEvent.done(detected_language="fr"),
    ]

    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _ScriptedSession(events)
        delivered = await _collect(session)
        return delivered, session.result()

    delivered, result = asyncio.run(run())
    assert delivered[-1].detected_language == "fr"
    assert result.detected_language == "fr"


def test_silent_session_terminal_carries_no_language() -> None:
    # No admitted event carried a language -> nothing to stamp; the done
    # event stays language-less (never fabricate).
    async def run() -> list[TranscriptionEvent]:
        return await _collect(_ScriptedSession([]))

    delivered = asyncio.run(run())
    assert delivered[-1].type == "done"
    assert delivered[-1].detected_language is None


def test_backpressure_terminal_is_stamped_with_session_language() -> None:
    # The forced backpressure error runs the same terminal funnel: the
    # admitted language survives onto the terminal even when the session
    # ends by overflow.
    class _LanguageThenFlood(TranscriptionSession):
        def __init__(self) -> None:
            super().__init__(event_buffer_capacity=2)

        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.partial("s0", "x", detected_language="en")
            for i in range(4):
                yield TranscriptionEvent.partial(f"flood-{i}", "y")

    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _LanguageThenFlood()
        delivered = await _collect_after_producer_runs(session)
        return delivered, session.partial_result()

    delivered, result = asyncio.run(run())
    assert delivered[-1].code == "backpressure"
    assert delivered[-1].detected_language == "en"
    assert result.detected_language == "en"
    replay = StreamReducer()
    for event in delivered:
        replay.add(event)
    assert replay.result().detected_language == "en"


def test_delivered_stream_preserves_reading_order_under_backpressure() -> None:
    # Out-of-order finalization with a slow consumer: with the pending
    # declarations dropped, the delivered stream's first mentions moved to
    # the finals' arrival order, so an app-side reduce of the delivered
    # events read "world hello" while session.result() read "hello world" --
    # a silently reordered transcript (the cardinal sin), reachable with
    # nothing but a slow consumer and finals completing out of order.
    class _OutOfOrderFinals(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.partial("s0", "he")
            yield TranscriptionEvent.partial("s1", "wo")
            yield TranscriptionEvent.final("s1", "world")
            yield TranscriptionEvent.final("s0", "hello")

    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _OutOfOrderFinals()
        events = await _collect_after_producer_runs(session)
        return events, session.partial_result()

    events, result = asyncio.run(run())
    assert result.text == "hello world"
    replay = StreamReducer()
    for event in events:
        replay.add(event)
    replayed = replay.result()
    assert replayed.text == result.text
    assert replayed.segments is not None and result.segments is not None
    assert [s.text for s in replayed.segments] == [s.text for s in result.segments]


def test_delivered_supersede_reduces_identically_when_old_segment_never_delivered() -> None:
    # A supersede retiring a segment whose only mention was still pending:
    # with that declaration dropped, the delivered supersede carried a
    # never-declared old_id -- the app-side reduce SUPPRESSED it (and
    # appended the replacement at the tail: "one two") while the session
    # spliced it in place ("two one"). The kept declaration makes both
    # reads identical, with no suppression diagnostics on the replay.
    class _FastSupersede(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.partial("b", "tw")
            yield TranscriptionEvent.final("a", "one")
            yield TranscriptionEvent.supersede(["b"], ["c"])
            yield TranscriptionEvent.final("c", "two")

    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _FastSupersede()
        events = await _collect_after_producer_runs(session)
        return events, session.result()

    events, result = asyncio.run(run())
    assert result.text == "two one"
    replay = StreamReducer()
    for event in events:
        replay.add(event)
    replayed = replay.result()
    assert replayed.text == result.text
    assert not any(
        d.code.startswith("supersede") or d.code.startswith("lifecycle")
        for d in replayed.diagnostics
    )


def test_overflow_event_is_refused_entirely_never_polluting_result() -> None:
    # The buffer-overflow commit-ordering bug: the producer reduced FIRST and
    # buffered second, so the very event whose put() raised overflow -- an
    # event the consumer can never see -- was already part of result().
    # A refused event must be refused ENTIRELY: its detected_language must
    # not stick, its segment must not claim a reading-order position, and
    # everything actually admitted must still be delivered ahead of the
    # backpressure terminal (delivered-reduce == result()).
    async def run() -> tuple[list[TranscriptionEvent], Any]:
        session = _FloodThenEventSession(
            TranscriptionEvent.partial("s-refused", "x", detected_language="de"),
            event_buffer_capacity=2,
        )
        events = await _collect_after_producer_runs(session)
        return events, session.partial_result()

    events, result = asyncio.run(run())
    assert events[-1].code == "backpressure"
    assert all(event.segment_id != "s-refused" for event in events)
    assert result.detected_language is None
    # Everything the reducer admitted was delivered (the admitted flooding
    # partials precede the terminal), so an app-side reduce of the delivered
    # events agrees with result().
    replay = StreamReducer()
    for event in events:
        replay.add(event)
    assert replay.result().detected_language == result.detected_language
    assert replay.result().text == result.text


def test_non_terminal_overflow_still_emits_backpressure_when_consumer_is_slow() -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _FloodThenEventSession(
            TranscriptionEvent.progress(),
            event_buffer_capacity=2,
        )
        return await _collect_after_producer_runs(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "backpressure"
    assert all(event.type != "progress" for event in events)


def test_same_segment_partials_still_coalesce_when_consumer_is_slow() -> None:
    class _SameSegmentBurstSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.partial("s0", "a")
            yield TranscriptionEvent.partial("s0", "ab")
            yield TranscriptionEvent.partial("s0", "abc")
            yield TranscriptionEvent.done()

    async def run() -> list[TranscriptionEvent]:
        session = _SameSegmentBurstSession(event_buffer_capacity=1)
        return await _collect_after_producer_runs(session)

    events = asyncio.run(run())
    assert [event.type for event in events] == ["partial", "done"]
    assert events[0].text == "abc"


def test_audio_queue_is_bounded() -> None:
    async def run() -> None:
        session = _EchoSession(audio_queue_maxsize=2)
        assert session._audio_queue.maxsize == 2  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Teardown -- sync bridge timeout / no leak
# --------------------------------------------------------------------------- #
class _HangOpenSession(TranscriptionSession):
    async def _open(self) -> None:
        await asyncio.sleep(100)

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        yield TranscriptionEvent.done()  # pragma: no cover


def test_sync_bridge_open_timeout_no_deadlock() -> None:
    sync = SyncSession(_HangOpenSession(), submit_timeout=0.1)
    with pytest.raises(TimeoutError):
        sync.__enter__()
    # Background thread must be torn down, not leaked.
    assert sync._thread.is_alive() is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert sync.is_loop_alive() is False


class _RaiseOpenSession(TranscriptionSession):
    async def _open(self) -> None:
        raise RuntimeError("open boom")

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        return
        yield  # pragma: no cover - makes this an async generator


def test_sync_bridge_open_raise_tears_down_thread() -> None:
    # A non-timeout raise from the engine's _open: __enter__ must tear down its
    # owned loop thread before propagating (a context manager whose __enter__ raises
    # never receives __exit__), so the regular-exception path leaks nothing either.
    sync = SyncSession(_RaiseOpenSession(), submit_timeout=5.0)
    with pytest.raises(RuntimeError, match="open boom"):
        sync.__enter__()
    assert sync.is_loop_alive() is False


# --------------------------------------------------------------------------- #
# Reserved private-attribute guard
# --------------------------------------------------------------------------- #
def test_reserved_session_attrs_matches_base_init() -> None:
    # Drift guard: _RESERVED_SESSION_ATTRS must list exactly the private attributes
    # the base __init__ creates -- otherwise a future base attribute silently falls
    # outside the reserved-attribute guard. vars() after construction is the 21 reserved names
    # plus the name-mangled snapshot store (excluded).
    session = _EchoSession()
    present = {name for name in vars(session) if not name.startswith("_TranscriptionSession__")}
    assert present == set(streaming_module._RESERVED_SESSION_ATTRS)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


class _ClobberBufferSession(TranscriptionSession):
    """A subclass that accidentally reuses the reserved ``_buffer`` name."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer: object = {}  # type: ignore[assignment]  # clobbers base state

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        return
        yield  # pragma: no cover - makes this an async generator


def test_session_clobbering_reserved_attr_fails_loudly() -> None:
    # The cardinal-rule fix: a reserved-attribute clobber must fail with a named,
    # actionable error at session open -- not a cryptic crash deep in the producer.
    session = _ClobberBufferSession()

    async def run() -> None:
        async with session:
            pass  # pragma: no cover - __aenter__ must raise before the body

    with pytest.raises(TypeError, match=r"reserved TranscriptionSession attribute.*_buffer"):
        asyncio.run(run())


def test_session_subclass_own_attributes_pass_guard() -> None:
    # A well-behaved subclass with its own (non-reserved) attributes opens cleanly:
    # the guard must not over-fire on attributes the base does not own.
    class _OwnAttrSession(TranscriptionSession):
        def __init__(self) -> None:
            super().__init__()
            self._my_engine_state = "ok"  # not a reserved name

        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            return
            yield  # pragma: no cover - makes this an async generator

    async def run() -> list[TranscriptionEvent]:
        events: list[TranscriptionEvent] = []
        async with _OwnAttrSession() as session:
            async for event in session:
                events.append(event)
        return events

    events = asyncio.run(run())
    assert events[-1].type == "done"


def test_replace_reserved_attr_rejects_unknown_name() -> None:
    # The guard-aware setter only accepts genuine reserved names: a typo'd or
    # non-reserved name is a programming error, not a silent no-op.
    session = _EchoSession()
    with pytest.raises(KeyError):
        session._replace_reserved_attr("_not_a_reserved_name", object())  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_max_guard_diagnostics_must_be_positive() -> None:
    # The diagnostics bound is validated in __init__ itself (like the other bounds),
    # so a non-positive cap fails loudly -- and the error names the public parameter,
    # not the guard's internal one.
    with pytest.raises(ValueError, match="max_guard_diagnostics must be > 0"):
        _EchoSession(max_guard_diagnostics=0)


def test_sync_bridge_exit_always_shuts_down() -> None:
    sync = SyncSession(_EchoSession(), submit_timeout=5.0)
    sync.__enter__()
    sync.feed([b"hi"])
    list(sync)
    sync.__exit__(None, None, None)
    assert sync._thread.is_alive() is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_sync_bridge_shutdown_is_idempotent() -> None:
    # A second __exit__ / _shutdown must be a no-op (the already-closed guard),
    # never re-tearing-down a stopped loop.
    sync = SyncSession(_EchoSession(), submit_timeout=5.0)
    sync.__enter__()
    sync.feed([b"hi"])
    list(sync)
    sync.__exit__(None, None, None)
    # Second teardown returns immediately via the _closed guard.
    sync._shutdown()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert sync._thread.is_alive() is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_sync_bridge_unbounded_submit_timeout() -> None:
    # submit_timeout=None means the pump never imposes its own deadline (it relies
    # on the session's own terminal-event guarantees). The bridge still completes.
    sync = SyncSession(_EchoSession(), submit_timeout=None)
    with sync:
        sync.feed([b"hello"])
        events = list(sync)
    assert events[-1].type == "done"


def test_cancel_all_tasks_cancels_outstanding() -> None:
    # The teardown helper cancels and awaits every task on the loop except the
    # caller, so no task is destroyed while pending.
    async def run() -> bool:
        async def _forever() -> None:
            await asyncio.sleep(100)

        task = asyncio.ensure_future(_forever())
        await asyncio.sleep(0)  # let it start
        await _cancel_all_tasks()
        return task.cancelled()

    assert asyncio.run(run()) is True


def test_aexit_without_aenter_has_no_tasks_to_cancel() -> None:
    # __aexit__ before __aenter__ ran: there is no producer/feed task, so the
    # cancel/gather is skipped and _close still runs cleanly.
    async def run() -> None:
        session = _EchoSession()
        await session.__aexit__(None, None, None)

    asyncio.run(run())


def test_iterate_stops_on_closed_empty_buffer() -> None:
    # Teardown race guard: if the event buffer is closed with no terminal event
    # ever landing (for example, the producer was canceled mid-flight), the iterator
    # must end cleanly when get() returns None rather than hang.
    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession(done_timeout=5.0)
        # Close the buffer directly without any events -> get() yields None.
        session._buffer.close()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        events: list[TranscriptionEvent] = []
        async for ev in session:
            events.append(ev)
        return events

    assert asyncio.run(run()) == []


def test_cancel_all_tasks_no_other_tasks_is_noop() -> None:
    # With no outstanding tasks the gather branch is skipped without error.
    asyncio.run(_cancel_all_tasks())


def test_abstract_produce_raises_not_implemented() -> None:
    # The abstract base _produce body raises NotImplementedError when invoked
    # directly (for example, a subclass that delegates to super() instead of overriding).
    class _Concrete(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.done()  # pragma: no cover

    with pytest.raises(NotImplementedError):
        # Deliberately invoke the abstract base body (which raises) to prove the
        # contract for a subclass that wrongly delegates to super().
        TranscriptionSession._produce(_Concrete())  # pyright: ignore[reportPrivateUsage, reportAbstractUsage]


# --------------------------------------------------------------------------- #
# Reconnect scaffolding
# --------------------------------------------------------------------------- #
class _ReconnectSession(TranscriptionSession):
    """Drains all audio, notes a reconnect, then finalizes -- continuity test."""

    def __init__(
        self, gap: tuple[float, float], *, content_lost: bool = False, **kw: object
    ) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._gap = gap
        self._content_lost = content_lost

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        chunks: list[bytes] = []
        async for chunk in self.audio_chunks():
            chunks.append(chunk)
        # Simulate the engine re-establishing, replaying the rolling buffer,
        # and signaling the bridged gap after audio has been processed. The
        # engine explicitly decides whether content was lost.
        _ = self.replay_buffer()
        self.note_reconnect(self._gap[0], self._gap[1], content_lost=self._content_lost)
        yield TranscriptionEvent.final("seg-0", b"".join(chunks).decode(), start=0.0)


def test_reconnect_emits_progress_replayable_no_content_lost() -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _ReconnectSession((1.0, 2.0))
        session.feed([b"ab", b"cd"])  # list -> replayable
        assert session.replayable is True
        return await _collect(session)

    events = asyncio.run(run())
    progress = [e for e in events if e.type == "progress"]
    assert len(progress) == 1
    assert progress[0].reconnect is True
    assert progress[0].gap_start == 1.0 and progress[0].gap_end == 2.0
    # Replayable source -> NO content_lost error.
    assert not any(e.type == "error" for e in events)
    assert events[-1].type == "done"


def test_reconnect_adapter_signals_content_lost_emits_progress_then_content_lost() -> None:
    async def run() -> list[TranscriptionEvent]:
        # Async generator source -> non-replayable; engine decides content was
        # lost (the gap could not be replayed) and passes content_lost=True.
        async def gen() -> AsyncIterator[bytes]:
            for _ in range(5):
                yield b"x"

        session = _ReconnectSession((1.0, 2.0), content_lost=True, audio_history_maxlen=1)
        session.feed(gen())
        assert session.replayable is False
        return await _collect(session)

    events = asyncio.run(run())
    progress = [e for e in events if e.type == "progress" and e.reconnect]
    errors = [e for e in events if e.type == "error"]
    assert len(progress) == 1
    # content_lost MUST IMMEDIATELY follow the reconnect progress.
    assert any(e.code == "content_lost" for e in errors)
    cl = next(e for e in errors if e.code == "content_lost")
    assert cl.recoverable is True
    assert cl.is_terminal is False
    assert events.index(cl) == events.index(progress[0]) + 1
    assert events[-1].type == "done"


class _LossyReconnectContinuationSession(TranscriptionSession):
    """Emits more content after a lossy reconnect warning."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        yield TranscriptionEvent.partial("seg-0", "he")
        await asyncio.sleep(0)
        yield TranscriptionEvent.final("seg-0", "hello", start=0.0, end=1.0)
        self.note_reconnect(1.0, 2.0, content_lost=True)
        yield TranscriptionEvent.partial("seg-1", "wo")
        await asyncio.sleep(0)
        yield TranscriptionEvent.final("seg-1", "world", start=2.0, end=3.0)


def test_lossy_reconnect_content_lost_is_non_terminal_and_stream_matches_result() -> None:
    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _LossyReconnectContinuationSession()
        session.feed([])
        events = await _collect(session)
        return events, session.result().text

    events, result_text = asyncio.run(run())
    progress_i = next(i for i, e in enumerate(events) if e.type == "progress" and e.reconnect)
    content_lost_i = next(
        i for i, e in enumerate(events) if e.type == "error" and e.code == "content_lost"
    )
    content_lost = events[content_lost_i]
    stream_text = " ".join(e.text or "" for e in events if e.type == "final")

    assert [e.text for e in events if e.type == "final"] == ["hello", "world"]
    assert content_lost_i == progress_i + 1
    assert content_lost.recoverable is True
    assert content_lost.is_terminal is False
    # content_lost is non-terminal, so the session keeps emitting content after it.
    # The post-reconnect segment's partial ("wo") MAY be coalesced into its final
    # ("world") depending on how promptly the consumer drains -- a legitimate
    # coalescing behavior (only reconnect events are drop-proof;
    # content partials are not) that varies with async scheduling across Python
    # versions. So assert the continuation via the drop-proof final, not the
    # transient partial.
    after = events[content_lost_i + 1 :]
    assert [e.text for e in after if e.type == "final"] == ["world"]
    assert after[-1].type == "done"
    assert result_text == "hello world"
    assert stream_text == result_text


def test_reconnect_no_content_lost_even_after_ring_wraps_many_times() -> None:
    # With content_lost defaulting False, a non-replayable source whose
    # rolling ring has wrapped many times during NORMAL operation MUST NOT get a
    # fabricated content_lost -- the old eviction-based false positive is gone.
    async def run() -> list[TranscriptionEvent]:
        async def gen() -> AsyncIterator[bytes]:
            for _ in range(100):  # far exceeds the tiny ring -> many evictions
                yield b"x"

        session = _ReconnectSession((1.0, 2.0), audio_history_maxlen=2)
        session.feed(gen())
        assert session.replayable is False
        return await _collect(session)

    events = asyncio.run(run())
    assert any(e.type == "progress" and e.reconnect for e in events)
    # No content_lost despite the ring wrapping ~50 times.
    assert not any(e.type == "error" for e in events)
    assert events[-1].type == "done"


class _ReplayCaptureSession(TranscriptionSession):
    """Drains all audio, then records replay_buffer() for inspection."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.captured_replay: list[bytes] = []

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        async for _ in self.audio_chunks():
            pass
        self.captured_replay = self.replay_buffer()
        yield TranscriptionEvent.final("seg-0", "x", start=0.0)


def test_replayable_source_longer_than_ring_replays_in_full() -> None:
    # A replayable list longer than audio_history_maxlen MUST replay in full:
    # "replayable" promises loss-free replay, not just the rolling tail.
    async def run() -> _ReplayCaptureSession:
        chunks = [bytes([i]) for i in range(10)]
        session = _ReplayCaptureSession(audio_history_maxlen=3)
        session.feed(chunks)  # list -> replayable
        assert session.replayable is True
        await _collect(session)
        return session

    session = asyncio.run(run())
    assert session.captured_replay == [bytes([i]) for i in range(10)]


def test_nonreplayable_source_replays_only_bounded_ring() -> None:
    # A non-replayable (live) source can only offer the bounded rolling window.
    async def run() -> _ReplayCaptureSession:
        async def gen() -> AsyncIterator[bytes]:
            for i in range(10):
                yield bytes([i])

        session = _ReplayCaptureSession(audio_history_maxlen=3)
        session.feed(gen())
        assert session.replayable is False
        await _collect(session)
        return session

    session = asyncio.run(run())
    # Only the last 3 chunks survived the bounded ring.
    assert session.captured_replay == [bytes([7]), bytes([8]), bytes([9])]


def test_reconnect_pair_survives_full_buffer_and_stays_adjacent() -> None:
    # A pending progress + content_lost pair MUST survive (and stay adjacent)
    # even when the bounded buffer is already full: reconnect events bypass the
    # capacity bound (reconnect adjacency + "error never dropped").
    async def run() -> list[TranscriptionEvent]:
        async def gen() -> AsyncIterator[bytes]:
            yield b"x"
            yield b"x"

        session = _ReconnectSession(
            (1.0, 2.0), content_lost=True, event_buffer_capacity=2, audio_history_maxlen=1
        )
        session.feed(gen())  # engine signals content_lost -> pair queued
        # Saturate the bounded event buffer so a non-drop-proof drain of the
        # queued reconnect pair would overflow / split them.
        session._buffer.put(TranscriptionEvent.partial("p0", "x"))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        session._buffer.put(TranscriptionEvent.partial("p1", "x"))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return await _collect(session)

    events = asyncio.run(run())
    progress = next(e for e in events if e.type == "progress" and e.reconnect)
    cl = next(e for e in events if e.type == "error" and e.code == "content_lost")
    # Adjacent: content_lost IMMEDIATELY follows the reconnect progress.
    assert events.index(cl) == events.index(progress) + 1


def test_reconnect_drained_on_early_terminal_return() -> None:
    # note_reconnect queued just before the producer yields a terminal event:
    # the reconnect events MUST still be delivered (ahead of the terminal).
    class _ReconnectThenTerminal(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            async for _ in self.audio_chunks():
                pass
            self.note_reconnect(1.0, 2.0)
            yield TranscriptionEvent.done()

    async def run() -> list[TranscriptionEvent]:
        session = _ReconnectThenTerminal()
        session.feed([b"x"])  # replayable -> progress only, no content_lost
        return await _collect(session)

    events = asyncio.run(run())
    assert any(e.type == "progress" and e.reconnect for e in events)
    # The reconnect progress is delivered before the terminal done.
    prog_i = next(i for i, e in enumerate(events) if e.type == "progress")
    done_i = next(i for i, e in enumerate(events) if e.type == "done")
    assert prog_i < done_i


def test_reconnect_drained_on_producer_exception() -> None:
    # note_reconnect queued, then the producer raises: the reconnect events MUST
    # still be delivered (the finally / except-path drain), ahead of the error.
    class _ReconnectThenRaise(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            async for _ in self.audio_chunks():
                pass
            self.note_reconnect(1.0, 2.0)
            raise RuntimeError("boom")
            yield TranscriptionEvent.done()  # pragma: no cover - unreachable

    async def run() -> list[TranscriptionEvent]:
        session = _ReconnectThenRaise()
        session.feed([b"x"])  # replayable -> progress only
        return await _collect(session)

    events = asyncio.run(run())
    prog_i = next(i for i, e in enumerate(events) if e.type == "progress" and e.reconnect)
    err_i = next(i for i, e in enumerate(events) if e.type == "error")
    assert prog_i < err_i
    assert events[err_i].code == "engine_error"


def test_reconnect_events_delivered_while_producer_blocked_on_slow_reconnect() -> None:
    # A7: note_reconnect MUST flush its progress + content_lost promptly so the
    # consumer sees the reconnect notification even while the producer is BLOCKED
    # mid-reconnect (awaiting indefinitely without yielding another event).
    class _BlockOnReconnect(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            self.note_reconnect(1.0, 2.0, content_lost=True)
            await asyncio.Event().wait()  # stuck reconnect: never yields again.
            yield TranscriptionEvent.done()  # pragma: no cover - unreachable

    async def run() -> tuple[TranscriptionEvent, TranscriptionEvent]:
        session = _BlockOnReconnect()
        session.feed([])
        async with session:
            ait = session.__aiter__()
            # Both events arrive WITHOUT any further produced event: a short
            # timeout proves they were flushed promptly, not on the next yield.
            progress = await asyncio.wait_for(ait.__anext__(), timeout=1.0)
            content_lost = await asyncio.wait_for(ait.__anext__(), timeout=1.0)
            # No third event is available (the producer is still blocked).
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ait.__anext__(), timeout=0.1)
            return progress, content_lost

    progress, content_lost = asyncio.run(run())
    assert progress.type == "progress" and progress.reconnect is True
    assert progress.gap_start == 1.0 and progress.gap_end == 2.0
    # content_lost IMMEDIATELY follows the progress and is non-terminal.
    assert content_lost.type == "error" and content_lost.code == "content_lost"
    assert content_lost.recoverable is True
    assert content_lost.is_terminal is False


# --------------------------------------------------------------------------- #
# Lifecycle enforcement and stable text that only grows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("capabilities", [None, StreamingCapabilities()])
def test_guard_suppresses_partial_after_final(capabilities: StreamingCapabilities | None) -> None:
    guard = _LifecycleGuard(capabilities=capabilities)
    assert guard.admit(TranscriptionEvent.final("s0", "done")) is not None
    assert guard.admit(TranscriptionEvent.partial("s0", "oops")) is None
    assert [d.code for d in guard.diagnostics] == ["lifecycle_partial_after_final"]


def test_guard_suppresses_events_after_closed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "x"))
    guard.admit(TranscriptionEvent.closed("s0", "X."))
    assert guard.admit(TranscriptionEvent.partial("s0", "y")) is None
    assert guard.admit(TranscriptionEvent.final("s0", "z")) is None


def test_guard_suppresses_closed_in_supersede_old_ids() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "x"))
    guard.admit(TranscriptionEvent.closed("s0", "X."))
    # closed segment MUST NOT appear in a later supersede old_ids.
    assert guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"])) is None
    assert any(d.code == "lifecycle_closed_superseded" for d in guard.diagnostics)


def test_guard_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.final("s0", "x"))
    with pytest.raises(ValueError):
        guard.admit(TranscriptionEvent.partial("s0", "y"))


def test_guard_accepts_growing_stable_text_unchanged() -> None:
    guard = _LifecycleGuard()
    first = guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="he"))
    assert first is not None and first.stable_text == "he"
    second = guard.admit(TranscriptionEvent.partial("s0", "hello world", stable_text="hello "))
    assert second is not None and second.stable_text == "hello "
    assert guard._stable_text["s0"] == "hello "  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert not guard.diagnostics


def test_guard_clamps_shrinking_stable_text() -> None:
    # A shorter stable text would retract a promise the application already
    # holds: the event is delivered with the earlier stable text instead.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hell"))
    shrunk = guard.admit(TranscriptionEvent.partial("s0", "hello!", stable_text="he"))
    assert shrunk is not None and shrunk.stable_text == "hell"
    assert shrunk.text == "hello!"
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED]
    assert "is shorter than" in guard.diagnostics[0].message
    # Omitting stable_text on a later partial is a shrink to "" and is
    # clamped the same way.
    omitted = guard.admit(TranscriptionEvent.partial("s0", "hello!!"))
    assert omitted is not None and omitted.stable_text == "hell"
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED] * 2


def test_guard_moves_stable_text_back_to_a_combining_boundary() -> None:
    # "e" + COMBINING ACUTE ACCENT: a stable text of "e" would end between
    # the letter and its accent, so it moves back to "".
    guard = _LifecycleGuard()
    admitted = guard.admit(TranscriptionEvent.partial("s0", "e" + _ACUTE + "x", stable_text="e"))
    assert admitted is not None and admitted.stable_text == ""
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED]
    assert "combining character sequence" in guard.diagnostics[0].message
    # A zero width joiner left at the end of the stable text is moved back
    # past the emoji it would join.
    joined = guard.admit(
        TranscriptionEvent.partial("s1", _TECHNOLOGIST + " ok", stable_text="\U0001f469" + _ZWJ)
    )
    assert joined is not None and joined.stable_text == ""


def test_guard_boundary_clamp_moves_back_but_not_below_the_earlier_stable_text() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "a b", stable_text="a "))
    # The proposed `"a bc"` ends between `c` and its accent. The nearest valid
    # boundary before it is `"a b"`, which is still longer than the earlier
    # stable text.
    moved = guard.admit(TranscriptionEvent.partial("s0", "a bc" + _ACUTE, stable_text="a bc"))
    assert moved is not None and moved.stable_text == "a b"
    # Here the only boundary between the earlier stable text and the
    # proposed one is the earlier stable text itself.
    guard.admit(TranscriptionEvent.partial("s1", "ab", stable_text="ab"))
    kept = guard.admit(
        TranscriptionEvent.partial("s1", "abc" + _ACUTE + "\u0302", stable_text="abc" + _ACUTE)
    )
    assert kept is not None and kept.stable_text == "ab"
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED] * 2


# An event built without validation (model_copy, model_construct) can carry a
# stable_text that construction would have rejected. In normal mode the guard
# repairs the shapes these tests cover and records a diagnostic, instead of
# forwarding the value as is. In strict mode it raises.
def test_guard_repairs_a_partial_whose_stable_text_is_not_the_start_of_its_text() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello wor", stable_text="hel"))
    forged = TranscriptionEvent.partial("s0", "hello world", stable_text="hello ").model_copy(
        update={"text": "help"}
    )
    out = guard.admit(forged)
    assert out is not None
    assert out.text == "help"
    assert out.stable_text == "hel"  # the earlier stable text
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED]
    assert "not the start of text" in guard.diagnostics[0].message
    # The repaired value is what later events are judged against.
    assert guard.admit(TranscriptionEvent.final("s0", "help me")) is not None


def test_guard_repairs_a_first_partial_whose_stable_text_is_longer_than_its_text() -> None:
    guard = _LifecycleGuard()
    forged = TranscriptionEvent.partial("s0", "hello world", stable_text="hello ").model_copy(
        update={"text": "hi"}
    )
    out = guard.admit(forged)
    assert out is not None
    assert out.stable_text == ""
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED]


def test_guard_repairs_a_partial_built_with_no_stable_text() -> None:
    guard = _LifecycleGuard()
    forged = TranscriptionEvent.model_construct(type="partial", segment_id="s0", text="hello")
    out = guard.admit(forged)
    assert out is not None
    assert out.stable_text == ""
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED]


@pytest.mark.parametrize("finality", ["final", "closed"])
def test_guard_repairs_a_final_whose_stable_text_is_not_its_whole_text(finality: str) -> None:
    guard = _LifecycleGuard()
    forged = TranscriptionEvent.partial("s0", "hello world", stable_text="hello ").model_copy(
        update={"type": "final", "finality": finality}
    )
    out = guard.admit(forged)
    assert out is not None
    assert out.stable_text == "hello world"
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED]
    assert "not its whole text" in guard.diagnostics[0].message


@pytest.mark.parametrize(
    "forged_value",
    [("hel",), ("hel",) * 10, 3, ["hel"]],
    ids=["tuple", "long-tuple", "number", "list"],
)
def test_guard_repairs_a_partial_whose_stable_text_is_not_a_string(forged_value: object) -> None:
    # str.startswith accepts a tuple, so a forged tuple would pass a bare
    # prefix test and be forwarded as is.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello wor", stable_text="hel"))
    forged = TranscriptionEvent.partial("s0", "hello world").model_copy(
        update={"stable_text": forged_value}
    )
    out = guard.admit(forged)
    assert out is not None
    assert out.stable_text == "hel"
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_CLAMPED]


def test_guard_messages_state_the_violation_and_name_the_segment() -> None:
    # In strict mode the same text is the error and nothing is delivered, so a
    # message does not say what happened to the event.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hell"))
    guard.admit(TranscriptionEvent.partial("s0", "hello!", stable_text="he"))
    guard.admit(TranscriptionEvent.partial("s0", "goodbye"))
    clamped, rewritten = (d.message for d in guard.diagnostics)
    assert clamped.startswith("partial for segment 's0': ")
    assert "The stable text that passes these checks is 'hell'." in clamped
    assert rewritten.startswith("partial for segment 's0' changes")
    assert "'goodbye' no longer starts with it" in rewritten
    for message in (clamped, rewritten):
        assert "delivered" not in message
        assert "suppressed" not in message


def test_guard_strict_raises_for_a_stable_text_that_is_not_the_start_of_its_text() -> None:
    guard = _LifecycleGuard(strict=True)
    forged = TranscriptionEvent.partial("s0", "hello world", stable_text="hello ").model_copy(
        update={"text": "hi"}
    )
    with pytest.raises(ValueError, match="not the start of text"):
        guard.admit(forged)


def test_session_survives_a_partial_built_with_model_copy() -> None:
    # Reusing an event through model_copy is an easy mistake for an engine
    # author. In normal mode, this stable text that is not the start of its
    # text costs a diagnostic, not the session; in strict mode the session
    # ends with engine_error.
    first = TranscriptionEvent.partial("s0", "hello world", stable_text="hello ")
    script = [first.model_copy(update={"text": "hi"}), TranscriptionEvent.final("s0", "hi")]

    async def run() -> tuple[list[TranscriptionEvent], str, list[str]]:
        session = _ScriptedSession(script)
        events = await _collect(session)
        return events, session.result().text, [d.code for d in session.diagnostics()]

    events, text, codes = asyncio.run(run())
    assert [e.type for e in events] == ["partial", "final", "done"]
    assert events[0].stable_text == ""
    assert text == "hi"
    assert codes == [DIAG_STABLE_TEXT_CLAMPED]


def test_guard_closed_may_rewrite_stable_text_and_is_not_clamped() -> None:
    # The closed restatement may reformat stable text once ("twenty twenty"
    # becomes "2020"), and its stable text is the whole restated text.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "twenty twenty", stable_text="twenty "))
    guard.admit(TranscriptionEvent.final("s0", "twenty twenty"))
    closed = guard.admit(TranscriptionEvent.closed("s0", "2020"))
    assert closed is not None and closed.stable_text == "2020"
    assert not guard.diagnostics
    # A closed restatement straight after partials is exempt as well.
    guard.admit(TranscriptionEvent.partial("s1", "hello", stable_text="hello"))
    assert guard.admit(TranscriptionEvent.closed("s1", "Hi.")) is not None
    assert not guard.diagnostics


@pytest.mark.parametrize(
    ("events", "match"),
    [
        pytest.param(
            [
                TranscriptionEvent.partial("s0", "hello", stable_text="hel"),
                TranscriptionEvent.partial("s0", "goodbye"),
            ],
            "no longer starts with it",
            id="rewrite",
        ),
        pytest.param(
            [
                TranscriptionEvent.partial("s0", "a", stable_text="a"),
                TranscriptionEvent.partial("s0", "a" + _ACUTE),
            ],
            "adds a combining mark",
            id="combining-mark-appended",
        ),
        pytest.param(
            [
                TranscriptionEvent.partial("s0", "hello", stable_text="hel"),
                TranscriptionEvent.partial("s0", "hello", stable_text="he"),
            ],
            "is shorter than",
            id="shrink",
        ),
        pytest.param(
            [TranscriptionEvent.partial("s0", "e" + _ACUTE, stable_text="e")],
            "combining character sequence",
            id="boundary",
        ),
    ],
)
def test_guard_strict_raises_for_each_stable_text_violation(
    events: list[TranscriptionEvent], match: str
) -> None:
    guard = _LifecycleGuard(strict=True)
    *setup, offending = events
    for event in setup:
        assert guard.admit(event) is not None
    with pytest.raises(ValueError, match=match):
        guard.admit(offending)


# --------------------------------------------------------------------------- #
# Stable text through the session
# --------------------------------------------------------------------------- #
def _assert_stable_text_only_grows(events: Iterable[TranscriptionEvent]) -> None:
    """Assert each partial or plain final keeps and extends its segment's stable text.

    A closed final is skipped: it may reformat the stable text once.
    """
    delivered: dict[str, str] = {}
    for event in events:
        if event.type not in ("partial", "final") or event.finality == "closed":
            continue
        assert event.segment_id is not None and event.stable_text is not None
        earlier = delivered.get(event.segment_id, "")
        assert event.stable_text.startswith(earlier), (earlier, event.stable_text)
        delivered[event.segment_id] = event.stable_text


def test_session_result_and_delivered_stream_agree_on_growing_stable_text() -> None:
    from standard_asr.compliance import assert_stable_text_invariant

    script = [
        TranscriptionEvent.partial("s0", "hel"),
        TranscriptionEvent.partial("s0", "hello wo", stable_text="hello "),
        # The engine retracts part of its stable text; the session delivers
        # the earlier stable text instead.
        TranscriptionEvent.partial("s0", "hello world", stable_text="hel"),
        TranscriptionEvent.final("s0", "hello world"),
        TranscriptionEvent.partial("s1", "\U00020bb7野", stable_text="\U00020bb7"),
        TranscriptionEvent.final("s1", "\U00020bb7野家"),
    ]

    async def run() -> tuple[list[TranscriptionEvent], Any, list[str]]:
        session = _ScriptedSession(script)
        events = await _collect(session)
        return events, session.result(), [d.code for d in session.diagnostics()]

    events, result, codes = asyncio.run(run())
    assert codes == [DIAG_STABLE_TEXT_CLAMPED]
    _assert_stable_text_only_grows(events)
    assert_stable_text_invariant(events)
    # Coalescing may drop some partials, so only the values the guard decided
    # are pinned: no partial carries the retracted `"hel"`, and each final is
    # stable as a whole.
    assert all(e.stable_text != "hel" for e in events)
    finals = [(e.segment_id, e.stable_text) for e in events if e.type == "final"]
    assert finals == [("s0", "hello world"), ("s1", "\U00020bb7野家")]
    # An application that reduces the delivered stream gets the session's
    # own result.
    replay = StreamReducer()
    for event in events:
        replay.add(event)
    assert replay.result().segments == result.segments
    assert result.segments is not None
    assert [s.text for s in result.segments] == ["hello world", "\U00020bb7野家"]


def test_coalescing_under_a_slow_consumer_never_delivers_shorter_stable_text() -> None:
    # Coalescing drops pending partials when the consumer falls behind. The
    # guard clamps each partial before it reaches the buffer, so whichever
    # partials survive, a later one never carries shorter stable text than
    # an earlier one, even when the engine itself retracted some.
    words = "one two three four five six seven eight nine ten".split()

    class _BurstySession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            text = ""
            for index, word in enumerate(words):
                stable = text
                text = f"{text}{word} "
                # Every third partial retracts its stable text to "".
                yield TranscriptionEvent.partial(
                    "s0", text, stable_text="" if index % 3 == 2 else stable
                )
                if index % 4 == 3:
                    await asyncio.sleep(0.03)
            yield TranscriptionEvent.final("s0", text.strip())

    async def run() -> tuple[list[TranscriptionEvent], list[str]]:
        session = _BurstySession()
        delivered: list[TranscriptionEvent] = []
        async with session:
            async for event in session:
                delivered.append(event)
                await asyncio.sleep(0.02)
        return delivered, [d.code for d in session.diagnostics()]

    delivered, codes = asyncio.run(run())
    partials = [e for e in delivered if e.type == "partial"]
    # Some partials were coalesced away, and the engine's retractions were
    # clamped.
    assert len(partials) < len(words)
    assert DIAG_STABLE_TEXT_CLAMPED in codes
    _assert_stable_text_only_grows(delivered)
    assert delivered[-2].type == "final" and delivered[-2].stable_text == delivered[-2].text


# --------------------------------------------------------------------------- #
# Stable text abandoned: an open segment with stable text when done arrives
# --------------------------------------------------------------------------- #
def test_guard_reports_stable_text_abandoned_at_done() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    guard.admit(TranscriptionEvent.partial("s1", "draft"))
    guard.admit(TranscriptionEvent.partial("s2", "world", stable_text="wor"))
    guard.admit(TranscriptionEvent.final("s2", "world"))
    done = guard.admit(TranscriptionEvent.done())
    # The done itself is delivered: the guard reports, it does not suppress.
    assert done is not None and done.type == "done"
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_ABANDONED]
    # Only the open segment with stable text is named.
    message = guard.diagnostics[0].message
    assert "['s0']" in message
    assert "s1" not in message and "s2" not in message


def test_guard_no_stable_text_abandoned_when_every_stable_segment_is_final() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    guard.admit(TranscriptionEvent.final("s0", "hello"))
    guard.admit(TranscriptionEvent.partial("s1", "never stable"))
    assert guard.admit(TranscriptionEvent.done()) is not None
    assert not guard.diagnostics


def test_guard_no_stable_text_abandoned_on_an_error_terminal() -> None:
    # A session that fails is already explicit about losing text.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    assert guard.admit(TranscriptionEvent.make_error("boom", recoverable=False)) is not None
    assert not guard.diagnostics


def test_guard_stable_text_abandoned_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    with pytest.raises(ValueError, match="never reached final"):
        guard.admit(TranscriptionEvent.done())


@pytest.mark.parametrize("engine_sends_done", [False, True], ids=["base-done", "engine-done"])
def test_session_reports_stable_text_abandoned(engine_sends_done: bool) -> None:
    script = [
        TranscriptionEvent.partial("s0", "hello wor", stable_text="hello "),
        TranscriptionEvent.final("s1", "kept"),
    ]
    if engine_sends_done:
        script.append(TranscriptionEvent.done())

    async def run() -> tuple[list[TranscriptionEvent], str, list[str]]:
        session = _ScriptedSession(script)
        events = await _collect(session)
        return events, session.result().text, [d.code for d in session.diagnostics()]

    events, text, codes = asyncio.run(run())
    assert codes == [DIAG_STABLE_TEXT_ABANDONED]
    assert events[-1].type == "done"
    # The result holds finalized segments only: the abandoned stable text is
    # absent, which is what the diagnostic reports.
    assert text == "kept"


def test_session_no_stable_text_abandoned_when_the_session_ends_with_an_error() -> None:
    async def run() -> tuple[list[TranscriptionEvent], list[str]]:
        session = _ScriptedSession(
            [
                TranscriptionEvent.partial("s0", "hello", stable_text="hel"),
                TranscriptionEvent.make_error("engine_gave_up", recoverable=False),
            ]
        )
        events = await _collect(session)
        return events, [d.code for d in session.diagnostics()]

    events, codes = asyncio.run(run())
    assert events[-1].code == "engine_gave_up"
    assert codes == []


@pytest.mark.parametrize("engine_sends_done", [False, True])
def test_session_stable_text_abandoned_strict_ends_with_engine_error(
    engine_sends_done: bool,
) -> None:
    script = [TranscriptionEvent.partial("s0", "hello", stable_text="hel")]
    if engine_sends_done:
        script.append(TranscriptionEvent.done())

    async def run() -> list[TranscriptionEvent]:
        session = _ScriptedSession(script, strict_lifecycle=True)
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "engine_error"
    assert not any(e.type == "done" for e in events)


# --------------------------------------------------------------------------- #
# Events checked against the engine's declared streaming capabilities
# --------------------------------------------------------------------------- #
_WORD = Word(text="hi", start=0.0, end=0.5)


@pytest.mark.parametrize(
    ("event", "code", "emits_partials"),
    [
        pytest.param(
            TranscriptionEvent.partial("s0", "hello"),
            "stream_exceeds_emits_partials",
            False,
            id="partial",
        ),
        pytest.param(
            TranscriptionEvent.partial("s0", "hello", stable_text="hel"),
            "stream_exceeds_partial_stability",
            True,
            id="stable-text-on-a-partial",
        ),
        pytest.param(
            TranscriptionEvent.supersede(["s0"], ["s1"]),
            "stream_exceeds_re_segments",
            True,
            id="supersede",
        ),
        pytest.param(
            TranscriptionEvent.progress(audio_processed_until=1.0),
            "stream_exceeds_audio_progress",
            True,
            id="audio-cursor",
        ),
        pytest.param(
            TranscriptionEvent.final("s0", "hi", start=0.0, end=1.0),
            "stream_exceeds_timestamps",
            True,
            id="segment-timestamps",
        ),
        pytest.param(
            TranscriptionEvent.partial("s0", "hi", words=[_WORD]),
            "stream_exceeds_word_timestamps",
            True,
            id="words",
        ),
        pytest.param(
            TranscriptionEvent.partial("s0", "hi", speaker="A"),
            "stream_exceeds_diarization",
            True,
            id="speaker",
        ),
    ],
)
def test_guard_records_an_event_the_declared_capabilities_do_not_cover(
    event: TranscriptionEvent, code: str, emits_partials: bool
) -> None:
    # The event is forwarded unchanged: removing the field would hide the
    # engine's fault, and dropping a supersede would duplicate text.
    guard = _LifecycleGuard(
        capabilities=StreamingCapabilities(emits_partials=FlagCap(supported=emits_partials))
    )
    if event.type == "supersede":
        guard.admit(TranscriptionEvent.final("s0", "h"))
    out = guard.admit(event)
    assert out == event
    assert [d.code for d in guard.diagnostics] == [code]
    assert guard.diagnostics[0].level == "warning"
    if code == "stream_exceeds_emits_partials":
        assert guard.diagnostics[0].message == (
            "partial for segment 's0' carries text that may still change, but "
            "streaming.emits_partials is unsupported in the engine's capabilities. "
            "The engine must support emits_partials, or not send partial events."
        )


@pytest.mark.parametrize(
    ("events", "code"),
    [
        pytest.param(
            [TranscriptionEvent.partial("s0", text) for text in ("h", "hi", "hi!")],
            "stream_exceeds_emits_partials",
            id="partial",
        ),
        pytest.param(
            [TranscriptionEvent.progress(audio_processed_until=s) for s in (1.0, 2.0, 3.0)],
            "stream_exceeds_audio_progress",
            id="audio-cursor",
        ),
    ],
)
def test_guard_records_each_capability_mismatch_once_per_session(
    events: list[TranscriptionEvent], code: str
) -> None:
    # The mismatch is a fact about the engine, not about one event.
    guard = _LifecycleGuard(capabilities=StreamingCapabilities())
    for event in events:
        assert guard.admit(event) == event
    assert [d.code for d in guard.diagnostics] == [code]
    # The message states the violation only: in strict mode it is the error.
    assert "once per session" not in guard.diagnostics[0].message


def test_guard_accepts_events_the_declared_capabilities_cover() -> None:
    capabilities = StreamingCapabilities(
        emits_partials=FlagCap(supported=True),
        partial_stability=FlagCap(supported=True),
        re_segments=FlagCap(supported=True),
        timestamps=StreamTimestampsCap(mode="post_align"),
        audio_progress=FlagCap(supported=True),
        word_timestamps=WordTimestampsCap(supported=True, granularities=["word"]),
        diarization=DiarizationCap(supported=True),
    )
    guard = _LifecycleGuard(capabilities=capabilities)
    guard.admit(
        TranscriptionEvent.partial(
            "s0",
            "hi there",
            stable_text="hi ",
            words=[_WORD],
            speaker="A",
            audio_processed_until=1.0,
        )
    )
    guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"]))
    guard.admit(TranscriptionEvent.final("s1", "hi there"))
    guard.admit(TranscriptionEvent.done())
    assert guard.diagnostics == []


def test_guard_accepts_a_final_without_emits_partials() -> None:
    guard = _LifecycleGuard(capabilities=StreamingCapabilities())
    event = TranscriptionEvent.final("s0", "hello")
    assert guard.admit(event) == event
    assert guard.diagnostics == []


def test_guard_without_capabilities_does_not_check_them() -> None:
    # A session built directly, as in a unit test, has no declaration to
    # check against. "No declaration" is not "nothing supported".
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"]))
    assert guard.diagnostics == []


def test_guard_strict_raises_for_a_capability_mismatch() -> None:
    guard = _LifecycleGuard(strict=True, capabilities=StreamingCapabilities())
    guard.admit(TranscriptionEvent.final("s0", "hello"))
    with pytest.raises(ValueError, match=r"streaming\.re_segments is unsupported"):
        guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"]))


def test_guard_reports_finals_left_unclosed_under_finality_level_closed() -> None:
    capabilities = StreamingCapabilities(finality_level=FinalityCap(mode="closed"))
    guard = _LifecycleGuard(capabilities=capabilities)
    guard.admit(TranscriptionEvent.final("s0", "hello"))
    guard.admit(TranscriptionEvent.closed("s0", "Hello."))
    guard.admit(TranscriptionEvent.final("s1", "world"))
    guard.admit(TranscriptionEvent.done())
    assert [d.code for d in guard.diagnostics] == ["finality_level_not_reached"]
    assert "'s1'" in guard.diagnostics[0].message and "'s0'" not in guard.diagnostics[0].message


def test_guard_records_several_capability_mismatches_of_one_event_in_a_fixed_order() -> None:
    event = TranscriptionEvent.partial(
        "s0",
        "hi there",
        stable_text="hi ",
        words=[_WORD.model_copy(update={"speaker": "A"})],
        audio_processed_until=1.0,
    )
    guard = _LifecycleGuard(capabilities=StreamingCapabilities())
    assert guard.admit(event) == event
    assert [d.code for d in guard.diagnostics] == [
        "stream_exceeds_emits_partials",
        "stream_exceeds_partial_stability",
        "stream_exceeds_audio_progress",
        "stream_exceeds_word_timestamps",
        "stream_exceeds_diarization",
    ]
    # Strict mode raises for the first one in that order.
    strict = _LifecycleGuard(strict=True, capabilities=StreamingCapabilities())
    with pytest.raises(ValueError, match=r"streaming\.emits_partials is unsupported"):
        strict.admit(event)


def test_guard_judges_the_event_as_sent_not_as_repaired() -> None:
    # The boundary repair empties this partial's stable text. The engine
    # still sent stable text that its capabilities do not support.
    guard = _LifecycleGuard(
        capabilities=StreamingCapabilities(emits_partials=FlagCap(supported=True))
    )
    out = guard.admit(TranscriptionEvent.partial("s0", "e" + _ACUTE + "x", stable_text="e"))
    assert out is not None and out.stable_text == ""
    assert [d.code for d in guard.diagnostics] == [
        DIAG_STABLE_TEXT_CLAMPED,
        "stream_exceeds_partial_stability",
    ]


@pytest.mark.parametrize("engine_sends_done", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_session_reports_finals_left_unclosed_under_finality_level_closed(
    engine_sends_done: bool, strict: bool
) -> None:
    # The check runs when the session reaches done, whether the engine sent
    # done or the session added it after the engine's last event.
    script = [
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.closed("s0", "Hello."),
        TranscriptionEvent.final("s1", "world"),
    ]
    if engine_sends_done:
        script.append(TranscriptionEvent.done())

    async def run() -> tuple[list[TranscriptionEvent], list[str]]:
        session = _ScriptedSession(script, strict_lifecycle=strict)
        session._bind_streaming_capabilities(  # pyright: ignore[reportPrivateUsage]
            StreamingCapabilities(finality_level=FinalityCap(mode="closed"))
        )
        events = await _collect(session)
        return events, [d.code for d in session.diagnostics()]

    events, codes = asyncio.run(run())
    if strict:
        # engine_error takes the place of done, and nothing is recorded.
        assert [e.type for e in events] == ["final", "final", "final", "error"]
        assert events[-1].code == "engine_error"
        assert codes == []
    else:
        assert [e.type for e in events] == ["final", "final", "final", "done"]
        assert codes == ["finality_level_not_reached"]


@pytest.mark.parametrize(
    ("script", "code"),
    [
        pytest.param(
            [
                TranscriptionEvent.partial("s0", "hello"),
                TranscriptionEvent.final("s0", "goodbye"),
            ],
            "stream_exceeds_emits_partials",
            id="partial",
        ),
        pytest.param(
            [
                TranscriptionEvent.final("s0", "hello"),
                TranscriptionEvent.supersede(["s0"], ["s1"]),
                TranscriptionEvent.final("s1", "goodbye"),
            ],
            "stream_exceeds_re_segments",
            id="supersede",
        ),
    ],
)
def test_bound_capabilities_reach_the_session_diagnostics(
    script: list[TranscriptionEvent], code: str
) -> None:
    # The engine base, or bind_session_capabilities for an engine that does
    # not derive from it, binds the capabilities after the session is built.
    async def run() -> tuple[list[TranscriptionEvent], str, list[str]]:
        session = _ScriptedSession(script)
        bind_session_capabilities(
            session,
            cast("StandardASR", SimpleNamespace(declared_capabilities=DeclaredCapabilities())),
        )
        events = await _collect(session)
        return events, session.result().text, [d.code for d in session.diagnostics()]

    events, text, codes = asyncio.run(run())
    # Every event is delivered, so the result holds the replacement only.
    assert events == [*script, TranscriptionEvent.done()]
    assert text == "goodbye"
    assert codes == [code]


@pytest.mark.parametrize(
    "script",
    [
        pytest.param([TranscriptionEvent.partial("s0", "hello")], id="partial"),
        pytest.param(
            [
                TranscriptionEvent.final("s0", "hello"),
                TranscriptionEvent.supersede(["s0"], ["s1"]),
            ],
            id="supersede",
        ),
    ],
)
def test_bound_capabilities_in_strict_mode_end_the_session(
    script: list[TranscriptionEvent],
) -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _ScriptedSession(script, strict_lifecycle=True)
        bind_session_capabilities(
            session,
            cast("StandardASR", SimpleNamespace(declared_capabilities=DeclaredCapabilities())),
        )
        return await _collect(session)

    events = asyncio.run(run())
    assert events[:-1] == script[:-1]
    assert events[-1].type == "error"
    assert events[-1].code == "engine_error"


def test_guard_clamps_decreasing_audio_cursor() -> None:
    # audio_processed_until is monotonic across the whole session; a decrease is
    # clamped to the prior value with a diagnostic.
    guard = _LifecycleGuard()
    e1 = guard.admit(TranscriptionEvent.progress(audio_processed_until=2.0))
    assert e1 is not None and e1.audio_processed_until == 2.0
    e2 = guard.admit(TranscriptionEvent.progress(audio_processed_until=1.0))
    assert e2 is not None and e2.audio_processed_until == 2.0
    assert any(d.code == "audio_cursor_decreased" for d in guard.diagnostics)


def test_guard_reports_undeclared_audio_progress_without_losing_content() -> None:
    guard = _LifecycleGuard(
        capabilities=StreamingCapabilities(emits_partials=FlagCap(supported=True))
    )
    event = guard.admit(TranscriptionEvent.partial("s0", "text", audio_processed_until=2.0))
    assert event is not None
    assert event.text == "text"
    assert event.audio_processed_until == 2.0
    assert [d.code for d in guard.diagnostics] == ["stream_exceeds_audio_progress"]


def test_guard_raises_on_decreasing_audio_cursor_strict() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.progress(audio_processed_until=2.0))
    with pytest.raises(ValueError, match="cursor is monotonic"):
        guard.admit(TranscriptionEvent.progress(audio_processed_until=1.0))


def test_guard_suppresses_stable_text_rewrite() -> None:
    # Stable text does not change: extending the text after it is fine, but an
    # event whose text no longer starts with it is suppressed.
    guard = _LifecycleGuard()
    first = guard.admit(TranscriptionEvent.partial("s0", "the cat", stable_text="the "))
    assert first is not None
    extend = guard.admit(TranscriptionEvent.partial("s0", "the cattle", stable_text="the "))
    assert extend is not None
    rewrite = guard.admit(TranscriptionEvent.partial("s0", "a dog runs"))
    assert rewrite is None
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_REWRITTEN]


def test_guard_non_closed_final_rewrite_of_stable_text_suppressed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hello"))
    rejected = guard.admit(TranscriptionEvent.final("s0", "Hello."))
    assert rejected is None
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_REWRITTEN]
    # The suppressed final changed nothing: a final that keeps the stable
    # text is still accepted.
    assert guard.admit(TranscriptionEvent.final("s0", "hello there")) is not None


def test_guard_rejects_a_combining_mark_appended_to_a_stable_character() -> None:
    # "a" is stable. Text "a" + COMBINING ACUTE ACCENT + "b" still starts with
    # "a", but the accent changes the character the application already saw,
    # so the event is a rewrite, not an extension.
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.partial("s0", "a", stable_text="a")) is not None
    assert guard.admit(TranscriptionEvent.partial("s0", "a" + _ACUTE + "b")) is None
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_REWRITTEN]
    # The same holds for a zero width joiner that would fuse a stable emoji
    # with the next one.
    woman = "\U0001f469"
    assert guard.admit(TranscriptionEvent.partial("s1", woman, stable_text=woman)) is not None
    assert guard.admit(TranscriptionEvent.partial("s1", _TECHNOLOGIST)) is None
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_REWRITTEN] * 2
    # A plain extension of the stable character is still accepted.
    extended = guard.admit(TranscriptionEvent.partial("s0", "ab", stable_text="a"))
    assert extended is not None and extended.stable_text == "a"


def test_guard_supersede_new_ids_open_then_partial_allowed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "x"))
    assert guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"])) is not None
    # s1 was started open by supersede; a partial for it is legal.
    assert guard.admit(TranscriptionEvent.partial("s1", "new")) is not None
    assert not guard.diagnostics


# --------------------------------------------------------------------------- #
# Supersede rule: a supersede withdraws the retired segments, stable text
# included
# --------------------------------------------------------------------------- #
# The guard compares no text across a supersede: the retired segments and every
# promise attached to them end with the event, and each replacement segment
# starts with no stable text.
def test_guard_supersede_retiring_a_partial_with_stable_text_is_admitted() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("a", "hello wor", stable_text="hello "))
    assert guard.admit(TranscriptionEvent.supersede(["a"], ["b"])) is not None
    # The replacement is judged only against its own history: text unrelated
    # to the retired stable text is accepted, and from then on b's own stable
    # text only grows.
    assert guard.admit(TranscriptionEvent.partial("b", "goodbye")) is not None
    assert (
        guard.admit(TranscriptionEvent.partial("b", "goodbye all", stable_text="good")) is not None
    )
    assert not guard.diagnostics
    assert guard.admit(TranscriptionEvent.partial("b", "bye")) is None
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_REWRITTEN]
    # The retired segment is over: a later event for it is a lifecycle error.
    assert guard.admit(TranscriptionEvent.partial("a", "hello world", stable_text="hello ")) is None
    assert guard.diagnostics[-1].code == DIAG_LIFECYCLE_AFTER_TERMINAL


def test_guard_supersede_retiring_a_final_is_admitted() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("a", "hello"))
    assert guard.admit(TranscriptionEvent.supersede(["a"], ["b"])) is not None
    assert guard.admit(TranscriptionEvent.final("b", "goodbye")) is not None
    assert not guard.diagnostics


def test_guard_supersede_pure_deletion_of_stable_text_is_admitted() -> None:
    # A supersede with empty new_ids retires segments without replacing them,
    # whether or not they have stable text.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("p", "um", stable_text="u"))
    guard.admit(TranscriptionEvent.final("f", "uh"))
    assert guard.admit(TranscriptionEvent.supersede(["p"], [])) is not None
    assert guard.admit(TranscriptionEvent.supersede(["f"], [])) is not None
    assert not guard.diagnostics


def test_guard_forgets_the_text_and_speaker_of_a_retired_segment() -> None:
    # Only the lifecycle state outlives a retired segment. A long session that
    # revises often does not keep every withdrawn text.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("old", "abc", stable_text="abc", speaker="X"))
    guard.admit(TranscriptionEvent.supersede(["old"], []))
    assert guard._stable_text == {}  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert guard._last_speaker == {}  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    # The id still cannot be reused.
    assert guard.admit(TranscriptionEvent.partial("old", "abc")) is None
    assert guard.diagnostics[-1].code == DIAG_LIFECYCLE_AFTER_TERMINAL


def test_guard_keeps_only_what_is_read_again_after_a_final() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("a", "hi", stable_text="hi", speaker="X"))
    guard.admit(TranscriptionEvent.final("a", "hi", speaker="X"))
    # No rule reads a finalized segment's stable text again. Its speaker is
    # still read when a later supersede retires the segment.
    assert guard._stable_text == {}  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert guard._last_speaker == {"a": "X"}  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    guard.admit(TranscriptionEvent.closed("a", "Hi.", speaker="X"))
    assert guard._last_speaker == {}  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert guard.segments_in_state("closed") == ["a"]


def test_guard_chained_supersedes_each_start_from_empty_stable_text() -> None:
    guard = _LifecycleGuard()
    reducer = StreamReducer()
    events = [
        TranscriptionEvent.partial("a", "the cat", stable_text="the "),
        TranscriptionEvent.final("a", "the cat sat"),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.partial("b", "a cat", stable_text="a "),
        TranscriptionEvent.supersede(["b"], ["c"]),
        TranscriptionEvent.final("c", "that cat sat"),
    ]
    for event in events:
        admitted = guard.admit(event)
        assert admitted is not None
        reducer.add(admitted)
    assert not guard.diagnostics
    assert reducer.result().text == "that cat sat"


def test_guard_issue_81_case_1_replacement_keeps_stable_parts_around_unstable_text() -> None:
    # "hello" stable with " world" not yet stable, then "foo" stable with
    # " bar" not yet stable, replaced by one segment that keeps both stable
    # parts and rewrites the rest.
    guard = _LifecycleGuard()
    reducer = StreamReducer()
    events = [
        TranscriptionEvent.partial("a", "hello world", stable_text="hello"),
        TranscriptionEvent.partial("b", "foo bar", stable_text="foo"),
        TranscriptionEvent.supersede(["a", "b"], ["c"]),
        TranscriptionEvent.partial("c", "hello there foo bar", stable_text="hello there foo"),
        TranscriptionEvent.final("c", "hello there foo bar"),
    ]
    for event in events:
        admitted = guard.admit(event)
        assert admitted is not None
        reducer.add(admitted)
    assert not guard.diagnostics
    assert reducer.result().text == "hello there foo bar"


def test_guard_issue_81_case_2_merge_of_settled_segments_is_admitted() -> None:
    # No text is joined or compared, so the separator between the retired
    # texts never matters.
    guard = _LifecycleGuard()
    reducer = StreamReducer()
    events = [
        TranscriptionEvent.final("a", "hello"),
        TranscriptionEvent.final("b", "foo"),
        TranscriptionEvent.supersede(["a", "b"], ["c"]),
        TranscriptionEvent.final("c", "hello foo"),
    ]
    for event in events:
        admitted = guard.admit(event)
        assert admitted is not None
        reducer.add(admitted)
    assert not guard.diagnostics
    assert reducer.result().text == "hello foo"


def test_guard_issue_81_case_3_stable_text_replaced_by_unrelated_text() -> None:
    # The supersede withdraws `"abc"` explicitly, so its replacement by `"xyz"`
    # is admitted with no diagnostic.
    guard = _LifecycleGuard()
    reducer = StreamReducer()
    events = [
        TranscriptionEvent.partial("a", "abc", stable_text="abc"),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.final("b", "xyz"),
        TranscriptionEvent.done(),
    ]
    for event in events:
        admitted = guard.admit(event)
        assert admitted is not None
        reducer.add(admitted)
    assert not guard.diagnostics
    assert reducer.result().text == "xyz"


def test_session_supersede_of_stable_text_lands_in_place() -> None:
    # Through the public session: the replacement takes the retired
    # segment's place in reading order, the delivered stream reduces to the
    # session's own result, and a retired segment whose stable text never
    # reached a final is not reported as abandoned.
    async def run() -> tuple[list[TranscriptionEvent], Any, list[str]]:
        session = _ScriptedSession(
            [
                TranscriptionEvent.partial("a", "hello wor", stable_text="hello "),
                TranscriptionEvent.final("x", "keep"),
                TranscriptionEvent.supersede(["a"], ["a2"]),
                TranscriptionEvent.final("a2", "hi"),
            ]
        )
        events = await _collect(session)
        return events, session.result(), [d.code for d in session.diagnostics()]

    events, result, codes = asyncio.run(run())
    assert codes == []
    assert result.segments is not None
    assert [s.text for s in result.segments] == ["hi", "keep"]
    replay = StreamReducer()
    for event in events:
        replay.add(event)
    assert replay.result().segments == result.segments


# --------------------------------------------------------------------------- #
# Finals: a final is wholly stable
# --------------------------------------------------------------------------- #
def test_final_stable_text_must_be_the_whole_text() -> None:
    assert TranscriptionEvent.final("s0", "hello", stable_text="hello").stable_text == "hello"
    with pytest.raises(ValidationError, match="MUST be the whole text"):
        TranscriptionEvent.final("s0", "hello world", stable_text="hello")
    with pytest.raises(ValidationError, match="MUST be the whole text"):
        TranscriptionEvent.final("s0", "hello", stable_text="")
    # An empty final is wholly stable with an empty stable text.
    assert TranscriptionEvent.final("s0", "").stable_text == ""


def test_closed_stable_text_must_be_the_whole_text() -> None:
    with pytest.raises(ValidationError, match="MUST be the whole text"):
        TranscriptionEvent.closed("s0", "Hello.", stable_text="")
    with pytest.raises(ValidationError, match="MUST be the whole text"):
        TranscriptionEvent.closed("s0", "Hello.", stable_text="Hello")


@pytest.mark.parametrize("finality", ["final", "closed"])
def test_final_whose_text_ends_with_a_joiner_is_admitted(
    finality: Literal["final", "closed"],
) -> None:
    # A final has no boundary inside its text, so the boundary check does not
    # apply to it, although the standalone check on the same strings fails.
    text = "a" + _ZWJ
    assert validate_stable_text(text, text) is False
    event = TranscriptionEvent(type="final", segment_id="s0", text=text, finality=finality)
    guard = _LifecycleGuard()
    out = guard.admit(event)
    assert out is not None and out.stable_text == text
    assert guard.diagnostics == []

    async def run() -> tuple[list[TranscriptionEvent], str, list[str]]:
        session = _ScriptedSession([event])
        events = await _collect(session)
        return events, session.result().text, [d.code for d in session.diagnostics()]

    events, result_text, codes = asyncio.run(run())
    assert [e.type for e in events] == ["final", "done"]
    assert result_text == text.strip()
    assert codes == []


def test_guard_delivers_a_final_as_wholly_stable_after_partial_stable_text() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    admitted = guard.admit(TranscriptionEvent.final("s0", "hello world"))
    assert admitted is not None and admitted.stable_text == "hello world"
    assert not guard.diagnostics


def test_final_that_rewrites_stable_text_is_rejected_and_then_abandoned() -> None:
    # A final must keep the stable text its partials published. When it does
    # not, the final is suppressed; the segment stays open, so the session's
    # done also reports the stable text as abandoned.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    assert guard.admit(TranscriptionEvent.final("s0", "goodbye")) is None
    assert guard.admit(TranscriptionEvent.done()) is not None
    assert [d.code for d in guard.diagnostics] == [
        DIAG_STABLE_TEXT_REWRITTEN,
        DIAG_STABLE_TEXT_ABANDONED,
    ]


# --------------------------------------------------------------------------- #
# Locked speaker + cross-speaker supersede
# --------------------------------------------------------------------------- #
def test_guard_locked_speaker_change_suppressed() -> None:
    # Once a segment has stable text and an accepted speaker, X->Y is an
    # illegal rewrite: the whole event is suppressed (never clamped -- a clamp
    # would keep presenting stale attribution).
    guard = _LifecycleGuard()
    first = guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel", speaker="A"))
    assert first is not None
    rejected = guard.admit(
        TranscriptionEvent.partial("s0", "hello!", stable_text="hel", speaker="B")
    )
    assert rejected is None
    assert [d.code for d in guard.diagnostics] == [DIAG_LOCKED_SPEAKER_REWRITTEN]


def test_guard_locked_speaker_retraction_suppressed() -> None:
    # X->None on a segment with stable text is a retraction by rewrite: also
    # illegal.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel", speaker="A"))
    rejected = guard.admit(TranscriptionEvent.final("s0", "hello!"))
    assert rejected is None
    diag = next(d for d in guard.diagnostics if d.code == DIAG_LOCKED_SPEAKER_REWRITTEN)
    assert "X->None" in diag.message


def test_guard_speaker_none_to_x_after_stable_text_allowed() -> None:
    # None->X after the segment has stable text is the recommended
    # delay-speaker-to-final engine strategy and MUST be admitted.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel"))
    accepted = guard.admit(TranscriptionEvent.final("s0", "hello!", speaker="A"))
    assert accepted is not None and accepted.speaker == "A"
    assert not guard.diagnostics


def test_guard_speaker_floats_until_the_segment_has_stable_text() -> None:
    # The lock is keyed on non-empty stable text from an EARLIER event. While
    # the stable text is empty the speaker floats freely (A->B admitted). An
    # event that both makes text stable and sets the speaker is legal; only
    # afterward is the last accepted speaker locked.
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.partial("s0", "he", speaker="A")) is not None
    assert (
        guard.admit(TranscriptionEvent.partial("s0", "hel", stable_text="", speaker="B"))
        is not None
    )
    made_stable = guard.admit(
        TranscriptionEvent.partial("s0", "hello", stable_text="hel", speaker="A")
    )
    assert made_stable is not None
    assert not guard.diagnostics
    rejected = guard.admit(
        TranscriptionEvent.partial("s0", "hello!", stable_text="hel", speaker="B")
    )
    assert rejected is None
    assert [d.code for d in guard.diagnostics] == [DIAG_LOCKED_SPEAKER_REWRITTEN]


def test_guard_closed_final_exempt_from_locked_speaker() -> None:
    # closed is the terminal post-processing restatement: it may settle a
    # different speaker, just as it may reformat stable text.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel", speaker="A"))
    accepted = guard.admit(TranscriptionEvent.closed("s0", "Hello.", speaker="B"))
    assert accepted is not None and accepted.speaker == "B"
    assert not guard.diagnostics


def test_guard_locked_speaker_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.partial("s0", "hello", stable_text="hel", speaker="A"))
    with pytest.raises(ValueError, match="speaker of a segment with stable text is locked"):
        guard.admit(TranscriptionEvent.partial("s0", "hello!", stable_text="hel", speaker="B"))


def test_guard_rejected_event_does_not_poison_speaker_ledger() -> None:
    # The ledger commit sits AFTER every reject path: a rejected event that
    # carries a speaker must not register it, and follow-up events are judged
    # against the ledger as it stood before the rejection.
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.partial("c", "hello", stable_text="hel")) is not None
    # This event rewrites the stable text while carrying speaker "C": it is
    # rejected AFTER the point where a naive implementation would have
    # recorded the speaker.
    rejected = guard.admit(TranscriptionEvent.partial("c", "goodbye", speaker="C"))
    assert rejected is None
    assert [d.code for d in guard.diagnostics] == [DIAG_STABLE_TEXT_REWRITTEN]
    ledger = guard._last_speaker  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert "c" not in ledger
    # Follow-up: c may still take ANY first speaker (the rejected "C" never
    # locked in) ...
    ok = guard.admit(TranscriptionEvent.partial("c", "hello there", stable_text="hel", speaker="D"))
    assert ok is not None and ok.speaker == "D"
    # ... and is thereafter judged against the ACCEPTED "D", not the
    # rejected "C".
    rejected_again = guard.admit(
        TranscriptionEvent.partial("c", "hello there!", stable_text="hel", speaker="C")
    )
    assert rejected_again is None
    assert guard.diagnostics[-1].code == DIAG_LOCKED_SPEAKER_REWRITTEN


def test_guard_supersede_cross_speaker_merge_suppressed() -> None:
    # Two retired segments with distinct last-known speakers merged into one
    # new id: pigeonhole forces a cross-speaker merge -> whole supersede
    # suppressed.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s1", "hello", speaker="A"))
    guard.admit(TranscriptionEvent.partial("s2", "world", speaker="B"))
    rejected = guard.admit(TranscriptionEvent.supersede(["s1", "s2"], ["s3"]))
    assert rejected is None
    assert any(d.code == "supersede_cross_speaker_merge" for d in guard.diagnostics)
    # Documented suppression side effect: the old segments stay alive (the
    # supersede never took effect), so a later partial on an old id is still
    # admitted -- reduced text may be duplicated once s3 arrives fresh.
    follow_up = guard.admit(TranscriptionEvent.partial("s1", "hello there", speaker="A"))
    assert follow_up is not None
    state = guard._state  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert "s3" not in state


def test_guard_supersede_same_speaker_merge_allowed() -> None:
    # A single distinct speaker cannot be cross-merged: admitted.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s1", "hello", speaker="A"))
    guard.admit(TranscriptionEvent.partial("s2", "world", speaker="A"))
    accepted = guard.admit(TranscriptionEvent.supersede(["s1", "s2"], ["s3"]))
    assert accepted is not None
    assert not guard.diagnostics


def test_guard_supersede_split_with_distinct_speakers_allowed() -> None:
    # As many new ids as distinct retired speakers: no pigeonhole violation
    # (a speaker-preserving re-segmentation is representable), admitted.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s1", "hello", speaker="A"))
    guard.admit(TranscriptionEvent.partial("s2", "world", speaker="B"))
    accepted = guard.admit(TranscriptionEvent.supersede(["s1", "s2"], ["s3", "s4"]))
    assert accepted is not None
    assert not guard.diagnostics


def test_guard_supersede_duplicate_new_ids_evasion_still_suppressed() -> None:
    # The model validator rejects duplicate new_ids at construction, but the
    # guard independently defends against events materialized through
    # validator-bypassing paths (model_construct/model_copy): counting the RAW
    # new_ids length would read ["s3", "s3"] as two replacement slots and admit
    # the exact many->1 cross-speaker merge the pigeonhole exists to suppress.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s1", "hello", speaker="A"))
    guard.admit(TranscriptionEvent.partial("s2", "world", speaker="B"))
    forged = TranscriptionEvent.model_construct(
        type="supersede", old_ids=["s1", "s2"], new_ids=["s3", "s3"]
    )
    assert guard.admit(forged) is None
    diag = next(d for d in guard.diagnostics if d.code == "supersede_cross_speaker_merge")
    # The message must report the DISTINCT replacement count (1) the pigeonhole
    # actually compared, not the raw forged length (2): reporting 2 against
    # 2 distinct speakers would contradict the very rule the diagnostic cites
    # and read like a guard misfire.
    assert "fewer distinct segments (1:" in diag.message
    assert "['s3', 's3']" in diag.message  # raw list keeps the forgery visible
    state = guard._state  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert "s3" not in state


def test_forged_empty_old_ids_supersede_degrades_never_crashes() -> None:
    """An empty-``old_ids`` supersede is suppressed, not an ``IndexError``.

    The model validator requires a supersede to retire at least one id, but
    the guard and reducer independently defend against events materialized
    through validator-bypassing paths (model_construct/model_copy). An empty
    block sailed past every per-old-id loop straight into
    ``block_start([])``'s ``positions[0]`` -- and the resulting IndexError
    escaped into ``_run_producer``'s generic handler, terminating the WHOLE
    session with ``engine_error`` for one malformed event.
    """
    forged = TranscriptionEvent.model_construct(type="supersede", old_ids=[], new_ids=["b"])

    # Guard: suppressed with the no-defined-placement diagnostic.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s1", "hello"))
    assert guard.admit(forged) is None
    assert any(d.code == "supersede_noncontiguous_old_ids" for d in guard.diagnostics)

    # Reducer: same degradation; the live segment survives untouched.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello", start=0.0, end=1.0))
    reducer.add(forged)
    result = reducer.result()
    assert result.text == "hello"
    assert any(d.code == "supersede_noncontiguous_old_ids" for d in result.diagnostics)

    # The spec §5.2 reference reduce raises its DOCUMENTED type, ValueError
    # -- callers wrote `except ValueError` against its contract.
    order = ["s1"]
    texts = {"s1": "hello"}
    with pytest.raises(ValueError, match="MUST retire at least one segment"):
        reduce_event(order, texts, forged)
    assert order == ["s1"] and texts == {"s1": "hello"}


def test_guard_supersede_unknown_speakers_allowed() -> None:
    # Segments that never carried a speaker do not count as distinct: a merge
    # of unlabeled segments (or one labeled + one unlabeled) is admitted.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("s1", "hello"))
    guard.admit(TranscriptionEvent.partial("s2", "world"))
    assert guard.admit(TranscriptionEvent.supersede(["s1", "s2"], ["s3"])) is not None
    guard.admit(TranscriptionEvent.partial("s4", "one", speaker="A"))
    guard.admit(TranscriptionEvent.partial("s5", "two"))
    assert guard.admit(TranscriptionEvent.supersede(["s4", "s5"], ["s6"])) is not None
    assert not guard.diagnostics


def test_guard_diagnostics_bounded_aggregates_overflow_by_code() -> None:
    # The diagnostic channel is bounded. A misbehaving engine that
    # trips a clamp on every event must not grow diagnostics without limit; past
    # the cap the guard keeps a single trailing diagnostics_truncated summary
    # (per-code counts) instead of retaining each entry.
    guard = _LifecycleGuard(max_diagnostics=5)
    for _ in range(100):
        guard._reject("audio_cursor_decreased", "down")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    # Never exceeds the cap, and the last entry is the aggregated summary.
    assert len(guard.diagnostics) == 5
    assert guard.diagnostics[-1].code == "diagnostics_truncated"
    # 4 real entries retained + (100 - 4 = 96) aggregated under that code.
    assert [d.code for d in guard.diagnostics[:4]] == ["audio_cursor_decreased"] * 4
    assert "audio_cursor_decreased': 96" in guard.diagnostics[-1].message


def test_guard_diagnostics_bound_holds_via_admit_clamp_path() -> None:
    # A real misbehaving engine -- a perpetually decreasing audio
    # cursor that clamps on every admit -- stays bounded through admit(), not
    # just the internal _reject helper.
    guard = _LifecycleGuard(max_diagnostics=5)
    # First event sets the cursor to a high value; every later event reports a
    # lower cursor and is clamped (audio_cursor_decreased) on admit.
    guard.admit(TranscriptionEvent.partial("s0", "x", audio_processed_until=100.0))
    for i in range(100):
        guard.admit(TranscriptionEvent.partial("s0", "x", audio_processed_until=float(i)))
    assert len(guard.diagnostics) == 5
    assert guard.diagnostics[-1].code == "diagnostics_truncated"
    assert "audio_cursor_decreased" in guard.diagnostics[-1].message


def test_guard_diagnostics_bound_aggregates_distinct_codes_separately() -> None:
    # The overflow summary tallies each code independently, so a
    # mix of violation kinds is reported with per-code counts.
    guard = _LifecycleGuard(max_diagnostics=3)
    for _ in range(5):
        guard._reject("code_a", "a")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    for _ in range(7):
        guard._reject("code_b", "b")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert len(guard.diagnostics) == 3
    summary = guard.diagnostics[-1]
    assert summary.code == "diagnostics_truncated"
    # 2 real (both code_a) retained; 3 code_a + 7 code_b overflowed.
    assert "code_a': 3" in summary.message
    assert "code_b': 7" in summary.message


def test_guard_diagnostics_no_truncation_below_cap() -> None:
    # Below the cap, diagnostics are retained verbatim with no
    # summary entry -- the common case is unaffected.
    guard = _LifecycleGuard(max_diagnostics=1000)
    guard._reject("audio_cursor_decreased", "down")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert len(guard.diagnostics) == 1
    assert guard.diagnostics[0].code == "audio_cursor_decreased"
    assert not any(d.code == "diagnostics_truncated" for d in guard.diagnostics)


def test_guard_rejects_nonpositive_max_diagnostics() -> None:
    # The bound must be a positive integer (mirrors the other
    # bounded-resource guards that reject <= 0).
    with pytest.raises(ValueError, match="max_diagnostics must be > 0"):
        _LifecycleGuard(max_diagnostics=0)


def test_session_diagnostics_bounded_end_to_end() -> None:
    # The cap is enforced through the live session, not just the
    # guard in isolation. A scripted session whose engine reports a perpetually
    # decreasing audio cursor clamps on every event; session.diagnostics() stays
    # bounded and ends in the aggregated overflow summary (after the standard
    # layer's initial gating diagnostics, of which a scripted session has none).
    from standard_asr.contract.results import Diagnostic

    async def run() -> list[Diagnostic]:
        events = [TranscriptionEvent.partial("s0", "x", audio_processed_until=100.0)]
        events += [
            TranscriptionEvent.partial("s0", "x", audio_processed_until=float(i))
            for i in range(200)
        ]
        # Cap the diagnostics bound small so it is reached without 1000 events;
        # the production default is DEFAULT_MAX_GUARD_DIAGNOSTICS.
        session = _ScriptedSession(events, max_guard_diagnostics=5)
        await _collect(session)
        return session.diagnostics()

    diags = asyncio.run(run())
    assert len(diags) == 5
    assert diags[-1].code == "diagnostics_truncated"
    assert "audio_cursor_decreased" in diags[-1].message


# --------------------------------------------------------------------------- #
# Supersede ordering & disjointness invariants
# --------------------------------------------------------------------------- #
def test_supersede_disjoint_enforced_at_construction() -> None:
    # old_ids n new_ids = empty MUST hold; the event model refuses to build one.
    with pytest.raises(ValueError, match="disjoint"):
        TranscriptionEvent.supersede(["a"], ["a"])
    with pytest.raises(ValueError, match="disjoint"):
        TranscriptionEvent(type="supersede", old_ids=["a"], new_ids=["a"])


def test_supersede_duplicate_old_id_rejected_at_construction() -> None:
    # A duplicate id within old_ids retires the same segment twice
    # (retire-once) -- structurally malformed, like an old/new overlap. The
    # event model refuses to build one via both the classmethod and raw
    # constructor, so it never reaches the guard.
    with pytest.raises(ValueError, match="MUST NOT repeat a segment id"):
        TranscriptionEvent.supersede(["a", "a"], ["b"])
    with pytest.raises(ValueError, match="MUST NOT repeat a segment id"):
        TranscriptionEvent(type="supersede", old_ids=["a", "a"], new_ids=["b"])


def test_supersede_duplicate_new_id_rejected_at_construction() -> None:
    # A duplicate id within new_ids introduces the same replacement segment
    # twice -- malformed under set-to-set lineage (new_ids is
    # semantically a set). Left constructible it would inflate the raw new_ids
    # length and evade the guard's cross-speaker pigeonhole count.
    with pytest.raises(ValueError, match="new_ids MUST NOT repeat a segment id"):
        TranscriptionEvent.supersede(["a"], ["b", "b"])
    with pytest.raises(ValueError, match="new_ids MUST NOT repeat a segment id"):
        TranscriptionEvent(type="supersede", old_ids=["a"], new_ids=["b", "b"])


def test_guard_supersede_unknown_old_id_suppressed() -> None:
    guard = _LifecycleGuard()
    rejected = guard.admit(TranscriptionEvent.supersede(["never-seen"], ["b"]))
    assert rejected is None
    assert any(d.code == "supersede_unknown_old_id" for d in guard.diagnostics)


def test_guard_supersede_unknown_old_id_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    # "never-declared", not "never-announced": the shared admission wording
    # is accurate for both consumers -- an id can be declared by an earlier
    # supersede's new_ids without ever receiving a partial/final.
    with pytest.raises(ValueError, match="never-declared"):
        guard.admit(TranscriptionEvent.supersede(["never-seen"], ["b"]))


def test_guard_supersede_reintroduces_known_new_id_suppressed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("a", "x"))
    guard.admit(TranscriptionEvent.partial("b", "y"))  # b already open
    rejected = guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))
    assert rejected is None
    assert any(d.code == "supersede_reintroduces_segment" for d in guard.diagnostics)


def test_guard_supersede_reintroduces_known_new_id_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.partial("a", "x"))
    guard.admit(TranscriptionEvent.partial("b", "y"))
    with pytest.raises(ValueError, match="MUST be fresh"):
        guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))


# --------------------------------------------------------------------------- #
# Illegal final-after-final
# --------------------------------------------------------------------------- #
def test_guard_suppresses_final_after_final() -> None:
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.final("s0", "done")) is not None
    rejected = guard.admit(TranscriptionEvent.final("s0", "rewritten"))
    assert rejected is None
    assert any(d.code == "lifecycle_final_after_final" for d in guard.diagnostics)


def test_guard_final_after_final_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.final("s0", "done"))
    with pytest.raises(ValueError, match="only supersede or a"):
        guard.admit(TranscriptionEvent.final("s0", "again"))


def test_guard_closed_after_final_is_legal() -> None:
    # A closed event (finality="closed") after a plain final is the legal
    # in-place post-processing correction.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "hello"))
    closed = guard.admit(TranscriptionEvent.closed("s0", "Hello."))
    assert closed is not None
    assert not any(d.code == "lifecycle_final_after_final" for d in guard.diagnostics)


def test_session_accepts_closed_rewrite_of_stable_text_and_updates_result() -> None:
    async def run() -> tuple[list[TranscriptionEvent], str, list[str]]:
        session = _ScriptedSession(
            [
                TranscriptionEvent.final("s0", "hello"),
                TranscriptionEvent.closed("s0", "Hello."),
            ]
        )
        events = await _collect(session)
        return events, session.result().text, [d.code for d in session.diagnostics()]

    events, text, diagnostic_codes = asyncio.run(run())
    assert any(e.type == "final" and e.finality == "closed" for e in events)
    assert text == "Hello."
    assert diagnostic_codes == []


def test_session_accepts_closed_punctuation_itn_of_stable_text() -> None:
    # The closed restatement may shorten the text ("twenty dollars" becomes
    # "$20."). Its stable text is the whole corrected text, forwarded as-is.
    raw = "i owe twenty dollars"
    corrected = "I owe $20."

    async def run() -> tuple[list[TranscriptionEvent], str, list[str]]:
        session = _ScriptedSession(
            [
                TranscriptionEvent.partial("s0", raw, stable_text="i owe "),
                TranscriptionEvent.final("s0", raw),
                TranscriptionEvent.closed("s0", corrected),
            ]
        )
        events = await _collect(session)
        return events, session.result().text, [d.code for d in session.diagnostics()]

    events, text, diagnostic_codes = asyncio.run(run())
    closed_events = [e for e in events if e.type == "final" and e.finality == "closed"]
    assert [e.stable_text for e in closed_events] == [corrected]
    assert text == corrected
    assert diagnostic_codes == []


def test_check_event_sequence_accepts_closed_itn_restatement() -> None:
    # The compliance half of the closed restatement: the suite must NOT fail
    # an engine whose closed final restates stable text in a new format.
    from standard_asr.compliance import check_event_sequence

    report = check_event_sequence(
        [
            TranscriptionEvent.partial("s0", "twenty twenty", stable_text="twenty "),
            TranscriptionEvent.final("s0", "twenty twenty"),
            TranscriptionEvent.closed("s0", "2020"),
            TranscriptionEvent.done(),
        ]
    )
    assert report.passed is True, [i.message for i in report.issues]


def test_event_model_validate_passes_non_dict_through() -> None:
    # The before-validator must pass non-dict input through untouched and let
    # pydantic's own type validation reject it (not mask it with a TypeError).
    with pytest.raises(ValueError):
        TranscriptionEvent.model_validate(123)


def test_event_extra_rejects_bytes_object_keys() -> None:
    # The event's wire-visible extra shares the result models' string key
    # domain: a bytes key (coerced to str by lax pydantic, colliding silently)
    # must fail loudly, at every nesting depth.
    with pytest.raises(ValidationError) as exc_info:
        TranscriptionEvent.partial("s", "t", extra={b"k": 1})
    assert exc_info.value.errors()[0]["type"] == "standard_asr_json_object_key"
    with pytest.raises(ValidationError):
        TranscriptionEvent.partial("s", "t", extra={"ok": {b"nested": 1}})
    # Legit string-keyed extra is untouched.
    assert TranscriptionEvent.partial("s", "t", extra={"k": {"n": 1}}).extra == {"k": {"n": 1}}


def test_event_explicit_none_detected_language_passes() -> None:
    # Explicit None runs the field validator (unlike the omitted default).
    assert TranscriptionEvent.partial("s", "hi", detected_language=None).detected_language is None


def test_error_event_unset_recoverable_defaults_to_terminal() -> None:
    # recoverable=None would be an undefined third state that is_terminal
    # silently reads as "recoverable"; unknown recoverability must fail closed
    # to terminal.
    event = TranscriptionEvent(type="error", code="boom")
    assert event.recoverable is False
    assert event.is_terminal is True
    explicit = TranscriptionEvent(type="error", code="boom", recoverable=True)
    assert explicit.recoverable is True
    assert explicit.is_terminal is False


def test_event_detected_language_is_validated_and_canonicalized() -> None:
    # The event field is the reconnect-continuity mechanism and
    # must hold a concrete BCP-47 tag, like the result model.
    event = TranscriptionEvent.partial("s", "hola", detected_language="ES-es")
    assert event.detected_language == "es-ES"
    with pytest.raises(ValueError, match="well-formed BCP-47"):
        TranscriptionEvent.partial("s", "hi", detected_language="English")
    with pytest.raises(ValueError, match="reserved 'auto'"):
        TranscriptionEvent.partial("s", "hi", detected_language="auto")
    assert TranscriptionEvent.partial("s", "hi").detected_language is None


def test_guard_supersede_out_of_order_diagnostic_names_both_causes() -> None:
    # A supersede arriving AFTER its new_id's first partial/final is an
    # ordering violation, not (only) an id-reuse one; the diagnostic must name
    # the out-of-order cause instead of mislabeling it.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("a", "hello"))
    guard.admit(TranscriptionEvent.partial("b", "wor"))  # new_id announced early
    rejected = guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))
    assert rejected is None
    diag = next(d for d in guard.diagnostics if d.code == "supersede_reintroduces_segment")
    assert "out of order" in diag.message


def test_session_constructor_rejects_degenerate_bounds() -> None:
    # audio_queue_maxsize=0 would mean an UNBOUNDED ``asyncio.Queue`` (silently
    # disabling feed backpressure); zero/negative deadlines and buffer bounds
    # are configuration bugs, not no-ops.
    with pytest.raises(ValueError, match="audio_queue_maxsize"):
        _ScriptedSession([], audio_queue_maxsize=0)
    with pytest.raises(ValueError, match="done_timeout"):
        _ScriptedSession([], done_timeout=0)
    with pytest.raises(ValueError, match="max_idle"):
        _ScriptedSession([], max_idle=0)
    with pytest.raises(ValueError, match="max_session_seconds"):
        _ScriptedSession([], max_session_seconds=-1)
    with pytest.raises(ValueError, match="event_buffer_capacity"):
        _ScriptedSession([], event_buffer_capacity=0)


def test_deadline_terminal_stops_producer_so_result_matches_stream() -> None:
    # A synthesized deadline terminal ends iteration; the producer must be
    # stopped with it, or it keeps feeding the reducer events the consumer
    # never saw and result() diverges from the delivered stream.
    class _LateContentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.final("s0", "delivered")
            await asyncio.sleep(0.2)
            yield TranscriptionEvent.final("s1", "never delivered")

    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _LateContentSession(done_timeout=0.05)
        events: list[TranscriptionEvent] = []
        async with session:
            session.feed([])
            async for event in session:
                events.append(event)
            # Give a not-canceled producer ample time to emit the late final
            # before reducing -- without the fix this makes result() diverge.
            await asyncio.sleep(0.3)
        return events, session.partial_result().text

    events, text = asyncio.run(run())
    assert any(e.type == "error" and e.code == "done_timeout" for e in events)
    assert "never delivered" not in text
    assert text == "delivered"


def test_guard_suppresses_closed_after_superseded_segment() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "hello"))
    assert guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"])) is not None
    rejected = guard.admit(TranscriptionEvent.closed("s0", "Hello."))
    assert rejected is None
    assert any(d.code == "lifecycle_after_terminal" for d in guard.diagnostics)


# --------------------------------------------------------------------------- #
# Supersede with empty new_ids (pure deletion)
# --------------------------------------------------------------------------- #
def test_guard_supersede_empty_new_ids_without_stable_text_is_allowed() -> None:
    # Pure deletion is fine when the retired segment has no stable text.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("a", "draft"))
    accepted = guard.admit(TranscriptionEvent.supersede(["a"], []))
    assert accepted is not None
    assert not guard.diagnostics


def test_session_suppresses_illegal_transition_in_stream() -> None:
    class _BadSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.final("s0", "good")
            yield TranscriptionEvent.partial("s0", "revived")  # illegal

    async def run() -> tuple[list[TranscriptionEvent], int]:
        session = _BadSession()
        session.feed([])
        events = await _collect(session)
        return events, len(session.diagnostics())

    events, ndiag = asyncio.run(run())
    # The revived partial must NOT be forwarded.
    assert not any(e.type == "partial" for e in events)
    assert ndiag >= 1


# --------------------------------------------------------------------------- #
# Termination guarantees (idle / wall clock) beyond per-event gap
# --------------------------------------------------------------------------- #
def test_heartbeat_only_engine_still_terminates() -> None:
    class _HeartbeatSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            while True:
                await asyncio.sleep(0.01)
                yield TranscriptionEvent.progress(audio_processed_until=1.0)

    async def run() -> list[TranscriptionEvent]:
        # Frequent heartbeats keep done_timeout alive, but max_idle (no content)
        # MUST still terminate the iterator.
        session = _HeartbeatSession(done_timeout=5.0, max_idle=0.1)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "stream_stalled"


def test_max_session_seconds_caps_wall_time() -> None:
    class _ChattySession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            i = 0
            while True:
                await asyncio.sleep(0.01)
                yield TranscriptionEvent.final(f"s{i}", "x")
                i += 1

    async def run() -> list[TranscriptionEvent]:
        # Continuous content events keep both done_timeout and max_idle alive;
        # only the wall-clock cap guarantees termination.
        session = _ChattySession(done_timeout=5.0, max_idle=5.0, max_session_seconds=0.1)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"


def test_max_session_seconds_without_max_idle() -> None:
    # max_idle is None (1041 False branch); only the wall-clock cap terminates a
    # continuously chatty session.
    class _ChattySession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            i = 0
            while True:
                await asyncio.sleep(0.01)
                yield TranscriptionEvent.final(f"s{i}", "x")
                i += 1

    async def run() -> list[TranscriptionEvent]:
        session = _ChattySession(done_timeout=5.0, max_idle=None, max_session_seconds=0.1)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"


def test_max_session_seconds_anchored_at_session_start_not_iteration() -> None:
    # The absolute wall-clock cap is measured from session
    # establishment (__aenter__), NOT from the first __anext__. The gap between
    # opening the session and starting to iterate MUST count against the cap.
    #
    # The fake clock distinguishes the two anchorings: __aenter__ reads the
    # origin at t=0, and by the time _iterate reads the clock the session wall
    # time is already 2.0 s > the 1.0 s budget. With the correct session anchor
    # (now - session_start = 2.0 > 1.0) the cap trips on the first loop
    # iteration. With the old iteration anchor (start = now = 2.0, so
    # now - start = 0) it would NOT trip and -- with every other deadline
    # disabled -- the iterator would block on the silent producer indefinitely;
    # the test would then hang rather than return a session_timeout.
    class _SilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)  # never yields: only the cap can terminate.
            yield TranscriptionEvent.done()  # pragma: no cover - unreachable

    async def run() -> list[TranscriptionEvent]:
        session = _SilentSession(done_timeout=None, max_idle=None, max_session_seconds=1.0)
        ticks = iter([0.0, 2.0, 2.0, 2.0, 2.0])

        def _clock() -> float:
            try:
                return next(ticks)
            except StopIteration:  # pragma: no cover - safety for extra reads
                return 2.0

        # White-box: inject a deterministic clock through the guard-aware setter so
        # the reserved-attribute guard tracks it instead of flagging a clobber.
        session._replace_reserved_attr("_monotonic", _clock)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        session.feed([])
        return await asyncio.wait_for(_collect(session), timeout=5.0)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"


def test_max_session_seconds_not_reset_by_reiteration() -> None:
    # Re-entering __aiter__ MUST NOT reset the wall-clock origin and
    # renew the session indefinitely. The session anchors the cap once at
    # __aenter__; a second __aiter__ is rejected (single-consumer contract), so
    # the cap can never be reset by re-iterating.
    class _SilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover - unreachable

    async def run() -> str:
        session = _SilentSession(done_timeout=None, max_idle=None, max_session_seconds=0.05)
        session.feed([])
        async with session:
            # First (and only legal) iteration: the cap, anchored at __aenter__,
            # terminates the silent session.
            events = [event async for event in session]
            # A second __aiter__ MUST fail loudly rather than hand back a
            # competing iterator with a fresh, renewed wall-clock origin.
            with pytest.raises(InvalidSessionUseError, match="already being iterated"):
                session.__aiter__()
            return events[-1].code or ""

    code = asyncio.run(run())
    assert code == "session_timeout"


def test_session_single_consumer_rejects_concurrent_iterators() -> None:
    # A session has one event stream and one consumer. A second
    # concurrent __aiter__ would race the first for buffered events, silently
    # splitting the stream. The second call MUST raise instead.
    class _SlowSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(0.01)
            yield TranscriptionEvent.final("s0", "x")
            yield TranscriptionEvent.done()

    async def run() -> None:
        session = _SlowSession(done_timeout=5.0)
        session.feed([])
        async with session:
            first = session.__aiter__()
            assert first is not None
            with pytest.raises(InvalidSessionUseError, match="single event stream"):
                session.__aiter__()
            # Draining the first iterator still works normally.
            events = [event async for event in first]
            assert events[-1].type == "done"

    asyncio.run(run())


def test_session_timeout_checked_at_loop_top_with_buffered_events() -> None:
    # The wall-clock cap is detected at the TOP of the loop (remaining <= 0)
    # before any wait, when the clock has already advanced past the budget. A
    # deterministic fake clock removes the timing race.
    class _OneShotSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.final("s0", "x")
            await asyncio.sleep(10)  # then go silent

    async def run() -> list[TranscriptionEvent]:
        session = _OneShotSession(done_timeout=5.0, max_idle=None, max_session_seconds=1.0)
        # Deterministic clock: start at 0, then jump past the 1.0 s budget so the
        # second loop iteration's top-of-loop check sees remaining <= 0.
        ticks = iter([0.0, 0.0, 2.0, 2.0, 2.0, 2.0])

        def _clock() -> float:
            try:
                return next(ticks)
            except StopIteration:  # pragma: no cover - safety for extra reads
                return 2.0

        session._replace_reserved_attr("_monotonic", _clock)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"


def test_session_timeout_on_silence_in_timeout_handler() -> None:
    # Total silence: the per-event wait times out exactly at the wall-clock cap so
    # the TimeoutError handler synthesizes session_timeout (not done_timeout).
    class _SilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        # max_session < done_timeout, no idle cap: the wait is bounded by the
        # remaining wall-clock budget and the handler picks session_timeout.
        session = _SilentSession(done_timeout=0.2, max_idle=None, max_session_seconds=0.05)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].code == "session_timeout"


def test_done_timeout_still_fires_on_total_silence() -> None:
    class _SilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _SilentSession(done_timeout=0.05, max_idle=5.0)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].code == "done_timeout"


# --------------------------------------------------------------------------- #
# Liveness: done_timeout is a pipeline-inactivity backstop
# --------------------------------------------------------------------------- #
def test_silent_engine_survives_while_audio_is_consumed() -> None:
    # The scenario: a cloud-WS-style engine emits NOTHING while the
    # user is silent, but the engine keeps consuming fed audio. Consumption
    # advances the liveness anchor, so the session outlives done_timeout many
    # times over without a manufactured terminal; ending input then yields the
    # engine's own final + done.
    class _SilentConsumerSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            async for _chunk in self.audio_chunks():
                pass  # consume forever, emit nothing (user silence)
            yield TranscriptionEvent.final("s0", "after silence")

    async def run() -> list[TranscriptionEvent]:
        session = _SilentConsumerSession(done_timeout=0.5)
        events: list[TranscriptionEvent] = []
        async with session:

            async def _feed_silence() -> None:
                # Keep feeding well past done_timeout (silence is audio too).
                for _ in range(20):
                    await session.send_audio(b"\x00")
                    await asyncio.sleep(0.03)
                await session.end_audio()

            feeder = asyncio.create_task(_feed_silence())
            async for event in session:
                events.append(event)
            await feeder
        return events

    events = asyncio.run(run())
    assert not any(e.type == "error" for e in events)
    assert [e.type for e in events][-2:] == ["final", "done"]


def test_done_timeout_fires_after_end_of_input_when_done_never_arrives() -> None:
    # After end-of-input there is nothing left to consume: the liveness anchor
    # freezes and done_timeout bounds the engine's flush-and-done window -- the
    # original "done MUST arrive, bounded by a timeout" promise.
    class _ConsumeThenHangSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            async for _chunk in self.audio_chunks():
                pass
            await asyncio.sleep(10)  # never delivers done
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _ConsumeThenHangSession(done_timeout=0.1)
        session.feed([b"x", b"y"])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "done_timeout"


def test_done_timeout_none_disables_backstop_other_deadlines_rule() -> None:
    # Explicit opt-out: done_timeout=None never synthesizes done_timeout; a
    # remaining (here: wall-clock) deadline still terminates the session.
    class _SilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _SilentSession(done_timeout=None, max_idle=None, max_session_seconds=0.05)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].code == "session_timeout"


def test_all_deadlines_disabled_waits_without_timeout() -> None:
    # done_timeout=None + max_idle=None + max_session_seconds=None: the event
    # wait is plain (no timeout) and an engine terminal still ends the session.
    class _SlowFinalSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(0.05)
            yield TranscriptionEvent.final("s0", "x")

    async def run() -> list[TranscriptionEvent]:
        session = _SlowFinalSession(done_timeout=None, max_idle=None, max_session_seconds=None)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert [e.type for e in events] == ["final", "done"]


def test_session_accepts_none_done_timeout_rejects_nonpositive() -> None:
    session = _EchoSession(done_timeout=None)
    assert session.done_timeout is None
    with pytest.raises(ValueError, match="done_timeout"):
        _EchoSession(done_timeout=-1.0)


def test_stream_deadlines_model_validates_and_tracks_explicit_fields() -> None:
    deadlines = StreamDeadlines(max_idle=0.5)
    assert deadlines.model_fields_set == {"max_idle"}
    assert deadlines.done_timeout == DEFAULT_DONE_TIMEOUT
    assert StreamDeadlines(done_timeout=None).done_timeout is None
    with pytest.raises(ValidationError):
        StreamDeadlines(done_timeout=0.0)
    with pytest.raises(ValidationError):
        StreamDeadlines(max_session_seconds=-1.0)


def test_stream_deadlines_rejects_unknown_field() -> None:
    """An unknown StreamDeadlines field fails construction (``extra='forbid'``)."""
    # extra="forbid": a misspelled deadline is a silently dropped SAFETY
    # parameter -- under pydantic's default extra="ignore" the typo below was
    # swallowed and the session ran with max_idle at its engine default, the
    # exact opposite of what the caller wrote. It MUST fail at construction.
    with pytest.raises(ValidationError) as excinfo:
        StreamDeadlines(max_idle_seconds=5.0)  # type: ignore[call-arg]

    errors = excinfo.value.errors()
    assert [e["type"] for e in errors] == ["extra_forbidden"]
    assert errors[0]["loc"] == ("max_idle_seconds",)
    # The correctly spelled field is still accepted (the guard is not blanket).
    assert StreamDeadlines(max_idle=5.0).max_idle == 5.0


def test_apply_deadline_overrides_touches_only_explicit_fields() -> None:
    session = _EchoSession(done_timeout=7.0, max_idle=9.0)
    overrides = StreamDeadlines(max_idle=0.5, max_session_seconds=11.0)
    session._apply_deadline_overrides(overrides)  # pyright: ignore[reportPrivateUsage]
    assert session.done_timeout == 7.0  # engine choice kept (field unset)
    assert session.max_idle == 0.5  # explicit override applied
    assert session.max_session_seconds == 11.0
    disable = StreamDeadlines(done_timeout=None)
    session._apply_deadline_overrides(disable)  # pyright: ignore[reportPrivateUsage]
    assert session.done_timeout is None  # explicit disable applied


def test_sync_pump_slices_do_not_kill_quiet_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pump slices are pure loop-liveness polls: a session quiet for several
    # slices still delivers its events -- no manufactured timeout, bridge
    # intact (a live fed session may legitimately be quiet far longer than
    # any slice).
    monkeypatch.setattr(streaming_module, "_SYNC_PUMP_POLL_SECONDS", 0.02)

    class _QuietThenFinalSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(0.15)  # several pump slices of silence
            yield TranscriptionEvent.final("s0", "late")

    with SyncSession(_QuietThenFinalSession()) as sync:
        sync.feed([])
        events = list(sync)
    assert [e.type for e in events] == ["final", "done"]


def test_sync_pump_detects_dead_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    # A dead bridge loop is the one failure the in-loop deadlines cannot
    # report (they run ON that loop): the pump's slice-poll must notice the
    # dead thread, tear the bridge down, and raise instead of waiting forever.
    monkeypatch.setattr(streaming_module, "_SYNC_PUMP_POLL_SECONDS", 0.02)

    class _NeverEventSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    # RR-009: the REAL _shutdown is exercised here (not stubbed). On this
    # already-stopped-loop "died" path it must drain the pending tasks
    # synchronously (run_until_complete) and close the loop, with NO un-awaited
    # coroutine / pending-task warnings -- which the suite's filterwarnings=error
    # would otherwise turn into failures.
    sync = SyncSession(_NeverEventSession(), submit_timeout=5.0)
    sync.__enter__()
    outcome: list[BaseException | TranscriptionEvent] = []

    def _pump() -> None:
        try:
            outcome.append(next(iter(sync)))
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            outcome.append(exc)

    pump_thread = threading.Thread(target=_pump)
    pump_thread.start()
    time.sleep(0.1)  # let the pump submit __anext__ and enter a slice wait
    # Simulate the loop thread dying mid-session (not a clean teardown:
    # _closed stays False, so this is not the StreamClosedError path).
    sync._loop.call_soon_threadsafe(sync._loop.stop)  # pyright: ignore[reportPrivateUsage]
    sync._thread.join(timeout=2.0)  # pyright: ignore[reportPrivateUsage]
    assert not sync._thread.is_alive()  # pyright: ignore[reportPrivateUsage]
    pump_thread.join(timeout=5.0)
    assert not pump_thread.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], TimeoutError)
    assert "event-loop thread died" in str(outcome[0])
    # The real _shutdown drained the stopped loop and closed it (no leak / warning).
    assert not sync._thread.is_alive()  # pyright: ignore[reportPrivateUsage]
    assert sync._loop.is_closed()  # pyright: ignore[reportPrivateUsage]


def test_sync_pump_after_teardown_raises_stream_closed() -> None:
    # A prior lifecycle timeout tears the bridge down; pumping events
    # afterward must fail with the contracted StreamClosedError, not hang or
    # raise an unrelated loop error.
    sync = SyncSession(_HangEndAudioSession(), submit_timeout=0.1)
    with pytest.raises(TimeoutError):
        with sync:
            sync.end_audio()  # hangs -> tears the bridge down
    with pytest.raises(StreamClosedError, match="cannot pump events"):
        next(iter(sync))


def test_sync_iter_without_enter_raises_stream_closed() -> None:
    # RR-008: iterating a SyncSession that was never entered (so no event stream
    # exists to pump) MUST raise the contracted StreamClosedError -- not a bare
    # AssertionError. An earlier campaign change left the assert shadowing the
    # contract; under `python -O` the assert is stripped and it degraded to an
    # AttributeError on None. The explicit raise fixes both interpreters.
    sync = SyncSession(_EchoSession())
    try:
        with pytest.raises(StreamClosedError, match="no event stream to pump"):
            next(iter(sync))
    finally:
        sync._shutdown()  # pyright: ignore[reportPrivateUsage]  # close the owned loop/thread


# --------------------------------------------------------------------------- #
# MEDIUM -- end_audio() first then feed() rejected (mode claimed atomically)
# --------------------------------------------------------------------------- #
def test_end_audio_first_then_feed_raises() -> None:
    # End_audio claims manual mode; a later feed is mixing ->
    # InvalidSessionUseError.
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.end_audio()  # claims manual mode
            session.feed([b"x"])  # mixing -> must raise

    with pytest.raises(InvalidSessionUseError):
        asyncio.run(run())


def test_feed_twice_raises() -> None:
    # A session owns at most one fed source; a second feed is a usage
    # error -> InvalidSessionUseError.
    async def run() -> None:
        session = _EchoSession()
        session.feed([b"a"])
        session.feed([b"b"])

    with pytest.raises(InvalidSessionUseError):
        asyncio.run(run())


# --------------------------------------------------------------------------- #
# MEDIUM -- StreamReducer: no fabricated 0.0 timestamps; arrival order kept
# --------------------------------------------------------------------------- #
def test_streaming_diagnostic_code_constants_match_their_wire_literals() -> None:
    """Pin each streaming DIAG_* constant to its exact wire literal."""
    # The lifecycle guard's rejection verdicts and the reducer's timestamp
    # disclosure are wire-visible ``Diagnostic.code`` values consumers match on.
    # The emission sites now reference these constants, so constant and literal
    # are pinned together here exactly once -- a rename breaks loudly instead of
    # silently changing what applications receive. (Terminal EVENT codes such as
    # done_timeout / backpressure are a different namespace and not listed.)
    assert DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE == "segment_timestamps_unavailable"
    assert DIAG_SUPERSEDE_UNKNOWN_OLD_ID == "supersede_unknown_old_id"
    assert DIAG_LIFECYCLE_CLOSED_SUPERSEDED == "lifecycle_closed_superseded"
    assert DIAG_LIFECYCLE_RETIRED_RESUPERSEDED == "lifecycle_retired_resuperseded"
    assert DIAG_SUPERSEDE_REINTRODUCES_SEGMENT == "supersede_reintroduces_segment"
    assert DIAG_SUPERSEDE_CROSS_SPEAKER_MERGE == "supersede_cross_speaker_merge"
    assert DIAG_LIFECYCLE_AFTER_TERMINAL == "lifecycle_after_terminal"
    assert DIAG_LIFECYCLE_PARTIAL_AFTER_FINAL == "lifecycle_partial_after_final"
    assert DIAG_LIFECYCLE_FINAL_AFTER_FINAL == "lifecycle_final_after_final"
    assert DIAG_STABLE_TEXT_REWRITTEN == "stable_text_rewritten"
    assert DIAG_LOCKED_SPEAKER_REWRITTEN == "locked_speaker_rewritten"
    assert DIAG_AUDIO_CURSOR_DECREASED == "audio_cursor_decreased"
    assert DIAG_STABLE_TEXT_CLAMPED == "stable_text_clamped"
    assert DIAG_STABLE_TEXT_ABANDONED == "stable_text_abandoned"


def test_reducer_preserves_arrival_order_without_timestamps() -> None:
    """Timestamp-less finals keep arrival order and store real ``None`` timing."""
    reducer = StreamReducer()
    # No start/end given (timestamp-less engine like Qwen streaming).
    reducer.add(TranscriptionEvent.final("s1", "world"))
    reducer.add(TranscriptionEvent.final("s2", "hello"))
    result = reducer.result()
    # Arrival order preserved (there are no starts to sort on, and nothing is
    # fabricated to make them sortable).
    assert result.text == "world hello"
    # The unmeasured spans are stored VERBATIM as None (the values are the
    # per-segment truth); exactly one warning diagnostic aggregates the count.
    assert [d.code for d in result.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]
    diag = result.diagnostics[0]
    assert diag.level == "warning"
    assert diag.param == "segments"
    assert "2 of 2" in diag.message
    assert result.segments is not None
    assert [s.text for s in result.segments] == ["world", "hello"]
    assert all(s.start is None and s.end is None for s in result.segments)
    assert all(s.timestamp_status == "unavailable" for s in result.segments)


def test_event_end_without_start_is_rejected() -> None:
    """A content event carrying ``end`` but no ``start`` is malformed.

    Mirrors ``Segment``'s shape invariant (measured / start-only /
    unavailable). Pre-guard, the reducer silently fabricated ``start=0.0``
    for this shape -- a silent wrong timestamp.
    """
    for factory in (TranscriptionEvent.partial, TranscriptionEvent.final):
        with pytest.raises(ValidationError, match="end without start"):
            factory("s1", "text", end=2.0)
    # start-only and both-bounds stay legal on events, as on segments.
    assert TranscriptionEvent.final("s1", "text", start=1.0).end is None
    assert TranscriptionEvent.final("s1", "text", start=1.0, end=2.0).end == 2.0


def test_event_end_before_start_is_rejected() -> None:
    """A content event whose span runs backwards is malformed at construction.

    The other half of the ``Segment`` mirror (``end >= start``,
    zero-duration allowed). Unchecked, the shape passed the event model and
    the guard, then crashed only when the reducer built the ``Segment`` --
    deferring one malformed event into a whole-session failure.
    """
    for factory in (TranscriptionEvent.partial, TranscriptionEvent.final):
        with pytest.raises(ValidationError, match="end >= start"):
            factory("s1", "text", start=2.0, end=1.0)
    # A zero-duration span stays legal, as on segments.
    assert TranscriptionEvent.final("s1", "text", start=2.0, end=2.0).end == 2.0


def test_reducer_sorts_when_all_have_timestamps() -> None:
    """Fully timestamped finals sort by ``start`` with nothing to disclose."""
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "second", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "first", start=1.0, end=2.0))
    result = reducer.result()
    assert result.text == "first second"
    # Every retained segment carries real timing -> nothing to disclose.
    assert result.diagnostics == []


def test_reducer_no_sort_when_mixed_timestamps() -> None:
    """Mixed timestamps preserve arrival order and disclose the placeholder count."""
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "b", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "a"))  # no timestamp
    # Mixed -> preserve arrival order, do not sort on a fabricated 0.0.
    result = reducer.result()
    assert result.text == "b a"
    # A partial disclosure is still a disclosure: one segment of two is a
    # placeholder, and the count says so.
    assert [d.code for d in result.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]
    assert "1 of 2" in result.diagnostics[0].message
    assert result.segments is not None
    assert [s.text for s in result.segments] == ["b", "a"]


def test_reducer_timestamp_diagnostic_ignores_superseded_segments() -> None:
    """Only retained segments count toward the timestamp disclosure."""
    # Only RETAINED segments are disclosed: a timestamp-less segment that was
    # superseded away is not counted (and an all-timestamped remainder emits no
    # diagnostic at all).
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "dropped"))  # no timestamp
    reducer.add(TranscriptionEvent.final("s2", "kept", start=1.0, end=2.0))
    reducer.add(TranscriptionEvent.supersede(["s1"], ["s3"]))
    result = reducer.result()
    assert result.text == "kept"
    assert result.diagnostics == []


# --------------------------------------------------------------------------- #
# LOW -- coalescing buffer drains O(n) without per-pop reindex
# --------------------------------------------------------------------------- #
def test_coalescing_buffer_large_drain_order() -> None:
    async def run() -> list[str | None]:
        buf = _CoalescingBuffer(capacity=10_000)
        for i in range(5000):
            buf.put(TranscriptionEvent.final(f"s{i}", str(i)))
        buf.close()
        events = await _drain_buffer(buf)
        return [e.text for e in events]

    texts = asyncio.run(run())
    assert texts == [str(i) for i in range(5000)]


# --------------------------------------------------------------------------- #
# SURVEY -- a two-pass supersede reduce; DSM; FireRed
# --------------------------------------------------------------------------- #
def test_survey_wenet_two_pass_supersede_reduce() -> None:
    # First pass finalizes seg-3/seg-4; second pass merges them into seg-5,
    # which takes over the retired block's reading-order position in place.
    order: list[str] = []
    texts: dict[str, str] = {}
    reduce_event(order, texts, TranscriptionEvent.final("seg-3", "hello"))
    reduce_event(order, texts, TranscriptionEvent.final("seg-4", "world"))
    reduce_event(order, texts, TranscriptionEvent.supersede(["seg-3", "seg-4"], ["seg-5"]))
    assert (order, texts) == (["seg-5"], {})
    reduce_event(order, texts, TranscriptionEvent.final("seg-5", "hello world"))
    assert (order, texts) == (["seg-5"], {"seg-5": "hello world"})


def test_reduce_event_ignores_non_text_events() -> None:
    # done / error / heartbeat carry no segment text -> the state is untouched.
    order: list[str] = ["s1"]
    texts: dict[str, str] = {"s1": "kept"}
    reduce_event(order, texts, TranscriptionEvent.done())
    reduce_event(order, texts, TranscriptionEvent.make_error("x", recoverable=False))
    assert (order, texts) == (["s1"], {"s1": "kept"})


def test_reduce_event_supersede_unknown_old_id_raises() -> None:
    # A supersede retiring an id that holds no live position has no defined
    # placement: the helper (which has no diagnostics channel) fails loudly
    # instead of corrupting the order. The guard-filtered session path never
    # delivers such an event to application reduces.
    order: list[str] = ["s1"]
    texts: dict[str, str] = {"s1": "a"}
    with pytest.raises(ValueError, match="ghost.*no live reading-order position"):
        reduce_event(order, texts, TranscriptionEvent.supersede(["ghost"], ["s2"]))
    assert (order, texts) == (["s1"], {"s1": "a"})


def test_reducer_records_detected_language() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.partial("s0", "hola", detected_language="es"))
    reducer.add(TranscriptionEvent.final("s0", "hola amigo"))
    result = reducer.result()
    assert result.detected_language == "es"


def test_reducer_refinalize_same_segment_keeps_single_slot() -> None:
    # A second final for the same segment_id overwrites in place (it is already in
    # _order), it must not append a duplicate ordering entry.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s0", "first"))
    reducer.add(TranscriptionEvent.final("s0", "second"))
    result = reducer.result()
    assert result.text == "second"
    assert result.segments is not None
    assert len(result.segments) == 1


def test_reducer_supersede_removes_committed_segment() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s0", "old", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.supersede(["s0"], ["s1"]))
    reducer.add(TranscriptionEvent.final("s1", "new", start=0.0, end=1.0))
    assert reducer.result().text == "new"


def test_reducer_supersede_unknown_id_is_noop() -> None:
    # Superseding an id the reducer never committed must be skipped silently (the
    # `if old_id in self._segments` guard), leaving committed segments intact.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s0", "kept", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.supersede(["never-seen"], ["s9"]))
    assert reducer.result().text == "kept"


def test_survey_fireredasr_no_interim_only_finals() -> None:
    # no_interim engine: each segment emits exactly one final, no partial.
    class _NoInterim(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            async for chunk in self.audio_chunks():
                yield TranscriptionEvent.final("seg-0", chunk.decode(), start=0.0)

    async def run() -> list[TranscriptionEvent]:
        session = _NoInterim()
        session.feed([b"sentence"])
        return await _collect(session)

    events = asyncio.run(run())
    assert not any(e.type == "partial" for e in events)
    assert any(e.type == "final" for e in events)


def test_survey_dsm_heartbeat_progress_does_not_reset_idle() -> None:
    # A DSM-style heartbeat (progress only) is not a content event.
    assert TranscriptionEvent.progress(audio_processed_until=1.0).is_content is False
    assert TranscriptionEvent.partial("s", "x").is_content is True
    assert TranscriptionEvent.final("s", "x").is_content is True
    assert TranscriptionEvent.supersede(["a"], ["b"]).is_content is True


# --------------------------------------------------------------------------- #
# A terminal releases the audio-input side (no feeder deadlock)
# --------------------------------------------------------------------------- #
def test_deadline_terminal_wakes_feeder_blocked_in_send_audio() -> None:
    # A deadline terminal cancels the producer -- the bounded audio
    # queue's only drainer -- so a feeder task blocked in send_audio() MUST be
    # released (and subsequent sends MUST raise StreamClosedError), or the
    # documented feeder+consumer pattern deadlocks inside `async with session:`.
    class _SilentEngineSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)  # consumes no audio, emits nothing
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _SilentEngineSession(done_timeout=0.05, audio_queue_maxsize=1)
        async with session:

            async def _feeder() -> str:
                try:
                    while True:
                        await session.send_audio(b"x")
                except StreamClosedError:
                    return "released"

            feeder = asyncio.create_task(_feeder())
            events = [event async for event in session]
            # Without the input release this await never completes (deadlock).
            outcome = await asyncio.wait_for(feeder, timeout=2.0)
        return events, outcome

    events, outcome = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "done_timeout"
    assert outcome == "released"


def test_engine_terminal_wakes_feeder_blocked_in_send_audio() -> None:
    # Funnel breadth: an ENGINE-emitted terminal also ends the producer, so
    # the same input release MUST happen on that path too.
    class _FailFastSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(0.05)  # let the feeder fill the queue and block
            yield TranscriptionEvent.make_error(code="engine_gave_up", recoverable=False)

    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _FailFastSession(audio_queue_maxsize=1)
        async with session:

            async def _feeder() -> str:
                try:
                    while True:
                        await session.send_audio(b"x")
                except StreamClosedError:
                    return "released"

            feeder = asyncio.create_task(_feeder())
            events = [event async for event in session]
            outcome = await asyncio.wait_for(feeder, timeout=2.0)
        return events, outcome

    events, outcome = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "engine_gave_up"
    assert outcome == "released"


# --------------------------------------------------------------------------- #
# Deadline drains admitted-but-undelivered events (stream == result)
# --------------------------------------------------------------------------- #
def test_deadline_terminal_drains_admitted_but_undelivered_events_first() -> None:
    # An event admitted to the reducer but still undelivered in the
    # buffer when a deadline fires MUST be delivered ahead of the synthesized
    # terminal, so result() equals the streamed content (stream == result).
    class _BurstThenSilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.final("s0", "seen")
            yield TranscriptionEvent.final("s1", "buffered")
            await asyncio.sleep(10)  # then go silent (no terminal)

    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _BurstThenSilentSession(done_timeout=5.0, max_idle=None, max_session_seconds=1.0)
        # Deterministic clock: deliver s0, then jump past the wall-clock budget
        # so the next top-of-loop check fires with s1 still buffered.
        ticks = iter([0.0, 0.0, 0.0, 2.0])

        def _clock() -> float:
            try:
                return next(ticks)
            except StopIteration:  # pragma: no cover - safety for extra reads
                return 2.0

        session._replace_reserved_attr("_monotonic", _clock)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        session.feed([])
        events = await _collect(session)
        return events, session.partial_result().text

    events, text = asyncio.run(run())
    # Both admitted finals are delivered, in order, before the terminal.
    assert [e.text for e in events if e.type == "final"] == ["seen", "buffered"]
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"
    assert text == "seen buffered"


def test_deadline_drain_stops_at_real_buffered_terminal() -> None:
    # Drain edge case: when the wall-clock cap fires with a REAL terminal already
    # buffered, the drain ends with it -- exactly one terminal is delivered,
    # never a second synthesized one after it.
    class _FastDoneSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.final("s0", "x")

    async def run() -> list[TranscriptionEvent]:
        session = _FastDoneSession(done_timeout=5.0, max_idle=None, max_session_seconds=1.0)
        # The producer finishes (final + done buffered) before the first get;
        # the clock then jumps past the budget at the top of the loop.
        ticks = iter([0.0, 2.0])

        def _clock() -> float:
            try:
                return next(ticks)
            except StopIteration:  # pragma: no cover - safety for extra reads
                return 2.0

        session._replace_reserved_attr("_monotonic", _clock)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        session.feed([])
        async with session:
            await asyncio.sleep(0.05)  # let the producer buffer final + done
            return [event async for event in session]

    events = asyncio.run(run())
    assert [event.type for event in events] == ["final", "done"]


# --------------------------------------------------------------------------- #
# Feed-source completion after terminal events
# --------------------------------------------------------------------------- #


def test_feed_source_drains_without_blocking_after_terminal() -> None:
    # Feed-mode input release: when the engine terminates while the fed source
    # has chunks, the feed task discards the remainder instead of blocking
    # forever on the dead queue -- it completes on its own.
    class _FailFastSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.make_error(code="engine_gave_up", recoverable=False)

    async def run() -> None:
        session = _FailFastSession(audio_queue_maxsize=1)
        session.feed([b"x"] * 8)
        async with session:
            events = [event async for event in session]
            assert events[-1].code == "engine_gave_up"
            feed_task = session._feed_task  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            assert feed_task is not None
            await asyncio.wait_for(feed_task, timeout=2.0)

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Ultra review fixes: regressions
# --------------------------------------------------------------------------- #
def test_supersede_of_superseded_segment_is_rejected() -> None:
    # Superseded is a terminal state -- an id retires
    # the moment it appears in old_ids and MUST NOT be retired a second time
    # (a double retirement would give one segment two replacement lineages).
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.partial("a", "hello")) is not None
    assert guard.admit(TranscriptionEvent.supersede(["a"], ["b"])) is not None
    assert guard.admit(TranscriptionEvent.supersede(["a"], ["c"])) is None
    assert any(d.code == "lifecycle_retired_resuperseded" for d in guard.diagnostics)


def test_supersede_of_superseded_segment_raises_in_strict() -> None:
    guard = _LifecycleGuard(strict=True)
    assert guard.admit(TranscriptionEvent.partial("a", "hello")) is not None
    assert guard.admit(TranscriptionEvent.supersede(["a"], ["b"])) is not None
    with pytest.raises(ValueError, match="already-superseded"):
        guard.admit(TranscriptionEvent.supersede(["a"], ["c"]))


def test_suppressed_event_does_not_advance_audio_cursor() -> None:
    # (cursor half): a suppressed illegal event must not poison the
    # session audio cursor -- later legal events would otherwise be clamped up
    # to the rejected event's cursor with a misleading diagnostic.
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.final("s0", "x")) is not None
    suppressed = guard.admit(TranscriptionEvent.partial("s0", "y", audio_processed_until=99.0))
    assert suppressed is None  # partial after final: illegal transition
    ok = guard.admit(TranscriptionEvent.progress(audio_processed_until=2.0))
    assert ok is not None
    assert ok.audio_processed_until == 2.0  # NOT clamped to the rejected 99.0
    assert not any(d.code == "audio_cursor_decreased" for d in guard.diagnostics)


def test_admitted_event_still_advances_audio_cursor() -> None:
    # The compute/commit split must not break normal monotonic enforcement.
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.progress(audio_processed_until=5.0)) is not None
    clamped = guard.admit(TranscriptionEvent.progress(audio_processed_until=1.0))
    assert clamped is not None
    assert clamped.audio_processed_until == 5.0
    assert any(d.code == "audio_cursor_decreased" for d in guard.diagnostics)


def test_recoverable_error_never_dropped_at_capacity() -> None:
    # The "never drop" rule does not distinguish recoverable from
    # terminal errors. At capacity an engine-yielded recoverable error must
    # bypass the bound, not be replaced by a backpressure overflow error.
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer(capacity=2)
        buf.put(TranscriptionEvent.partial("s0", "a"))
        buf.put(TranscriptionEvent.partial("s1", "b"))  # at capacity now
        buf.put(TranscriptionEvent.make_error("transient_glitch", recoverable=True))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert any(e.type == "error" and e.code == "transient_glitch" for e in events)


class _HangEndAudioSession(TranscriptionSession):
    """Session whose ``end_audio`` hangs (simulates a stuck engine in the body)."""

    async def end_audio(self) -> None:
        await asyncio.sleep(100)

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        yield TranscriptionEvent.done()  # pragma: no cover


def test_sync_bridge_body_timeout_propagates_unmasked() -> None:
    # When a lifecycle call times out INSIDE the with body, the
    # bridge tears the loop down; __exit__ must then be a no-op so the original
    # TimeoutError propagates -- not RuntimeError("Event loop is closed"), and
    # without leaving a never-awaited __aexit__ coroutine behind (the suite
    # runs with warnings as errors).
    with pytest.raises(TimeoutError, match="lifecycle call timed out"):
        with SyncSession(_HangEndAudioSession(), submit_timeout=0.1) as sync:
            sync.end_audio()


def test_sync_bridge_calls_after_teardown_raise_stream_closed() -> None:
    # Lifecycle calls after the teardown must fail with the
    # contracted StreamClosedError, not an unrelated loop RuntimeError.
    sync = SyncSession(_HangEndAudioSession(), submit_timeout=0.1)
    sync.__enter__()
    with pytest.raises(TimeoutError):
        sync.end_audio()
    with pytest.raises(StreamClosedError, match="already torn down"):
        sync.feed([b"x"])
    # __exit__ after teardown is a silent no-op (no masking, no leak).
    sync.__exit__(None, None, None)
    assert sync._thread.is_alive() is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_sync_pump_detects_frozen_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    # An engine running blocking
    # (non-async) code mid-session freezes the bridge loop while its thread
    # stays ALIVE; the in-loop deadlines cannot fire on a frozen loop, so the
    # pump's responsiveness probe is the only thing standing between the sync
    # caller and an infinite hang (the no-hang contract). Brief stalls
    # are tolerated (consecutive-probe threshold); a persistent freeze must
    # tear down and raise.
    monkeypatch.setattr(streaming_module, "_SYNC_PUMP_POLL_SECONDS", 0.05)

    class _BlocksMidSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.partial("s0", "hi")
            await asyncio.sleep(0.1)  # let the lifecycle submits finish first
            time.sleep(1.2)  # sync sleep: freezes the loop, thread stays alive
            yield TranscriptionEvent.done()  # pragma: no cover

    sync = SyncSession(_BlocksMidSession(), submit_timeout=5.0)
    sync.__enter__()
    sync.feed([b"x"])
    events: list[TranscriptionEvent] = []
    with pytest.raises(TimeoutError, match="frozen by blocking engine code"):
        for ev in sync:
            events.append(ev)
    assert [e.type for e in events] == ["partial"]


# --------------------------------------------------------------------------- #
# supersede placement golden traces (wire-shaped events -> reading order)
# --------------------------------------------------------------------------- #
#: Wire-shaped (JSON-dict) event traces with the expected reduced reading
#: order and text. Each trace drives BOTH canonical reduces -- the
#: ``reduce_event`` helper and ``StreamReducer`` -- through
#: ``TranscriptionEvent.model_validate`` (the wire layer), pinning the
#: two-layer isomorphism: same vectors, same order, same text. All segments
#: are untimestamped unless a trace says otherwise, so list order IS the
#: reading order (the spec's null-timestamp rule) and nothing is rescued by
#: a timestamp sort.
_PLACEMENT_TRACES: list[dict[str, Any]] = [
    {
        "name": "replace-first-block",
        "events": [
            {"type": "final", "segment_id": "a", "text": "hello"},
            {"type": "final", "segment_id": "b", "text": "world"},
            {"type": "supersede", "old_ids": ["a"], "new_ids": ["a2"]},
            {"type": "final", "segment_id": "a2", "text": "hi"},
        ],
        "order": ["a2", "b"],
        "text": "hi world",
    },
    {
        "name": "replace-middle-block",
        "events": [
            {"type": "final", "segment_id": "a", "text": "one"},
            {"type": "final", "segment_id": "b", "text": "two"},
            {"type": "final", "segment_id": "c", "text": "three"},
            {"type": "supersede", "old_ids": ["b"], "new_ids": ["b2"]},
            {"type": "final", "segment_id": "b2", "text": "TWO"},
        ],
        "order": ["a", "b2", "c"],
        "text": "one TWO three",
    },
    {
        "name": "replace-last-block",
        "events": [
            {"type": "final", "segment_id": "a", "text": "one"},
            {"type": "final", "segment_id": "b", "text": "two"},
            {"type": "supersede", "old_ids": ["b"], "new_ids": ["b2"]},
            {"type": "final", "segment_id": "b2", "text": "TWO"},
        ],
        "order": ["a", "b2"],
        "text": "one TWO",
    },
    {
        "name": "merge-many-to-one-mid-stream",
        "events": [
            {"type": "final", "segment_id": "s1", "text": "alpha"},
            {"type": "final", "segment_id": "s2", "text": "beta"},
            {"type": "final", "segment_id": "s3", "text": "gamma"},
            {"type": "final", "segment_id": "s4", "text": "delta"},
            {"type": "supersede", "old_ids": ["s2", "s3"], "new_ids": ["m"]},
            {"type": "final", "segment_id": "m", "text": "beta-gamma"},
        ],
        "order": ["s1", "m", "s4"],
        "text": "alpha beta-gamma delta",
    },
    {
        "name": "split-one-to-many-mid-stream",
        "events": [
            {"type": "final", "segment_id": "s1", "text": "alpha"},
            {"type": "final", "segment_id": "s2", "text": "betagamma"},
            {"type": "final", "segment_id": "s3", "text": "delta"},
            {"type": "supersede", "old_ids": ["s2"], "new_ids": ["x", "y"]},
            {"type": "final", "segment_id": "x", "text": "beta"},
            {"type": "final", "segment_id": "y", "text": "gamma"},
        ],
        "order": ["s1", "x", "y", "s3"],
        "text": "alpha beta gamma delta",
    },
    {
        "name": "many-to-many-mid-stream",
        "events": [
            {"type": "final", "segment_id": "s1", "text": "a"},
            {"type": "final", "segment_id": "s2", "text": "b"},
            {"type": "final", "segment_id": "s3", "text": "c"},
            {"type": "final", "segment_id": "s4", "text": "d"},
            {"type": "supersede", "old_ids": ["s2", "s3"], "new_ids": ["t", "u"]},
            {"type": "final", "segment_id": "t", "text": "B"},
            {"type": "final", "segment_id": "u", "text": "C"},
        ],
        "order": ["s1", "t", "u", "s4"],
        "text": "a B C d",
    },
    {
        "name": "pure-deletion-mid-stream",
        "events": [
            {"type": "final", "segment_id": "s1", "text": "keep"},
            {"type": "final", "segment_id": "s2", "text": "drop"},
            {"type": "final", "segment_id": "s3", "text": "tail"},
            {"type": "supersede", "old_ids": ["s2"], "new_ids": []},
        ],
        "order": ["s1", "s3"],
        "text": "keep tail",
    },
    {
        "name": "replacement-partial-before-final",
        "events": [
            {"type": "final", "segment_id": "a", "text": "hello"},
            {"type": "final", "segment_id": "b", "text": "world"},
            {"type": "supersede", "old_ids": ["a"], "new_ids": ["a2"]},
            {"type": "partial", "segment_id": "a2", "text": "h"},
            {"type": "final", "segment_id": "a2", "text": "hi"},
        ],
        "order": ["a2", "b"],
        "text": "hi world",
    },
    {
        "name": "chained-supersede-contentless-link",
        "events": [
            # A -> B -> C where B never produces a partial/final: its
            # supersede-declared position still anchors the second splice.
            {"type": "final", "segment_id": "A", "text": "first"},
            {"type": "final", "segment_id": "tail", "text": "last"},
            {"type": "supersede", "old_ids": ["A"], "new_ids": ["B"]},
            {"type": "supersede", "old_ids": ["B"], "new_ids": ["C"]},
            {"type": "final", "segment_id": "C", "text": "FIRST"},
        ],
        "order": ["C", "tail"],
        "text": "FIRST last",
    },
    {
        "name": "partial-declares-position",
        "events": [
            # "a" is declared by a partial only; the replacement still takes
            # a's position, ahead of the finalized "b".
            {"type": "partial", "segment_id": "a", "text": "he"},
            {"type": "final", "segment_id": "b", "text": "world"},
            {"type": "supersede", "old_ids": ["a"], "new_ids": ["a2"]},
            {"type": "final", "segment_id": "a2", "text": "hi"},
        ],
        "order": ["a2", "b"],
        "text": "hi world",
    },
]


def _wire_events(trace: dict[str, Any]) -> list[TranscriptionEvent]:
    """Validate a trace's dict events through the wire model.

    Args:
        trace: One golden trace.

    Returns:
        The validated events.
    """
    return [TranscriptionEvent.model_validate(raw) for raw in trace["events"]]


@pytest.mark.parametrize("trace", _PLACEMENT_TRACES, ids=lambda t: t["name"])
def test_supersede_placement_golden_traces_reduce_event(trace: dict[str, Any]) -> None:
    """The reduce_event helper reproduces every golden reading order."""
    order: list[str] = []
    texts: dict[str, str] = {}
    for event in _wire_events(trace):
        reduce_event(order, texts, event)
    assert order == trace["order"]
    display = " ".join(texts[sid] for sid in order if sid in texts)
    assert display == trace["text"]


@pytest.mark.parametrize("trace", _PLACEMENT_TRACES, ids=lambda t: t["name"])
def test_supersede_placement_golden_traces_stream_reducer(trace: dict[str, Any]) -> None:
    """StreamReducer agrees with the helper on every golden trace."""
    reducer = StreamReducer()
    for event in _wire_events(trace):
        reducer.add(event)
    result = reducer.result()
    assert result.text == trace["text"]
    # The reduced segments are the FINALIZED ids of the golden order, with
    # each segment carrying its final text.
    final_texts = {
        raw["segment_id"]: raw["text"] for raw in trace["events"] if raw["type"] == "final"
    }
    expected = [(sid, final_texts[sid]) for sid in trace["order"] if sid in final_texts]
    assert result.segments is not None
    assert [segment.text for segment in result.segments] == [text for _, text in expected]


def test_supersede_placement_with_timestamps_still_time_sorts() -> None:
    # When EVERY retained segment carries a start, the timestamp sort stays
    # authoritative (replacement carries the real re-measured span).
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("a", "hello", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.final("b", "world", start=1.0, end=2.0))
    reducer.add(TranscriptionEvent.supersede(["a"], ["a2"]))
    reducer.add(TranscriptionEvent.final("a2", "hi", start=0.0, end=1.0))
    assert reducer.result().text == "hi world"

    # MIXED spans (one segment unmeasured) -> reading order governs, which
    # after the in-place splice is already the correct order.
    mixed = StreamReducer()
    mixed.add(TranscriptionEvent.final("a", "hello", start=0.0, end=1.0))
    mixed.add(TranscriptionEvent.final("b", "world"))
    mixed.add(TranscriptionEvent.supersede(["a"], ["a2"]))
    mixed.add(TranscriptionEvent.final("a2", "hi"))
    assert mixed.result().text == "hi world"


def test_supersede_noncontiguous_old_block_is_suppressed_everywhere() -> None:
    # old_ids skipping over a live unrelated segment have no defined
    # placement: the reducer suppresses (duplicate-text side effect, like
    # every suppressed supersede) and diagnoses; the helper raises; the
    # session guard suppresses with the same code (strict raises below).
    events = [
        TranscriptionEvent.final("a", "one"),
        TranscriptionEvent.final("x", "keep"),
        TranscriptionEvent.final("b", "two"),
        TranscriptionEvent.supersede(["a", "b"], ["m"]),
        TranscriptionEvent.final("m", "merged"),
    ]
    reducer = StreamReducer()
    for event in events:
        reducer.add(event)
    result = reducer.result()
    # Suppressed supersede -> retired-nothing; m arrives as a fresh segment.
    assert result.text == "one keep two merged"
    assert any(d.code == "supersede_noncontiguous_old_ids" for d in result.diagnostics)

    order: list[str] = []
    texts: dict[str, str] = {}
    with pytest.raises(ValueError, match="contiguous block"):
        for event in events:
            reduce_event(order, texts, event)


def test_supersede_misordered_old_ids_are_suppressed() -> None:
    # Positions contiguous but listed AGAINST reading order: old_ids MUST be
    # in reading order (the replacement takes the block's place in it).
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("a", "one"))
    reducer.add(TranscriptionEvent.final("b", "two"))
    reducer.add(TranscriptionEvent.supersede(["b", "a"], ["m"]))
    reducer.add(TranscriptionEvent.final("m", "merged"))
    result = reducer.result()
    assert result.text == "one two merged"
    assert any(d.code == "supersede_noncontiguous_old_ids" for d in result.diagnostics)


def test_reducer_suppresses_unknown_retired_and_reintroduced_ids() -> None:
    # The reducer mirrors the guard's order-integrity rejections so a
    # standalone (guardless) reduce degrades identically: unknown old id,
    # double retirement, reintroduced new id, and a final resurrecting a
    # retired id are each suppressed with the guard's diagnostic code.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("a", "one"))
    reducer.add(TranscriptionEvent.supersede(["ghost"], ["g2"]))
    reducer.add(TranscriptionEvent.supersede(["a"], ["a2"]))
    reducer.add(TranscriptionEvent.supersede(["a"], ["a3"]))  # retired twice
    reducer.add(TranscriptionEvent.supersede(["a2"], ["a"]))  # reintroduces a
    reducer.add(TranscriptionEvent.final("a", "zombie"))  # resurrects a
    reducer.add(TranscriptionEvent.final("a2", "ONE"))
    result = reducer.result()
    assert result.text == "ONE"
    codes = [d.code for d in result.diagnostics]
    assert codes.count("supersede_unknown_old_id") == 1
    assert codes.count("lifecycle_retired_resuperseded") == 1
    assert codes.count("supersede_reintroduces_segment") == 1
    assert codes.count("lifecycle_after_terminal") == 1
    # result() is idempotent: a second call reports the same state.
    assert reducer.result().text == "ONE"
    assert [d.code for d in reducer.result().diagnostics] == codes


def test_reducer_suppressed_events_do_not_commit_detected_language() -> None:
    # A suppressed event must change NOTHING -- the sticky session language
    # included. Committing it on entry let a refused supersede rewrite
    # result().detected_language while its content was suppressed: the event
    # stream and its reduction silently disagreeing (the drift this pins).
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s0", "hello", detected_language="en"))
    # Suppressed: unknown old id -- its 'fr' must not stick.
    reducer.add(TranscriptionEvent.supersede(["never-declared"], ["s1"], detected_language="fr"))
    assert reducer.result().detected_language == "en"
    # Suppressed: final resurrecting a retired id.
    reducer.add(TranscriptionEvent.supersede(["s0"], ["s2"]))
    reducer.add(TranscriptionEvent.final("s0", "zombie", detected_language="de"))
    assert reducer.result().detected_language == "en"
    # Suppressed: partial for a retired id (guard's after-terminal mirror).
    reducer.add(TranscriptionEvent.partial("s0", "zombie", detected_language="it"))
    result = reducer.result()
    assert result.detected_language == "en"
    assert [d.code for d in result.diagnostics].count("lifecycle_after_terminal") == 2
    # An admitted event still commits (last non-None wins), and the sticky
    # accessor reports the same value result() does.
    reducer.add(TranscriptionEvent.final("s2", "bonjour", detected_language="fr"))
    assert reducer.detected_language == "fr"
    assert reducer.result().detected_language == "fr"


def test_reducer_suppressed_partial_does_not_declare_position() -> None:
    # The retired-partial suppression must also keep its (refused)
    # declaration out of the reading order -- identical to the pre-existing
    # silent ignore, now diagnosed.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("a", "one"))
    reducer.add(TranscriptionEvent.supersede(["a"], ["b"]))
    reducer.add(TranscriptionEvent.partial("a", "ghost"))  # suppressed
    reducer.add(TranscriptionEvent.final("b", "two"))
    result = reducer.result()
    assert result.text == "two"
    assert [d.code for d in result.diagnostics].count("lifecycle_after_terminal") == 1


def test_guard_suppresses_noncontiguous_supersede_and_strict_raises() -> None:
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.final("a", "one")) is not None
    assert guard.admit(TranscriptionEvent.final("x", "keep")) is not None
    assert guard.admit(TranscriptionEvent.final("b", "two")) is not None
    assert guard.admit(TranscriptionEvent.supersede(["a", "b"], ["m"])) is None
    assert any(d.code == "supersede_noncontiguous_old_ids" for d in guard.diagnostics)
    # The suppressed event mutated nothing: a and b are still live and a
    # CONTIGUOUS retirement of a alone still works.
    assert guard.admit(TranscriptionEvent.supersede(["a"], ["a2"])) is not None

    strict = _LifecycleGuard(strict=True)
    assert strict.admit(TranscriptionEvent.final("a", "one")) is not None
    assert strict.admit(TranscriptionEvent.final("x", "keep")) is not None
    assert strict.admit(TranscriptionEvent.final("b", "two")) is not None
    with pytest.raises(ValueError, match="contiguous block"):
        strict.admit(TranscriptionEvent.supersede(["a", "b"], ["m"]))


def test_session_reduces_mid_stream_supersede_in_place() -> None:
    # End-to-end through TranscriptionSession: the guard admits the legal
    # splice and the session's reducer places the replacement in reading
    # order (no timestamps anywhere).
    events = [
        TranscriptionEvent.final("a", "hello"),
        TranscriptionEvent.final("b", "world"),
        TranscriptionEvent.supersede(["a"], ["a2"]),
        TranscriptionEvent.final("a2", "hi"),
    ]

    async def _run() -> str:
        session = _ScriptedSession(events)
        async with session:
            async for _ in session:
                pass
        return session.result().text

    assert asyncio.run(_run()) == "hi world"


def test_reading_order_ledger_and_helper_edge_paths() -> None:
    # The ledger's own unknown-id answer (its callers pre-check for precise
    # diagnostics, but the primitive must be safe standalone).
    ledger = streaming_module._ReadingOrderLedger()  # pyright: ignore[reportPrivateUsage]
    ledger.declare("a")
    assert ledger.block_start(["ghost"]) is None
    assert "a" in ledger and "ghost" not in ledger

    # reduce_event: a reintroduced new_id fails loudly (the helper has no
    # diagnostics channel to suppress into).
    order: list[str] = []
    texts: dict[str, str] = {}
    reduce_event(order, texts, TranscriptionEvent.final("a", "one"))
    reduce_event(order, texts, TranscriptionEvent.final("b", "two"))
    with pytest.raises(ValueError, match="already holds a reading-order position"):
        reduce_event(order, texts, TranscriptionEvent.supersede(["a"], ["b"]))

    # A partial for a RETIRED id cannot resurrect a reading-order position.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("a", "one"))
    reducer.add(TranscriptionEvent.final("b", "two"))
    reducer.add(TranscriptionEvent.supersede(["a"], ["a2"]))
    reducer.add(TranscriptionEvent.partial("a", "zombie"))
    reducer.add(TranscriptionEvent.final("a2", "ONE"))
    assert reducer.result().text == "ONE two"


def test_empty_reduced_result_carries_an_empty_segments_list() -> None:
    """An empty reduction is "performed but empty" -- [] on the wire, not null.

    The spec's null rule: ``None`` = not requested / not applicable (the
    renderers may synthesize a whole-text fallback cue over it), ``[]`` =
    requested-and-performed but empty (zero cues, never fabricated). The
    reducer ran the segment lifecycle, so its emptiness is the second
    state: a fresh reducer, a silence-only session, and a
    delete-everything supersede all reduce to ``segments == []``.
    """
    from standard_asr.renderers import to_srt, to_vtt

    fresh = StreamReducer().result()
    assert fresh.segments == []
    assert fresh.text == ""
    assert fresh.model_dump(mode="json")["segments"] == []

    silence = StreamReducer()
    silence.add(TranscriptionEvent.progress(audio_processed_until=5.0))
    silence.add(TranscriptionEvent.done())
    assert silence.result().segments == []

    deleted = StreamReducer()
    deleted.add(TranscriptionEvent.final("s1", "gone"))
    deleted.add(TranscriptionEvent.supersede(["s1"], []))
    result = deleted.result()
    assert result.segments == []
    assert result.text == ""
    # Zero cues, never a fabricated whole-text fallback (that is the
    # segments-is-None shape, which a reduced result can no longer be).
    assert to_srt(result) == ""
    assert to_vtt(result) == "WEBVTT\n"

    # The batch side is untouched: an engine that was not asked for
    # segments still models the "not requested" state as None.
    from standard_asr.contract.results import TranscriptionResult

    assert TranscriptionResult(text="hi").segments is None


def test_silent_session_end_to_end_reduces_to_empty_list() -> None:
    async def _run() -> Any:
        session = _ScriptedSession([TranscriptionEvent.progress(audio_processed_until=1.0)])
        async with session:
            async for _ in session:
                pass
        return session.result()

    result = asyncio.run(_run())
    assert result.segments == []
    assert result.text == ""
