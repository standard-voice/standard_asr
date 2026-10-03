# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared stable-text cases in ``fixtures/streaming/stable_text.json``.

The JSON file is written for consumers in any language: each case lists
wire-shaped events and what the lifecycle guard does with each of them. These
tests replay every case through the runtime's lifecycle guard, the session's
reducer, and the reference reduce, so the file cannot disagree with those
three. They do not run a session. An event that fails construction is
skipped here and the next one is replayed, while a session whose engine
fails to construct an event ends with ``engine_error``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from standard_asr.runtime.streaming import (
    StreamReducer,
    TranscriptionEvent,
    _LifecycleGuard,  # pyright: ignore[reportPrivateUsage]
    reduce_event,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "streaming" / "stable_text.json"
_DOCUMENT: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_CASES: list[dict[str, Any]] = _DOCUMENT["cases"]


def test_fixture_case_names_are_unique_and_groups_are_known() -> None:
    names = [case["name"] for case in _CASES]
    assert len(names) == len(set(names))
    assert {case["group"] for case in _CASES} == {"core", "final", "supersede"}


def test_fixture_content_events_always_carry_stable_text() -> None:
    # A consumer in another language reads stable_text from the frame and
    # never applies a default, so every partial and final names it.
    for case in _CASES:
        for event in case["events"]:
            if event["type"] in ("partial", "final"):
                assert "stable_text" in event, (case["name"], event)


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_fixture_case(case: dict[str, Any]) -> None:
    expected = case["expected"]
    assert len(case["events"]) == len(expected["events"])
    guard = _LifecycleGuard()
    reducer = StreamReducer()
    order: list[str] = []
    texts: dict[str, str] = {}
    for raw, want in zip(case["events"], expected["events"], strict=True):
        try:
            event = TranscriptionEvent.model_validate(raw)
        except ValidationError:
            assert want["outcome"] == "invalid", raw
            assert want["diagnostics"] == [], raw
            continue
        before = len(guard.diagnostics)
        delivered = guard.admit(event)
        codes = [d.code for d in guard.diagnostics[before:]]
        assert codes == want["diagnostics"], raw
        if delivered is None:
            assert want["outcome"] == "rejected", raw
            continue
        assert want["outcome"] == "admitted", raw
        assert delivered.stable_text == want.get("stable_text"), raw
        reducer.add(delivered)
        reduce_event(order, texts, delivered)
    # The reference reduce keeps every live segment's latest text in reading
    # order; the session's reducer keeps the finalized segments.
    assert [[sid, texts[sid]] for sid in order] == expected["display"]
    result = reducer.result()
    assert [s.text for s in result.segments or []] == expected["result_segments"]
    if "result_text" in expected:
        # Only single-segment cases pin the reduced text: how the texts of
        # several segments are joined is not settled.
        assert len(expected["result_segments"]) <= 1
        assert result.text == expected["result_text"]
