---
title: Streaming
---

# Streaming

Standard ASR unifies the widely divergent streaming behaviors of 30+ ASR engines under one event protocol. This guide covers everything an application developer needs to build a robust streaming integration.

## Opening a session

Ask the engine for the PCM wire format it wants, then open a session:

```python
audio_format = engine.recommended_wire_format()

async with engine.start_transcription(audio_format=audio_format) as session:
    ...
```

`recommended_wire_format()` returns the engine's preferred sample rate and encoding as an `AudioFormat`, or `None` when the engine declares no usable positive sample rate (no bare-frame session can be opened then). If you need a specific format (for example, 8 kHz for telephony), construct one yourself -- the engine raises `UnsupportedFeatureError` if it cannot accept it. The recommendation is derived from the engine's static Properties; whether a bare-frame session can be opened at all is a capability question -- gate on `engine.supports("streaming_input")` first.

Entering the `async with` block is where the engine opens its resources, so it is also where an inference-artifact failure surfaces: `ArtifactUnavailableError` when a required artifact is gone, `ArtifactAcquisitionError` when an acquisition the engine was allowed to attempt failed. Both are raised rather than delivered as events, because no event stream exists yet; the `artifact_unavailable` and `artifact_acquisition_failed` codes below carry the same failures once the engine is producing events. See [Inference artifacts](../reference/artifacts.md).

> **Known pre-1.0 limitation.** The recommendation is a format the engine's session-establishment *validator* accepts -- for the rare self-managed-wire engine (an engine that manages its own wire format and opens sessions with a bare `start_transcription()`, taking no `audio_format` at all), that is not the same thing as the right way to open the session. How such engines declare their transport is being settled in the capability-ontology ADR ([#45](https://github.com/standard-voice/standard_asr/issues/45)); until then, follow the engine's own documentation for the no-argument open path.

For whole-input streaming (the engine streams *output* over a complete audio file), pass `audio=` instead of `audio_format=`:

```python
async with engine.start_transcription(audio="meeting.wav") as session:
    async for event in session:
        ...
```

## Feeding audio

For live-input streaming there are two mutually exclusive input modes:

**Managed mode** -- hand the session an iterable of PCM byte chunks and let it drive the input side for you:

```python
session.feed(microphone)  # any sync or async iterable of bytes chunks
```

`feed()` consumes the source and signals end-of-input automatically when the iterable finishes. Do **not** call `end_audio()` yourself in this mode -- a session is owned by exactly one input mode, and mixing them raises `InvalidSessionUseError`.

**Manual mode** -- push chunks yourself and signal the end explicitly:

```python
await session.send_audio(chunk)  # repeat per chunk
await session.end_audio()  # signal end-of-input
```

## The event protocol

Every streaming session emits a sequence of `TranscriptionEvent` objects. The `type` field tells you what happened:

| Type | Meaning | `text` | `segment_id` | `speaker` |
| ---- | ------- | ------ | ------------- | --------- |
| `partial` | Interim text that **may change** with the next event on this segment. | Current best guess. | The segment this partial belongs to. | Segment speaker label when diarized, else `None`. |
| `final` | This segment's text is **settled against new audio** -- more audio does not change it. It can still be replaced by a `supersede`, or restated once by a terminal `final` with `finality="closed"` (see "Finality" below). | Final text. | The segment that is now final. | Segment speaker label when diarized, else `None`. |
| `supersede` | The engine re-segmented: one or more previously emitted segments are **replaced**. The replacement segments' own `partial`/`final` events arrive afterward -- a `supersede` always precedes any event of its `new_ids`. | `None` | `None` (check `old_ids`). | `None` |
| `progress` | A progress heartbeat (for example, audio position). No transcript content. | `None` | `None`, or the segment it reports on. | `None` |
| `done` | The session is complete. No more events follow. | `None` | `None` | `None` |
| `error` | An engine error mid-stream. Machine-readable code in `event.code`; when the standard layer synthesized the error from an exception, human detail in `event.extra.get("detail")` (deadline and engine-authored errors may carry no detail); `event.recoverable` says whether the session may continue. Inference-artifact failures use `artifact_unavailable` or `artifact_acquisition_failed`, not `engine_error`. | `None` | `None`, or the segment the error concerns. | `None` |

Request diarization when opening the session (`RuntimeParams(diarization=DIARIZE)`, gated by `streaming.diarization`); `event.speaker` then carries the segment-level speaker label on `partial` / `final` events. It stays `None` when diarization was not requested or is unsupported — except on engines whose diarization is `always_on`, which may emit speaker labels unrequested.

## The core reduce

Handle `partial`, `final`, and `supersede`, and your app is safe on every compliant engine -- including ones that rewrite interim text or merge segments after emitting them:

```python
order: list[str] = []  # reading order of live segment ids
texts: dict[str, str] = {}

async for event in session:
    if event.type in ("partial", "final"):
        if event.segment_id not in order:
            order.append(event.segment_id)  # first mention claims a position
        texts[event.segment_id] = event.text
    elif event.type == "supersede":
        pos = order.index(event.old_ids[0])  # the retired block's position
        for old_id in event.old_ids:
            order.remove(old_id)
            texts.pop(old_id, None)
        order[pos:pos] = event.new_ids  # replacements take its place
```

Display text is `texts` joined in `order`. The state is a reading-order list plus a text map, not a bare map. For engines that emit no timestamps, **list order is the reading order**. A mid-stream `supersede` must splice its replacements into the retired block's position. A bare dict can only append, which would silently reorder the transcript. This exact reduce ships as `standard_asr.runtime.streaming.reduce_event`, and `StreamReducer` / `session.result()` build a full `TranscriptionResult` the same way.

Engines that never revise or re-segment never emit `supersede`. Your code does not need to know which engine is running.

## Stable text

Some engines can tell you which part of a segment's current text is settled before the segment is final. Every `partial` and `final` event carries that part as `event.stable_text`: a string that `event.text` starts with. The rest of `event.text` may still change.

```python
if event.type == "partial":
    stable = event.stable_text  # only a closed final may reformat it
    tentative = event.text[len(stable) :]  # may still change
```

The field is a string, not a count, so programming languages do not have to agree on a unit for counting characters. If you split `text` into the stable text and the rest, remove exactly the code points of `stable_text` from the start of `text`. Work on Unicode code points, or on the code units of one encoding used for both strings. Do not find the split by counting user-perceived characters alone, as Swift's `String.count` does. The [streaming specification](../specification/protocol.md#streaming), section 4.2, requires this of every client that splits `text`. A tool that splits text into user-perceived characters still has a use: holding back part of the stable text, as "Where stable text ends" describes.

What the engine promises:

- Stable text only grows, except as the two events below allow. A later `partial` or plain `final` of the segment starts its `text` with the segment's earlier stable text.
- A `partial` carries non-empty `stable_text` only if the engine declares the `streaming.partial_stability` capability.
- A `final` is stable as a whole: its `stable_text` equals its `text`.

Two events can end the promise, and your code sees both:

- A `supersede` withdraws the segments in `old_ids`, stable text included. Drop what you derived from them and start over on `new_ids`. An engine sends one only if it declares the `streaming.re_segments` capability.
- A `final` with `finality="closed"` may restate the segment once (see "Finality").

The session checks both capability declarations while it runs. If the engine breaks one, the session does not change or drop the event because of it. It records `stream_exceeds_partial_stability` or `stream_exceeds_re_segments` in `session.diagnostics()`, once per session. A session built with `strict_lifecycle=True` ends instead, with an `error` event whose code is `engine_error`. The compliance suite runs the same check on a recorded stream.

The session can check only when it has the engine's capabilities. `EngineBase.start_transcription` gives them to every session it opens. An engine that implements the protocol without deriving from `EngineBase` does not. The reference server gives them to every session it opens, and you can do the same with `bind_session_capabilities(session, engine)` from `standard_asr`. A session that never gets them is not checked.

The session does enforce the stable-text rules it can check. It suppresses or repairs an event that breaks them and records `stable_text_rewritten` or `stable_text_clamped` in `session.diagnostics()`. If the session reaches `done` while a segment that has stable text is still open, it records `stable_text_abandoned` there too: text the engine marked stable is missing from `session.result()`. These diagnostics are in `session.diagnostics()`, not in `session.result().diagnostics`. A session built with `strict_lifecycle=True` instead ends at the first violation, with an `error` event whose code is `engine_error`.

How to use it:

- **To display text**, ignore the field and use the core reduce above.
- **To start work early**, such as intent detection or translation, process the part of `stable_text` you have not processed yet. This suits work you can cancel or redo. Stable text does not mean that a word, a sentence, or an intent is complete. Cancel that work when a `supersede` retires the segment:

  ```python
  acted: dict[str, str] = {}  # segment_id -> stable text already processed

  async for event in session:
      if event.type in ("partial", "final"):
          done = acted.get(event.segment_id, "")
          if not event.stable_text.startswith(done):
              # Only a `closed` restatement rewrites stable text: redo the segment.
              cancel_processing(event.segment_id)
              done = ""
          new_text = event.stable_text[len(done) :]
          if new_text:
              start_processing(event.segment_id, new_text)
              acted[event.segment_id] = event.stable_text
      elif event.type == "supersede":
          for old_id in event.old_ids:
              cancel_processing(old_id)
              acted.pop(old_id, None)
  ```

- **For an action you cannot undo**, such as typing the text into another program, writing to a database, or sending a message, choose the point to act from what can still change (see "Finality"). The protocol says what an engine may still change; it does not say when your application acts, with one exception about speakers: the speaker routing warning in section 7.2 of the [streaming specification](../specification/protocol.md#streaming) forbids a voice assistant from taking an irreversible action, such as routing, on the speaker of a `partial` in a segment that has no stable text yet.
  - If you need a segment's text never to change again, wait for its `closed` final. An engine whose `streaming.finality_level` mode is `"final"` does not promise one, so on that engine the wait can last until the session ends.
  - If you need the session's final result, wait for `done` and use `session.result()`. The result holds only segments that reached `final` and were not superseded; `done` does not turn an open `partial` into part of the result.
  - On an engine that does not declare `streaming.re_segments`, the only change to the text after a `final` is one `closed` restatement of how it is written. That restatement may also correct the segment's `speaker`. If your action tolerates that, you can act on the `final`. If it needs the exact final spelling, punctuation, or number format, do not treat a plain `final` as `closed`. If such an engine sends a `supersede` anyway, the event still arrives, and `session.diagnostics()` holds `stream_exceeds_re_segments`. That diagnostic tells you that the engine withdrew text it declared it would not withdraw.
  - If you act earlier, or act on a `final` from an engine that may send `supersede`, be ready for a later event to disagree with what you already did.
- **For output in reading order**, remember that stable text is per segment. Two segments can be open at the same time, and the later one can become stable first. Keep the reading order with the core reduce above, which also puts replacement segments where the retired ones were. If your output can only append, and cannot take text back or replace it, also wait until the segments before it are as settled as that output needs. A plain `final` can still be superseded or restated (see "Finality").

### Where stable text ends

An engine must end stable text between two user-perceived characters, never inside one. A user-perceived character can be several code points: a letter with its accent, a consonant with its vowel sign, or an emoji built from several parts. Unicode calls such a unit an extended grapheme cluster.

The standard layer checks only part of this rule. It catches stable text that ends before a combining mark, such as an accent or most vowel signs. It also catches stable text that ends before a zero width joiner or a zero width non-joiner, or right after a zero width joiner. It does not catch these splits inside one character:

- Thai before the vowel SARA AM (`ำ`), which is in everyday words such as `ทำ`, "do", and `น้ำ`, "water". Lao has the same vowel.
- A joined pair of consonants such as `क्ष`, which is in most sentences of Hindi, Marathi, Nepali, Bengali, Gujarati, Odia, Telugu, and Malayalam. Under the ICU library's segmentation, Khmer and Myanmar stacked consonants are the same case.
- Korean written as separate letters (jamo) instead of whole syllables.
- Flags, emoji with a skin tone, emoji tag sequences such as the flag of Scotland, half-width katakana with a voiced sound mark, and a CR LF line break.

An engine that marks stable text a whole word at a time does not split a character. The risk comes from an engine that places the boundary inside a word, for example after each model token, and gets it wrong. The [streaming section of the specification](../specification/protocol.md#streaming) owns the rule and the exact check.

Such a split never changes the transcript: `text` is always the whole segment text, and the result is built from it. Whether your application is affected depends on what it does with stable text:

- **Not affected:** an application that ignores `stable_text`, or appends each newly stable piece to one place, such as a text field. The same code points arrive in the same order.
- **Affected:** an application that draws the stable part and the rest as two separately styled runs. Text rendering usually does not join a character across two runs. The reader sees a broken character at the boundary, such as a vowel sign drawn on a dotted circle. It repairs itself once the stable text grows past that character.
- **Affected:** an application that hands each newly stable piece to something that treats it as complete text. Examples are speech synthesis, translation, a command parser, a search query, Unicode normalization, and a length measurement. One piece ends with half a character, and the next piece starts with the other half.

In both affected cases, the last stable character can also change: stable `ท` becomes `ทำ` when the next piece arrives.

If your application is affected and serves one of these languages, you can check the boundary yourself. Split `text` into user-perceived characters with the tool your platform provides: `Intl.Segmenter` in JavaScript, `Character` in Swift, ICU `BreakIterator` in Java and C++, the `unicode-segmentation` crate in Rust, or `\X` in the third-party `regex` module in Python. If `stable_text` ends inside a character, treat it as ending where that character starts. This hold-back is always safe, because it only treats less text as stable.

## Finality

A `final` settles a segment against new audio. Two events can still follow it. A `supersede` can replace the segment. An engine may also send the segment once more as a `final` with `finality="closed"`. That is a post-processing restatement that changes how the text is written (punctuation, numbers, casing), not what was said. The restated text can be shorter: "twenty twenty" becomes "2020". It may also correct the segment's `speaker`. Replace the displayed text when a `closed` final arrives; do not append. After `closed`, the segment does not change and cannot be superseded. That holds for this segment only, not for other segments or the session.

The engine's `streaming.finality_level` capability says whether every segment reaches `closed`:

- With mode `"final"`, the engine does not promise a `closed` final for each segment, and it may still send one.
- With mode `"closed"`, the engine brings every segment that reached `final`, and that no `supersede` retired, to `closed` before the session ends with `done`.

The session checks this declaration when it reaches `done`, whether the engine sent `done` or the session added it after the engine's last event. Under mode `"closed"`, if a segment that no `supersede` retired is still `final` and not `closed` at that point, the session still delivers `done` and records `finality_level_not_reached` in `session.diagnostics()`. With `strict_lifecycle=True`, the session ends with a terminal `error` event whose code is `engine_error`, in place of `done`. The compliance suite runs the same check on a recorded stream.

## Collapsing a session into a result

After the session ends, collapse all events into a standard `TranscriptionResult`:

```python
result = session.result()
print(result.text)
print(result.segments)
```

This gives you the same constant-shape result you get from `engine.transcribe()`, so your downstream code (subtitle rendering, search, etc.) works identically whether the input was batch or streamed.

One honesty note: some engines omit timestamps (or one of the two bounds) while streaming. The reducer stores the engine's measurement verbatim: `Segment.start`/`end` are `float | None`, and `None` means "not measured" (check `segment.timestamp_status`: `"measured"`, `"start_only"`, or `"unavailable"`). Nothing is fabricated. A result with any unmeasured span also carries a `segment_timestamps_unavailable` warning diagnostic as the aggregate disclosure. The renderers read the values themselves. A result whose every segment is *renderable* (a measured span that survives the output's millisecond grid) renders per-segment faithfully. An unmeasured span makes `to_srt`/`to_vtt` raise `SubtitleRenderingError` by default. A measured span that quantizes to zero milliseconds does the same, because players silently drop a `T --> T` cue. Rendering such a segment would mean silently dropping, hiding, or fabricating timing, and that trade-off is yours to make. Pass `on_unrenderable="omit"` to keep only the renderable cues (the other segments' text stays in `result.text` but not in the file), or `"collapse"` to render one whole-text cue with no per-segment timeline.

## Synchronous bridge

If you cannot use `async`, wrap the session in `SyncSession`:

```python
from standard_asr import SyncSession

audio_format = engine.recommended_wire_format()
sync = SyncSession(engine.start_transcription(audio_format=audio_format))

with sync:
    sync.feed(pcm_chunks)  # an iterable of bytes chunks (or one bytes chunk)
    for event in sync:
        print(event.type, event.text)
```

`SyncSession` mirrors the async session's input modes: `feed(...)` for managed input (end-of-input is signaled automatically), or `send_audio(chunk)` + `end_audio()` for manual input. As with the async session, the two modes must not be mixed.

`SyncSession` runs the async session on a background thread and exposes a blocking iterator. See the [API reference](../reference/streaming.md) for the full interface.

## Deadlines

Application-level deadlines control how long a session waits for the engine:

```python
from standard_asr import StreamDeadlines

async with engine.start_transcription(
    audio_format=audio_format,
    deadlines=StreamDeadlines(max_idle=5.0, max_session_seconds=60.0),
) as session:
    ...
```

The three deadlines are `done_timeout` (pipeline-inactivity hang backstop), `max_idle` (content-stall detector), and `max_session_seconds` (absolute wall-clock cap); each accepts `None` to disable it.

When a deadline fires, the session terminates with a terminal **`error`** event (`code` = `done_timeout` / `stream_stalled` / `session_timeout`) -- not a `done` event -- so a deadline-killed session is never mistaken for normal completion. Handle the `error` event's `code` to distinguish the cases.

## Diagnostics mid-stream

The standard layer attaches parameter-gating and language-resolution diagnostics to the session at `start_transcription` (for example, a best-effort drop of an unsupported feature -- always disclosed, never silent). Engines add their own mid-stream notes via `session.emit_diagnostic()` (for example, a lossy fallback). The session adds its own when the engine breaks a protocol rule the session checks, such as the stable-text rules above. All of them arrive through `session.diagnostics()`, without interrupting the event flow.

## Further reading

- [API Reference: streaming](../reference/streaming.md) -- full type signatures.
- [Specification](../specification/protocol.md) -- the normative segment lifecycle, event ordering, and backpressure rules.
