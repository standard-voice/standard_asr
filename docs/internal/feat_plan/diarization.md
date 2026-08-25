# Diarization (Speaker Identification) — Design Case

**Status:** Implementation underway on `feat/diarization` (2026-07-01). All
eight §7 decisions adopted as recommended (num_speakers deferred → empty
marker; `always_on` a queryable `FlagCap` child of the cap node (pre-merge
revision — see §6 revision log; was a bare bool); frozen-speaker guard +
carry-forward + cross-speaker supersede ban as one package; label validators;
reconnect mint-fresh default; granularity deferred). Normative spec text has
landed in `docs/content/specification/protocol.md`; code chunks follow. This document
captures the cross-engine survey, design decisions, risk analysis, and
implementation plan.

**Branch:** `feat/diarization`

Builds on and must stay consistent with:
- `docs/content/specification/protocol.md` — §Capabilities (DiarizationCap), §Runtime
  Parameters (RuntimeParams), §Transcription Result (TR.5 speaker reserve),
  §Streaming (TranscriptionEvent).
- `src/standard_asr/contract/capabilities.py` — `DiarizationCap`,
  `DiarizationConstraints`, `BatchCapabilities.diarization`.
- `src/standard_asr/contract/results.py` — `Segment.speaker`, `Word.speaker`.
- `src/standard_asr/contract/params.py` — `RuntimeParams` (closed, no
  diarization field yet).
- `src/standard_asr/runtime/streaming.py` — `TranscriptionEvent` (no speaker field),
  `StreamReducer` (does not propagate speaker).

---

## 1. Problem statement

Speaker diarization ("who said what") is a core ASR feature supported by the
majority of cloud and local engines. Standard ASR already reserves the result
shape (`Segment.speaker`, `Word.speaker`) and has a capability shell
(`DiarizationCap`), but there is **no portable way to request diarization**.
An application must drop into engine-specific `provider_params` — defeating the
"one interface, any engine" promise.

The gap: capability exists → result shape exists → **request path missing** →
gating missing → streaming support missing → compliance tests missing.

---

## 2. Cross-engine survey (38 entries, June 2026)

> Covers the complete ASR model inventory: 15 cloud APIs + 7 local frameworks
> + 16 mlx-audio STT families (20 presets). See
> `memory/research-report-model-inventory.md` for the canonical list.

### 2.1 Engines with diarization support

| Engine | Enable mechanism | Count hint | Label type | Granularity | Named speakers | Batch | Streaming | Max spk |
|---|---|---|---|---|---|---|---|---|
| **OpenAI** | Separate model ID | None | `str` (`"A"`,`"B"` or names) | per-segment | Yes (4 names + audio refs) | Yes | SSE only | ~unlimited |
| **ElevenLabs** | `diarize` bool | `num_speakers` ≤32 | `str` (`"speaker_0"` / roles) | per-word | Via speaker library | Yes | No | 32 |
| **Google STT** v1/v2 | `enable_speaker_diarization` / config presence | `min/max_speaker_count` | `str` (`"1"`,`"2"` / roles) | per-word | No | Yes | Yes (not Chirp 3) | ~10 |
| **AWS Transcribe** | `ShowSpeakerLabels` bool | `MaxSpeakerLabels` 2–10 (batch) | `str` (`"spk_0"` / `"0"`) | per-word | No | Yes | Yes | 10 |
| **Azure Speech** | `diarization.enabled` / class | `maxSpeakers` 2–35 | `int` / `str` (`"Guest-1"`) | per-segment | No (enrollment retired) | Yes | Yes | 35 |
| **Deepgram** | `diarize_model` enum | None (auto) | `int` (0-indexed) | per-word | No | Yes | Yes (v1, buggy) | undoc. |
| **AssemblyAI** | `speaker_labels` bool | `speakers_expected`, `min/max` | `str` (`"A"`,`"B"`) | per-word + utterance | Via add-on | Yes | Yes | 30/10 |
| **Speechmatics** | `diarization` enum | `max_speakers` | `str` (`"S1"`,`"S2"`,`"UU"`) | per-word | Via enrolled speakers | Yes | Yes | 20/unlimited |
| **Rev.ai** | `skip_diarization` (on by default) | `speakers_count` | `int` | per-monologue | No | Yes | Switch only | undoc. |
| **Gladia** | `diarization` bool | `number_of_speakers`, `min/max` | `int` (0-based) | per-utterance | No | Yes | No | undoc. |
| **阿里雲** | `diarization_enabled` / `auto_split` bool | `speaker_count` 2–100 | `int` (0-indexed) | per-sentence | No | Yes | No | 100 |
| **火山引擎** | `enable_speaker_info` bool (default true) | None (auto) | `str` (1-indexed) | per-utterance | No | Yes | unconfirmed | undoc. |
| **騰訊雲** | `SpeakerDiarization` 0/1/3 | `SpeakerNumber` 0–10 (8k) | `int` (0-indexed) | per-sentence | Yes (mode 3: enrolled voices) | Yes | No | 20 (auto) |
| **whisper.cpp** | `--diarize`/`--tinydiarize` | None | `str`/turn marker | per-segment/turn | No | Yes | PoC only | 2 |
| **FunASR** | `spk_model` init | `preset_spk_num` | `int` (0-indexed) | per-sentence | No | Yes | No | undoc. |
| **sherpa-onnx** | `OfflineSpeakerDiarization` | `num_clusters` / `threshold` | `int` (0-indexed) | per-segment | No | Yes | No | undoc. |
| **NVIDIA NeMo** | ClusteringDiarizer / Sortformer | config-based | `int` | per-word (post-hoc) | No | Yes | Sortformer streaming | 4 (Sortformer) |

**Total: 19/38 entries support diarization** (17 cloud/local + 2 mlx-audio
model-level).

### 2.2 Engines without diarization support

| Engine | Notes |
|---|---|
| **Groq** | OpenAI-compatible subset; no speaker params or fields. Groq's own demo uses client-side pyannote. |
| **Fish Audio** | Simple REST ASR; explicitly listed as limitation. |
| **OpenAI Whisper (local)** | No native support. Standard workaround: pyannote post-hoc merge (WhisperX). |
| **faster-whisper** | No native support. Same pyannote workaround; WhisperX uses faster-whisper internally. |
| **Nemotron ASR** | TDT architecture; token + duration heads only, no speaker head. Separate `diar_sortformer_4spk-v1` exists. |
| **SenseVoice** | 4 prefix tokens (lang/emotion/event/ITN) + text; no speaker token. FunASR diarization requires separate CAM++ model. |
| **Cohere ASR** | Model card explicitly states no diarization support. |
| **Voxtral Mini / Realtime** | Open-weights versions have no diarization. (Mistral's proprietary API may add it server-side.) |
| **Canary** | Tasks: ASR + AST + PnC/timestamps. NVIDIA confirmed no diarization. |
| **Qwen2-Audio** | Evaluated tasks: ASR, translation, emotion, sound. No speaker labels. |
| **GLM-ASR Nano** | Whisper encoder + Llama decoder; outputs plain text only. |
| **Moonshine** | Pure encoder-decoder ASR. Moonshine Voice product uses separate ONNX embedding model — pipeline-level, not model-level. |
| **MMS** | wav2vec 2.0 + CTC; flat text output. |
| **FireRedASR2** | 4 modules: ASR + VAD + LID + punctuation. No diarization. |
| **Fun-ASR Nano** | Paraformer text tokens only. Research variant SA-Paraformer exists but is not standard Paraformer. |
| **Qwen3-ASR (local)** | AuT encoder + Qwen3 LLM decoder; output is language tag + text only. No speaker head, no speaker tokens. Timestamps require separate forced-alignment model. |
| **Qwen3-ASR (mlx-audio)** | Same model via std-mlx-audio; adapter declares no diarization capability. |
| **Whisper (mlx-audio)** | Same Whisper architecture as OpenAI Whisper local (#16); std-mlx-audio adapter declares no diarization capability. (Covers mlx-audio presets #24.) |
| **Parakeet TDT (mlx-audio)** | TDT decoder outputs token + duration only; diarization requires separate NeMo MSDD/Sortformer pipeline (not part of the model). (Covers mlx-audio preset #25.) |

### 2.3 mlx-audio model-level diarization (special cases)

Two mlx-audio families have model-native diarization, but the `std-mlx-audio`
adapter layer does **not** expose it today:

| Model | Native capability | Adapter status |
|---|---|---|
| **VibeVoice ASR** (Microsoft) | Joint end-to-end ASR + diarization + timestamping. Qwen2-based decoder autoregressively generates JSON with `Speaker`, `Start`, `End`, `Content` fields. True joint model, not a pipeline. | `GenericSttBackend` with `text_from_segments=True` parses structured JSON segments but does NOT map `Speaker` → `Segment.speaker`. This is the natural landing point for diarization support. |
| **Granite Speech 4.1-2b-plus** (IBM) | Inline `[Speaker N]:` tags in decoded text (ICASSP 2026). Only the `4.1-2b-plus` variant; other Granite variants do not support it. | Not currently an mlx-audio preset. Would need `text_from_segments=True` + speaker tag parsing. |

### 2.4 Universality assessment (updated)

| Dimension | Coverage | Verdict |
|---|---|---|
| Enable toggle (bool/presence) | **17/17** diarizing engines | **Universal — standardize** |
| Speaker count hint (num/max) | **13/17** (all except OpenAI, Deepgram, whisper.cpp, 火山) | **Widespread — standardize** |
| Speaker labels (any format) | **17/17** | **Universal — standardize** (normalize format) |
| Named/known speakers | **4/17** (OpenAI, ElevenLabs, Speechmatics, 騰訊 mode 3) | **Engine-specific — `provider_params`** |
| Speaker roles | **2/17** (ElevenLabs, Google medical) | **Engine-specific — `provider_params`** |
| Speaker confidence | **4/17** (Deepgram, AssemblyAI, Rev.ai, Gladia) | **Engine-specific — `provider_params` for v1** |
| Batch diarization | **17/17** | **Universal** |
| Streaming diarization | **8/17** (Google, AWS, AssemblyAI, Speechmatics, Deepgram, OpenAI SSE, Azure, NeMo Sortformer) | **Widespread — standardize capability** |
| No diarization at all | **19/38** (50%) | These engines declare `diarization.supported=False` |

### 2.5 Attribution granularity split

- **Per-word** (8): ElevenLabs, Google, AWS, Deepgram, AssemblyAI,
  Speechmatics, NeMo (post-hoc), Granite (inline `[Speaker N]:` tags) — each
  word carries a speaker label.
- **Per-segment/utterance** (7): OpenAI, Azure, Gladia, 阿里雲, 火山, 騰訊,
  VibeVoice — speaker assigned at segment/utterance level.
- **Per-sentence** (2): FunASR, sherpa-onnx.
- **Per-monologue** (1): Rev.ai.
- **Turn boundary only** (1): whisper.cpp `tinydiarize`.

The 8:7 word:segment split (19 diarizing entries total across all rows)
reinforces the need for both levels.
`Segment.speaker` (authoritative) + `Word.speaker` (optional refinement)
handles this.

### 2.6 Label format divergence

Every engine uses a different scheme: `"A"/"B"`, `"spk_0"`, `"S1"`,
`"speaker_0"`, bare integers `0`/`1`, `"Guest-1"`, 1-indexed strings.
The standard normalizes to `str | None` — adapters convert native labels.
No prescribed format (engines may use names when known-speaker features are
active).

**Normative requirement**: labels MUST be consistent within a single result
(same string = same speaker, different speakers = different strings). Labels
are NOT stable across sessions, NOT identity-linked, NOT comparable across
engines.

**Consistency scope (Round-4):** "within a single result" spans ALL label
carriers — top-level `segments[]`, `words[]`, AND the per-channel
`channels[]` sub-results. Telephony engines that diarize each channel
independently (both channels labelling their first speaker `"0"`) MUST be
re-labelled by the adapter into one result-wide namespace before assembly,
or the same string would silently denote two different people.

---

## 3. Design decisions

### D1. Request path: `diarization: DiarizationRequest | None` on RuntimeParams

**Chosen**: Option B (richer object). Presence = enable; `None` = not
requested.

```python
class DiarizationRequest(BaseModel):
    """Portable diarization request parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    num_speakers: int | None = Field(default=None, ge=2)
```

**Rationale** (vs alternatives):

| Option | Pros | Cons |
|---|---|---|
| A: `diarize: bool \| None` | Simplest | Cannot carry `num_speakers`; bool cannot evolve without breaking |
| **B: `DiarizationRequest \| None`** | **Forward-compatible; presence = enable; carries portable knobs** | **Moderate complexity (one new model)** |
| C: Defer to `provider_params` | Zero spec work | Breaks "one interface, any engine" for a universal feature |

Option B matches the existing pattern: `word_timestamps` uses a typed value
(granularity enum) where presence = enable. `DiarizationRequest()` (bare) =
"enable with defaults"; `DiarizationRequest(num_speakers=4)` = "enable with
hint".

**v1 field: `num_speakers: int | None`** — a best-effort hint, `ge=2` (review
S3: requesting diarization with 1 expected speaker is semantically
meaningless — diarization exists to distinguish multiple speakers). The engine
SHOULD use it to improve accuracy but MUST NOT fail if the actual count
differs. Gated by `DiarizationConstraints.accepts_speaker_count_hint` (bool).

**Convenience constant (review #5):** Add `DIARIZE = DiarizationRequest()` at
module level, paralleling `AUTO = "auto"` for language. Common case reads as
`RuntimeParams(diarization=DIARIZE)` instead of the more verbose
`RuntimeParams(diarization=DiarizationRequest())`.

**Null semantics (review G3)**: `DiarizationRequest` has **no `[]`-analogue**
unlike `phrase_hints`. `None` = not requested; any `DiarizationRequest`
instance = diarization requested. `DiarizationRequest()` (bare, all defaults)
= "enable with defaults." There is no "requested-but-empty" state.

**Deferred to `provider_params`** (engine-specific, not portable):
- `known_speaker_names: list[str]` (OpenAI only)
- `known_speaker_references` (OpenAI only)
- `detect_speaker_roles: bool` (ElevenLabs only)
- `diarization_threshold: float` (ElevenLabs only)
- `min_speakers: int` (some engines — can graduate later if widespread)
- `speaker_sensitivity: float` (Speechmatics only)

### D2. Capability model

#### D2.1 Batch — hint capability (REVERTED to bool after Round-3 verification)

`accepts_speaker_count_hint: bool` lives on `DiarizationConstraints`;
`DiarizationCap` is unchanged from the shipped shape (`supported` +
`constraints`):

```python
class DiarizationConstraints(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    max_speakers: int | None = Field(default=None, gt=0)
    accepts_speaker_count_hint: bool = Field(default=False)  # NEW
```

- `max_speakers`: upper bound on speakers the engine can produce in results.
  `None` = no declared limit. Constrains the request hint value only when
  `accepts_speaker_count_hint=True`; otherwise it is a result-capacity figure.
- `accepts_speaker_count_hint`: whether the engine honors
  `DiarizationRequest.num_speakers`.

**Why bool-on-constraints, NOT a `FlagCap` sub-node (Round-2 consensus was
verified WRONG in Round 3).** Round 2's five reviewers recommended replacing the
bool with a `speaker_count_hint: FlagCap` sub-node on `DiarizationCap`, claiming
"zero new infrastructure / consistent with `mutable_mid_stream` / uses
`covers()` for free." **Round 3 empirically disproved all three claims** by
running the proposed topology against the real `capabilities.py` machinery:

1. **It breaks the two-layer isomorphism (M.1.2 / spec §C R6, CRITICAL).** A
   `FlagCap` child under a `_FlagLikeNode` parent that carries its own
   `supported` gate desyncs across surfaces. **Verified** on the real full tree
   for the state `(diarization.supported=False, speaker_count_hint.supported=
   True)`: `canonical_json()` → `supported: true`, `supports(...)` → `True`, but
   `iter_supported_paths()` → **absent**. Three surfaces disagree — the exact bug
   class the codebase fixed twice (`_coerce_streaming_guidance`, the dict
   fail-closed branch) and guards with `test_mutable_mid_stream_survives_json_
   round_trip`.
2. **It constructs a contradiction with no validator.** `DiarizationCap(
   supported=False, speaker_count_hint=FlagCap(supported=True))` builds without
   error (**verified**), and `covers()` passes a contradictory effective
   narrowing silently (**verified** `covers()` → `True`). Closing it needs a NEW
   `@model_validator` — contradicting the "zero new infrastructure" claim.
3. **The `mutable_mid_stream` precedent does NOT apply.** `mutable_mid_stream`'s
   parent (`StreamingGuidanceCaps`) is a `_Container` (no `supported` gate);
   `DiarizationCap` is a `_FlagLikeNode` (**verified**). There is **zero
   precedent** in the entire tree for a `FlagCap` capability node under a
   `_FlagLikeNode` parent.

The bool on `constraints` is **inert on the path layer** (a plain `BaseModel`
field, never a capability node) — **verified** to produce no desync surface.
Its one real cost: `covers()` does not catch an `accepts_speaker_count_hint`
`False→True` widening (**verified** — Round-1's gap is real). Close it with a
contained `_node_narrows()` bool branch (analogous to the existing
`granularities` branch — no registry needed, the `_gate_granularity` pattern
proves a per-field check needs no `_BOOL_CONSTRAINT_FIELDS` registry). This is
far cheaper than the FlagCap version's validator + isomorphism break.

> **Open decision (surfaced to maintainer):** an even simpler path is to
> **defer `num_speakers` to v1.1 / `provider_params`** (like `min_speakers`
> already is), making `DiarizationRequest` an empty marker in v1. This deletes
> the hint capability, the gating `model_copy`, and the `_node_narrows` bool
> branch entirely. Trade-off: 13/17 engines carry `num_speakers` in
> `provider_params` interim — and (R4-A) `provider_params` is **not
> wire-constructible** (`WireRuntimeParams` rejects it by design,
> discover-only), so HTTP/WS cross-language clients have NO count-hint
> channel at all until graduation. See §7 decisions.

Gating logic (reads `node_at("<mode>.diarization").constraints`, the
`_gate_granularity` pattern):
1. If `diarization is not None` and `capabilities.<mode>.diarization.supported`
   is False → strict raises `UnsupportedFeatureError`; best_effort drops +
   diagnostic.
2. If `diarization.num_speakers is not None` and
   `constraints.accepts_speaker_count_hint` is False → strict raises;
   best_effort drops `num_speakers` (sets to None) + diagnostic.
3. If `accepts_speaker_count_hint` and `constraints.max_speakers` is set and
   `diarization.num_speakers > max_speakers` → strict raises; best_effort
   clamps + diagnostic. **Corner (verified F3):** `max_speakers` is `gt=0`, so
   `max_speakers=1` is declarable while `num_speakers` is `ge=2`; clamping a
   hint down to `max_speakers=1` would produce an invalid `num_speakers=1`, and
   `model_copy` does NOT re-run validators. The clamp MUST floor at 2 (or drop +
   diagnostic when `max_speakers < 2`).

**`on_unsupported` does NOT apply to diarization** (review #4). The
`on_unsupported` field governs only the `guidance` family's degradation path
(phrase_hints → prompt). Diarization has no degradation target — it is either
supported or not. This should be explicitly documented.

**Gating sub-object mutation (review E2).** `DiarizationRequest` is `frozen=
True`, so the gating function cannot set `num_speakers = None` in-place.
Unlike existing gated parameters (all scalars or lists that can be set directly
to `None` via `updates[field_name] = None`), gating `num_speakers` requires
constructing a new `DiarizationRequest` via `model_copy(update=
{"num_speakers": None})` and storing it as `updates["diarization"]`. This is a
**novel gating pattern** — the first nested-model portable field (other than
`provider_params`). The `_gate_diarization()` function must handle this
explicitly. When the entire diarization feature is unsupported (step 1), the
outer gate sets `updates["diarization"] = None` (drops the whole request) —
this follows the existing scalar pattern.

**DiarizationRequest field drift guard (review #4).** Add an assertion (or
test) that every field on `DiarizationRequest` is covered by
`_gate_diarization`, parallel to the `_GATED_PARAMS` drift guard. Without
this, a future `min_speakers` field could silently bypass gating.

#### D2.2 Streaming (new)

Add `diarization: DiarizationCap` to `StreamingCapabilities`:

```python
class StreamingCapabilities(_Container):
    # ... existing fields ...
    diarization: DiarizationCap = Field(default_factory=DiarizationCap)
```

Defaults to `supported=False` (fail-closed). 8/17 surveyed engines support
streaming diarization — this is not premature.

No new streaming-specific diarization capabilities needed:
- Speaker stability in partials follows the existing `partial`-is-tentative /
  `final`-is-committed contract. No separate flag.
- Speaker re-assignment through re-segmentation uses the existing `supersede`
  mechanism.
- The `word_stability` flag already communicates whether partial content is
  meaningful; speaker assignment follows the same stability contract.

### D3. Result model: shape sufficient, but VALIDATORS are required

> **CORRECTION (Round-3 verified F2):** the earlier "no changes needed" was
> WRONG. The result *shape* (TR.5 reserve) is sufficient, but `Segment.speaker`
> and `Word.speaker` are bare `str | None` with **no validator** — `Segment(
> speaker="")` and `speaker="   "` both construct (verified), producing silent
> bad data. `""` is a third, undefined state (neither `None` "not determined"
> nor a real label) and breaks the design's own "labels consistent" requirement
> and any `"" ∈ label` check. `phrase_hints` already has exactly this validator
> (`contract/params.py:330-364`) for the identical reason. **Add a
> `field_validator` rejecting empty, whitespace-only, AND whitespace-padded
> `speaker`** (R4-F tightening: `"A "` and `"A"` are two different strings =
> two different speakers under the consistency rule, so one adapter
> off-by-one silently breaks within-result consistency; reject rather than
> normalize, matching the phrase_hints stance) on `Segment`,
> `Word`, AND `TranscriptionEvent.speaker` (symmetric — else a malformed
> `event.speaker=""` passes event validation then crashes in the reducer's
> `Segment(speaker="")`, the deferred-crash pattern the codebase already
> designs against). `None` stays valid.

The existing shape (TR.5 reserve) is sufficient:

- **`Segment.speaker: str | None`** — authoritative segment-level speaker.
  When diarization is active, segments SHOULD be split at speaker boundaries.
- **`Word.speaker: str | None`** — optional word-level refinement.

**Speaker inheritance rule** (new normative, review #3): when `Word.speaker`
is `None`, the speaker is inherited from the enclosing `Segment.speaker`.
Making this normative (spec-level, not app guesswork) is actually *more*
explicit than leaving it undefined — every consumer knows the rule. When
`Word.speaker` is populated, it overrides the segment-level speaker (mixed-
speaker segment where the engine provides word-level detail).

**`Segment.speaker = None` semantics when diarization active (review #3):**
- Diarization NOT requested → `Segment.speaker = None` (not applicable)
- Diarization requested + active → `Segment.speaker = None` means "engine
  could not determine speaker for this segment" (explicit "I don't know",
  not "I didn't try")

This distinction matters for apps that need to differentiate "diarization not
requested" from "diarization failed on this segment."

**Segment-speaker synthesis: STANDARD-LAYER step with ONE pinned rule
(R4-D correction — was: per-adapter obligation, "majority/first").** Two fixes
to the Round-2 formulation:

1. "Majority/first" is two rules, not one — different adapters picking
   different rules would make the same engine + audio produce a different
   `Segment.speaker` across adapters, defeating the portability the feature
   exists for. The rule is pinned: **majority by word count; tie broken by
   the earliest word's speaker; `None`-speaker words do not vote.**
2. The synthesis moves OUT of per-adapter obligations INTO the standard
   layer. The streaming side already lives in the reducer (D4); the batch
   side belongs in the `EngineBase.transcribe` template's result
   post-processing (the template already wraps `_transcribe` and attaches
   diagnostics — `runtime/interface.py:388`), same philosophy as the renderers'
   defensive re-sort: implemented once, consistent everywhere, zero
   per-adapter cost. Adapters MAY pre-populate `Segment.speaker` themselves
   (native segment-level engines do); the standard layer synthesizes only
   when `Segment.speaker is None` and the segment's `words` carry speakers.
   Word-only engines (ElevenLabs, Deepgram, AssemblyAI) then get the
   authoritative `Segment.speaker` populated for free.

**No top-level `speakers[]` roster** (YAGNI — but the prior justification was
WRONG). The earlier claim that `detected_speaker_count` is "trivially derivable
from segments via `len(set(seg.speaker ...))`" is **false (Round-3 verified
F4)**: that formula returns 0 for a word-only-speaker segment whose true count
is 2 (speakers reachable only through `Word.speaker` or `channels[]` are
invisible to a segment-only scan). The roster deferral still stands on pure
YAGNI grounds (additive-safe, no current consumer needs it), but do **not** ship
the cited one-liner as a derivation — a correct count must scan segments + their
words + channels.

**Channel vs speaker** (orthogonal): `Segment.channel` is physical provenance
(which microphone); `Segment.speaker` is logical identity (who is talking).
Both can be present simultaneously. No model change needed, but the spec
SHOULD add a clarifying note. Engine-specific mutual exclusivity (ElevenLabs,
AWS) is handled via `effective_capabilities` narrowing, not standard-level
enforcement.

### D4. Streaming events: add `speaker` to `TranscriptionEvent`

```python
class TranscriptionEvent(BaseModel):
    # ... existing fields ...
    speaker: str | None = None
```

**Rationale**: `Segment.speaker` is spec-authoritative (TR.5), but the event
currently has no way to express segment-level speaker without embedding it in
every word. Many engines assign speaker per-segment, not per-word — forcing
adapters to synthesize per-word labels is wasteful and error-prone.

**`TranscriptionEvent.speaker` follows the same inheritance rule as
`Segment.speaker`** (review #3): authoritative at the segment level,
overridable at the word level by `event.words[i].speaker`.

**`StreamReducer.add()` must propagate `event.speaker`** into the reduced
`Segment` (constructor at `streaming.py:538-543`, the `Segment(...)` opener is
line 538; `speaker=` lands beside `words=event.words` at line 542). This is a
latent bug that silently erases speaker data once any engine populates it.

> **CORRECTION (Round-3 verified F1):** the earlier note "also propagate
> `channel`" was WRONG. `TranscriptionEvent` has **no `channel` field**
> (verified — its fields are `segment_id, text, stable_until, finality, words,
> start, end, audio_processed_until, old_ids, new_ids, code, recoverable,
> retriable_after, reconnect, gap_start, gap_end, detected_language, extra`,
> with `extra="forbid"`). `Segment(channel=event.channel)` would raise
> `AttributeError`. Channel-in-streaming is a separate future feature: it
> requires FIRST adding `channel: int | None` to `TranscriptionEvent`. Do NOT
> bundle it into the speaker fix.

**Reducer MUST synthesize `Segment.speaker` from words when `event.speaker is
None`** (Round-3 streaming finding), using the SAME pinned rule as the batch
standard-layer synthesis (D3 / R4-D: majority by word count, tie → earliest
word's speaker). The batch-side synthesis is bypassed by the streaming
reducer, which builds `Segment` directly from the event. For word-only engines
(ElevenLabs/Deepgram/AssemblyAI) that set `event.speaker=None` while populating
`event.words[].speaker`, a naive `Segment(speaker=event.speaker)` leaves the
authoritative segment speaker `None` over non-null words — inverting the
inheritance rule. The reducer must do the synthesis, not leave it to adapters.

**Speaker stability in partial events**: speaker assignments on `partial`
events are provisional and may change (same as text). Speaker assignments on
`final` events are committed. This follows the existing partial/final contract
naturally — no new mechanism needed.

### D5. Wire protocol

#### WireRuntimeParams

Add `diarization: DiarizationRequest | None = None`. The existing drift
assertion (`WireRuntimeParams.model_fields == RuntimeParams.model_fields -
{"provider_params"}`) enforces this automatically — adding to RuntimeParams
without WireRuntimeParams causes an import-time crash. **Both models MUST be
updated in the same commit** (review S4).

#### HTTP batch — three-way wire mapping (review G1, normative)

```json
POST /transcribe
{
  "language": "en",
  "diarization": { "num_speakers": 3 },
  "word_timestamps": "word"
}
```

| Wire JSON | Python value | Meaning |
|---|---|---|
| `"diarization": {"num_speakers": 3}` | `DiarizationRequest(num_speakers=3)` | Enable with 3-speaker hint |
| `"diarization": {}` | `DiarizationRequest()` | Enable with defaults |
| `"diarization": null` | `None` | Not requested |
| key absent | `None` (field default) | Not requested |

Pydantic handles all four cases correctly: `DiarizationRequest` uses
`extra="forbid"`, so unknown keys inside the nested object (e.g.,
`{"diarization": {"unknown": true}}`) are rejected at wire validation.

#### WebSocket streaming

```json
{ "type": "start", "params": { "diarization": {} } }
```

Events gain `speaker`:

```json
{ "type": "final", "segment_id": "seg-3", "text": "I agree.",
  "speaker": "speaker_1", "start": 12.4, "end": 13.8 }
```

#### Capability advertisement

No wire-format change. `DiarizationCap` and `DiarizationConstraints` already
serialize via `canonical_json()`. The new `accepts_speaker_count_hint` bool
appears inside `constraints` (a plain submodel — no injected `supported`,
verified inert on the path layer):

```json
{
  "batch": {
    "diarization": {
      "supported": true,
      "constraints": { "max_speakers": 32, "accepts_speaker_count_hint": true }
    }
  },
  "streaming": {
    "diarization": {
      "supported": false,
      "constraints": { "max_speakers": null, "accepts_speaker_count_hint": false }
    }
  }
}
```

### D6. SRT/VTT renderers

Speaker labels on diarized results are not emitted by the renderers (they
only emit timestamp + text).

**Decision**: Add `include_speakers: bool = False` parameter to `to_srt()` /
`to_vtt()`:
- Default `False`, justified by **text purity — NOT backward compatibility**
  (R4-F correction: this repo is pre-release and backward compat is
  explicitly not a design driver). SRT has no standard speaker syntax, so
  injecting `[Speaker N]: ` mutates the cue *text* itself and pollutes any
  downstream text processing; and a renderer is a projection of the result,
  not the result — the caller still holds the full data, so `False` is not
  silent loss. A VTT-only `True` default (native, display-safe `<v>` markup)
  was considered and rejected: asymmetric defaults across the two renderers
  surprise more than they help.
- When `True` and `segment.speaker is not None`:
  - SRT: prefix each cue with `[Speaker N]: `.
  - VTT: use `<v Speaker N>` voice tags (W3C WebVTT spec).

This is explicit opt-in, consistent with the "never silently degrade"
philosophy.

**Sanitization ordering (review G6).** The `<v>` voice tag in VTT is cue-span
markup that must be injected **after** the existing `_sanitize_cue_text()`
escapes `<`/`>` in segment text — otherwise the voice tag itself gets escaped.
Implementation order: (1) sanitize `segment.text`, (2) wrap in `<v label>`
tag. The **speaker label itself** must also be sanitized for the VTT voice-tag
context (the `<v>` tag's annotation text has its own constraints per W3C spec
— notably, it must not contain `>`).

---

## 4. Risk analysis

### P0 — Must resolve before implementation

| Risk | Description | Mitigation |
|---|---|---|
| **StreamReducer erases speaker** | `StreamReducer.add()` constructs `Segment(...)` without propagating `event.speaker` — silently loses all speaker data on reduction | Fix: propagate `event.speaker` into `Segment(speaker=event.speaker)` |
| **TranscriptionEvent has no segment-level speaker** | Only carried via `Word.speaker`; many engines assign per-segment | Fix: add `speaker: str \| None = None` to `TranscriptionEvent` |
| **No request path** | `DiarizationCap` exists but cannot be activated portably | Fix: add `diarization: DiarizationRequest \| None` to `RuntimeParams` |
| **No gating** | No entry in `_GATED_PARAMS`, no constraint sub-checker | Fix: add gating entry + `_gate_diarization()` |

### P1 — Should resolve alongside

| Risk | Description | Mitigation |
|---|---|---|
| **Speaker label portability** | Engines use incompatible label formats | Adapters normalize to `str`; normative requirement for within-result consistency |
| **`num_speakers` hint semantics** | Engines treat count as exact / range / hint | Define as best-effort hint; validation against `max_speakers`; clamp in best_effort |
| **Streaming diarization latency** | Speaker assignment requires context; early partials may have no speaker | Document: partial speaker is provisional; final is committed |
| **SRT/VTT speaker omission** | Renderers do not emit `.speaker` | Add `include_speakers` opt-in parameter (default `False` on text-purity grounds, D6) |
| **Diarization + word_timestamps interaction** | App cannot predict whether word-level speakers are available | Normative population rule (open Q1 interaction note) defines it; granularity capability deferred (R4-F, §5.3 Q1) |

### P2 — Spec clarification

| Risk | Description | Mitigation |
|---|---|---|
| **Channel vs speaker confusion** | Two orthogonal axes on same model, no guidance | Add spec note clarifying physical (channel) vs logical (speaker) |
| **Speaker stability contract in lifecycle guard** | Guard enforces text stability but not speaker stability | Document: speaker on partial is provisional; only text has frozen-prefix guarantee |

### P3 — Correctly deferred

| Risk | Description | Mitigation |
|---|---|---|
| **Known speakers / identification** | Only 4/17 engines; fundamentally different from blind diarization | Keep in `provider_params`; graduate if 2+ engines converge |
| **Top-level `speakers[]` roster** | YAGNI confirmed; additive-safe | Defer; add when known-speaker-names support is needed for 2+ engines |

---

## 5. Implementation plan

### 5.1 Files affected

| Change | File(s) | Complexity |
|---|---|---|
| `DiarizationRequest` model + `DIARIZE` constant | `contract/params.py` | Low |
| `diarization` on `RuntimeParams` + `WireRuntimeParams` | `contract/params.py` | Low (drift guard; MUST update both in same commit) |
| `accepts_speaker_count_hint: bool` on `DiarizationConstraints` | `contract/capabilities.py` | Low (bool on constraints + `_node_narrows` bool branch) |
| `diarization` on `StreamingCapabilities` | `contract/capabilities.py` | Low |
| `speaker` on `TranscriptionEvent` | `runtime/streaming.py` | Low |
| `StreamReducer` speaker propagation (line ~538) | `runtime/streaming.py` | Low |
| Gating entry + `_gate_diarization()` (novel `model_copy` pattern) | `runtime/gating.py` | Moderate |
| `include_speakers` on renderers + sanitization ordering | `renderers.py` | Moderate |
| Wire model update | `audio/wire.py` (if separate) or `toolchain/server.py` | Low |
| `__init__.py` re-exports | `__init__.py` | Trivial |
| Batch segment-speaker synthesis (pinned rule, R4-D) | `runtime/interface.py` (`EngineBase.transcribe` post-processing) | Low–Moderate |
| Tests | `tests/test_*.py` | Moderate–High |
| Spec update (§Capabilities tree + §Runtime Params + TR.5 + §Streaming + TR.1/TR.3 always-on exemption) | `docs/content/specification/protocol.md` | High (normative text) |
| Compliance tests (incl. batch-only vs streaming-only diarization; batch + streaming `*_exceeds_diarization` cross-checks) | `compliance.py` | Moderate |

### 5.2 Sequencing

1. **Spec update** — Add diarization section to specification.md (normative
   text for request path, gating, result semantics, streaming behavior).
   Update §Capabilities tree (line ~348) to add `diarization` under
   `streaming` (review G7). Add speaker inheritance rule to TR.5 (review S1).
2. **Core models** — `DiarizationRequest` + `DIARIZE` constant, add
   `accepts_speaker_count_hint: bool` to `DiarizationConstraints`, add
   `diarization` to `StreamingCapabilities`, add `speaker` to
   `TranscriptionEvent`. **Simultaneously** update `RuntimeParams` +
   `WireRuntimeParams` (same commit — drift guard enforces).
3. **Gating** — `_gate_diarization()` in `runtime/gating.py`. Uses
   `node_at("<mode>.diarization").constraints.accepts_speaker_count_hint` for hint acceptance (the `_gate_granularity` pattern). Uses
   `DiarizationRequest.model_copy(update={"num_speakers": None})` for sub-
   object mutation. Add `DiarizationRequest` field drift guard. Position in
   `_GATED_PARAMS`: after existing entries (no ordering dependency).
4. **Standard-layer synthesis** — batch segment-speaker synthesis in
   `EngineBase.transcribe` post-processing + the reducer-side synthesis,
   both using the single pinned rule (R4-D).
5. **StreamReducer fix** — propagate `event.speaker` into reduced `Segment`
   at `streaming.py:~538`.
6. **Renderers** — `include_speakers` parameter. Voice tag injection AFTER
   text sanitization (review G6).
7. **Compliance** — diarization compliance tests. Include explicit test for
   batch-supported + streaming-unsupported engines (review G8) and the
   batch + streaming `*_exceeds_diarization` cross-checks (R3-C / R4-E).
8. **Tests** — unit tests for all new paths.

### 5.3 Open questions

1. **Diarization granularity capability**: Should `DiarizationCap` declare
   whether it provides word-level or segment-level speaker attribution? The
   survey shows an 8:7 split (§2.5; the earlier "6:4" figure was stale).
   **Recommendation (REVERSED in Round 4): defer, alongside `num_speakers`.**
   With the v1 `DiarizationRequest` as a pure enable marker an app cannot
   *request* a granularity, so the capability would be purely informational;
   the interaction rule below already defines what is actually populated;
   and adding it needs the `WordTimestampsCap` supported⇒non-empty validator
   while making the node a multi-archetype hybrid (Round-3 lower fork).
   Purely additive later.

   **Interaction with `word_timestamps` (review G4)**: when an app requests
   `word_timestamps="segment"` (segment-level timestamps only), `Segment.words`
   is `None` (TR.3). Even if the engine supports word-level diarization,
   `Word.speaker` is inaccessible because there are no `Word` objects. The
   normative rule should be: if diarization is requested, `Segment.speaker`
   MUST always be populated (it is the authoritative shape). `Word.speaker` is
   populated only when `words` is also populated (i.e., `word_timestamps` was
   requested at word level). The diarization granularity capability declares
   what the engine *can produce*, not what it will always populate — the
   actual population depends on the interaction with `word_timestamps`.

2. **`num_speakers` vs `min_speakers` / `max_speakers` request**: Google and
   Gladia accept a range (`min/max`). Should the standard offer a range?
   **Recommendation**: start with `num_speakers` only (hint); add `min/max` if
   multiple engines converge on range semantics. Simpler is better for v1.

3. **Diarization + multi-channel mutual exclusivity**: ElevenLabs and AWS
   make diarization and multi-channel mutually exclusive. Should the standard
   enforce or declare this? **Recommendation**: no standard-level enforcement
   — but the mechanism is a **request-time adapter gate**, NOT
   `effective_capabilities` narrowing (R4-E correction: effective caps are an
   *instance-level*, config-time narrowing, while channel count is a
   *request-level* property arriving with the audio — an instance-level
   mechanism cannot express "this request is stereo, so diarization is
   unavailable"). The adapter MUST fail the conflicting request itself:
   strict raises `UnsupportedFeatureError`; best_effort drops diarization
   with a diagnostic. The standard should not bake in engine-specific
   constraints.

4. **Speaker label normative format**: Should the standard prescribe a format
   like `"speaker_0"`, `"speaker_1"`? **Recommendation**: no — labels are
   `str`, adapters convert native format to strings, engines using
   known-speaker features may return human-readable names. Only require
   within-result consistency.

5. **`num_speakers` naming (review S2)**: The design uses `num_speakers`
   (matching ElevenLabs, FunASR). Alternatives: `speaker_count_hint` (clearer
   hint semantics), `expected_speakers` (matching AssemblyAI). The name should
   avoid confusion with `DiarizationConstraints.max_speakers` (a constraint,
   not a request). **Recommendation**: `num_speakers` is fine — it's the most
   common name across engines, and the hint semantics are documented. The
   `constraints.max_speakers` vs `request.num_speakers` distinction is clear
   from context (constraints live on the engine, request comes from the app).

### 5.4 Non-goals (recorded so they are deliberate, not overlooked — R4-F)

- **Post-hoc diarization pipelines** (pyannote / WhisperX style: any ASR +
  a separate diarizer model, merged by timestamps). This is the real-world
  workaround for the 19/38 engines with no native support (Groq's own demo
  uses client-side pyannote; whisper.cpp users reach for WhisperX), and
  standardizing it would give *every* engine diarization — but it is a
  second protocol surface (a diarizer plugin type + a standard merge step),
  a major scope expansion. v1 deliberately does not attempt it. Nothing in
  this design blocks it: the result shape (timestamped `words` +
  `Segment.speaker`) is exactly the merge target a future
  `standard-diarization` companion package would write into.
- **`speaker_on_partials` capability** (does the engine attach speakers to
  partial events, or only at final?). Would let a voice assistant know in
  advance whether early routing is possible — `word_stability` is the
  precedent for declaring a stability signal. Deferred until a real
  streaming-diarization adapter needs it; additive.
- **A standard "suppress diarization" switch** for always-on engines
  (privacy-motivated apps that want NO speaker labels). No standard
  off-switch exists in v1; such apps strip the fields client-side. Recorded
  as a consequence of §7 Decision 2.

---

## 6. Review log

### Round 1: Adversarial review (single agent)

Errors E1–E3, gaps G1–G8, suggestions S1–S5 from the initial adversarial
review. All incorporated into sections D1–D6 and 5.1–5.2.

### Round 2: Five independent deep reviewers

Five agents independently reviewed the design against the spec, codebase,
mission, goals, and research reports. Each read all relevant source files.

| Reviewer | Score | Verdict |
|---|---|---|
| #1 Request path | High confidence | No changes. `DiarizationRequest \| None` is optimal |
| #2 Capability model | 75-80% | Sound. `max_speakers` reframed as result-capacity constraint |
| #3 Result & streaming | High confidence | Near-optimal. Added speaker-None semantics, adapter obligation |
| #4 Gating & wire | 90-98% | Ready. Added DiarizationRequest drift guard, VTT sanitization spec |
| #5 Mission & alternatives | 8/10 mission, 7→9/10 quality | **Key: replace bool constraint with FlagCap sub-node** |

**Cross-reviewer consensus (applied to design):**

1. ~~**`accepts_speaker_count_hint` → `speaker_count_hint: FlagCap` sub-node**~~
   **REVERTED by Round 3 (empirically disproved).** The FlagCap topology breaks
   two-layer isomorphism; "zero new patterns" was false. Back to bool on
   constraints. See §6 Round 3 and D2.1.
2. **`max_speakers` reframed** as result-capacity constraint (reviewer #2).
   Round 3 (F7) notes this reframing pulls it toward a Property under R7 —
   re-justify placement; tension is open.
3. **`Segment.speaker = None` when diarization active** = "engine could not
   determine speaker" (reviewer #3). Spec-level normative statement.
4. **Adapter SHOULD synthesize `Segment.speaker`** from word-level speakers
   for word-only engines (reviewer #3). Round 3: the **streaming reducer** must
   also do this (the obligation is bypassed on the streaming path). Round 4
   (R4-D): superseded — synthesis relocated to the standard layer with ONE
   pinned rule; see D3.
5. **`TranscriptionEvent.speaker` inheritance rule** mirrors `Segment.speaker`
   exactly (reviewer #3).
6. ~~**StreamReducer also propagate `channel`**~~ **WRONG — reverted by Round 3
   (verified F1).** `TranscriptionEvent` has no `channel` field; the line would
   not compile. Channel-in-streaming needs the event field added first; not
   part of this work.
7. **DiarizationRequest field drift guard** needed (reviewer #4).
8. **`on_unsupported` does not apply to diarization** — explicitly document
   (reviewer #4).
9. **VTT voice-label sanitization**: strip/replace `>` and newlines from
   speaker label (reviewer #4).
10. **`DIARIZE = DiarizationRequest()` convenience constant** (reviewer #5).

### Round 3: Five STRICT reviewers + empirical verification

Round 3 targeted what Round 2 missed: the just-applied FlagCap mutation (which
nobody had reviewed), the compliance suite's actual probe mechanics, the real
adapter code, and adversarial streaming. **Every empirically-checkable finding
was then verified by running it against the real code** (capabilities machinery,
`results.py`, `streaming.py`, std-mlx-audio backends). Verification confirmed
all of them; it also caught that Round 2 had written two defects into the design
(the FlagCap isomorphism break and the `channel=event.channel` compile error).

**R3-A — FlagCap mutation is verified WRONG (reverted in D2.1).** Two-layer
desync confirmed on the real tree (`canonical_json`=true, `supports()`=true,
`iter_supported_paths()`=absent for `parent=False,hint=True`); contradiction
constructs without a validator; `covers()` passes it silently; the
`mutable_mid_stream` precedent doesn't apply (`_Container` vs `_FlagLikeNode`).
→ Reverted to bool on constraints + a contained `_node_narrows` bool branch.

**R3-B — Always-on diarization conflict (HIGH, NEW, verified).** VibeVoice
(joint model, best DER in survey 3.42%), Rev.ai (default-on), 火山 (default
true) **cannot not diarize**, contradicting the "diarization=None → speaker=
None" contract. Verified in shipped code: all 5 `ModelBackend.to_result`
signatures lack a diarization-requested flag, and the VibeVoice adapter parses
the Speaker JSON for text only and discards it. The adapter physically cannot
implement "strip when not requested." → Needs a `DiarizationCap.always_on`
concept (always-on engines MAY populate `speaker` with `diarization=None`; that
is not a violation) + the requested-flag plumbed into `to_result`. **Blocks the
design's own flagship local diarizer from being compliant.** See §7 decision.

**R3-C — Compliance has no teeth for diarization correctness (HIGH).** Every
existing probe is structural or feeds silent audio that gating rejects pre-
inference; none can observe transcript content. So `diarization.supported=True`
is unverifiable — an adapter returning `speaker=None` forever passes. It is the
first capability with no swap-safety defense. `accepts_speaker_count_hint=True`
is similarly vacuous (best-effort honoring is unverifiable; it only means "won't
raise"). → (a) honest scoping admission in §5.1; (b) add the ONE checkable
invariant — `stream_exceeds_diarization` cross-check (`event.speaker != None`
while `diarization.supported=False`) to `_cross_check_event_capabilities`;
(c) optional opt-in golden-fixture helper for authors; (d) the batch-side
twin (R4-E): any `speaker != None` across `segments[]`/`words[]`/`channels[]`
of a batch result while `batch.diarization.supported=False` → error
(opportunistic — probes feed silent audio, so it will not fire on the
standard probes, but it catches egregious violations for free wherever a
result is observed).

**R3-D — Construction validators required (HIGH, verified F2).** `Segment(
speaker="")` / whitespace construct as silent bad data; design's "no changes
needed" was wrong. Add validators (D3, corrected). The "within-result
consistency" MUST is unenforceable/unverifiable beyond non-emptiness — relabel
the unverifiable half as an adapter obligation (like TR.2 sorting), not a MUST
the standard pretends to enforce.

**R3-E — Streaming silent-corruption risks (speaker ≠ just text-metadata).**
The whole stability/continuity/frozen-prefix machinery is text-only; speaker is
a routing key with stronger semantics. Verified code paths:
- **Backpressure coalescing** (`streaming.py:651` `slot.event = event`): blind
  whole-event replace → a kept partial with `speaker=None` silently drops a
  dropped partial's `speaker="A"`. (Same latent bug already exists for
  `detected_language`.) → carry-forward last-non-null, or mandate adapters
  repeat speaker on every partial.
- **Frozen-word speaker change**: `_frozen_prefix_rewritten` checks text only; a
  voice assistant that acted on "Speaker A: Hello" sees it flip to B with text
  unchanged. → decide whether `stable_until` protects speaker over the frozen
  region (recommend yes; add a guard).
- **Supersede cross-speaker merge**: frozen text preserved but speaker silently
  rewritten; set-to-set lineage can't preserve per-speaker attribution. →
  forbid cross-speaker merges OR require word-level speakers survive + segment
  speaker → None for mixed merges.
- **`closed` event**: may rewrite text (ITN) but MUST NOT re-diarize.
- **Reconnect**: `speaker` absent from the §6.3 MUST-stay-continuous list;
  re-clustering after reconnect breaks within-result consistency silently. →
  add `speaker` to the continuity contract + diagnostic when unmappable.
- **`reduce_event`** (`streaming.py:486`, the documented app-side reduce) is a
  `dict[str,str]` — structurally speaker-blind; spec §5.2's canonical reduce is
  text-only. Document or restructure so the canonical reduce isn't itself a
  silent-loss example.

**R3-F — Other verified concrete defects.**
- TR.2 `(start, channel)` sort is non-deterministic for single-channel
  overlapping speakers (`channel=None`, no tie-break) — needs a `speaker`
  tie-break or an explicit "unspecified order" note.
- `max_speakers` gt=0 vs `num_speakers` ge=2: clamping to `max_speakers=1`
  yields an invalid hint; `model_copy` won't catch it (clamp must floor at 2).
- Exports are **4 surfaces** (`__init__.py`, `runtime_params.__all__`,
  `engine.py` import + `__all__`), not "trivial / one file."
- Granite `[Speaker N]` inline tags: adapter MUST strip from `text` + diagnose
  parse failure (else markup leaks into the transcript — silent contamination).
- "&lt;1 hour adapter effort" (Round-2 #5) is contradicted by real code for
  joint/inline/word-only engines (VibeVoice signature change, Granite parser,
  word→segment synthesis). Re-scope per engine class.
- `provider_formats` renderer interaction is a phantom — it's docstring-only in
  `renderers.py`, no passthrough exists. Drop that open question.
- VTT `<v>` wrap of a multi-line cue body is ambiguous; specify per-line vs
  whole-body.

### Round 4: Independent fourth-party review + code re-verification

Round 4 independently re-verified every load-bearing Round-3 claim against
the shipped code — **all confirmed** (reducer non-propagation at
`streaming.py:538-543`; no `channel` on the event; bare `speaker` fields
without validators; blind coalescing replace at `streaming.py:651`;
text-only frozen-prefix guard; `_Container` vs `_FlagLikeNode` topology;
std-mlx-audio `to_result` signatures and the VibeVoice Speaker discard) —
and audited the design against `docs/content/mission.md` / `docs/content/goals.md`. Verdict:
architecture sound (direct delivery of G.1.4 with zero shape change), all
seven decision *directions* endorsed; the following corrections are
incorporated into this document and `diarization-decisions.md`.

**R4-A — Decision-1 wire blind spot (analysis gap).** Deferring
`num_speakers` to `provider_params` removes it from the wire entirely:
`WireRuntimeParams` rejects `provider_params` by design (discover-only,
`contract/params.py:369-404`), so HTTP/WS cross-language clients have NO
count-hint channel until graduation. This is a systemic property of
`provider_params`, not diarization-specific, but under G.5 ("wire contract
as first-class spec") it MUST be a recorded cost of the defer option, and
wire-client demand becomes one of the graduation triggers. Recommendation
stays defer (the hint is an accuracy nudge, not a function).

**R4-B — `always_on` home correction.** *(Representation superseded by the
pre-merge revision below — `always_on` is now a queryable `FlagCap` node, not a
bare bool; the "path fail-closed / no `supported` injection" claims here no
longer hold. The home correction, validator, and normative obligations stand.)*
`DiarizationConstraints` is the
wrong home: spec §C 3.3 defines `constraints` as *machine-checkable limits*,
and `always_on` is a behavioural fact — its kin is `self_resamples`, which
the spec explicitly flags as the single behavioural capability and houses
with care (§AI 3.2 / §C R7). Corrected home: a **bare `bool` field on the
`DiarizationCap` node itself** — verified inert on the path layer against
the real machinery (`_derive_supported` fail-closes non-node values, so
`supports("….always_on")` is `False`; `_iter_paths` yields no path for it;
`canonical_json` emits it as a plain field) — no Round-3-class isomorphism
hazard. A one-line `@model_validator` rejects the contradiction
`supported=False ∧ always_on=True`. Spec text MUST additionally state:
(1) engines that CAN disable diarization (Rev.ai `skip_diarization`, 火山
`enable_speaker_info=false`) MUST disable it when not requested —
`always_on` is reserved for architecturally non-disableable models
(VibeVoice), never adapter convenience (unverifiable, but the same class of
declaration-honesty norm as elsewhere); (2) always-on population is an
explicit, named exemption to TR.1/TR.3's "unrequested data MUST NOT be
populated" stance (TR.3 forbids exactly this for word timestamps — without
the named exemption the two sections contradict); (3) apps wanting no
speaker labels on an always-on engine strip them client-side (§5.4).

**R4-C — Decision-3 semantics pinned + 3↔4↔5 coupling.** The frozen-speaker
guard is **segment-level ONLY** (`event.speaker`; per-word frozen tracking
would need per-event word alignment — out of v1 scope, stays an adapter
obligation). `None→X` after freezing MUST be legal (it is the
delay-to-final adapter strategy this very design recommends); `X→None` and
`X→Y` are both violations (a retraction is a rewrite). On violation the
guard SUPPRESSES the whole event, not clamps: clamping would keep presenting
a stale attribution — the opposite silent-wrong-result — matching
`_frozen_prefix_rewritten`. The cost of suppression is two-tier: a
suppressed in-flight `partial` is an ephemeral loss (the next partial
reshows the segment), but a suppressed violating plain `final` is NOT —
the reducer commits only finals, so that segment never commits and its
text is permanently absent from `session.result()` (only the
`frozen_speaker_rewritten` warning diagnostic remains). Compliant escape
hatches so the final lands: restate the locked speaker on the final,
delay speaker to the final (`None→X` is legal), or settle the corrected
speaker on the guard-exempt `closed` terminal event. Side effect (a
blindly-forwarding Google/AWS adapter loses consecutive partials, and
loses the whole segment if its final also carries the rewritten speaker)
is the established "only harms non-compliant adapters" stance. Coupling: Decision 4's carry-forward is safe ONLY because
Decision 3 forbids `X→None` (otherwise carry-forward resurrects a
deliberately retracted speaker); Decision 5's enforcement needs the
per-segment last-speaker tracking Decision 3's guard introduces.
**Decisions 3+4+5 ship together or defer together.**

**R4-D — Synthesis rule pinned and moved to the standard layer.**
"Majority/first" was two rules; pinned to ONE (majority by word count, tie →
earliest word's speaker, `None` words don't vote) and relocated from
per-adapter obligation to the standard layer (batch: `EngineBase.transcribe`
post-processing; streaming: the reducer). See D3/D4.

**R4-E — Consistency scope, exclusivity mechanism, batch cross-check.**
Within-result label consistency explicitly spans `segments[]` / `words[]` /
`channels[]` (§2.6 — telephony per-channel diarizers both label their first
speaker `"0"`). ElevenLabs/AWS diarization×multi-channel exclusivity is a
request-time adapter gate, not `effective_capabilities` narrowing (§5.3 Q3
corrected — effective caps are instance-level; channel count is
request-level). Compliance gains the batch-side twin of the stream
cross-check (§R3-C d).

**R4-F — Smaller corrections.** Speaker validator also rejects
whitespace-padded labels (D3). `include_speakers=False` re-justified on
text purity — the "backward compatibility" rationale contradicted the
repo's pre-release philosophy (D6). Granularity-capability recommendation
reversed to defer (§5.3 Q1, aligned with `num_speakers`). Reconnect gets a
mechanical safe default: mint fresh labels after reconnect, never reuse a
pre-reconnect label without identity evidence (§7 lower forks). Non-goals
recorded (§5.4). Survey nits fixed: decisions-doc count 17/38 → 19/38;
stale "6:4" → 8:7; Rev.ai added to §2.5; §5.2 renumbered (missing step 4).

### Pre-merge revision (2026-07-02): `always_on` → queryable `FlagCap` node

An in-PR pre-merge design review found two gaps in the shipped `always_on`
design (the R4-B "bare bool on the node" correction):

1. **The justification was false.** "always_on is a behavioural fact, unlike
   requestable capabilities" does not distinguish it — the tree already holds
   many non-requestable behavioural facts (`self_resamples`, `emits_partials`,
   `re_segments`, `word_stability`, `reconnect`, `finality_level`,
   `timestamps`).
2. **Representation inconsistency.** `self_resamples` — which R4-B itself calls
   always_on's "kin" — is a queryable `FlagCap` node participating in
   `supports()`, `iter_supported_paths()`, `canonical_json()` (injected
   `supported`), and `covers()`; `always_on` was a bare bool excluded from all
   four. Same kind of fact, two representations.

**Resolution.** `always_on` becomes a `FlagCap` child node of `DiarizationCap`,
uniform with `self_resamples`. Now working: `supports("<mode>.diarization.
always_on")` is a valid query path; it appears in `iter_supported_paths()`
when supported; `canonical_json()` injects `supported`; `covers()` rejects a
declared-unsupported → effective-supported widening as declaration drift (the
bare bool had no such guard). The construction-time validator still rejects
`supported=False ∧ always_on supported`. Wire shape:
`{"supported": true, "always_on": {"supported": true}}`; a bare bool now fails
pydantic validation loudly (pre-release, no coercion shim). The only surviving
distinction is documented prose: the **semantic inversion** — for every other
flag `True` = "you may request this", for `always_on` `True` = "this is imposed
on you" (labels may appear unrequested). The old "path is fail-closed by
design" and "canonical JSON does not inject supported" claims are retired.

---

## 7. Open decisions (maintainer's call)

Round 3 surfaced material forks that change scope. These are spec-shaping
("infrastructure for 10 years") and are the maintainer's call, not the
reviewers'.

### Decision 1 — `num_speakers`: ship in v1 (bool-gated) or defer?

| Option | Pros | Cons |
|---|---|---|
| **A: Ship `num_speakers` (bool-on-constraints gate)** | 13/17 engines support it; portable now; matches Round-1/#1 endorsement | Adds the hint capability + gating `model_copy` + `_node_narrows` bool branch + clamp-floor corner; the hint capability is unverifiable (R3-C) |
| **B: Defer to `provider_params` (empty `DiarizationRequest` marker in v1)** | Deletes ALL hint machinery; `DiarizationRequest()` is a pure enable marker; graduates later with field evidence (like `min_speakers` already deferred) | 13 engines carry `num_speakers` in `provider_params` interim (they already need `provider_params` for thresholds/roles anyway); **wire clients lose the hint entirely** — `provider_params` is not wire-constructible (R4-A) |

**Recommendation: lean B.** Round 3 showed the hint machinery is the most
over-engineered part and its capability is vacuous (only means "won't raise").
A clean enable-marker for v1 + graduate `num_speakers` when there is field
evidence is the most defensible. But A is viable if portability-now is
prioritized. **R4-A caveat:** under B, HTTP/WS cross-language clients have NO
count-hint channel at all until graduation (`WireRuntimeParams` rejects
`provider_params` by design); the graduation trigger therefore includes
wire-client demand, not only in-process engine-semantics convergence.

### Decision 2 — Always-on diarizers (BLOCKING for VibeVoice)

The "diarization=None → speaker=None" contract is contradicted by always-on
engines (VibeVoice, Rev.ai, 火山). Options:

| Option | Behavior |
|---|---|
| **A: Add `always_on` as a `FlagCap` child node of `DiarizationCap` (home corrected by R4-B — NOT on `constraints`; representation corrected pre-merge — a queryable node, NOT a bare bool)** | Always-on engines declare it; they MAY populate `speaker` with `diarization=None` and that is NOT a violation. Uniform with `self_resamples`: queryable via `supports()`, appears in `iter_supported_paths()`, `canonical_json()` injects `supported`, and `covers()` rejects declared→effective drift. Requires: one-line validator rejecting `supported=False ∧ always_on supported`; "can-disable ⇒ MUST disable when not requested" normative text; a named TR.1/TR.3 exemption in the spec; documented semantic inversion (True = imposed, not requestable). |
| **B: Require adapters to strip speaker when not requested** | Honors the contract literally, but is wasteful and bug-prone (the strip path is the default); needs a requested-flag plumbed into every adapter `to_result`. |
| **C: Redefine the result contract** | `Segment.speaker != None` no longer implies "diarization was requested"; apps use capability, not field-nullness, to know. Simplest contract, but loses the "field tells you what happened" signal. |

A lighter variant of A was also considered in Round 4: no capability field
at all, just the normative MAY + an `info` diagnostic
(`code="unrequested_speaker_labels"`) whenever unrequested speakers are
populated — uses the existing structured-diagnostics channel instead of a
new declaration. Rejected as the primary mechanism because it loses
*pre-transcription* discoverability, but the diagnostic MAY be emitted in
addition to the flag.

**Recommendation: A** (declare always-on honestly, on the node per R4-B)
**+ C's clarification** (apps query capability, not field-nullness — already
the TR.1 stance). This unblocks VibeVoice — the design's best local diarizer
— which today cannot be compliant.

### Decision 3 — Does `stable_until` protect `speaker` over the frozen region?

| Option | Behavior |
|---|---|
| **A: Yes — add a frozen-speaker guard** | Once a segment has frozen text, its speaker MUST NOT change (except `closed`). Rigorous; protects voice-assistant routing. New guard alongside `_frozen_prefix_rewritten`. |
| **B: No — document that `stable_until` covers text only** | Speaker may float on partials; §7.2 voice-assistant guidance MUST warn never to key irreversible action on a partial's speaker. Cheaper; weaker guarantee. |

**Recommendation: A** for correctness ("correctness wins over DX"), given
speaker is a routing key, not display polish. **With R4-C's pinned
semantics:** guard is segment-level only (`event.speaker`); `None→X` after
freezing is legal (delay-to-final strategy); `X→None` and `X→Y` are
violations; violation SUPPRESSES the whole event (not clamps — a clamp
presents stale attribution, the opposite silent wrong result). Ships as a
package with Decisions 4 and 5 (see R4-C coupling).

### Lower forks (recommendations, not blocking)

- **Supersede cross-speaker merge:** forbid cross-speaker merges.
  Enforcement (R4-C): the normative MUST NOT + a compliance test are
  primary; runtime guard suppression is secondary and its side effect MUST
  be documented — a suppressed supersede leaves the old segments alive in
  the reducer while the new segments later arrive as fresh opens, so the
  reduced result carries duplicated text (the established "only harms
  non-compliant adapters" stance for illegal supersedes). Needs Decision 3's
  per-segment speaker tracking — implement together.
- **Reconnect:** add `speaker` to the §6.3 continuity contract, with a
  mechanical safe default (R4-F): after a reconnect the adapter MUST NOT
  reuse a pre-reconnect label for a new cluster unless it has identity
  evidence; otherwise it MUST mint fresh labels (`speaker_2`, `speaker_3`,
  …) and emit a diagnostic. Blind-clustering engines cannot re-map as the
  *common* case, not the exception — over-counting speakers is the safe
  direction; silently merging different people under one label is not.
- **TR.2 sort:** add `speaker` tie-break after `(start, channel)`.
- **Granularity capability (old open Q1):** **defer** (Round-4 reversal,
  aligned with `num_speakers` — §5.3 Q1 updated to match): with a pure
  enable-marker request an app cannot request a granularity, so the node
  would be purely informational; the population rule already defines the
  result shape; and it needs a new validator while making the node a
  multi-archetype hybrid. Additive later.
