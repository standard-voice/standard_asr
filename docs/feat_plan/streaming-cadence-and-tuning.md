# Streaming Cadence & Tuning (Design Case)

**Status:** Design settled 2026-07-05 (decisions recorded below; engine facts
verified against official upstream sources the same day). NOT yet implemented —
it changes the engine protocol/spec (new capability node, new runtime param,
one new properties field) and rolls out across the core *and* every plugin at
once. Spec updates at implementation time MUST carry the design rationale into
the spec text (maintainer directive 2026-07-05).

> Rebuilt 2026-07-05 (an earlier version was lost, never committed), then
> settled the same day in maintainer discussion. Re-derived from the
> 2026-06-15 investigation record, the design-notes/research archive, a fresh
> inventory of current declaration surfaces, and an official-source
> verification pass (per-engine sources cited in §3).

Delivers the roadmap line (#27): **"Streaming cadence"**.

Mission fit: *streaming semantics are the core value proposition*
(`docs/mission.md`); the metadata surfaces exist so apps never guess (G.1.3,
G.1.5); every new field gets a wire expression (G.5.2).

Builds on and must stay consistent with:
- `docs/spec/specification.md` — §4 streaming events/stability, R5 param
  freeze, IC.7 init/runtime boundary, §9 capability table, §10 ship/defer.
- `docs/research/3 streaming 补充调查 2026-06-05.md` §3.1/§3.7 and
  `docs/design-notes/6–8` — the original design record.
- `docs/feat_plan/model-management.md` — the *other* pending
  properties/capabilities growth wave (see D7).

---

## 1. Problem statement

An application opening a streaming session cannot answer three basic questions
through the standard:

1. **How fast is this engine?** Expected text lag, and whether results arrive
   continuously, in ticks, or only at segment ends.
2. **Can I trade latency for accuracy, and how?** Several engines expose a
   latency dial; the standard has no portable way to see or set it.
3. **How should I feed audio?** What chunk size the engine prefers — today
   apps hard-code a guess (`standard-asr-live` uses `chunk_ms=100`).

Observed consequence (2026-06-15 investigation): `standard-asr-live` over
`std-mlx-audio`'s nemotron preset "felt like a fixed 5-second interval". The
model has a native cache-aware incremental decoder; the plugin ran a generic
5-second windowed re-decode, its knobs (`redecode_interval_s=5.0`) trapped
inside the plugin, and the app guessed at chunk size. The user experienced the
worst cadence of the three layers and could not tell which layer was
responsible.

There is also a **standing spec/implementation inconsistency**: spec §10
(`specification.md:1275`) defers "runtime `target_latency` adjustment — *v1
fixes it at construction time only*" — but no construction-time field exists
anywhere. An engine with four latency tiers cannot even advertise that it has
them. The 2026-06 audit record already sketched the remedy (P3); this design
adopts and refines it. The original review cut only the **mid-session
adjustment** ("无控制通道 → 砍到构造期"); construction-time declaration was
deferred, never rejected.

## 2. Context — what exists today

| Surface | Cadence content today |
|---|---|
| `StreamingCapabilities` (`contract/capabilities.py:515`) | 10 fields, all flags/enums — no numeric timing anywhere |
| `BaseProperties` (`contract/properties.py:185`) | sample rates only — waveform identity, not chunking/pacing |
| `RuntimeParams` (`contract/params.py:129`) | closed portable set + `provider_params`; frozen at session open (**R5**) |
| Session API (`runtime/streaming.py:2074`) | `feed()`/`send_audio()` accept arbitrary chunks; only *termination* deadlines exist |
| `recommended_wire_format()` (`runtime/interface.py:976`) | rate/encoding/channels — never a chunk duration |
| Heartbeat (`specification.md:1016-1021`) | `progress`/`audio_processed_until` are MAY, reporting-only; core imposes no interval |
| Escape hatches (IC.7) | engine-opaque `BaseConfig`/`ProviderParams` knobs — not cross-engine-portable |

## 3. Evidence — verified cadence shapes (official sources, 2026-07-05)

URLs are linked inline where a stable one exists; other rows name their
official source type (model card / paper / docs) — re-verify from the engine's
official page, not from secondary surveys.

| Shape | Engine | Verified facts (official model cards / papers / docs) |
|---|---|---|
| **Tiered** | NVIDIA Nemotron streaming (en 0.6B) | exactly 4 att-context working points → 0.08/0.16/0.56/1.12 s ("Dynamic Runtime Flexibility… Pareto curve", HF card). Multilingual 3.5 sibling: **5** points, [56,N]. Reference implementation selects the point **once, before the streaming loop** — construction-time in practice. |
| **Configurable, stepped** | Mistral Voxtral Mini Realtime | delay = any multiple of 80 ms in 80–1200 ms, plus 2400 ms standalone; 480 ms recommended (≈Whisper accuracy). Set at input-preparation/config time; no mid-stream API. ⚠ correction to the 2026-06 survey: **not continuous**. Append-only emission is architecturally implied but *never officially stated* — do not cite as fact. ([HF card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602), [paper](https://arxiv.org/abs/2602.11298)) |
| **Configurable, stepped** | NVIDIA Parakeet unified | 160–2080 ms in 80 ms steps ("choose the optimal streaming latency… with step of 80ms"), plus a curated preset table — sits between "tiered" and "configurable"; step-aware `range_ms` represents it honestly. |
| **Configurable, continuous** | NVIDIA Canary streaming | float `chunk_secs`(2.0) / `left_context_secs`(10.0) / `right_context_secs`(2.0); Wait-K vs AlignAtt strategies. (The survey's "same position re-predicted within a chunk" claim is NOT in official docs — dropped.) |
| **Fixed** (per preset) | Kyutai STT | 0.5 s (`stt-1b-en_fr`, semantic VAD) vs 2.5 s (`stt-2.6b-en`) — delay baked per checkpoint; model choice, not a dial. Frame-aligned timestamps confirmed. |
| **Fixed** | Moonshine v2 streaming | ~80 ms algorithmic lookahead; finalization needs ~320 ms future context; fixed per architecture/checkpoint. |
| **Fixed + commit axis** | ElevenLabs Scribe v2 Realtime | ~150 ms partials; separate client-controlled **commit boundary** (`commit_strategy: manual\|vad` + thresholds). The commit boundary is a *finality* mechanism, not output latency — deliberately out of this design's scope (see D1). |
| **Finals-only** | FireRedASR2S | streaming VAD + offline beam-search decode per segment; no partials. Already representable: `emits_partials=false`. |
| — | SenseVoice | officially **no streaming mode at all** ("not designed for streaming transcription", FunAudioLLM paper); "SenseVoice+VAD" pseudo-streaming is a community deployment pattern, not official — cite with that caveat only. |
| **Plugin-synthesized** | std-mlx-audio windowed re-decode | cadence is an artifact of `redecode_interval_s`, not the model — declared `typical` ≈ 5000 ms makes the problem visible and attributable. |

Two orthogonal axes: **declaration** (engine → app: shape + numbers) and
**tuning** (app → engine: a portable dial where one exists). Plus one honesty
axis: declared cadence makes plugin-synthesized cadence visible.

---

## 4. Decisions (2026-07-05)

**D1 — Declaration: `StreamingCapabilities.output_latency` node.**
`mode: "fixed" | "configurable" | "tiered" | "none"` (default `none` — degrade
to unknown, never false precision), with constraint payload:
`points_ms: list[int]` (tiered — non-empty, sorted), `range_ms: {min, max,
step?}` (configurable — `step` optional, for the verified 80 ms-stepped dials),
and `typical_ms: int | None` (all modes except `none`; the default/expected
operating point). Lives in **capabilities** because it gates the tuning param
(D3) — and it flows into `supports()`, canonical JSON, and the wire for free.
The ElevenLabs-style client-commit boundary is a *finality* axis and stays out
(recorded as future work, next to `finality_level`).

**D2 — No declared latency-class enum.** Coarse classes (realtime/low/…) are
*derived* by consumers (e.g. the catalog #28 buckets `typical_ms`); declaring
both a class and numbers would be a dual source of truth. An earlier roadmap
draft's "latency class enum" idea is delivered as a derived facet, not a
protocol field.

**D3 — Tuning: `RuntimeParams.target_latency_ms: int | None`.** Portable,
capability-gated (meaningful only for `configurable`/`tiered`), **target
semantics with snapping**: the engine picks the nearest achievable operating
point; the gating diagnostic reports `requested` vs `effective_ms` (this also
absorbs real-world dial irregularities like Voxtral's standalone 2400 ms
value). Strict policy → structured error when unsupported or out of
range/points; best-effort → snap/clamp + diagnostic. Frozen at session open
per **R5** — construction-time only, which matches how the verified engines
actually work (Nemotron's pre-loop setter, Voxtral's config-time delay).
Mid-session adjustment stays deferred with the original rationale (no control
channel); revisit only together with `mutable_mid_stream`.

**D4 — Chunk hint: `BaseProperties.preferred_stream_chunk_ms: int | None`.**
Static I/O identity, next to the sample-rate fields. Informational only —
`feed()`/`send_audio()` stay free-form; negotiation unchanged. `AudioFormat`
is **not** touched (format is not pacing). Unit: milliseconds.

**D5 — One number, precisely defined.** `typical_ms` = *typical delay between
audio entering the session and the corresponding text event* (end-to-end text
lag — the quantity users feel, and the same quantity Nemotron/Voxtral/Kyutai
publish). No separate update-interval field: steady-state event spacing is
app-observable at runtime and not worth a second declaration.

**D6 — Compliance: observe, don't assert (mostly).** Static checks assert
declaration well-formedness (tiered ⇒ valid `points_ms`; configurable ⇒ valid
`range_ms`; gating honored for `target_latency_ms`). Runtime probes (#33)
*report* observed first-event latency and cadence — never assert them (CI
hardware varies; assertions would flake). One behavioral assertion IS added to
#33's probe set: the `word_stability` honesty probe (declared `true` while
emitting `stable_until=0` everywhere = declared-honest, behaviorally dishonest
— today's one-directional checks pass it).

**D7 — Rollout: standalone wave, ahead of the metadata wave.** This design is
capabilities-centric with one properties field; it lands on its own and
unblocks streaming DX + resolves #33's pacing question now. Locality
(model-management) + hardware (#8) + model card (#19) form a second,
coordinated properties/spec wave (#19 design pending). Whichever lands first
must not paint the other into a corner; both waves update the spec §9 table,
params/gating docs, and every plugin.

---

## 5. Ecosystem effects

- **#33 runtime compliance**: pacing observations + the honesty probe land
  there (its open question 4 is resolved by this document).
- **#28 catalog**: latency facet derived from `typical_ms` (D2).
- **#7 live view / CLI `show`**: display declared vs observed cadence;
  `standard-asr-live` replaces its `chunk_ms=100` guess with the D4 hint.
- **Plugins**: every streaming plugin re-declares; `std-mlx-audio` becomes the
  honest worst case (`typical_ms≈5000`) until it adopts nemotron's native
  incremental decoder — a plugin-quality issue this design makes visible, not
  one it fixes.
- **Spec**: new §9 rows, R5 text extension, §10 rewritten to match reality
  (fixing the `specification.md:1275` inconsistency), **with the D1–D7
  rationale summarized in the spec text** per the maintainer's directive.

## 6. Remaining open items (implementation-level)

1. Exact validator set for `range_ms.step` (must `typical_ms` land on a valid
   point/step? — lean yes, validator-enforced).
2. Snap-diagnostic payload field names (`requested_ms`/`effective_ms` vs
   reusing an existing diagnostic shape).
3. Whether `preferred_stream_chunk_ms` also surfaces through CLI `show` output
   formatting in the same PR or later.
