# Changelog

All notable changes to **Standard ASR** (the `standard-asr` core package) are
documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While on `0.x` the public API is still stabilising toward `1.0`, so minor
releases may include breaking changes.

## [Unreleased]

### Changed (breaking — pre-1.0 policy: long-term design over compatibility)

- **`TranscriptionResult.metadata` removed.** The free-form "standardized
  metadata" dict had no standardized keys, no writer, and no reader — the same
  blanket-metadata channel the spec already removed from Properties and
  Capabilities. Engine-specific data belongs in `extra`; future standardized
  result data will land as named fields. A plugin still populating `metadata=`
  now fails loudly: the engine template wraps the resulting `ValidationError`
  as a `TranscriptionError` naming the plugin/core version mismatch (HTTP 5xx
  through the server — never a client-blaming 422).
- **Strict-mode candidate-language rejections raise `UnsupportedFeatureError`**
  (`param="candidate_languages"`, with `mode`) instead of a bare `ValueError`,
  matching every other strict-gate rejection (spec §RT R2) so all transports
  map it to a client-error verdict (REST 422 / WS `unsupported` / CLI exit 2).
  Malformed or `"auto"` candidate entries still raise `ValueError`
  unconditionally (caller code bugs). Update `except ValueError` handlers
  written against the old contract.
- **CLI discovery flag renamed `--strict` → `--strict-discovery`** on `list` /
  `show` / `prepare` / `transcribe` / `compliance`, and argparse prefix
  abbreviation is disabled: bare `--strict` is now a loud usage error instead
  of silently meaning discovery strictness while `strict` (the engine's
  parameter-gating policy, `--set strict=...`) means something else on the
  same command line.
- **Streaming-reduced results disclose placeholder timing.** Timestamp-less
  segments carry the reserved `timestamp_placeholder` marker in
  `Segment.extra` plus a result-level `segment_timestamps_unavailable`
  diagnostic; `to_srt`/`to_vtt` key on those signals (marked placeholders are
  omitted from timed cues; all-placeholder results render one whole-text cue)
  instead of emitting zero-duration cues players silently drop.

## [0.1.1] - 2026-06-16

Initial public release: a universal, plug-and-play interface protocol for ASR
(speech-to-text) inference, plus the runtime library and toolchain that enforce
it. No ASR models ship here — each engine is a separate, pip-installable plugin
that Standard ASR discovers automatically.

### Added

- **Universal engine interface.** The `StandardASR` protocol and `EngineBase`
  template define one contract for batch (`transcribe` / `transcribe_async`) and
  streaming (`start_transcription`) inference, so application code works
  unchanged across every compliant engine.
- **Audio input negotiation & constant output.** An `AudioInput` discriminated
  union (local path, encoded bytes, waveform array, fetchable URL, base64, cloud
  URI) is converted to what each engine accepts via a deterministic negotiation
  matrix; lossy steps emit structured diagnostics and impossible conversions fail
  loudly — never a silent wrong result. Output is always the constant-schema
  `TranscriptionResult` (the return type never varies with parameters).
- **Machine-readable engine metadata.** Engines declare static **Properties**
  (I/O bounds, sample-rate limits, BCP-47 `selectable_languages`), a
  hierarchical, fail-closed **Capabilities** tree (queried via
  `engine.supports("dot.path")`), and Pydantic **Config** models (with standard
  mixins such as device selection and `SecretStr` credential fields) for
  auto-generated configuration UIs.
- **Closed runtime parameters with capability gating.** A single closed
  `RuntimeParams` model is gated against each engine's declared capabilities in
  `strict` or `best_effort` mode — unsupported parameters fail loudly or are
  dropped with a diagnostic, never silently ignored.
- **Unified streaming semantics.** A single event protocol (`partial` / `final`
  / `supersede` / `progress` / `done` / `error`), segment lifecycle, and explicit
  stability guarantees (`stable_until` frozen prefixes; `final` / `closed`
  terminal states) normalise wildly different real-time engine behaviours so
  streaming application code is also "write once, run on any engine".
- **Zero-config plugin discovery.** Installed engines are found automatically via
  `standard_asr.models` entry points — install a plugin, use it immediately, no
  application changes.
- **Toolchain.** A CLI (`standard-asr`) to discover engines, transcribe files,
  manage/warm up models, run the compliance suite, and diagnose conflicts; a
  FastAPI **server** exposing any engine over HTTP + WebSocket with a non-leaking
  error contract; a one-command **compliance** suite that shares its validation
  logic with the runtime so verdicts and behaviour cannot drift; and `doctor` for
  read-only dependency-conflict diagnosis.
- **Batteries-included extras.** SRT/VTT renderers and audio loading, with heavy
  dependencies kept optional behind the `[audio]` and `[server]` extras.
- **Security by default.** Credentials use `SecretStr`; URL inputs are validated
  (HTTPS, SSRF guard) with unsafe options requiring explicit opt-in.

### Engineering

- Pure-Python core with a near-zero dependency footprint (`numpy` + `pydantic`).
- Typed end to end (`py.typed`, pyright strict), 100% test coverage, tested on
  CPython 3.10–3.14 across Linux/macOS/Windows and against numpy 1.26 and 2.x.

[Unreleased]: https://github.com/standard-voice/standard_asr/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/standard-voice/standard_asr/releases/tag/v0.1.1
