# Changelog

All notable changes to **Standard ASR** (the `standard-asr` core package) are
documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While on `0.x` the public API is still stabilising toward `1.0`, so minor
releases may include breaking changes.

## [Unreleased]

### Added

- **Speaker diarization ("who said what").** An opt-in, per-request
  `RuntimeParams.diarization` marker (the `DIARIZE` constant / `DiarizationRequest`)
  requests speaker labels, gated by a new `<mode>.diarization` capability. Results
  carry them on `Segment.speaker` / `Word.speaker` (and `TranscriptionEvent.speaker`
  while streaming), and a single pinned rule synthesizes the segment label from its
  words so batch and streaming agree. (#31)

### Changed (breaking — pre-1.0 policy: long-term design over compatibility)

- **Credential safety is redesigned around a pinned trust model: plugins are
  trusted code, and the security layer defends against ACCIDENTS, not
  adversaries** (AGENTS.md). What ships is a set of cheap, total rules
  (`runtime/redaction.py`): validation-error detail is rebuilt from
  `type`/`loc`/`msg` only (the `input` echo, `ctx`, and `url` are dropped;
  credential-named fields and input-echoing validator messages are redacted;
  a `loc` component not shaped like a field name — pasted key material is
  long or punctuated — is masked as `[redacted-key]`); operator-bound
  exception text goes through a bounded, cycle-safe cause/context chain
  summary (`safe_exception_summary`) that renders `ValidationError` links
  through the sanitized entries and withholds a wrapper message that
  byte-for-byte interpolated one; and `log_exception_safely` logs a full
  traceback only when the active chain carries no `ValidationError`.
  `BaseConfig`'s definition-time guards keep the honest-author dump rules
  as plain enumerations (undeclared value shapes, `SerializeAsAny`, nested
  submodel hooks, `exclude=True`). Documented accepted limits: a
  paraphrased or re-encoded echo, an author who copies an error's text and
  then discards the chain, an identifier-shaped sensitive value in a `loc`.
  (During pre-release review rounds this layer briefly grew into a
  1,600-line exception-provenance prover with core-schema closure proofs;
  it is removed under the AGENTS.md hard budget — its own complexity
  produced more confirmed defects than the residual risks it closed.)
- **`BaseConfig`'s serialization surface is closed by DISPATCH as well as by
  schema.** `public_dump()`'s "safe for `/v1/models`" contract rested on the
  definition-time serializer guards (rejecting `@computed_field` / `@model_serializer`
  / `@field_serializer` / `Annotated` serializers / `SerializeAsAny`), but an
  ordinary Python override of `model_dump` / `model_dump_json` / `__iter__` —
  or `public_dump` / `reveal_dump` themselves — never enters the decorator
  registry or the annotations, so no decorator or annotation scan can
  see it. A plugin adding
  `dumped["authorization"] = "Bearer " + self.api_key.get_secret_value()` in a
  `model_dump` override leaked the plaintext through `public_dump` under a key
  the by-name mask never checks (confirmed across all five entry points,
  including a mixin-carried override). These methods are now refused at class
  definition (statically over the whole MRO, so a mixin/intermediate-base
  override is caught too), AND `public_dump` / `reveal_dump` call the base
  implementations unbound at runtime (`BaseModel.model_dump(self, …)`;
  `reveal_dump` reads declared fields from instance state, not `dict(self)`),
  so a bypass of the definition-time gate still cannot dispatch to a plugin
  override — defense in depth.
- **Wire-visible JSON slots enforce a string key domain at construction**
  (spec §TR.1 + `contract.results.require_json_string_keys`). Every `extra`
  mapping (`Word` / `Segment` / `TranscriptionResult` / `TranscriptionEvent`)
  and `Diagnostic.provided` / `effective` now rejects any non-string object
  key — at every nesting depth — instead of letting pydantic's lax
  `dict[str, JsonValue]` silently coerce a `bytes` key to `str`. That
  coercion admitted keys no JSON document can express (a Python-only key),
  and `{"x": 1, b"x": 2}` COLLAPSED to a single `"x"` — two distinct keys
  silently becoming one, a silent wrong result on a wire-visible slot and a
  break of the Python/JSON two-layer isomorphism (G.5.2). An exact `str` is
  required (a `str` subclass is refused too: a hostile `__eq__`/`__hash__`
  could keep two keys that both serialize to `"x"` distinct in the input
  mapping, reintroducing the wire collision). A wire document's object keys
  are always strings, so the rule only ever fires for a Python caller reaching
  past the key domain; the fixed `standard_asr_json_object_key` error carries
  no input echo. (Capability nodes enforce the same rule through their extras
  validator — see the capability key-domain entry.)
- **Type names in protocol/compliance error surfaces are canonicalized**
  (`runtime.protocol_boundary`). `safe_type_name` / `_qualified_type_name`
  read a type's `__name__` / `__module__` / `__qualname__` through the
  interpreter's own getset descriptors (past a metaclass hijack) and require
  an exact-`str` identifier within a length bound; a non-conforming name
  (a newline from `type("A\nB", (), {})`, control characters, payload text)
  renders as a fixed `<unnameable type>` placeholder instead of forging a
  second log/report record on the `EngineContractError` surface. A new
  `safe_class_name` names a class directly, and the compliance suite's
  remaining direct type-name reads route through both.
- **The delivered event stream reduces to `session.result()` — the write
  side is transactional, and delivery preserves declarations** (spec §6.4 +
  `StreamReducer` + `TranscriptionSession`). Four repairs close every found
  divergence between the delivered stream and its reduction: (1) the
  reducer commits the sticky `detected_language` only on ADMITTED events —
  a suppressed supersede/final/partial no longer rewrites the result's
  language while its content is refused (the same admitted-only commit
  discipline the guard applies to its audio cursor; the retired-partial
  case now carries the guard's `lifecycle_after_terminal` diagnostic); (2)
  the producer commits to the buffer FIRST and reduces the CANONICAL
  committed event, so a buffer-overflow refusal refuses the event
  *entirely* — the old reduce-then-buffer order left an event the consumer
  could never see inside `result()`; (3) a pending partial that is the
  consumer's ONLY mention of its segment is its reading-order DECLARATION
  and is delivered ahead of the invalidating final/supersede instead of
  dropped — dropping it silently reordered the delivered transcript
  relative to `result()` for untimestamped streams and starved a delivered
  supersede of its `old_ids` declaration (suppressed consumer-side, spliced
  session-side: textual drift); (4) the terminal event is stamped with the
  session's reduced `detected_language` when it carries none — sticky
  language is order-sensitive while coalescing is not, so no per-event
  carry-forward converges in every interleaving, and the terminal is the
  one event always delivered last. A backwards event span (`end < start`)
  is now rejected at event construction (completing the `Segment` mirror
  the model already claimed), which is what makes the reducer total on
  admitted events and ordering (2) sound.
- **`EngineBase.transcribe_async` enforces the sync-call boundary** on the
  `transcribe` it consumes: an `async def` override (or a sync override
  returning a coroutine or wrong-typed value) now raises
  `EngineContractError` with the stray coroutine closed, instead of handing
  the caller a coroutine object as the "result".
- **`BaseConfig`'s input domain is closed to mappings.**
  `model_validate(obj, from_attributes=True)` extracted raw attribute
  strings behind the whitespace-preserving secret wrap and silently trimmed
  a padded credential; it now fails loudly
  (`standard_asr_config_mapping_required`). `model_validate_json` — which
  pydantic's native JSON pipeline broke for every secret-carrying document
  (the wrap's `SecretStr` instance is not valid JSON-mode input) — now
  validates the parsed document in python mode, preserving credential
  whitespace end-to-end with pinned grammar parity.
- **`supersede` places its replacements IN PLACE, on a reading-order
  ledger** (spec §5.2 + `reduce_event` + `StreamReducer` +
  `_LifecycleGuard`). The spec's ordering rule for untimestamped segments
  is "list order IS reading order", but both canonical reduces appended
  replacements at the end: `final(a,"hi") final(b,"world")
  supersede([a],[a2]) final(a2,"HI")` reduced to `"world HI"` — a
  syntactically valid, silently word-reordered transcript (the cardinal
  sin), and the dict-shaped core reduce could not even express placement.
  The spec now pins the missing semantics: `old_ids` MUST form a
  contiguous block of the LIVE reading order, in matching order (the
  invariant the frozen-prefix concatenation rule already presumed — the
  retired texts are adjacent), and `new_ids` take over the block's
  position in place; every id claims its position at first declaration
  (`partial`/`final`, or introduction via a supersede's `new_ids`, so a
  chained `A→B`,`B→C` with a contentless `B` still anchors correctly). A
  gapped or reordered block has no defined placement and is SUPPRESSED
  with the new `supersede_noncontiguous_old_ids` diagnostic (strict
  raises) under the same suppressed-supersede semantics as the
  cross-speaker ban. `reduce_event` is reshaped to the spec snippet's
  `(order, texts)` state and fails loudly on order-impossible shapes;
  `StreamReducer` orders by the shared ledger, tracks retired ids (no
  resurrection), and mirrors the guard's order-integrity rejections as
  result diagnostics for standalone (guardless) use; compliance replay
  inherits the placement check through the shared guard. Golden
  wire-shaped traces drive both reduces to identical results.
- **An empty reduced result carries `segments=[]`, never `null`**
  (`StreamReducer.result()` / `session.result()`). The spec's null rule
  separates `None` ("not requested / not applicable" — renderers may
  synthesize the whole-text fallback cue) from `[]`
  ("requested-and-performed but empty" — zero cues, never fabricated);
  the reducer performs the segment lifecycle, so a fresh reducer, a
  silence-only session, and a delete-everything supersede are the second
  state. The batch side's `None` ("segments not requested") is untouched.
- **A config's input surface must stay closed (`extra="forbid"`).** The
  same contract from the key side: a subclass reopening `extra` breaks the
  flat input-key vocabulary, the absent-vs-invalid classifier, the
  typo-names-the-key DX, and `public_dump`'s mask at once — `allow` stores
  undeclared caller data (a misplaced credential included) on the instance
  and emits it verbatim, where a by-name mask that knows only declared
  fields cannot reach it, while `ignore` silently swallows a mistyped
  credential key so it reads as an absent credential instead of a loud
  error. Both are refused at class definition, and a nested submodel with
  `extra="allow"` — which carries undeclared keys into the parent's dump
  just the same — is refused by the schema proof. The three
  decorator checks remain as a fast, precisely-worded front end; the proof
  is the backstop. `Field(exclude=True)` is refused for the round-trip half
  of the same contract: a dump documented for persistence must not silently
  drop a declared input. Derived values belong on a plain `@property` or in
  the engine.
- **The input surface is closed at EVERY depth, not only at the root**
  (`BaseConfig`). The previous rule stopped one level down: a nested
  options submodel left on pydantic's default (`extra="ignore"` — and the
  same silent-drop default for `TypedDict` and dataclasses) accepted a
  typo'd nested key and dropped it with no diagnostic, so
  `{"decode": {"baem": 8}}` read as applied while the engine ran on the
  field's default — the silent wrong result the top-level rule exists to
  prevent, one level down, and the same hole swallowed a credential key
  misplaced into a submodel. A config class is now refused at definition
  unless every nested input container its core schema reaches (submodel,
  `TypedDict`, pydantic dataclass — through containers, unions, and depth
  alike) forbids undeclared keys. The check reads the EFFECTIVE policy
  from the schema artifact (`config.extra_fields_behavior`, the same key
  on the 2.5 floor and current pydantic), so pydantic's config-propagation
  rules are honored rather than re-derived: a bare `TypedDict` and a bare
  stdlib dataclass inherit the config's `extra="forbid"` and stay closed
  for free (pinned by test), while a pydantic dataclass owns its config
  and must declare `config=ConfigDict(extra="forbid")` itself. A policy
  the walk cannot read as `forbid` — including a future pydantic that
  stops emitting the key — fails closed at class definition.
- **The env codec is read off the core schema, not an annotation-origin
  whitelist** (`BaseConfig.from_env`). The whitelist matched `list`/`dict`/
  `set`/`tuple`/`frozenset` origins and bare `BaseModel` subclasses — and
  missed `Mapping[str, int]`, `Sequence[str]`, a `TypedDict` and a
  dataclass, all of which pass every class-definition guard (they are
  ordinary, fully-declared config shapes) yet were unreachable through
  their own documented `STANDARD_ASR_<ENGINE>__<FIELD>` convention: the
  bare string hit the container validator and raised. Widening the list
  would only postpone the next miss, so the classification now walks the
  field's schema and stops at the first structured kind. The bias is
  deliberate and asymmetric: guessing "raw" for something structured fails
  LOUDLY at construction, while guessing "json" for a scalar silently
  reinterprets it (`"123"` → the integer 123), so "json" is returned only
  on positive evidence and anything unrecognized stays raw. A field
  reaching BOTH (`str | list[str]`) has no defined reading — either choice
  disagrees with the explicit constructor, which always takes the string —
  and is now refused at class definition instead of silently decoded when
  some env var happens to be set.
- **Every wire-visible slot is declared in the JSON value space**
  (`Diagnostic.provided`/`effective`, `Word`/`Segment`/`TranscriptionResult`/
  `TranscriptionEvent` `extra`, `emit_diagnostic`). Declared `Any`, these
  admitted Python objects with **no JSON representation at all** while the
  server spec simultaneously promised to forward diagnostics and non-error
  `extra` verbatim — two promises that cannot both hold. A `socket`, a numpy
  array, or a bare `object()` parked in `extra` constructed happily and then
  failed during the wire projection, *after* an endpoint had committed to a
  response. That is not two-layer isomorphism (G5.2), and the hidden
  "JSON-safe Any" precondition appeared in no type, validator, or document.
  The slots are now `JsonValue`, so the failure lands where an engine author
  can act on it: at construction, naming the field. Non-finite floats are
  excluded for the same reason (`NaN`/`Infinity` are Python floats but not
  JSON — a conforming parser rejects the whole document). Three concrete
  failures close with it: an `error` event now **drops `extra` before**
  serializing rather than overwriting it after (the old order made the
  payload's fate depend on the very field being discarded, so an
  unserializable `extra` cost the client `code`, `recoverable`,
  `retriable_after` and the gap/reconnect fields); the WS initial
  diagnostics frame is projected inside its own fault boundary (it ran
  between two boundaries — establishment had returned, the forward loop's
  catch had not started — so it killed the route unhandled, bypassing the
  operator-log redaction); and the REST response's projection is completed
  inside `_run_transcription`'s fault-mapping region rather than left to the
  ASGI encoder after the endpoint returned. `to_json_value` is the one
  helper for handing a structured value or a typed container to such a slot.
- **The metadata endpoints wrap the whole operation, not just class
  resolution** (`/v1/capabilities`, `/v1/params-schema`,
  `/v1/config-schema`). Resolution was mapped (404 / scrubbed 500) but
  everything after it ran outside any boundary — and all of it is
  third-party code: reading `declared_capabilities` /
  `provider_params_type` / `config_type` dispatches whatever descriptor or
  metaclass property the plugin installed, `canonical_json()` and
  `model_json_schema()` are the plugin's methods, and a custom
  `__get_pydantic_json_schema__` runs inside the latter. A raise anywhere
  in that stretch left the endpoint through Starlette's unhandled path:
  the client got an undocumented plain 500 instead of the documented
  scrubbed body, and `log_exception_safely` never ran — so the ASGI
  server's own traceback logger rendered the chain natively and a
  `ValidationError` in it printed its input echo into the operator log.
  These are unauthenticated discovery surfaces, so "a plugin descriptor
  does not usually fail" was never a boundary. One shared helper now spans
  resolution → descriptor read → projection, passing the endpoints' own
  deliberate verdicts (the "no capabilities declared" 404) through
  unchanged and mapping every other `Exception` to a safe-logged scrubbed
  500. The JSON projection is finished inside the boundary too (encoded
  eagerly, non-finite numbers rejected — they are not JSON), so an
  unencodable payload is a scrubbed 500 rather than a crash after the
  endpoint returned.
- **One model's fault can no longer deny every other model its verdict**
  (`compliance run`). The command's whole point is a verdict for the
  installed plugin set in one invocation (G2.1), but the construction arm
  caught a NAMED type list (`DiscoveryError`, `FactoryLoadError`,
  `ValueError`) while `registry.create` wraps only a construction-time
  `ValidationError`. A factory's own `RuntimeError`/`TypeError`/`OSError`
  — an SDK failing to initialize, a missing native library, an unreadable
  model directory — therefore escaped `for name in names` and aborted the
  run: every LATER model produced no report at all, and
  `check_entrypoints`'s earlier isolated call could not compensate
  (`_run_instance_checks` constructs a second, independent time). The arm
  now catches `Exception`, and a second per-model envelope wraps
  `_run_instance_checks` itself so even a check implementation crashing on
  an unanticipated shape stays one model's verdict, reported under its own
  `compliance_check_crashed` code so it remains distinguishable from a
  plugin's `engine_construction_failed`. `BaseException` is deliberately
  not caught: `KeyboardInterrupt` and `SystemExit` are the operator's
  control flow, not a plugin verdict.
- **A secret carrier requires the secret marker, top-level and nested**
  (`BaseConfig`). An UNMARKED `SecretStr`/`SecretBytes` field is
  half-protected: pydantic masks its dumps, but the whitespace-preserving
  pre-validator skips it — its raw string input is silently stripped by
  `str_strip_whitespace`, the exact credential rewrite the pipeline
  forbids — and the schema never renders it as a password/write-only
  input. A carrier annotation without `secret_field(...)` is now a
  definition-time `TypeError`; the nested-submodel rejection widens from
  marked secrets to bare carriers (a nested carrier is additionally
  unreachable by `reveal_dump` and unwrappable by the submodel's own
  hooks).

- **The sync-call boundary is total: plugin cleanup cannot displace the
  verdict.** Closing a stray coroutine throws `GeneratorExit` into a
  SUSPENDED coroutine, running the author's `finally` blocks on the calling
  thread — and an escaping cleanup error replaced the `EngineContractError`
  the boundary exists to produce, handing fault ownership back to whatever
  caught it next. The diagnosis is now formed before any cleanup runs,
  cleanup is contained at `BaseException`, and its failure is reported in
  the clause with fixed text (never the cleanup exception's own message).
  `safe_type_name` joins the module as the shared, total namer for the
  consumers that report a protocol violation.
- **The sync-call boundary's CLASSIFICATION is total too, and now
  returns a structured verdict.** `inspect.isawaitable`/`isinstance`
  read the value's type metadata, and a hostile metaclass `__mro__`
  property — or an ordinary broken `__class__` property — made the read
  raise OUTSIDE containment: the raw plugin exception escaped every
  consumer in place of the stable verdict. `sync_result_defect` now
  returns a `SyncResultDefect` (kind `awaitable` / `wrong_type` /
  `unclassifiable`; `str()` of it is the clause, so embedding messages
  are unchanged) and every metadata read is contained — an
  unclassifiable value IS a defect (fail-closed), surfaced by compliance
  as the new `protocol_member_unclassifiable_result` code. Consumers
  pick their surface from the verdict instead of re-inspecting the
  value: the CLI sync-bridge no longer re-runs `inspect.isawaitable` or
  reads `type(value).__name__`, and the CLI error path's last-resort
  rendering uses the shared `safe_type_name` instead of a raw
  `type(exc).__qualname__` (a hostile metaclass made error reporting
  itself crash). A bare-name collision whose qualification reads fail
  now renders `(module/qualname unreadable)` rather than the
  self-contradictory "returned bool, not bool".
- **The sync boundary starts at the `EngineBase` author hooks.** The
  template dereferences `_transcribe()`'s result and `_start_transcription()`'s
  session before either public method returns, so an `async def` hook
  surfaced as a secondary `AttributeError` on a coroutine (plus a
  never-awaited warning) before any consumer's boundary could classify it.
  A synchronous public `transcribe()` says nothing about the hook it
  delegates to, so no surface-level modality check could see this.
- **The config input-key vocabulary is validation-only.** `_flat_input_keys`
  included `field.alias` unconditionally, but pydantic treats `alias` as
  bidirectional only until a `validation_alias` overrides it — after which
  `alias` is serialization-only and REJECTED on input. All three consumers
  ask "can this key populate this field?", so a key that cannot was wrong
  for each; the visible damage was the definition-time collision guard
  rejecting a legal declaration whose serialization alias merely spelled
  another field's name.

### Changed (non-breaking)

- **Documentation site rebuilt on Fumadocs (Next.js, static export).** The
  site now lives in `docs/site`, renders the published Markdown content in
  `docs/content/`, generates the API reference from the source with Griffe,
  and deploys to GitHub Pages through the Pages deployment API instead of a
  `gh-pages` branch push. The `docs` extra now pulls in `griffelib` in
  place of `mkdocs-material` / `mkdocstrings`, and the previous site's
  `.html` URLs redirect to their new locations.
- **Internal package restructure.** The flat module layout is split into
  audience-signaling subpackages (`audio`, `contract`, `plugins`, `runtime`,
  `toolchain`). The public API is unchanged — every name still imports from
  `standard_asr` (application surface) and `standard_asr.engine` (engine-author
  surface). (#32)
- **Roadmap moved to GitHub issue [#27](https://github.com/standard-voice/standard_asr/issues/27).**
  The in-repo `docs/roadmap.md` was removed so there is a single, always-current
  source of truth for what is planned.

### Fixed

- **Security: a WS error event's detail is never repr'd into the operator
  log without shape vetting** (`toolchain.server`). The bridge logged
  `extra["detail"]` with `%r` *before* the client-side scrub — and a value
  installed past the `JsonValue` declaration (a mutated `extra` dict is
  plain Python; `frozen` guards the field, not the dict) reaches exactly
  that line: a held `ValidationError`'s `repr` echoes its input into the
  log the scrub exists to close. The bridge now logs only exact-`str`/
  `None` values (which dispatch no author code — `type is str`, never
  `isinstance`) and withholds everything else by shape, with a marker
  naming why.
- **Security: the metadata fault boundary starts at the model key, not at
  the resolved class** (`toolchain.server`).`_ensure_engine_class`'s
  `_is_protocol`/`transcribe` reads are metaclass/descriptor dispatch, and
  a failure there (e.g. a metaclass property carrying a `ValidationError`)
  escaped *before* the Round-15 boundary ever began — producing an
  undocumented plain-text 500 instead of the documented scrubbed body, and
  the ASGI server's native traceback logger rendering the raw chain
  (input echo included) because `log_exception_safely` never ran.
  Resolution is now inside the boundary try on `/v1/capabilities`,
  `/v1/params-schema`, and `/v1/config-schema`.
- **The env codec reads `Json[T]` as terminal-raw** (`BaseConfig.from_env`,
  spec IC.4). `Json[list[str]]`'s contract is that the input IS the JSON
  document text — pydantic's own `json` validator decodes it, and the
  explicit constructor REJECTS the decoded value (`json_type`). The codec
  descended into the annotation's inner document schema instead, saw its
  `list`, and pre-decoded the env string — so the field passed every
  class-definition guard and the constructor, yet failed construction from
  its own documented env convention (the decoded list hitting the `json`
  validator). The `json` core-schema kind now terminates the walk as
  terminal-raw, matching the constructor's own semantics on the 2.5 floor
  and current pydantic alike; `json-or-python` keeps descending (its two
  halves genuinely describe Python-side input).
- **Capability extension extras are closed to the JSON value space at
  construction** (`contract.capabilities`). Capability nodes parse unknown
  keys tolerantly (`extra="allow"`) for forward compatibility — but the
  VALUES were untyped, so any Python object (`object()`, `Path`, NaN/Inf)
  constructed happily through Python-land and only failed the wire
  projection at the metadata endpoint: a state the in-process tree accepts
  but the wire document cannot carry, breaking G.5.2's two-layer promise
  exactly where the wire-visible slots entry closed it for
  results/events/diagnostics. Every extra value is now validated into
  `dict[str, JsonValue]` (with `allow_inf_nan=False`) by a before-validator
  shared by `_CapNode`, `_Container`, and the constraint submodels — the
  canonicalized output is stored, so the tree and any later wire
  `model_validate` agree. The mechanism is a validator plus one module-level
  adapter rather than the `__pydantic_extra__` typed annotation because the
  native mechanism cannot express the floor contract: pydantic 2.5 builds
  typed extras only from an EAGER annotation and the module (project-wide)
  annotates lazily via `from __future__ import annotations`, which the
  floor rejects at class creation; the adapter was profiled to
  accept/reject identically on the floor and current pydantic. The
  tolerant-KEYS behavior is unchanged for JSON-string keys: unknown keys
  whose values are JSON still parse, and non-extension keys still answer
  fail-closed on probes (the key domain itself is pinned by the next entry).
- **Capability extension extras enforce the same string key domain as the
  wire slots** (`contract.capabilities` + spec R2). The extras value adapter
  above validates through pydantic's lax `dict[str, JsonValue]`, which
  DECODES a `bytes` key into its `str` spelling — and the canonicalizing
  merge then re-homed the laundered key: `{b"supported": True}` silently
  overrode a declared `supported=False` (in either insertion order: the
  bytes key never equals the declared str key at bucketing time, yet
  collides with it after coercion), and `b"x_vendor"` minted a canonical
  extension key the input never spelled — silent capability corruption on
  the exact surface gating decisions read. Extras now run
  `require_json_string_keys` (the same walker, rule, and
  `standard_asr_json_object_key` error as the results-layer wire slots)
  BEFORE the value adapter: every extra key must be an exact `str` at every
  depth, so no key can change spelling across canonicalization, and no
  order-sensitive collision with a declared field is possible. "Tolerant
  keys" tolerates UNKNOWN keys, not un-JSON ones.
- **The WS diagnostics delta sees an in-place overflow summary.** Past the
  channel cap the guard keeps a single aggregated `diagnostics_truncated`
  entry and rewrites its per-code tally in place, so a length-only cursor
  reported "nothing new" forever: the WS client kept the counts from the
  first overflow while the in-process and REST views converged on the final
  ones (a two-layer drift G.5.2 forbids). The summary is now a singleton on
  the wire — a later occurrence supersedes the delivered one — and the
  final tally always lands.
- **An empty segment is skipped by the subtitle renderers, not treated as
  unrenderable.** Both renderers already skip a payload-less segment, so an
  empty one produces no cue whether or not it was measured; judging
  renderability first made the same segment silently skipped when it had
  timestamps and a hard `SubtitleRenderingError` when it did not. Empty
  segments are filtered before the policy runs and excluded from its counts.
- **Server docstring: `FactoryLoadError` maps to a scrubbed 500, not 404.**
  The code was already right (a resolved key whose plugin fails to load is a
  deployment fault); the docstring described the mapping this PR replaced.
- **`standard-asr show` reports a broken plugin as an engine fault.** It was
  the one consumer that caught `FactoryLoadError`, printed
  `Capabilities: <unavailable: ...>`, and returned **0** — telling a script
  the model was usable when nothing about the engine could be read. It now
  prints the same diagnostics and exits **1**. The class-level
  `declared_capabilities` read runs through the engine-fault seam as well
  (it can be a metaclass property).
- **The `prepare` attribute LOOKUP runs inside the engine-fault seam.** The
  round-8 envelope wrapped the call but not the binding, and `prepare` may
  be a property whose body is plugin code — a descriptor raising
  `ValueError` was reported as the invoker's usage error (exit 2) instead of
  an engine fault (exit 1).

- **Engine DECLARATION defects raise `EngineContractError`, not
  `ConfigError`.** A malformed declared `selectable_languages` /
  `detectable_languages` tag, a language axis without the IC.6
  `default_language` obligation, and an unsatisfiable `prepare` shape
  (coroutine function / non-callable / parameter-requiring) are the engine
  author's contract violations — no configuration value fixes them.
  `ConfigError` now means what its name says: the supplied or ambient
  configuration is invalid, fixable by whoever supplies it. Update
  `except ConfigError` handlers that relied on catching declaration bugs.
- **CLI exit codes classify fault at the seam, not by exception class.**
  A registered model whose plugin fails to import (`FactoryLoadError`), the
  `ValueError` family escaping the engine execution seam (`transcribe()` /
  `prepare()` — a bare SDK `ValueError`, a raw engine-internal
  `ValidationError`, `InvalidProviderParamError`), and every
  `EngineContractError` now exit **1** (engine/deployment fault, the CLI
  twin of the server's scrubbed 500). Every `ConfigError` — including
  `ConfigurationRequiredError` surfacing lazily at first transcribe — stays
  exit **2**: at the CLI the invoker owns the flags AND the env, so invalid
  configuration is caller-actionable there (`docs/spec/cli.md` states the
  contract).

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
- **`Segment.start`/`end` are nullable — the placeholder-marker design is
  replaced.** `None` now means "the engine measured no such time" and is
  stored verbatim by the streaming reducer (nothing fabricates `0.0` spans
  anymore); the legal shapes are `(float, float)` (measured, `end >= start`),
  `(float, None)` (start-only — the real onset survives on the model), and
  `(None, None)` (unavailable); `(None, float)` is construction-rejected, on
  `TranscriptionEvent` `partial`/`final` too (a previously silently-mangled
  adapter shape). The derived read-only `Segment.timestamp_status`
  (`"measured"` / `"start_only"` / `"unavailable"`) can never disagree with
  the values. **Removed**: the reserved `SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER`
  key, its construction validator, and its exports — timing truth lived in a
  mutable side-channel dict that `frozen=True` never protected and the wire
  spec never documented; `Segment.extra` is engine-owned again with no
  reserved keys. The wire schema now renders `start`/`end` as number-or-null
  (two-layer isomorphism: the schema is the documentation). The result-level
  `segment_timestamps_unavailable` diagnostic remains as the aggregate
  disclosure, derived from the values; the renderers no longer read it.
- **`to_srt`/`to_vtt` never silently drop text: missing timing is the
  caller's decision.** The renderers previously omitted unmeasured segments'
  text from the file by default — with the only disclosure living on the
  result object, not in the returned string (a marker-only result had NO
  disclosure at all). New keyword `on_unrenderable: "error" | "omit" |
  "collapse"` (type `UnrenderablePolicy`), default `"error"`: any segment
  that cannot render as a **visible** cue — no measured span, or a measured
  span that quantizes to zero milliseconds on the output grid (players
  silently drop a `T --> T` cue, so the old "render zero-length cues
  faithfully" contract produced successful files whose text never appeared)
  — raises the new `SubtitleRenderingError` (with `.unrenderable`/`.total`),
  and the caller explicitly chooses `"omit"` (renderable cues only; possibly
  zero) or `"collapse"` (one synthetic whole-text cue — the previous
  all-placeholder behavior, now opt-in). The renderer never widens a span on
  its own (no fabricated 1 ms). A result whose every span survives the grid
  renders identically to before under every policy, and a stale diagnostic
  can no longer collapse a real timeline (values are the only signal). Code
  that rendered timestamp-less streaming results must now pass a policy.
- **`start_transcription` is unconditionally required by compliance.** The
  `StandardASR` protocol has always pinned the member as present on every
  engine (batch-only raises `UnsupportedFeatureError` from it); the compliance
  surface check now enforces that instead of waiving it for batch-only
  engines. A structural plugin that omitted the method must add it (raise
  `UnsupportedFeatureError`); `EngineBase` subclasses are unaffected (the
  template provides it).
- **Engine-side `ValidationError` is an engine fault on every transport.** A
  bare pydantic `ValidationError` escaping `transcribe()` /
  `start_transcription()` after the request's options were validated maps to a
  scrubbed HTTP 500 (was: 422 blaming the client's `options`) and a scrubbed
  WS `internal_error` frame (was: `unsupported` with `str(exc)` echoing
  pydantic's `input_value`). Fault ownership no longer depends on whether the
  engine inherits `EngineBase`. API clients matching on the old 422 for this
  case must update.
- **`StandardASR.config` is a read-only protocol property.** Mutable protocol
  members are invariant under strict typing, so a real plugin annotating its
  own config subtype could not be typed as `StandardASR` without a cast —
  defeating the protocol's no-cast promise. Reading `engine.config` is
  unchanged; assigning it *through the protocol type* is no longer legal
  (config is constructor-injected).
- **`ConfigurationRequiredError` narrows the compliance credential skip.** The
  new `ConfigError` subtype means exactly "required configuration absent from
  this environment"; `BaseConfig.from_env` raises it automatically when
  construction fails solely on missing required fields. Compliance skips
  ONLY that state — a plain `ConfigError` or a raw `ValidationError` from a
  zero-arg factory is now a compliance **failure** (`factory_config_invalid`
  / `engine_construction_failed`), not a "needs credentials" warning: waiving
  every config failure let a broken plugin read as green-with-warning. An
  engine that raises `ConfigError` by hand for its missing-credential state
  must switch to `ConfigurationRequiredError` (engines using `from_env` need
  no change). The absence classification is deliberately narrow: only
  top-level `missing` failures on environment-fillable own fields qualify —
  a subclass that forgot to pin its `engine` discriminator default (an
  env-excluded field no environment variable can supply) or a
  supplied-but-incomplete nested value stays a plain `ConfigError` and fails
  compliance as the declaration bug it is. Aliased fields resolve correctly:
  pydantic keys its errors by alias, so a missing
  `Field(alias="xi-api-key")` credential is mapped back to its own-field
  name (unique field-name / `alias` / string `validation_alias` match;
  ambiguity stays fail-closed) and classified as absence — not rejected as
  an unknown name, which would have re-created the env-dependent verdict.
- **Compliance verifies the batch-only refusal behaviorally.** For an engine
  declaring no streaming axis, `check_entrypoints` now CALLS
  `start_transcription()` (construct-not-enter envelope; a compliant engine
  raises at the capability gate) and requires the protocol-pinned
  `UnsupportedFeatureError`: returning a session
  (`batch_only_streaming_not_refused`) or raising another type
  (`batch_only_streaming_refusal_wrong_error`) now fails — method presence
  alone certified engines that violate the promise callers rely on.
- **Synchronous protocol modality is enforced across the whole surface.**
  Every `StandardASR` member except `transcribe_async` is synchronous (async
  behavior lives in `transcribe_async` and inside the returned session).
  Detection lives in one shared, public home —
  `standard_asr.runtime.protocol_boundary.sync_result_defect` (a stray
  coroutine is CLOSED rather than leaked as a never-awaited
  `RuntimeWarning`; messages carry type names only, never the value) — with
  a raising adapter `require_sync_result` that surfaces a violation as the
  new `EngineContractError` (an engine fault, deliberately not a
  `ValueError`). Every consumer call site guards its result:
  - *compliance probes*: swap-safety's `transcribe()`, the streaming gating
    check's every `supports()` query plus its wire-format synthesis, the
    wire-format round-trip, the batch-only refusal probe, and the
    sync-bridge classification probe AND its `session_factory()`
    establishment boundary (`sync_bridge_invalid_session`);
  - *CLI*: the capability pre-gate and bridge setup, `transcribe` (a
    coroutine/wrong-typed result is an engine fault, exit 1, instead of a
    secondary `AttributeError`), and `prepare` (a sync wrapper delegating to
    an `async def` used to print a false "prepare complete" — the returned
    value must now be strictly `None`);
  - *server*: the REST `transcribe` path (boundary check AND
    `TranscribeResponse` construction now live inside the fault-mapping
    region, so a malformed engine result maps to the scrubbed 500 instead of
    a raw `ValidationError` escaping the route) and the WebSocket
    establishment path (a non-session return maps to the scrubbed
    `internal_error` frame instead of an `AttributeError` after the
    error-mapping block).
- **`supports()` MUST return a real `bool` — truthiness no longer negotiates.**
  Every capability consumer used to coerce `supports()` by truthiness, so a
  non-compliant `return "false"` read as "everything supported" — a silent
  wrong capability negotiation. Compliance now verifies the return type at
  the entrypoint layer and at every gating/bridge query
  (`protocol_member_wrong_return_type`; strict `isinstance` — a
  `numpy.bool_` is not a `bool`), and the CLI's pre-gate counts only a
  literal `True` as supported (fail-closed on any malformed answer).

- **Non-string validation aliases are rejected at config-class definition.**
  `BaseConfig`'s absent-vs-invalid classifier (the machinery behind
  `ConfigurationRequiredError`'s compliance *skip*) resolves every pydantic
  error `loc` to a single string token, and the env convention (IC.4) is
  flat — but a field declared with `AliasPath` reports a nested `loc` when
  it is simply ABSENT, so a pure missing-credential state was misclassified
  as a plugin defect (an environment-dependent compliance verdict: fail on a
  clean CI, pass on a credentialed machine). `__pydantic_init_subclass__`
  now rejects `AliasPath` (and any `AliasChoices` carrying one) loudly at
  class definition with the flat-mapping rationale; all-string
  `AliasChoices` is now genuinely supported (each choice resolves like a
  string alias, so pure absence classifies as `ConfigurationRequiredError`).
- **`doctor` never convicts on an approximation: satisfiability is
  three-state.** The emptiness probe collapsed "no witness found" into
  "unsatisfiable" — a logic error a finite search cannot back (and its
  release-derived candidates dropped the edge's epoch, so every epoch range
  over final releases, even the trivially satisfiable `>1!1.0`, was branded
  an internally-unsatisfiable hard conflict with a non-zero exit; on
  `packaging <= 25.0` an `===` specifier crashed the unguarded membership
  loop outright). Verdicts are now SAT / UNSAT / UNKNOWN: UNSAT comes only
  from `packaging`'s exact algebra (`SpecifierSet.is_unsatisfiable()`,
  `>= 26.2`) and is reported as an absolute verdict; the fallback witness
  search (older `packaging`) proves SAT — with epoch-preserving candidates —
  or answers UNKNOWN, which feeds the existing `analysis_unavailable`
  non-clean state with one note naming every undecided relation and the
  upgrade path. Honesty trade-off on legacy `packaging`: doctor no longer
  *convicts* range conflicts it cannot prove there (including the 1.x/2.x
  split), it discloses them as undecidable — still a non-zero exit, never a
  false clean. `packaging` remains optional, not a core dependency.
- **Server construction faults are never the caller's: 422 → 503/500.**
  Engine construction is `registry.create(model)` — zero-arg; the client
  chooses the model key and nothing else, so a construction failure is never
  a request error. `ConfigurationRequiredError` (required config absent from
  the server environment — the state compliance SKIPS) now maps to a **503**
  (REST) / `service_unavailable` frame (WS) with a stable generic detail —
  the absent field names are deployment detail, safe-logged for the operator,
  never sent. Every other construction failure (plain `ConfigError`,
  `InvalidProviderParamError`, a raw `ValidationError` — the state compliance
  FAILS as `engine_construction_failed`) maps to the scrubbed **500** (REST)
  / `internal_error` frame (WS). The old 422/`bad_request` mapping blamed the
  caller for faults it cannot see, reach, or fix — and surfaced server-side
  config field names to unauthenticated clients.
- **Pydantic input echoes no longer reach operator/CI logs.** The redaction
  boundary scrubbed every CLIENT-facing surface but the server's
  `logger.exception` calls still wrote raw `ValidationError`s — `input_value=`
  echo included, directly or via the `__cause__` chain of the standard
  layer's `raise TranscriptionError(...) from exc` wrap — into server logs,
  and the compliance suite embedded `repr(exc)` / `str(exc)` of raw
  validation errors into issue messages printed to terminals and CI logs.
  New `standard_asr.runtime.redaction.log_exception_safely` (used at every
  server exception-log site) logs the scrubbed one-line chain summary when
  the active chain carries a `ValidationError` — fault structure and stack
  context survive, the echoed input never does; generic exception text
  still reaches the operator log unchanged (the spec's deliberate channel,
  stated normatively in server.md §3.7 and IC.3). The compliance messages
  (`factory_config_invalid`, `properties_revalidation_failed`,
  `language_config_invalid`) use `sanitized_validation_message` for the
  same reason.
- **The CLI's normal error line goes through the same safe boundary.** Every
  `main()` catch arm previously printed bare `str(exc)` before any `--debug`
  logic ran, so a wrapper that copied a chained `ValidationError`'s
  (truncated) input echo into its own message leaked it to stderr with no
  flags at all. All arms now report via one helper: untainted chains keep
  their authored message, tainted chains render the input-echo-free summary.
  Discovery's entry-point `load()` wrapper builds its message with
  `safe_exception_summary` for the same reason (plugin module code can raise
  a `ValidationError` whose `repr` echoes input).
- **WS establishment maps a bare `ValueError` to `internal_error`, not
  `unsupported`.** By establishment every client input is already validated
  and a compliant engine signals unsupported features with
  `UnsupportedFeatureError`, so a surviving bare `ValueError` is an
  engine/adapter fault: the old arm blamed the caller and sent `str(exc)` —
  engine-internal, possibly credential-bearing text — to an unauthenticated
  client (REST's fault-ownership rule, now mirrored; spec §4.2 updated).
- **Secret fields resolve to exactly one carrier, and defaults are vetted.**
  The definition-time guard accepted plaintext unions (`SecretStr | int`
  left a constructed `int` unmasked in `repr`/`model_dump` while the schema
  advertised a password) and dual carriers (`SecretStr | SecretBytes`);
  `_secret_carrier` now requires exactly `SecretStr`/`SecretBytes`,
  optionally with `None`. Defaults are vetted at class definition (pydantic
  never validates them): a plain-string default leaked plaintext and crashed
  `model_dump_json`/`public_dump` inside the secret serializer;
  `default_factory` is rejected as unvettable. The whitespace-preserving
  pre-validator wraps raw strings into the field's *own* carrier — wrapping
  into `SecretStr` unconditionally made every env/alias string construction
  of a `SecretBytes` field fail where plain pydantic would have coerced it.
- **`from_env`'s explicit-wins is alias-aware.** An explicit value supplied
  under a field's alias / `validation_alias` / `AliasChoices` choice now
  suppresses that field's canonical env fallback; the old blind merge kept
  both keys and `extra="forbid"` loudly rejected the env key as extra where
  the documented contract says the explicit value wins (IC.4 updated).
  Passing two explicit keys for one field is still loudly rejected.
- **Synthetic-cue visibility is decided on the output millisecond grid.** A
  sub-millisecond `duration` (e.g. `0.0005`) passed the raw `> 0` float
  check yet formatted to `00:00:00,000 --> 00:00:00,000` — the invisible cue
  players silently drop. The fallback now fires whenever the duration
  quantizes to zero on the same `int(round(s * 1000))` grid the timestamp
  formatter renders. (Generalized to EVERY cue by the renderability
  redesign above: `on_unrenderable` covers measured spans that quantize to
  zero too.)
- **Compliance renders every embedded exception through the total boundary.**
  The remaining raw-`repr` sites (factory classification, sync-bridge
  establishment, strict discovery, class metadata) now use
  `safe_exception_summary`, and the bridge worker stores the exception
  OBJECT instead of freezing `repr(exc)` in-thread — a hostile `__repr__`
  previously crashed the worker before the error was recorded and the main
  thread mis-reported the crash as `sync_bridge_no_terminal`.
- **The WS config handshake is a closed request model.**
  `StreamConfigRequest` (frozen, `extra="forbid"`) replaces the ad-hoc
  parse: an unknown top-level key (a client typo like `"optinos"`) used to
  vanish silently and the session started on defaults; it is now a loud
  `bad_request` naming the key. The handshake catch-all no longer maps every
  exception to `bad_request` with raw `str(exc)`: only caller-fixable
  failures do; anything else is a scrubbed, safe-logged `internal_error`.
- **Bytes-input credentials keep exact contents; flat input keys have one
  owner.** pydantic's lax `bytes -> str` coercion for a `SecretStr` field
  ran the decoded text through `str_strip_whitespace` (silently trimming a
  padded credential); the pre-validator now wraps bytes/bytearray per
  carrier (UTF-8 for `SecretStr`, loud rejection for non-UTF-8; verbatim for
  `SecretBytes`). A field's alias colliding with another field's name or
  alias let ONE caller key silently populate TWO settings
  (`populate_by_name` fills both); every flat input key now has exactly one
  owning field, enforced at class definition. The pre-validator also
  covers **every `Mapping` input** (`MappingProxyType` and other read-only
  mappings previously bypassed the wrap and were silently stripped).
- **A rejected unknown key is masked in `loc` when shaped like key
  material.** `extra_forbidden`'s last `loc` component is the caller's own
  key text (the one structural path by which request input enters a
  `loc`), and FastAPI returns `loc` in the 422 body. A field-name-shaped
  key (identifier-ish, ≤ 32 chars) stays named — pointing at a typo'd
  `optinos` or a mis-placed `api_key` is deliberate DX — while a long or
  non-identifier-shaped key (the structural signature of pasted credential
  material) becomes `[redacted-key]`.
- **A registered model whose plugin fails to load is an engine fault, not
  an unknown model.** `FactoryLoadError` (the key resolved; a
  server-installed plugin failed to import/resolve/validate) mapped to
  404 / WS `unknown_model` with raw plugin-fault text — blaming the caller
  for a fault it cannot fix and crossing the trust boundary with
  import/annotation internals. All resolution surfaces (REST construction,
  WS, capabilities/params-schema/config-schema) now split the arms:
  `EntrypointValidationError` keeps its authored 404; `FactoryLoadError`
  maps to a scrubbed, safe-logged 500 / `internal_error`.
- **The CLI treats a raw `ValidationError` as an engine fault (exit 1).**
  Caller-originating pydantic failures are all classified upstream, so a raw
  one at the top level is a structural engine constructing an invalid
  internal model — the seam the server maps to a scrubbed 500. It was
  reported as usage exit 2 with a caller-audience rendering, which also
  split the trust boundary (the normal line used caller policy while
  `--debug` used the operator policy for the same error).
- **Every caller-authored `loc` component is filtered by shape.**
  `extra_forbidden` is not the only channel by which request text enters a
  `loc`: a `dict[str, T]` field contributes the caller's mapping KEY, and a
  non-string key contributes its `repr`. Both reached `ConfigError.details`,
  operator logs, and compliance output. Every string component is now
  filtered on every surface: field-name-shaped tokens and pydantic's
  `[key]` marker survive (naming a rejected key is deliberate typo DX — the
  sender already holds any key they placed in their own request), pasted
  key material becomes `[redacted-key]`.
- **The `supports()` classification probe contains `BaseException`.** It runs
  inside the establish worker's `except UnsupportedFeatureError` block, where
  Python does not route a raised exception to that try's sibling
  `BaseException` arm — the worker died unclassified and the run reported
  `sync_bridge_did_not_terminate` for what was really a broken `supports()`.
- **The sync-bridge drive worker contains `BaseException`.** A
  `CancelledError` (re-raised by `future.result()` in the drive thread)
  killed the daemon worker without recording the error, and the main
  thread mis-read the silent death as `sync_bridge_no_terminal` — a false
  verdict about the wrong defect. The drive worker now matches
  `_establish`'s deliberate `BaseException` containment.
- `check_streaming_param_gating` no longer false-fails a protocol-complete
  structural engine: the sub-constraint probe reads the non-protocol
  `effective_capabilities` defensively and falls back to the protocol's
  `declared_capabilities` (what `EngineBase` defaults to) instead of turning
  the missing convenience attribute into `gating_probe_selection_raised`.
  `check_provider_params_swap_safety` / `check_streaming_param_gating` are
  now typed against `StandardASR` (no `EngineBase` casts in the CLI).

- `check_recommended_wire_format` no longer requires `EngineBase`: the
  session-establishment format rule is the pure
  `ensure_wire_format_supported(properties, format)` shared with
  `EngineBase.ensure_stream_format_supported`, so a fully-compliant
  structural engine is no longer false-failed on an `AttributeError`.
- `compliance run` gives a credentialed engine one verdict: a zero-arg
  factory raising `ConfigurationRequiredError` (required configuration
  absent from the environment) is the same skipped-not-failed state
  `compliance entrypoints` already reports, instead of a contradictory
  per-model error in the same command. Any other `ConfigError` remains a
  failure (see the breaking `factory_config_invalid` entry above).
- `standard-asr doctor` no longer reports satisfiable pre/dev/post version
  windows (e.g. `>2.0rc1,<2.0rc3`) as hard conflicts: the emptiness probe now
  derives within-segment neighbor and edge-`.dev0` witnesses.

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
