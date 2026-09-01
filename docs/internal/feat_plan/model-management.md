# Inference artifact lifecycle

**Status:** Approved on 2026-08-26 after two BDFL review rounds. Implementation
is in progress on `feat/model-artifact-lifecycle`.

This design replaces the earlier locality-based model-management draft. It
defines a small contract for inspecting and acquiring persistent resources that
an engine needs for inference. The document calls such a resource an
**inference artifact**. Weights, tokenizers, aligners, and converted recognizer
bundles can be inference artifacts. The term does not mean a discoverable model
preset, an engine, or an arbitrary implementation cache. This design does not
classify an engine as local, cloud, or hybrid.

The design builds on these existing contracts and design records:

- [`lazy-load.md`](lazy-load.md) is a living background note about pure engine
  construction and guarded first-use downloads. The published protocol and
  CLI specification remain authoritative where the note is stale.
- [`../../content/specification/download-policy.md`](../../content/specification/download-policy.md)
  defines `STANDARD_ASR_ALLOW_DOWNLOAD`, cache-root precedence, and the current
  `prepare()` behavior.
- [`capability-kinds.md`](capability-kinds.md) separates static declarations
  from host observations.
- [`streaming-cadence-and-tuning.md`](streaming-cadence-and-tuning.md) D7
  requires this work, hardware declaration, and model-card metadata to share a
  coherent metadata architecture.
- `standard_asr.runtime.downloads` implements the download toggle and cache-root
  helpers.
- Protocol IC.9 and IC.11 define lazy construction and `prepare()`.

## 1. Decision summary

Standard ASR defines introspection and control. The engine performs every
engine-specific operation.

The first version has these parts:

1. A required `artifacts` section in a new class-level `declared_metadata`
   surface. It states three independent upper-bound facts about a model.
2. An instance-level `artifact_status()` method. It reports the artifact
   dependency closure for the resolved engine config and optional request
   context.
3. An instance-level `acquire_artifacts()` method. It delegates every explicit
   acquisition to the engine and returns a fresh report.
4. Structured progress and required-action data models.
5. Four artifact-specific errors for status, availability, acquisition, and a
   failed progress observer.
6. `standard-asr status` and `standard-asr pull` commands.

The first version does not include artifact removal, a cache-layout standard,
a remote management API, a package installer, license acceptance, or a
local/cloud/hybrid classification. It also does not change the library default
that permits first-use downloads. It makes that behavior inspectable and
controllable, and the CLI warns before a known first-use acquisition.

## 2. Problem

Plugin discovery lists every model that an installed plugin can provide. It
does not list only the model weights present on the machine. A plugin can
therefore add dozens of models to an application immediately after installation,
while the application cannot answer these questions:

- Does this model need persistent artifacts before it can run?
- Are the configured artifacts ready?
- Can the application acquire them explicitly?
- Can normal inference acquire them implicitly?
- Does the user or operator need to complete an external action first?
- How much data is present, and where is it stored?

The existing `prepare()` hook is not sufficient. It combines persistent
artifact acquisition with process-local loading, accelerator initialization,
kernel compilation, and other warm-up work. Its `None` return value gives an
application no status, progress, or required-action information. Its base
implementation is also a no-op, so an application cannot infer whether a model
needs artifacts from the presence of the method.

## 3. Scope and ownership

### 3.1 Standard ASR owns

Standard ASR owns the portable contract that an application consumes:

- the static declaration;
- the dynamic report and its state semantics;
- the operation that asks an engine to acquire artifacts;
- structured progress and required actions;
- the download-policy interaction;
- public errors, CLI behavior, and compliance checks.

### 3.2 The engine owns

The engine owns every recognizer-specific decision and side effect:

- determining the artifact dependency closure for its resolved config and
  request context;
- interpreting its recognizer's cache, archive, manifest, or directory layout;
- invoking a Hub SDK, a download script, an operating-system API, or another
  native mechanism;
- handling locks, resume files, temporary files, checksums, revisions, and
  shared blobs;
- translating native failures into the Standard ASR artifact contract.

### 3.3 Out of scope

The core does not:

- inspect a Hugging Face, ModelScope, NGC, Vosk, sherpa-onnx, or other cache;
- require one cache directory or physical bundle layout;
- accept license terms, request access, authenticate an account, or run a
  user-supplied shell command;
- install plugins or manage Python environments;
- download weights for a remote service that the engine only calls;
- deploy or manage a self-hosted inference server;
- convert or train a recognizer unless an engine chooses to do so inside its
  explicit acquisition operation;
- treat artifact readiness as proof of hardware compatibility, credentials,
  network availability, privacy, price, or inference correctness;
- remove shared artifacts in the first version.

## 4. Why locality is not part of this design

`local`, `cloud`, and `hybrid` combine unrelated facts. They can imply an
inference location, client-side weights, credentials, network use, privacy,
billing, hardware, or deployment ownership. None of those implications is
reliable.

For example, the `qwen3-asr/1.7b` model names an open-weight checkpoint, but the
current engine calls a separately deployed vLLM server. The Standard ASR process
does not own that server's weights. Artifact acquisition is therefore not
applicable to this engine, even when the server runs on the same machine.

This design answers only the question the application needs for inference-artifact
management: does the configured engine lifecycle consume, inspect, or acquire
separately supplied persistent artifacts?

Lifecycle ownership, not process topology, sets the boundary. The scope includes
persistent resources that the configured engine consumes or can ask another
system to supply, including weights, tokenizers, aligners, and converted
recognizer bundles. An operator-provided NFS path and an engine-managed sidecar
can both be in scope. Artifacts owned by an independent inference service that
the engine can only call are out of scope, even when that service runs on
loopback.

The scope excludes incidental implementation caches such as Python bytecode,
HTTP metadata, telemetry, temporary inference files, and opaque JIT caches.

If Standard ASR later needs to disclose network use, billing, or whether audio
leaves the process, each concern gets its own property and semantics. A locality
preset is not an escape hatch for those future contracts.

## 5. Evidence from real recognizers

The design was checked against the following acquisition shapes. The sources
are official project documentation or source code.

| Shape | Evidence | Consequence |
| --- | --- | --- |
| Explicit download and implicit load-time download both exist | [faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py) exposes `download_model()`, while `WhisperModel(...)` can also download. [ModelScope](https://modelscope.cn/docs/models/download) provides SDK and CLI downloads and also downloads from `Model.from_pretrained()`. | Acquisition controls are independent facts, not one mutually exclusive mode. |
| Access requires a human action before programmatic download | [Hugging Face gated models](https://huggingface.co/docs/hub/models-gated) require a browser access request and authentication. Approval can be automatic or manual. | Required actions are dynamic report data. A gated model is not permanently manual. |
| A recognizer accepts a downloaded file, a script output, or a converted file | [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md) documents a download script, manual download, and format conversion. | Acquisition is broader than one Hub download, and the effective behavior can depend on config. |
| Configuration changes acquisition behavior | [Vosk's Python binding](https://github.com/alphacep/vosk-api/blob/master/python/vosk/__init__.py) downloads and extracts a model for `Model(lang=...)` or `Model(model_name=...)`, but an explicit `model_path` is path-only. | Static facts are upper bounds. The instance report describes the resolved config. |
| One usable recognizer needs several coordinated files | [sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html) can require tokens, encoder, decoder, and joiner files. | A path existing is not proof of readiness. The engine reports a logical dependency, not individual files. |
| The source depends on a model identifier | [NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html) can resolve `from_pretrained()` through NGC or Hugging Face. | The core does not standardize one source backend. |
| A pipeline has request-dependent auxiliary models | [WhisperX](https://github.com/m-bain/whisperX) selects language-specific alignment models and uses a separately gated diarization model. [FunASR](https://github.com/modelscope/FunASR/blob/main/funasr/auto/auto_model.py) can compose ASR, VAD, punctuation, and speaker models. | Status is a dependency closure for a config and request context, not one boolean per entry point. |
| An auxiliary artifact is selected by a request or provider option | [pywhispercpp](https://github.com/absadiki/pywhispercpp/blob/f7bf62118c0a33a43cf8aabb58eef16cea5d16c4/pywhispercpp/model.py) exposes a separate `vad_model_path`, and [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) publishes a distinct forced-aligner checkpoint. | `ArtifactContext.params` is necessary even when the base recognizer preset is fixed. |
| The operating system owns shared speech assets | [Apple Speech `AssetInventory`](https://developer.apple.com/documentation/speech/assetinventory) installs and retains assets for sharing across applications. | An engine can delegate to an operating-system acquisition API without the core owning the files. |
| The Python distribution includes the default recognizer data | [PocketSphinx](https://github.com/cmusphinx/pocketsphinx/blob/main/docs/source/index.rst) includes a default model with the Python installation. | No separate acquisition is required, despite in-process inference. |
| The engine calls a separately operated inference service | The current [`std-qwen3-asr`](https://github.com/standard-voice/std-qwen3-asr/blob/main/src/std_qwen3_asr/engine.py) plugin is an HTTP and WebSocket client for DashScope or vLLM. | The artifact lifecycle is not applicable when the plugin cannot manage the service's weights. This says nothing about network use or billing. |
| Cache revisions and blobs are shared | [Hugging Face cache management](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) documents snapshots, shared blobs, cache inspection, corruption, and revision-aware deletion. | Logical model size can double-count shared storage. Generic removal is unsafe. |
| Incomplete and corrupt are different native conditions | Hugging Face documents `IncompleteSnapshotError` for a known partial snapshot and reports `CorruptedCacheException` separately from cache scanning. | The requirement state needs both `incomplete` and `corrupt`. |
| A boolean inventory exists but is not enough | [Wyoming model inventory](https://github.com/OHF-Voice/wyoming) exposes an `installed: bool`. | A boolean is useful prior art, but cannot express partial, corrupt, conditional, or unknown readiness. |
| Pulling a mutable name normally re-resolves it | [Docker pull](https://docs.docker.com/reference/cli/docker/image/pull/) updates mutable tags and pins digests. Hugging Face downloads resolve a branch or tag to its current commit. Ollama's pull path also fetches the remote manifest when a local manifest exists. | A pull operation needs an explicit refresh semantic; local readiness alone cannot prove freshness. |

The current Standard ASR plugins add two important config-dependent cases:

- `std-faster-whisper` and `std-mlx-audio` use a Hub model by default but accept
  a user-supplied `model_path`. Their static declarations describe every
  possible config. Their dynamic reports describe the resolved config.
- The current `std-faster-whisper` plugin has no artifact-only acquisition
  path. Its only implemented path constructs `WhisperModel(...)`, which can
  download and load. Calling upstream `download_model()` is a Phase D change,
  not a description of current plugin behavior.
- The current `std-mlx-audio` download-policy path has prerequisite defects:
  it discards the resolved `download_root`, does not consume `hf_token` or
  `dtype`, and passes `local_files_only` through an upstream loader that ignores
  it for most presets. Phase D must fix those defects before artifact status can
  claim to describe the same context as inference.
- `std-qwen3-asr` calls DashScope or a separately deployed vLLM server. It does
  not own inference artifacts in either case, including the open-weight vLLM model
  presets.

## 6. Conceptual model

### 6.1 A model is not a physical artifact

A Standard ASR model is a discoverable `engine_id/model_name` preset. One model
can require several logical artifacts. Several models can share the same
physical blobs. A logical artifact can contain many files.

The engine reports logical artifact requirements. The core never derives a
one-to-one mapping between models, directories, files, and storage use.

### 6.2 A report describes a resolved context

Artifact requirements can depend on:

- the model preset;
- init config, including a local path, cache root, revision, or selected
  recognizer family;
- batch or streaming mode;
- runtime parameters, including language, word timestamps, or diarization;
- information that is not available until inference, such as an automatically
  detected language.

`artifact_status()` therefore runs on an engine instance after config
resolution. It accepts an optional `ArtifactContext` carrying the mode and
`RuntimeParams`. The public
`EngineBase` template applies the same parameter gating and language resolution
used by inference before it calls the engine hook.

The returned requirements are the dependency closure for that context. They are
not a catalog of every artifact that the plugin could ever use.

When an input-dependent requirement cannot be resolved, the engine reports an
`unknown` logical requirement. It also reports whether inference can acquire
that requirement implicitly. It does not guess or download every possible
language model.

## 7. Static declaration

This feature is the first consumer of a fourth typed engine declaration
surface:

```python
class ArtifactDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    applicable: bool
    supports_explicit_acquisition: bool
    may_acquire_during_inference: bool


class DeclaredEngineMetadata(_JsonExtraModel):
    artifacts: ArtifactDeclaration


class ExampleASR(EngineBase):
    declared_metadata: ClassVar[DeclaredEngineMetadata]
```

`declared_metadata` is a structured aggregate, not a free-form metadata mapping.
Known sections are typed. Its root uses the same tolerant JSON-extra mechanism
as capabilities: unknown string keys with JSON values survive parsing and the
canonical wire projection, so #8 or #19 can add a sibling section without
breaking an older reader. Current authors use only standard sections or the
`x_<vendor>_*` namespace; compliance flags an unrecognized unprefixed producer
key as a likely typo. The typed `artifacts` section itself is frozen for protocol
1.x; changing its field set or meanings requires a major version.

This plan makes `artifacts` required in protocol 1.1. Future coordinated work
adds the following sibling sections instead of creating more unrelated class
variables:

- #19 adds static model-card identity and provenance;
- #8 adds declared hardware compatibility and requirements;
- current-host measurements and verification outcomes remain dynamic doctor
  and compliance reports, because they are observations with a different
  lifetime.

This is the total engine declaration architecture after this plan:

| Surface | Owns | Does not own |
| --- | --- | --- |
| `BaseProperties` | Static identity and audio I/O boundaries. | Lifecycle behavior, model-card prose, or host observations. |
| `DeclaredCapabilities` | Static transcription abilities, behaviors, and request limits. | Inference-artifact lifecycle or observed host state. |
| `config_type` | Inputs used to construct a configured engine. | Facts or observations. |
| `declared_metadata` | Structured static facts about the model preset and its operating requirements. | Runtime readiness, current hardware, or arbitrary plugin data. |

The new surface requires amendments to goals G.1.3 and G.5.2. Its wire
expression is `GET /v1/metadata/{model}`. Like the existing capabilities and
schema endpoints, this per-model endpoint resolves one entry point behind the
whole-operation metadata fault boundary. `standard-asr show` renders the same
canonical projection. Bulk `standard-asr list` and `GET /v1/models` remain
entry-point-only and do not import plugins.

The independent surface is deliberate. These are lifecycle declarations
consumed before or outside a transcription request. They are not I/O
properties, request limits, host observations, or behaviors that alter the
transcript. The previous rationale, which excluded them from capabilities only
because no `RuntimeParams` field gates on them, was incorrect: capability
behavior nodes can be informational.

The facts are deliberately independent:

- `applicable` is `True` when at least one supported config or
  request context uses a persistent artifact that is supplied separately from
  the installed plugin and falls inside the configured engine's lifecycle
  boundary. The artifact can be supplied externally by an operator or operating
  system.
- `supports_explicit_acquisition` is `True` when at least one supported context
  lets `acquire_artifacts()` ask the engine to materialize artifacts before
  inference.
- `may_acquire_during_inference` is `True` when at least one supported context
  can acquire artifacts from a normal `transcribe()` or streaming path.

If `applicable` is `False`, the other two fields must also be
`False`. The inverse is not required. An external-only engine declares
`True, False, False`.

These values are upper bounds over all supported configs. They let a per-model
metadata query show an honest static summary without constructing the engine.
They do not claim that the current config needs acquisition.

The `artifacts` section and all three values have no defaults. A default of
`False, False, False` would turn an unupdated engine into a false "no artifact
acquisition" claim. Every model must make the no-acquisition case explicit in
protocol 1.1.

The engine-author surface exports a frozen `NO_ARTIFACT_LIFECYCLE` constant
for the common explicit `False, False, False` declaration. Using the constant is
still an authored claim, not a silent inherited default.

Required authorship is enforced by static typing and compliance-time
`inspect.getattr_static`, not by `__init_subclass__`. Class creation therefore
stays cheap and avoids metaclass coupling. Compliance rejects a missing section,
a core-owned placeholder inherited from `EngineBase`, or a malformed
declaration. A plugin-owned base class can author one declaration for preset
subclasses that share it; compliance follows the MRO owner and does not require
21 identical declarations in a plugin family.

### 7.1 Example declarations

| Engine shape | `applicable` | `supports_explicit_acquisition` | `may_acquire_during_inference` |
| --- | ---: | ---: | ---: |
| Remote API, separately deployed server, or bundled model | `False` | `False` | `False` |
| Explicit pull required before inference | `True` | `True` | `False` |
| Explicit pull available, with automatic first-use fallback | `True` | `True` | `True` |
| Only first use can acquire | `True` | `False` | `True` |
| Operator or user must provide files | `True` | `False` | `False` |

### 7.2 Protocol generation and rollout

> **Amendment (2026-08-31, maintainer decision).** The protocol number line
> was re-based before any external release: the pre-artifact contract is
> `0.1.0`, this feature's contract is `0.2.0`, protocol major 0 is the
> pre-stable line with the minor as the breaking axis, and the first stable
> release promotes the then-current contract verbatim to `1.0.0`. The `1.0.0`
> / `1.1.0` numbers below are the historical plan as approved; spec AR.1 is
> authoritative.

This feature targets Standard ASR protocol `1.1.0` and core package `0.2.0`.
The two version numbers have different owners: `properties.protocol_version`
names the protocol implemented by an engine, while the package version names a
core release.

Phase A0 establishes the version mechanism before adding artifact APIs:

1. Validate `protocol_version` as `MAJOR.MINOR.PATCH` and define the core's
   supported protocol-major and feature-minor table.
2. Treat artifact lifecycle as introduced in protocol 1.1. A 1.0 engine is
   never interpreted as `NO_ARTIFACT_LIFECYCLE`; artifact tooling reports that
   the plugin must be upgraded by raising a new general
   `ProtocolCompatibilityError(StandardASRError)`. It has an explicit CLI arm
   that maps it to exit 2; it does not inherit `ValueError` and therefore does
   not couple unrelated transport mappings to this transition. The reference
   server does not use artifact management operations in this version.
3. Ship the new core surface in package 0.2.0. First-party plugin releases then
   require `standard-asr>=0.2.0`, declare protocol 1.1.0, and add the required
   metadata and methods in one coordinated rollout.
4. Compliance checks protocol-minor obligations. A plugin must not bump its
   declaration to 1.1.0 until the complete artifact contract is present.
5. A later additive protocol feature increments the minor version. Removing or
   changing an existing contract, or adding a new state that changes consumer
   control flow, increments the major version.

The core can keep ordinary inference available for a discovered protocol 1.0
plugin during the transition, but `status` and `pull` fail loudly with an
invoker-fixable compatibility error. This transition behavior avoids both a
false no-artifact claim and a bulk-list import.

The compatibility owner is explicit. `EngineBase.artifact_status()` and
`EngineBase.acquire_artifacts()` begin with the protocol-feature guard. The CLI
and other generic consumers run the same core guard before looking up a method,
so a structural protocol 1.0 engine also raises
`ProtocolCompatibilityError`, not `AttributeError`. An application that calls
a legacy structural object directly must check `properties.protocol_version`
first; protocol 1.1 objects have the constant method surface.

## 8. Dynamic data models

All author-side contract data models are frozen pydantic v2 models with
`extra="forbid"`. Unknown fields remain errors because they are normally plugin
typos. Output classification fields use bounded, open string vocabularies,
parallel to streaming error codes, rather than closed pydantic enums. This is
the tolerant-reader boundary: a newer producer's unknown value is preserved and
handled conservatively instead of making an older application reject the whole
report.

### 8.1 Request context

```python
class ArtifactContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ModeName | None = None
    params: RuntimeParams = Field(default_factory=RuntimeParams)
```

The context packages the request information known before audio arrives. It
keeps the public method signature extensible without adding unrelated keyword
parameters. The `EngineBase` template gates its params and resolves its language
axis before calling an engine hook.

`ModeName` is the existing shared mode vocabulary, not a new two-value artifact
type. When #35 adds `job`, it widens that one source and the artifact context in
the same protocol change. `mode=None` asks `EngineBase` to resolve a concrete
mode: prefer batch when declared, otherwise use the only declared inference
mode. A streaming-only engine therefore resolves to streaming instead of gating
against a mode it never declared. If several non-batch modes exist, the caller
must choose one; the template raises `ConfigError` rather than guessing. A
no-artifact engine with no mode domain returns a not-applicable batch report for
stable serialization.

The first version does not describe whole-input versus incremental streaming,
an audio format, an input kind, or audio-content-dependent selection. An engine
reports an unknown requirement when one of those facts is needed. A later
protocol version can add a context field without changing both operation
signatures.

### 8.2 Requirement state and aggregate readiness

```python
ArtifactState = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]
ArtifactReadiness = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]

ARTIFACT_READY: Final = "ready"
ARTIFACT_MISSING: Final = "missing"
ARTIFACT_INCOMPLETE: Final = "incomplete"
ARTIFACT_CORRUPT: Final = "corrupt"
ARTIFACT_UNKNOWN: Final = "unknown"

ARTIFACTS_READY: Final = "ready"
ARTIFACTS_UNAVAILABLE: Final = "unavailable"
ARTIFACTS_UNKNOWN: Final = "unknown"
ARTIFACTS_NOT_APPLICABLE: Final = "not_applicable"
```

Requirement states have narrow meanings:

- `ready`: The engine has reliable evidence that the configured logical
  requirement can resolve without acquiring new persistent artifacts.
- `missing`: The engine has reliable evidence that no usable artifact set is
  present for the requirement.
- `incomplete`: Some expected content is present, but the logical requirement
  is not complete. This includes a detectable interrupted acquisition.
- `corrupt`: The engine has reliable evidence that present content is unusable
  because an integrity or layout check failed.
- `unknown`: The engine cannot determine the state through a cheap,
  side-effect-free inspection.

The names are intentionally distinct: **status** is the inspection operation
and its report, **state** belongs to one logical requirement, and **readiness**
is the aggregate inference verdict.

Aggregate readiness has a separate vocabulary because an aggregate
`incomplete` had two competing meanings. `ready` means every required
requirement is ready. `unavailable` means at least one required requirement is
known not to be ready. `unknown` means readiness cannot be established.
`not_applicable` means the configured context has no inference-artifact
lifecycle in this engine.

`ready` does not promise that an online-enabled library skips metadata checks or
updates. A floating Hub branch can have a usable cached snapshot and still
resolve a newer revision during later inference. The separate
`may_acquire_during_inference` fact remains authoritative for that risk.

An engine does not hash several gigabytes on every status call merely to prove
integrity. It reports `corrupt` only when it already has reliable evidence. It
reports `unknown` when a cheap check cannot establish the state.

For forward compatibility, an unrecognized requirement state is treated as
`unknown`, an unrecognized aggregate readiness is not ready, an unrecognized
blocker prevents acquisition and projects to the standard `unsupported` reason
when an error is required, an unrecognized action uses its message as a generic
action, and an unrecognized progress phase is display-only. The raw blocker
token remains unchanged in the attached report; only the control-flow
projection is conservative. All raw tokens survive serialization. Current
protocol 1.1 engines emit only the standard tokens above or a namespaced
`x_<vendor>_*` token; compliance checks that producer obligation. Adding a
standard token with new control-flow semantics is a protocol-major change.

### 8.3 Required actions

```python
ArtifactActionKind = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]

ARTIFACT_ACTION_ACCEPT_TERMS: Final = "accept_terms"
ARTIFACT_ACTION_AUTHENTICATE: Final = "authenticate"
ARTIFACT_ACTION_REQUEST_ACCESS: Final = "request_access"
ARTIFACT_ACTION_PROVIDE_ARTIFACTS: Final = "provide_artifacts"
ARTIFACT_ACTION_INSTALL_EXTERNAL: Final = "install_external"
ARTIFACT_ACTION_OTHER: Final = "other"


class ArtifactAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ArtifactActionKind
    message: str
    url: HttpsUrl | None = None
```

`HttpsUrl` is a new `Annotated[AnyUrl, UrlConstraints(...)]` type, not an
existing repository type. It permits only the `https` scheme. A validator also
rejects URL user information so credentials cannot be embedded before the host.
The core displays but never dereferences this URL, so the audio-fetch SSRF DNS
and private-address checks do not apply. The data model does not carry a shell
command, credential value, signed download URL, or license-acceptance token.

Required actions describe prerequisites. They do not define the engine's
ultimate acquisition capability. A gated Hub model can statically support
explicit acquisition while its current requirement cannot run until an
`accept_terms`, `request_access`, or `authenticate` action is complete.

### 8.4 Logical requirements

```python
ArtifactAcquisitionBlocker = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]

ARTIFACT_BLOCKER_DOWNLOADS_DISABLED: Final = "downloads_disabled"
ARTIFACT_BLOCKER_ACTION_REQUIRED: Final = "action_required"
ARTIFACT_BLOCKER_UNSUPPORTED: Final = "unsupported"


class ArtifactRequirement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str
    label: str
    state: ArtifactState
    required_for_inference: bool
    can_acquire_now: bool
    may_acquire_during_inference: bool
    source_is_mutable: bool
    acquisition_blocker: ArtifactAcquisitionBlocker | None = None
    required_actions: tuple[ArtifactAction, ...] = ()
    location: Path | None = None
    size_bytes: int | None = None
    expected_size_bytes: int | None = None
    artifact_version: str | None = None
```

The fields mean:

- `artifact_id` is an opaque identifier scoped to the engine. It is stable
  enough to correlate status and progress during one configured lifecycle. It
  is not a global storage key or deletion authority. A final status query after
  acquisition must retain every target identifier passed to the hook. The
  dependency closure can add newly discovered children, but it cannot replace
  a target silently; a bootstrap implementation retains a parent ledger entry.
- `label` is display-only.
- `required_for_inference` states whether the selected execution path needs this
  logical requirement. An optional optimization or fallback artifact sets it to
  `False`.
- `can_acquire_now` states whether the engine can attempt explicit acquisition
  with the locally known config, policy, and prerequisites. It is not a
  guarantee that the source service accepts the request.
- `acquisition_blocker` tells the public template why a non-ready requirement
  cannot run. It distinguishes a disabled network path, a locally known action,
  and an unsupported explicit operation for the resolved config.
- `may_acquire_during_inference` is the effective fact for this requirement.
- `source_is_mutable` is `True` only when the requirement names a mutable
  source-service reference, such as a branch, tag, or model alias, that a
  refresh can re-resolve. A pinned commit or digest, an operator path, and an
  opaque installed asset set it to `False`. Local path changes are observed by
  status; they are not a source-service refresh.
- `location` is the logical root when the engine can report one. It can be a
  file or directory. Its absence does not imply remote inference.
- `size_bytes` is the logical present size. `expected_size_bytes` is the known
  complete size. Either can be `None`.
- `artifact_version` is an optional opaque revision, checksum, version, or
  equivalent display value.

`ArtifactRequirement.state` uses the requirement-state vocabulary and cannot be
`not_applicable`. Report applicability is explicit; an empty applicable report
means that this context currently has no requirements, not that artifact
management is outside the engine lifecycle.

A ready requirement cannot carry a required action, cannot acquire now, and has
no blocker because no acquisition is needed. Refresh is an explicit operation
request, not a property of readiness. For every non-ready requirement,
`can_acquire_now=True` exactly when `acquisition_blocker` is `None`. A known
action requires the `action_required` blocker. A `downloads_disabled` blocker
is valid only when the requirement's available explicit operation needs a
network transfer; a local extraction can still run. Sizes must be nonnegative.
`location` is absolute when present. `artifact_id` is a nonempty, bounded ASCII
identifier, not a file path. These invariants are pydantic validators, not
compliance-only conventions.

| Requirement condition | Valid acquisition fields |
| --- | --- |
| `state="ready"` | `required_actions=()`, `can_acquire_now=False`, and no blocker. |
| A locally known prerequisite is absent | At least one required action, `can_acquire_now=False`, and blocker `action_required`. |
| Native explicit acquisition can be attempted | No required actions, `can_acquire_now=True`, and no blocker. |
| Network acquisition is available but disabled | `can_acquire_now=False` and blocker `downloads_disabled`. |
| External requirement with a known user or operator action | At least one action such as `provide_artifacts`, `can_acquire_now=False`, and blocker `action_required`. |
| First-inference-only requirement, or no known action can improve it | `required_actions=()`, `can_acquire_now=False`, and blocker `unsupported`. |
| `state="unknown"` | Acquisition can remain runnable. `can_acquire_now=True` then has no blocker. |

Sizes are informational. Shared or deduplicated blobs can make the sum larger
than the physical disk use. A size does not predict how many bytes removal would
free.

### 8.5 Aggregate report

```python
class ArtifactReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ModeName
    applicable: bool
    requirements: tuple[ArtifactRequirement, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    readiness: ArtifactReadiness

    @model_validator(mode="after")
    def readiness_matches_requirements(self) -> Self: ...

    @classmethod
    def from_requirements(...) -> Self: ...
```

`from_requirements()` computes readiness. The after-validator checks recognized
protocol 1.1 values. It preserves a future unknown value and treats it as not
ready rather than rejecting the report. Current engines are still checked by
compliance for emitting the canonical value. Storing the value keeps JSON dump
and validation isomorphic; a pydantic `computed_field` would serialize a
read-only field that `extra="forbid"` then rejects on round-trip.

Aggregate readiness describes inference readiness, not whether every optional
artifact is present. It is derived from `applicable` and requirements with
`required_for_inference=True`:

1. `applicable=False` requires no requirements and means `not_applicable`.
2. `applicable=True` with no required requirements, or with all required
   requirements ready, means `ready`.
3. Any recognized required `missing`, `incomplete`, or `corrupt` requirement
   means `unavailable`.
4. Every remaining combination contains an `unknown` or future state and means
   `unknown`.

Aggregate readiness is a display and routing summary. Acquisition always iterates
the individual requirements and never chooses work from aggregate precedence.
An optional non-ready artifact does not make inference unavailable, but an
explicit `pull` can still acquire it.

Artifact identifiers must be unique inside one report.

The static declaration and dynamic report have fail-closed subset rules:

- A report's `applicable=True` requires the declaration's `applicable=True`.
- A declaration with `applicable=False` requires the report's `applicable=False`
  and an empty requirement tuple.
- `can_acquire_now=True` requires `supports_explicit_acquisition=True`, no
  blocker, and no required actions.
- A requirement with `may_acquire_during_inference=True` requires the matching
  static declaration to be `True`.
- A model that declares `applicable=False` always returns a
  not-applicable report.

The dynamic report can narrow the static declaration. For example, a
faster-whisper model class can support explicit Hub acquisition while an
instance configured with `model_path` reports an external requirement with
`can_acquire_now=False`.

`EngineBase` constructs the report from the engine hook's applicability value,
requirement tuple, and diagnostics, so
an `EngineBase` author does not manually duplicate aggregate readiness. A
structural engine that constructs a report directly gets the same validator.

A logical requirement can hide alternative physical layouts, mirrors, or file
sets that are equivalent from the application's perspective. Requirements form
an AND-set only after the engine resolves those alternatives. A local-to-remote
fallback or optional acceleration artifact uses `required_for_inference=False`,
or a config/preset that selects one execution path. The core does not model an
arbitrary dependency expression language.

## 9. Public engine operations

The full engine protocol gains two synchronous methods:

```python
def artifact_status(
    self,
    context: ArtifactContext | None = None,
) -> ArtifactReport: ...


def acquire_artifacts(
    self,
    context: ArtifactContext | None = None,
    *,
    refresh: bool = False,
    progress: ArtifactProgressCallback | None = None,
) -> ArtifactReport: ...
```

Both methods are part of protocol 1.1 `StandardASR`, including engines that have
no inference artifacts. This gives 1.1 applications one constant surface
without `hasattr()` or optional-protocol checks. An engine with no applicable
artifacts returns a not-applicable report with an empty requirement tuple.
`acquire_artifacts()` then succeeds as an idempotent no-op.

### 9.1 `artifact_status()` contract

`artifact_status()`:

- runs after engine config resolution;
- applies best-effort parameter gating for the requested mode, regardless of
  the engine's configured strict policy, and returns the resulting diagnostics;
- still raises `InvalidProviderParamError` for wrong-engine `provider_params`,
  because provider ownership is policy-independent;
- uses a best-effort language-resolution path as well as `gate_params(...,
  strict=False)`; it does not call the existing strict language helper
  unchanged;
- does not load weights, initialize an accelerator, or run inference;
- does not initiate an application-level network request;
- does not create, repair, update, or remove files;
- can inspect the filesystem, a local cache index, an operating-system asset
  inventory, or equivalent local state;
- can return `unknown` rather than performing an expensive integrity check;
- returns the same report when relevant external state has not changed.

Because status does not contact a source service, it does not promise to know
whether a remote access request was approved or whether terms were accepted.
It reports actions that are known from local config and cached state. An
acquisition failure can discover and return additional required actions.

A filesystem check can block when a path is on network-attached storage. The
protocol does not claim that ordinary filesystem access is local or bounded.
An application that scans many models should run status calls outside its UI
thread and apply its own deadline.

The public method's `Raises` list is complete:

- `ProtocolCompatibilityError` for a pre-1.1 engine;
- `InvalidProviderParamError` for wrong-engine provider params;
- `ValueError` for malformed request language data, independent of
  best-effort policy, or for an explicit mode the engine does not support;
- `ConfigError` or `EngineContractError` for invalid engine configuration or
  declarations;
- `ArtifactStatusError` for an unexpected inspection failure.

Unsupported portable values do not appear in this list because introspection
converges them through best-effort diagnostics.

### 9.2 `acquire_artifacts()` contract

`acquire_artifacts()`:

- is synchronous and idempotent while the config and upstream artifact
  resolution remain unchanged;
- attempts every non-ready requirement with `can_acquire_now=True` in the
  resolved dependency closure;
- when `refresh=True`, also passes requirements with
  `source_is_mutable=True` and no acquisition blocker to the explicit
  acquisition hook so it can re-resolve those references;
- does not transcribe audio or call a billable inference endpoint;
- can perform the minimum loading that a native acquisition API cannot separate
  from materialization, but does not promise process-local warm-up;
- applies `STANDARD_ASR_ALLOW_DOWNLOAD` before every network transfer;
- can perform a purely local copy, extraction, conversion, or verification when
  network downloads are disabled;
- delegates locking, resume behavior, temporary files, atomic replacement, and
  cross-process safety to the engine or its native library;
- returns a newly queried report after the operation.

The final report must contain every target passed to the acquisition hook, and
each target must be `ready`. Omitting a target violates `artifact_id` stability
and raises `EngineContractError`. A retained target that is still missing,
incomplete, corrupt, unknown, or in a future non-ready state raises
`ArtifactAcquisitionError(reason="failed")`, including when the target is
optional. A native API whose standalone cache inspection is opaque can use its
successful synchronous acquisition return as instance-local evidence for this
immediate final query. If it cannot establish readiness even then, explicit
acquisition cannot claim success.

If the report is not applicable, the operation is a no-op. With
`refresh=False`, it also returns when every reported requirement is ready.
Aggregate readiness alone does not skip an optional non-ready acquisition
target.

`refresh=True` means re-resolve a mutable branch, tag, model alias, or equivalent
source reference and acquire changed content. It does not mean retransferring
identical blobs. Refresh is per requirement. A pinned commit, digest, immutable
version, operator path, or other requirement with `source_is_mutable=False` is
a no-op, including when it appears beside a mutable Hub requirement. The engine
refreshes every mutable target it can refresh; the presence of an immutable or
external sibling never makes the whole operation fail.

An action-blocked mutable requirement is not reintroduced into the hook by
`refresh=True`; required and optional blockers keep the normal precedence and
final-report behavior below. `ArtifactAcquisitionError(reason="unsupported")`
is reserved for an unblocked requirement that actually has
`source_is_mutable=True` but whose engine cannot re-resolve that source. It is
not raised merely because a requirement has no mutable reference. When
`STANDARD_ASR_ALLOW_DOWNLOAD=0`, the template rejects any requested
mutable-source refresh before target filtering or the hook with
`reason="downloads_disabled"`: source re-resolution is a network metadata
request even when all blobs are already present. It must not silently skip the
lookup and claim freshness.

A deployment that needs deterministic prefetch still pins a revision:
an upstream mutable name can change again immediately after refresh. The CLI
spells the distinction as `pull` versus `pull --refresh`, and its help states
that plain `pull` does not check a ready floating source for updates.

An application can promise that the next inference does not acquire only when
every required artifact is ready, every effective
`may_acquire_during_inference` value is `False`, and any mutable source has been
pinned. A recent refresh alone is not a durability guarantee.

If no non-ready requirement can run now and at least one required requirement
is blocked, the operation raises an artifact acquisition error. If only
optional requirements are blocked, it returns the valid report. The blockers
select the machine-readable reason without guessing from the static upper bound
or the global download flag. When required blockers differ, `action_required`
takes precedence over `downloads_disabled`, which takes precedence over
`unsupported`. An unrecognized or `x_<vendor>_*` blocker participates in the
last `unsupported` bucket. The error retains every original blocker,
requirement, and action in its report while its closed `reason` remains
portable. This rule applies equally to missing, incomplete, corrupt, and
unknown requirements.

If some requirements can run and others are blocked, the operation acquires the
runnable set and queries final status. It then raises for any remaining blocked
required requirement, or returns with optional blockers intact. The caller
reads every requirement rather than assuming that a successful native download
made every conditional dependency ready.

A source service can reject acquisition for a prerequisite that status could
not discover without network access. The engine translates that failure into an
artifact acquisition error carrying newly discovered actions. It does not need
to mutate or fabricate the earlier status report.

The public `EngineBase` template owns the preflight report, progress boundary,
private engine hook, final status query, return-type check, and static/dynamic
subset checks. It passes the union of runnable non-ready requirements and
unblocked mutable refresh targets to the hook. The engine hook performs the
native acquisition and returns `None`.

The protected hooks are:

```python
def _artifact_requirements(
    self,
    context: ArtifactContext,
) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]: ...


def _acquire_artifacts(
    self,
    context: ArtifactContext,
    requirements: tuple[ArtifactRequirement, ...],
    refresh: bool,
    progress: ArtifactProgressCallback | None,
) -> None: ...
```

`EngineBase` merges parameter-gating diagnostics before engine inspection
diagnostics. A declaration with `applicable=False` gets the empty
requirements base implementation. A declaration with
`applicable=True` must override `_artifact_requirements`. A
declaration with `supports_explicit_acquisition=True` must also override
`_acquire_artifacts`. Missing required overrides are engine-contract defects.

### 9.3 Progress

```python
ArtifactProgressPhase = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]
ArtifactProgressUnit = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]

ARTIFACT_PROGRESS_RESOLVING: Final = "resolving"
ARTIFACT_PROGRESS_TRANSFERRING: Final = "transferring"
ARTIFACT_PROGRESS_EXTRACTING: Final = "extracting"
ARTIFACT_PROGRESS_CONVERTING: Final = "converting"
ARTIFACT_PROGRESS_VERIFYING: Final = "verifying"
ARTIFACT_PROGRESS_FINALIZING: Final = "finalizing"
ARTIFACT_PROGRESS_UNIT_BYTES: Final = "bytes"
ARTIFACT_PROGRESS_UNIT_FILES: Final = "files"


class ArtifactProgress(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: ArtifactProgressPhase
    artifact_id: str | None = None
    completed_units: int | None = None
    total_units: int | None = None
    unit: ArtifactProgressUnit | None = None


ArtifactProgressCallback = Callable[[ArtifactProgress], None]
```

The public template emits at least a resolving event before the engine hook and
a finalizing event before the final status query. An engine can emit more
events. An absent total produces indeterminate progress, never a fabricated
percentage. Events must be ordered and callback invocations must not overlap.
The callback execution thread is not portable, so an application callback must
be thread-safe.

The core wraps the callback before passing it to the engine. The wrapper
validates each event, serializes delivery with a lock, captures the first
callback exception, and suppresses later delivery. A callback failure is not a
cancellation request, so the native operation continues. After a successful
hook and final status query, the public method raises
`ArtifactProgressCallbackError` from the captured callback exception. An
acquisition or final-status failure takes precedence.

Progress counts are nonnegative. When both counts are present, completed units
cannot exceed total units. A unit is present exactly when at least one count is
present. Delivery order does not promise monotonic phases or byte counts unless
a later progress state machine specifies those guarantees.

The first version does not promise cancellation. Many native synchronous
downloaders cannot stop safely, and canceling an `asyncio.to_thread()` wrapper
does not stop its worker. A future cancellation contract needs an explicit
handle and post-cancel status semantics. Thread termination is not an acceptable
implementation.

The synchronous method and callback are deliberate first-version choices.
faster-whisper, Hugging Face, and Vosk expose synchronous acquisition entry
points, often with a callback or progress-class seam. A synchronous Standard ASR
operation can wrap all of them without claiming cancellation that the native
operation cannot honor. An async iterator cannot return the final report, and a
simple `acquire_artifacts_async()` implemented with `asyncio.to_thread()` would
mislead callers into thinking task cancellation stops the download. Applications
can run the synchronous method in their own worker and forward callbacks to an
event loop, but they must treat cancellation as detaching from observation, not
as stopping acquisition. A future async acquisition handle must define
cooperative cancellation and a separately awaitable final report before it is
added.

The callback exception rule is also intentional. Raising immediately would let
the caller leave while an uninterruptible native downloader keeps writing in an
unobserved worker. Capturing the first observer error, finishing the artifact
operation, querying final status, and then raising with that report means
`ArtifactProgressCallbackError` says exactly "the artifact operation succeeded,
but its observer failed." It is not an acquisition failure.

After an interrupted external process or native downloader failure, a later
status call can report `incomplete`, `corrupt`, `missing`, or `unknown`. It must
not assume that a failed operation left no files.

## 10. Relationship to `prepare()`

`acquire_artifacts()` and `prepare()` remain separate:

- `acquire_artifacts()` manages persistent artifacts and returns a report.
- `prepare()` performs process-local warm-up. It can load weights, initialize
  an accelerator, compile kernels, or perform a safe local priming operation.
- `prepare()` can acquire missing artifacts as part of warm-up when downloads
  are allowed.
- `standard-asr pull` calls `acquire_artifacts()`, never `prepare()`.
- `standard-asr prepare` keeps calling `prepare()`.

The separation lets an engine with a native download API populate persistent
storage without retaining a multi-gigabyte model in RAM. A native library that
cannot separate acquisition from loading can still load the minimum required
state, but no warm-up postcondition is portable and the engine releases
temporary loaded state when the native API permits it. The separation also
prevents a client-only remote engine's `prepare()` from being misread as a model
download operation when it only constructs an HTTP client.

The public artifact operation never fabricates audio. A recognizer whose only
safe acquisition path is genuine inference reports
`supports_explicit_acquisition=False` and
`may_acquire_during_inference=True`. Standard ASR discloses that limit and does
not issue a fake transcription.

## 11. Errors

The contract adds four public error types:

```python
class ArtifactStatusError(StructuredError):
    pass


class ArtifactUnavailableError(StructuredError):
    def __init__(
        self,
        message: str,
        *,
        reason: Literal[
            "missing",
            "incomplete",
            "corrupt",
            "unknown",
            "action_required",
            "downloads_disabled",
        ],
        report: ArtifactReport,
        hint: str | None = None,
    ) -> None:
        self.reason = reason
        self.report = report
        super().__init__(message, hint=hint)


class ArtifactAcquisitionError(StructuredError):
    def __init__(
        self,
        message: str,
        *,
        reason: Literal[
            "downloads_disabled",
            "action_required",
            "unsupported",
            "busy",
            "failed",
        ],
        report: ArtifactReport | None = None,
        required_actions: tuple[ArtifactAction, ...] = (),
        retriable_after: float | None = None,
        hint: str | None = None,
    ) -> None:
        self.reason = reason
        self.report = report
        self.required_actions = required_actions
        self.retriable_after = retriable_after
        super().__init__(message, hint=hint)


class ArtifactProgressCallbackError(StandardASRError):
    def __init__(self, message: str, *, report: ArtifactReport) -> None:
        self.report = report
        super().__init__(message)
```

`ArtifactStatusError` wraps an unexpected inspection failure and preserves the
native exception as `__cause__`. A predictable inability to inspect an opaque
cache is not an exception; the engine returns an unknown requirement and an
optional diagnostic.

`ArtifactUnavailableError` means a transcription or warm-up path cannot proceed
because a required artifact is not usable. It lets an application distinguish
"pull or complete an external action" from a generic engine failure.

`ArtifactAcquisitionError` means an explicit acquisition could not perform its
requested work, or an implicit inference acquisition failed before recognition.
Native exceptions remain in `__cause__`. `required_actions` can contain actions
learned from the failed source request when an offline status query could not
know them. The message, report, and actions must not contain credentials, signed
URLs, or raw validation-error input.

`retriable_after` replaces a weak retriable boolean. A nonnegative value says a
new operation or session can be attempted after that many seconds. `None` makes
no retry claim. It never makes the current streaming session recoverable.

The existing use of `DiscoveryError` for missing weights is replaced. Discovery
errors remain about plugin discovery and factory loading. Missing persistent
artifacts are a runtime availability state, not an entry-point discovery error.
Applications that currently catch `DiscoveryError` around `prepare()` or first
inference must migrate to the two artifact errors; applications that catch it
around registry discovery keep doing so. `errors.md`, the exception tree, CLI
arms, and examples make that split explicit.

The public template uses this failure precedence:

1. A preflight status failure raises `ArtifactStatusError`; acquisition does not
   start.
2. A native acquisition failure raises `ArtifactAcquisitionError`; the core
   does not mask it with a failing final query. The application can call status
   again.
3. After a successful hook, a final status failure raises
   `ArtifactStatusError`. The message states that acquisition might have
   succeeded but its final state is unknown.
4. A captured progress-callback failure raises
   `ArtifactProgressCallbackError` only after the native hook and final status
   both succeed. An acquisition or status failure takes precedence.

An engine raises `ArtifactUnavailableError` when it knows before recognition
that required artifacts cannot resolve and it cannot acquire them under the
current policy. It raises `ArtifactAcquisitionError` when an allowed implicit
acquisition was attempted and failed. These artifact errors are explicit
exceptions to the batch R7 `TranscriptionError` wrapper: they describe a
pre-inference availability operation, even when a native loader is called from
inside `_transcribe`. All other recognizer execution failures retain the R7
mapping. `prepare()` uses the same distinction.

`EngineBase` does not call `artifact_status()` before every transcription. A
universal preflight would add filesystem work to the hot path and could block a
valid implicit acquisition or fallback. Instead, each engine's native loading
guard translates a known missing, incomplete, or corrupt dependency into
`ArtifactUnavailableError`. A failed allowed implicit acquisition becomes
`ArtifactAcquisitionError`.

If `prepare()` can materialize persistent artifacts without genuine inference,
the engine declares explicit acquisition and implements the same materialization
through `_acquire_artifacts`. `prepare()` can then acquire and continue into
process-local warm-up. An engine whose only acquisition path is genuine
inference does not use fake audio in either method.

The reference server maps both artifact errors to a scrubbed HTTP 503 and a
`service_unavailable` pre-bridge streaming frame. This expands the normative
503 condition beyond `ConfigurationRequiredError` to operator-side
configuration or artifact unavailability. It never serializes the attached
local paths or actions to an unauthenticated wire client. A wire client cannot
repair the server's deployment state through a transcription request.

For streaming, a failure during session establishment maps to the existing
`service_unavailable` handshake error. A failure discovered after the session
starts produces a terminal engine event with code `artifact_unavailable` or
`artifact_acquisition_failed`. It is not the generic `engine_error`. The event
does not carry the local report, path, or action URL. An application must open a
new session after fixing the artifact state; the current session is not
recoverable.

Both new §6.2 codes set `recoverable=false`. `artifact_unavailable` sets
`retriable_after=null`. `artifact_acquisition_failed` can project a nonnegative
`retriable_after` from the error, but still terminates the current session. The
producer catches both artifact exceptions before the blanket `Exception` arm in
`streaming.py`; it constructs only the safe code fields and a scrubbed operator
detail. The producer must not place the report, path, action URL, or native
exception text in the event. The reference server later drops the operator
detail as it does for every error event.

## 12. CLI and application surface

### 12.1 Python

An application follows this flow:

```python
registry = discover_models()
engine = registry.create("faster-whisper/large-v3")

report = engine.artifact_status()
if any(
    requirement.can_acquire_now and requirement.state != ARTIFACT_READY
    for requirement in report.requirements
):
    report = engine.acquire_artifacts(progress=on_progress)
```

Applications display each required action from the report. They do not parse
exception messages, inspect cache directories, or call a recognizer-specific
downloader.

### 12.2 CLI

The first CLI version adds:

- `standard-asr status <engine/model>`;
- `standard-asr pull <engine/model>`.

Both commands accept the existing init-config flags. An omitted mode uses the
resolution rule in §8.1, and parameters default to `RuntimeParams()`. This
requires new shared parser infrastructure:
today `--json` and init-config options are not reusable across arbitrary
commands. Phase B extracts that infrastructure rather than claiming it already
exists. A later CLI extension can expose request-context flags without changing
the Python contract.

`status` exits with 0 when it produced a valid report, including a missing,
external, or unknown report. `--require-ready` instead exits with 1 unless
aggregate readiness is `ready` or `not_applicable`. `--json` emits the canonical
report for scripts; human diagnostics and notices remain on stderr.

`pull` calls `acquire_artifacts(refresh=False)`. `pull --refresh` passes
`refresh=True`; it re-resolves mutable sources but does not force identical
blobs to transfer again. Success means every required requirement is ready and
no attempted acquisition failed. An optional artifact that cannot be acquired
produces a warning and exit 0. A failed attempt remains a failure even when its
target was optional.

CLI results follow the existing fault-ownership rule:

| Result | Exit |
| --- | ---: |
| Valid `status` report | 0 |
| `status --require-ready` and required artifacts are not ready | 1 |
| `pull` leaves all required artifacts ready; optional artifact remains unavailable without an attempted failure | 0, with warning |
| A required artifact is blocked because downloads are disabled or a known external action is required | 2 |
| `pull --refresh` finds a mutable target while downloads are disabled | 2 |
| Protocol 1.0 plugin must be upgraded for artifact tooling | 2 |
| Native acquisition failure, busy acquisition, unsupported required acquisition, status failure, or progress observer failure | 1 |

`main()` gets explicit arms for `ArtifactStatusError`,
`ArtifactUnavailableError`, `ArtifactAcquisitionError`,
`ArtifactProgressCallbackError`, and the Phase A0 protocol-compatibility error.
It does not let the new errors fall through the bare `Exception` arm. For the
first two artifact error families, `downloads_disabled` and `action_required`
for a required artifact map to exit 2; engine/native failures map to exit 1.
Optional blockers return a report and warning instead of raising. `prepare` uses these same
arms, so replacing its current `DiscoveryError` does not silently change its
exit code.

Both operations are noninteractive library calls. `pull` does not read stdin,
accept terms, authenticate an account, or open a browser. It prints known
actions and returns the documented exit code.

`list` remains exactly discovery-only. It does not add metadata fields, resolve
an entry point, import a plugin, instantiate a model, or touch 30 cache roots.
An application can query selected models outside its UI thread. `show` resolves
one plugin and renders the full static artifact declaration behind the existing
metadata fault boundary.

The existing `cache` and `prepare` commands remain distinct. The current
`cache` wording is corrected: it reports only the Standard ASR fallback cache
root and does not claim that every engine uses that directory. Effective
artifact locations come from instance reports when the engine can provide
them.

The Python library retains the published default that an unset
`STANDARD_ASR_ALLOW_DOWNLOAD` permits downloads. It also retains no universal
status preflight on the hot path. The lifecycle API therefore does not, by
itself, eliminate first-use acquisition. The `transcribe` CLI performs one
status preflight for its selected model. When a required artifact is non-ready
and inference can acquire it, stderr says that first transcription can acquire
artifacts and points to `standard-asr pull`. If status is unknown but the
static or effective declaration allows inference acquisition, the wording says
"may acquire" rather than claiming a download will occur. The notice does not
block the existing default. This preflight is advisory: an unexpected
`ArtifactStatusError` emits a scrubbed warning and transcription continues. It
must not turn an introspection defect into a new inference outage. Caller input,
configuration, and protocol compatibility errors still follow their normal
fail-loud paths. An operator that needs a hard guarantee sets
`STANDARD_ASR_ALLOW_DOWNLOAD=0` and prefetches first.

### 12.3 Reference server

The first version adds no pull, status, or removal endpoint. Comparable servers
such as Speaches ship model download and removal endpoints, so this is a
deferral, not a claim that management is inherently outside an inference
server. A remote management endpoint can consume disk, trigger large downloads,
or disclose operator filesystem paths. The reference server has no operator
authorization, role separation, or deployment-ownership policy yet.

The server adds `GET /v1/metadata/{model}` and the 503 mappings for
`ArtifactUnavailableError` and `ArtifactAcquisitionError`. It does not change
bulk `GET /v1/models`. Operator-side management uses the Python API or local
CLI. Issue #7's daemon-hub decision is the forcing function for authenticated
management endpoints: if the reference server becomes the local lifecycle
owner, the endpoint and operator-auth design must be revisited together.

## 13. Compliance

Default, side-effect-free compliance checks inspect declarations and call
shapes only:

- The engine class carries a valid `declared_metadata.artifacts` section for
  protocol 1.1. Its MRO owner is plugin code, not a core placeholder; preset
  subclasses can inherit it from their plugin-owned base class.
- The declaration invariants hold.
- The full engine surface exposes synchronous `artifact_status()` and
  `acquire_artifacts()` methods with the specified signatures.
- An `EngineBase` model that declares explicit acquisition overrides the native
  acquisition hook.

Default compliance never calls either artifact operation. Calling a public hook
to prove that it has no side effects can trigger the exact side effect in a
noncompliant engine. This follows the side-effect boundary tracked in #53.

The implementation order is normative because this design depends on contracts
that do not exist yet:

1. Land #53 Part 1 first-class compliance outcomes.
2. Land protocol-version validation and artifact declaration/call-shape checks.
3. Land #53 Part 2's side-effect-free default and the shared `--runtime` flag
   namespace with #33.
4. Add artifact status to the `--runtime` profile. A skipped or unobserved path
   reports `skip` or `unknown`, never an unqualified pass.
5. Add real acquisition behind a separate `--acquire-artifacts` opt-in. That
   flag can use network and disk and never becomes an implicit part of
   `--runtime`.

#33's happy-path and R7 probes are revised in the same wave: a required artifact
that is unavailable or whose allowed implicit acquisition fails is a typed
artifact outcome, not a missing `TranscriptionError` wrapper. All other engine
execution failures remain R7 failures.

An opt-in runtime profile can call `artifact_status()`. It checks the following
observable properties, without claiming to prove arbitrary third-party code has
no hidden side effects:

- the return value and every nested data model are valid;
- a model declaring `applicable=False` returns a not-applicable
  report with an empty requirement tuple;
- dynamic requirements do not widen the static declaration;
- common Python socket and filesystem-write traps observe no prohibited work.

The trap result is evidence about the observed Python path, not proof about
native code, subprocesses, operating-system APIs, or network-attached
filesystems.

Real acquisition is a second, explicit opt-in because it can use network and
disk. When `--acquire-artifacts` is enabled for a configured model, it checks
only behavior that the
actual path reaches:

- progress values are valid and callback calls do not overlap;
- the final report is queried instead of assumed;
- returned data and raised artifact errors satisfy the public contract.

Compliance does not prove cache integrity, download an entire production model
matrix, accept terms, or test a paid inference endpoint.

Core unit tests use fake engines and temporary artifacts for every protocol
branch. Plugin unit tests remain responsible for native-library integration.
They use temporary directories and mocked downloaders for missing, ready,
incomplete, corrupt, gated, disabled, and concurrent acquisition paths.
Those controlled tests own repeated-status equality, download-toggle behavior,
idempotency, callback failures, native cause wrapping, and inference-error
translation. Generic compliance does not claim it forced an arbitrary
production engine through each branch.

## 14. Concurrency and safety

- Repeated acquisition is safe.
- Concurrent calls must not corrupt artifacts. An engine can coalesce, block on
  its native lock, or raise `ArtifactAcquisitionError(reason="busy",
  retriable_after=0)`. It must not run unsafe parallel writes merely because two
  applications requested the same model.
- Cross-process locking and atomicity belong to the engine or native library.
- Status remains read-only while another process acquires. It can report
  `incomplete` or `unknown` when it cannot observe a stable snapshot.
- Readiness is not monotonic. An operating system can reclaim a managed speech
  asset, an operator can delete a cache, a mounted path can disappear, and
  access to a gated source can be revoked. Every report is a point-in-time
  observation.
- Manual-action URLs use HTTPS and contain no embedded credentials.
- Paths and sizes are operator information. The first version does not expose
  them through an unauthenticated network endpoint.
- `STANDARD_ASR_ALLOW_DOWNLOAD=0` blocks network acquisition in explicit and
  inference paths. It does not prohibit reading, verifying, copying, extracting,
  or converting artifacts already supplied locally.
- A long local extraction or conversion is permitted under that setting only
  outside the pure constructor. An explicit `pull` already authorizes artifact
  work. An inference path can perform inseparable local materialization only
  when it declares `may_acquire_during_inference`; the CLI notice makes that
  possible cost visible. The download toggle remains a network-transfer policy,
  not a generic CPU or disk-work kill switch.
- An engine never logs a credential, signed URL, raw native exception chain, or
  validation-error input through artifact diagnostics.

## 15. Representative mappings

### 15.1 `std-faster-whisper`

Static declaration: `True, True, True`.

- Current reality: the plugin constructs `WhisperModel(...)`; that upstream
  constructor can download and load. `prepare()` triggers the same combined
  path. The plugin does not currently call `download_model()` itself.
- Phase D: the default Hub preset reports one recognizer requirement and uses
  upstream `download_model()` for artifact-only acquisition. Status uses a
  cache-only native resolution for the effective cache selection and revision.
  The common zero-config cache selection is the deliberate `download_root=None`
  passthrough to the Hugging Face default, not a configured directory.
- `model_path`: report an externally provided requirement.
  `can_acquire_now=False`.
  Report ready, missing, corrupt, or unknown from the directory checks that
  faster-whisper already needs before loading.
- `local_files_only=True`: effective
  `may_acquire_during_inference=False`, even though the static upper bound is
  `True`.
- `prepare()` continues to load the CTranslate2 model after acquisition.
- The upstream helper suppresses transfer progress and has no force-download
  option. Initial progress is therefore indeterminate. Repairing known corrupt
  content can require a direct Hub snapshot call rather than pretending the
  helper supports it.

### 15.2 `std-mlx-audio`

Static declaration: `True, True, True`.

- Current prerequisite: fix and test download-policy enforcement,
  `download_root`, `hf_token`, and `dtype`. Most current upstream loader paths
  ignore the forwarded `local_files_only`, and the plugin discards the resolved
  root. Until those defects are fixed, status and inference do not share an
  honest resolved context.
- Phase D default Hub preset: use snapshot acquisition without calling
  `model.generate`, with explicit root, native default, revision, token, and
  downloads-disabled integration tests.
- A preset with a checkpoint subfolder includes that subfolder in the logical
  requirement.
- `model_path` becomes an externally provided requirement.
- The current `prepare()` retains its MLX load and local priming behavior. `pull`
  never performs that priming inference.
- The current `Mms1BAll` preset has no language-adapter selector. It downloads
  one filtered `facebook/mms-1b-all` snapshot containing the base and all
  safetensors adapters, approximately 14.6 GB under the supported upstream
  loader patterns. It therefore reports one logical snapshot requirement.
  Splitting it into request-dependent adapter requirements is a future plugin
  optimization that first needs a language selector and selective acquisition;
  it is not the current architecture.

### 15.3 `std-qwen3-asr`

Static declaration: `False, False, False`.

DashScope and the separately deployed vLLM service own their artifact
lifecycles. The engine only constructs a client and requests a named
server-side model. The report is not applicable. This remains true when vLLM
runs on loopback or serves open weights. A different plugin that owns a vLLM
sidecar lifecycle can make the lifecycle applicable without changing this
topology-independent rule.

### 15.4 Future manual-file engine

A minimal whisper.cpp engine can declare `True, False, False`. It
reports a missing external requirement plus a `provide_artifacts` action and
the `action_required` blocker, so CLI pull exits 2. If a plugin later wraps the
official downloader safely, it changes the second value to `True` and sets
`can_acquire_now=True` for the default config.

## 16. Edge-case decisions

| Case | Decision |
| --- | --- |
| Hub model supports explicit download and automatic load-time download | Declare both independent facts. The effective requirement can acquire now and can also acquire during inference. |
| User selects a local path instead of the preset source | The static class declaration stays broad. The instance reports an external requirement for that path. |
| Terms must be accepted before a token can download | Report an action and set `can_acquire_now=False`. The static declaration preserves support after the action. Standard ASR never accepts the terms. |
| Access awaits manual approval | Report `request_access`; a later offline status can remain unchanged until acquisition retries. A rejected request is terminal unless the source owner changes it, and previously accepted access can be revoked. |
| Only genuine inference can trigger acquisition | Declare no explicit acquisition and disclose inference acquisition. Never send fake audio. |
| Loader acquisition cannot avoid loading weights | Explicit acquisition can perform the minimum inseparable loading, but its only portable postcondition concerns persistent artifacts. |
| Model is bundled in a wheel or container | No separate artifact acquisition lifecycle; report not applicable. A broken installation is a plugin fault. |
| Model is served by an independent local process | Report not applicable. Process proximity does not transfer lifecycle ownership. |
| Plugin owns a local sidecar and its model lifecycle | Acquisition is applicable even though another process stores or loads the artifacts. |
| Model path is on NFS or another shared mount | Report the external requirement and observed state. Do not claim offline behavior or bounded status latency. |
| Several presets share one snapshot or blob | Each model reports its logical requirement. IDs and sizes do not authorize generic deletion or promise additive disk use. |
| One preset needs base, VAD, punctuation, aligner, or diarizer artifacts | Return the resolved dependency closure as several requirements. |
| An aligner depends on auto-detected language | Return an unknown conditional requirement until the language is known. Do not download every language model. |
| Revision is a floating branch | A cached snapshot can be ready while inference still checks for and downloads an update. Keep inference-acquisition disclosure true. |
| User asks for `refresh=True` | Re-resolve only requirements with `source_is_mutable=True`; immutable and external siblings are no-ops. Do not force-transfer identical pinned blobs. A mutable target with downloads disabled raises `downloads_disabled`. |
| Cache has an interrupted temporary file | Report incomplete when the engine can identify it; otherwise report unknown. |
| Files exist but checksum or manifest validation failed | Report corrupt. Do not collapse it into incomplete. |
| Native downloader has no byte totals | Emit indeterminate progress with no total. Do not fabricate a percentage. |
| Native downloader cannot cancel safely | Do not promise cancellation in the first version. Never terminate its thread. |
| Acquisition was interrupted | Query status again. Do not assume missing or roll back files the core does not own. |
| Downloads are disabled but a local archive can be extracted | Permit the local operation. The toggle controls network downloads, not all filesystem materialization. |
| Engine cannot inspect an opaque native cache cheaply | Report unknown. Unknown never means ready. |
| Operating system reclaims a managed asset | A later ready report can regress to missing or unknown. Readiness is never a permanent lease. |
| Missing local optimization falls back to a remote path | Mark the local artifact `required_for_inference=False`, or resolve the selected path through config. Non-ready does not automatically mean unavailable. |
| Several physical layouts are interchangeable | Represent them inside one logical requirement after the engine resolves the alternatives. Do not expose an OR-expression language. |
| Durable converted weights or accelerator plan | Include them when the configured engine lifecycle treats them as a reusable inference artifact. |
| Ephemeral per-call transfer or opaque JIT cache | Exclude it from the first-version artifact report. It is not a durable inference artifact managed by this lifecycle. |

## 17. Rejected alternatives

### 17.1 `local | cloud | hybrid`

Rejected because the values entail unrelated assumptions and classify the
deployment rather than the artifact operation this feature owns.

### 17.2 One mutually exclusive acquisition mode

Rejected because explicit acquisition, automatic inference acquisition, and
external prerequisites coexist. Three orthogonal declarations plus dynamic
actions represent the combinations without an `other` or `hybrid` bucket.

### 17.3 `is_downloaded: bool`

Rejected because it cannot represent incomplete, corrupt, unknown, conditional,
external, or not-applicable states.

### 17.4 Class-level cache status

Rejected because init config changes the cache root, revision, credentials,
local path, recognizer source, and artifact closure. Only static upper-bound
facts belong on the engine class.

### 17.5 Core-specific Hub or cache adapters

Rejected because they couple the standard to a storage implementation and still
cannot cover archives, operating-system assets, conversions, or custom paths.

### 17.6 Triggering a fake transcription

Rejected because it can bill a user, create remote state, alter usage records,
or run an unintended recognizer path. An engine reports the absence of explicit
acquisition instead.

### 17.7 Making `pull` an alias for `prepare`

Rejected because persistent acquisition and process-local warm-up have different
postconditions, costs, and lifetimes.

### 17.8 Removal in the first version

Rejected because shared blobs, revisions, operator-provided paths, and
deduplicated caches need a separate plan-and-confirm contract. Status and
acquisition solve the immediate application gap without destructive behavior.

### 17.9 A remote management endpoint

Deferred for the first version. Speaches proves that model management endpoints
are useful and shippable. Standard ASR's reference server does not yet have the
operator authorization, role separation, deployment ownership, or disk policy
needed to expose them safely. Issue #7's daemon-hub decision can change that
ownership boundary.

### 17.10 Artifact lifecycle inside the capability tree

Rejected even though behavior nodes can be informational. The lifecycle facts
are consumed before and outside transcription, and adding them to
`DeclaredCapabilities` would make `supports()` cover a management operation.
The typed `declared_metadata` aggregate gives this family, #19, and the static
half of #8 one wire-visible home without weakening `BaseProperties` or creating
one ClassVar per feature.

### 17.11 Async acquisition in the first version

Deferred until the protocol can provide an acquisition handle with cooperative
cancellation and an independently awaitable final report. A thin
`asyncio.to_thread()` method would not stop native work when canceled. An async
iterator cannot return the final `ArtifactReport`. The synchronous operation and
progress callback match the native APIs this version must wrap.

## 18. Disposition of the superseded draft

The locality-based draft contained useful work in addition to the rejected
classification. Nothing disappears implicitly:

| Earlier item | Disposition |
| --- | --- |
| Local/cloud/hybrid classification and the related CLI filters | Rejected as a bundled preset. Network use, audio egress, credentials, operator ownership, and usage billing are separate facts. #28 must remove the stale locality facet; each fact needs its own evidence and owner before becoming catalog metadata. |
| “Will this download on first use?” | Delivered by static inference-acquisition disclosure, dynamic status, explicit pull, and the CLI preflight notice. The library default remains download-enabled. |
| `cache_status()` / downloaded boolean | Replaced by configured-instance `artifact_status()` and a dependency closure with missing, incomplete, corrupt, and unknown states. |
| Dynamic status and size columns in bulk `list` | Scope-reduced and deferred to #7 plus a future multi-model scanning helper. Bulk list must stay import-free; selected-model status belongs in `status` and application-owned background work. |
| Download directory introspection | Delivered only as optional per-requirement `location`, after config resolution. A class-level directory would be wrong for native defaults, custom paths, and shared stores. |
| `download_backend` label | Rejected. A backend label does not grant portable control and quickly becomes another locality-like preset. Required actions, locations, and effective operations expose the facts an application can act on. |
| Removal and reclaimed bytes | Deferred to a separate plan-and-execute deletion design. Shared blobs make a direct `remove_cache()` unsafe. #27 retains the roadmap item until a dedicated issue is approved. Current Hugging Face spelling is `hf cache ls/rm/prune`; the superseded draft's `scan-cache` and `delete-cache` names are legacy. |
| `pull` as `prepare` | Rejected. Persistent artifact acquisition and process-local warm-up remain distinct. `pull --refresh` supplies the update semantic missing from the earlier draft. |
| Proxy and mirror day-0 minimum | Split from this protocol feature because it does not depend on artifact APIs. The follow-up must document `HF_ENDPOINT` and standard proxy passthrough, add plugin passthrough tests, and add bounded `doctor` guidance for an unreachable Hugging Face endpoint. It must not automatically select or trust a third-party mirror. A dedicated GitHub issue is opened only after this design is approved; external issue mutation is intentionally outside this approval draft. |
| Standardized proxy or endpoint config | Deferred. One knob cannot faithfully cover Hugging Face, ModelScope, direct URLs, and native operating-system acquisition. It requires its own security and integrity design. |
| “Cloud or billed model?” | Not answered by artifact readiness. #28 and #27 must replace their stale locality wording with separate future questions: whether inference needs a network, whether audio leaves the controlled process boundary, and whether the engine can incur usage charges. This plan defines none of those fields. |
| Standard artifact path helper from IC.8 | The unimplemented promise is withdrawn as authority and becomes an optional convenience helper only. Engines own readiness, layout, and checksums. This protocol correction is an explicit approval item, not a silent deletion. |

## 19. Implementation plan after approval

### Phase A0: Version and prerequisite contracts

1. Define and test protocol-version parsing, the protocol 1.1 feature floor,
   the core 0.2.0 release relationship, and the compatibility error.
2. Land #53 Part 1 before changing compliance result semantics.
3. Reconcile IC.9 and download-policy cache precedence, including the native
   `None` passthrough.
4. Move the reusable `_JsonExtraModel` mechanism out of
   `contract/capabilities.py` into an internal shared contract module; do not
   import a private capability implementation across subject modules or expose
   that plumbing at the package top level.
5. Record the capability-kinds, streaming-cadence, #8, and #19 metadata
   decisions in one architecture note so no sibling wave creates a fifth
   surface.

### Phase A: Core contract

1. Add `DeclaredEngineMetadata` and its required `ArtifactDeclaration` section.
2. Add `contract/artifacts.py` with the context, open code vocabularies,
   requirement, report, and progress data models.
3. Add the two public operations to `StandardASR` and the template behavior to
   `EngineBase`, including best-effort introspection gating and refresh.
4. Add artifact status, availability, acquisition, and callback errors and
   application-facing top-level exports. `ArtifactContext`, report and
   requirement types, progress types, code constants, and errors are top-level;
   the shared `ModeName` alias also becomes top-level because it appears in the
   app-facing context. Protected hooks remain engine-author internals.
5. Add protocol-boundary return checks and static/dynamic subset checks.
6. Add unit tests for every state, aggregate readiness rule, declaration
   invariant, error path, callback boundary, refresh path, and context gating
   path.

### Phase B: Toolchain, streaming, and compliance

1. Keep CLI `list` and `GET /v1/models` byte-for-byte import-free in behavior;
   add metadata only to `show` and `GET /v1/metadata/{model}`.
2. Add reusable JSON and init-config CLI parsing, then implement `status`,
   `status --require-ready`, `pull`, and `pull --refresh` with the exit table in
   §12.2.
3. Add the CLI transcribe preflight notice without changing the library's
   download default.
4. Add explicit CLI exception arms, reference-server 503 mappings, streaming
   producer carve-outs, and the two §6.2 terminal codes.
5. After #53 Part 2 and #33 settle `--runtime`, add static compliance, runtime
   status, and separate `--acquire-artifacts` profiles in the order in §13.
6. Use fake engines and temporary artifacts for native failures, progress,
   refresh, optional requirements, and concurrent acquisition.

### Phase C: Specification, reference, and issue migration

1. Rewrite protocol IC.8, IC.9, and IC.11; RT R7; streaming §6.2; the engine
   metadata table; and version semantics.
2. Update the normative REST 503 table and WebSocket handshake mapping. Explain
   the coarse pre-bridge `service_unavailable` versus distinct post-bridge
   artifact codes.
3. Replace stale `DiscoveryError` use in `download-policy.md`,
   `EngineBase.prepare()`, `errors.md`, CLI help, and application guidance.
   Correct `EngineContractError` prose that currently calls `prepare()` a
   `StandardASR` member before the protocol actually contains it.
4. Add `docs/content/reference/artifacts.md` and update engine-author and
   application-developer guides. Correct the `cache` command description.
5. Add `inference artifact`, `artifact acquisition`, `artifact status`,
   artifact state/readiness, and pull to `TERMINOLOGY.md`; add the new code
   vocabularies to its controlled-vocabulary section.
6. Amend goals G.1.3 and G.5.2 for `declared_metadata` and its per-model wire
   projection. Do not add artifact fields to `ModelInfo`.
7. Update #27, #28, #8, #19, #33, #35, #7, and the streaming-cadence design
   wherever their wording or dependency order becomes stale. Create the
   dedicated proxy/mirror issue after this design is approved.

### Phase D: Coordinated plugin rollout

1. Implement faster-whisper cache-only status and upstream artifact-only
   acquisition. Test the native default cache, explicit root, revision, local
   path, disabled downloads, corrupt content, and indeterminate progress.
2. Before adding MLX status, fix `local_files_only` enforcement,
   `download_root`, `hf_token`, and `dtype`; then implement snapshot status and
   acquisition with native integration tests for every resolved-config axis.
3. Model current MMS as one filtered snapshot. Do not claim dynamic adapters
   until the plugin actually exposes selection and selective acquisition.
4. Declare Qwen3-ASR artifact acquisition not applicable for both DashScope and
   client-only vLLM presets.
5. Bump every first-party plugin to protocol 1.1.0 with a core 0.2.0 dependency
   floor only after its whole contract passes.
6. Re-verify every upstream call shape against the exact dependency versions
   locked for Phase D. The design evidence uses moving upstream references and
   does not substitute for release-time source verification.

### Deferred follow-ups

- safe removal through a plan-and-execute contract;
- authenticated operator endpoints, revisited with #7's daemon hub;
- cooperative cancellation handles and an async event stream;
- multi-model status scanning helpers and #7 list UX;
- richer remaining-download estimates that explicitly permit network access;
- separate network-use, audio-egress, and usage-billing declarations;
- standardized proxy and mirror configuration after the day-0 passthrough
  documentation and tests.

## 20. Approval gate

Implementation requires approval of these decisions:

1. Use inference-artifact lifecycle ownership, not locality, as the boundary.
2. Add one typed `declared_metadata` surface shared with the future static
   portions of #8 and #19; keep lifecycle facts out of Properties and
   Capabilities.
3. Keep bulk list surfaces import-free; expose static metadata only through
   `show` and a per-model wire endpoint.
4. Use three orthogonal static facts and configured-instance status with
   explicit applicability and a dependency closure.
5. Make status introspection best-effort for portable parameters while keeping
   wrong-engine provider params fail-loud.
6. Keep acquisition synchronous, separate from `prepare()`, and add explicit
   refresh semantics rather than redefining ready as fresh.
7. Use the CLI exit ownership table: user actions and disabled downloads are
   exit 2; native failures are exit 1; unavailable optional artifacts can warn
   and succeed.
8. Keep first-use downloads allowed by default, add a CLI notice, and define
   `STANDARD_ASR_ALLOW_DOWNLOAD` as a network-transfer gate rather than a ban on
   explicit local conversion.
9. Defer removal, cancellation, and authenticated remote management endpoints.
10. Replace missing-weight `DiscoveryError` use with artifact-specific errors
    and amend batch R7, streaming §6.2, server mappings, and migration docs.
11. Target protocol 1.1.0/core 0.2.0 with the coordinated bump procedure in
    §7.1.
12. Replace IC.8's unimplemented authoritative path/existence/checksum helper
    with engine-owned inspection; retain any generic path helper only as
    optional convenience.

No public API implementation or external issue mutation starts until this gate
is approved.
