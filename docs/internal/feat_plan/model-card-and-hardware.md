# Model card, hardware declaration, and preflight (design case)

**Status:** Draft of 2026-10-22 for item 11 of the roadmap (issue #27). Nothing here is implemented. Item 13 implements the three sections and item 14 the preflight. Section 12 lists the decisions the maintainer has to make.

Commits read: `standard_asr` 31612eb (`origin/main`), std-faster-whisper ef23a00, std-mlx-audio 080beb1, std-qwen3-asr 7b0bc54. Line numbers below refer to these commits.

## 1. Problem

Before an application downloads or loads a model's weights, it cannot tell what the model is or whether it runs on this machine. It learns an engine's hardware needs, and whether using it sends audio to a paid service, only after an install, a download, or a crash inside a native library.

Three requests meet here. Issue #19 asks for a model card: the identity of a model's weights, which is narrower than the model card of a Hugging Face repository. Issue #8, part A, asks for a hardware declaration. The design of the artifact lifecycle left three facts open (`docs/internal/feat_plan/model-management.md`, lines 147 to 149 and 1558): whether inference needs the network, whether audio leaves the process, and whether use can be charged. Roadmap item 14 adds the preflight: `doctor`, the CLI's diagnostic command, compares what the machine offers with what the selected engine declares.

The roadmap puts five questions to this design. A wrong answer to any of them makes the card lie.

1. Does each fact describe the engine, the model preset, or the configuration in use?
2. Which numbers are facts and which are estimates, and when does an estimate hold?
3. How is "unknown" answered, so that it never reads as "supported"?
4. What does an engine that runs no accelerator of its own declare?
5. Does inference need the network, does audio leave the machine, and can use cost money?

## 2. What exists today

- **`declared_metadata`** is one class variable that holds a `DeclaredEngineMetadata` (`src/standard_asr/contract/metadata.py`, lines 81 to 93). Its one section, `artifacts`, is required. Unknown sections with JSON values survive parsing and serialization, so a later protocol version can add a sibling section without breaking an older reader. A plugin's own base class can share one declaration with its subclasses (`docs/content/specification/protocol.md`, line 868). `GET /v1/metadata/{model}` and `standard-asr show` read it from the engine class, so an application reads it with nothing constructed, downloaded, or loaded.
- **The artifact section's pattern.** "Each field is an upper bound over every supported configuration and request context. The configured engine reports narrower effective behavior through its artifact status report" (`metadata.py`, lines 18 to 20). An upper bound here means the most the engine can do under any configuration it supports. A report that widens the declaration raises `EngineContractError` (AR.2, `protocol.md` line 882). The section has no defaults, because "a default of `False, False, False` would turn an unupdated engine into a false 'no artifact acquisition' claim" (`model-management.md`, lines 314 to 316).
- **Placement is decided.** The model card and the `hardware` section are sibling sections of `declared_metadata`. They are not new class variables or capability nodes (`docs/internal/feat_plan/engine-metadata-surface.md`, lines 29 to 33). They are not `BaseProperties` fields either: the protocol keeps those for identity and I/O facts (`protocol.md`, line 80). The comment on #19 that settles the placement adds: "the model card owns identity and provenance, #8 owns hardware, no field appears in both, and anything measured on the current machine stays in doctor and compliance output." After the first public release, an optional section costs a protocol patch release and an entry in `PROTOCOL_FEATURE_MINIMUMS`; a required one needs a new 0.x line.
- **A locality label was rejected.** `local`, `cloud`, and `hybrid` bundle unrelated facts (`model-management.md`, section 4). "Network use, audio egress, credentials, operator ownership, and usage billing are separate facts" (line 1548), and each "gets its own property and semantics" (line 148).
- **Configuration changes the facts.** std-faster-whisper's `model_path` loads other weights, `device` picks `cpu`, `cuda`, or `auto`, and `compute_type` changes the precision of the computation (`_config.py`, lines 51, 86, and 100). Its `revision` picks another version of the same Hugging Face repository (line 107). std-mlx-audio runs models through MLX, Apple's array framework for Apple silicon. It has `model_path` and `revision` too, and no `device` field: "MLX runs on Metal unconditionally" (`README.md`, line 229). Each std-qwen3-asr preset sets only the default of `backend` (`engine.py`, line 116). An explicit `backend="dashscope"` replaces it without a check, so every preset, the open-weight ones included, can send audio to DashScope, Alibaba Cloud's paid hosted API.
- **Two vocabularies say "device".** The configuration's `device` is a string the engine passes to its native library. The core's `DeviceConfigMixin` gives `"cpu"`, `"cuda"`, and `"mps"` as examples (`src/standard_asr/runtime/config.py`, line 1976), and `"mps"` is PyTorch's name for Apple GPUs.
- **What the machine shows cheaply.** The standard library reports the operating system, the architecture, and the number of CPUs, and on Linux and macOS the total memory; Windows needs a system call through `ctypes`. It cannot tell whether an accelerator works without loading the vendor's native code, and the preflight must start no accelerator runtime (roadmap item 14).

## 3. The answers in short

1. **Granularity.** Every class-level declaration belongs to one engine class, and this design requires that an engine class serve one model preset. IC.7 implies it, since model selection is the entry-point preset (`protocol.md`, line 809), and the first-party plugins already do it: a plugin gives each preset whose facts differ its own class. The model card describes the preset's default weights. The other two sections are upper bounds over every configuration and request the preset supports. A configured engine instance narrows them through a new property, `effective_metadata` (section 4.2).
2. **Facts and estimates.** The card's identity fields are facts the author states. Memory is an estimate: it always names the conditions it assumes and how it was obtained, and nothing ever gates on it (section 4.3).
3. **Unknown.** The new sections are optional, and so is every field in them. A missing section or field is unknown, and a default can only be unknown. No consumer reads unknown as supported, as compatible, or as "no" (section 4.1).
4. **No accelerator of its own.** An engine that only calls an external service declares `hardware.applicable = false` (section 6).
5. **Network, audio, money.** Three separate booleans in the section `external_service`, each with its own definition (section 7). `sends_audio` answers a narrower question than whether audio leaves the machine: it is true when audio reaches a process the engine did not start, a vLLM server on loopback included (section 4.5).

## 4. Rules shared by the new sections

### 4.1 Unknown

A missing section means that every fact in it is unknown. A missing field, or `null`, means that one fact is unknown. A field may have a default only when the default is unknown. One field is required inside its section: `hardware.applicable`, because the meaning of every other hardware field depends on it (section 6).

The artifact section is required because a default there would assert something false. An unknown asserts nothing, so a missing model card is honest. A required field in an optional section would push an author to invent a value to satisfy the validator, which is the false claim the artifact rule exists to prevent.

Every consumer, the preflight included, shows an unknown as unknown. Where a consumer must act on a fact it does not have, it takes the safe reading: an unknown `sends_audio` is treated as "may send audio", never as "does not".

### 4.2 The effective view: `effective_metadata`

The class-level declaration describes the preset under every configuration and request it supports, as the artifact section does (`metadata.py`, line 18). A configured engine instance describes itself through a new property, `effective_metadata`, named like `effective_capabilities` (`src/standard_asr/runtime/interface.py`, lines 893 to 903) and enforced like `artifact_status()`. It returns the three new sections for this instance. The artifact lifecycle keeps its own effective view, `artifact_status()`.

- **It reads the configuration only.** Like `artifact_status()` under AR.2, it loads no weights, starts no accelerator, and sends no request. IC.9's rule that construction is pure makes that possible (`protocol.md`, line 815). The process environment is not configuration, so a fact that an environment variable can change, such as where a proxy sends a request, keeps its upper bound. So does a behavior a request parameter can trigger: the view narrows by configuration only.
- **It only narrows.** Each value in it is an upper bound for this instance. Where the declaration says `false` ("never"), the effective value is `false`. Where it says `true` ("possible"), the effective value is `true`, or `false` when the configuration rules the behavior out. Where it says unknown, any value is allowed. A list may lose entries but never gain one. A view that widens the declaration is an engine defect, and the core owns the check. `effective_metadata` is a final property of `EngineBase` that validates what a protected hook returns, and it raises `EngineContractError` on a widening, as the `artifact_status()` template does for its report (AR.2). `effective_capabilities`, by contrast, is a plain property that only compliance checks.
- **The card is kept or dropped field by field.** The card describes the preset's default weights. Some configurations make inference run on other weights: a local checkpoint such as `model_path`, or another service. Under such a configuration, each card field the engine cannot vouch for becomes unknown. Another version of the same source, such as a configured `revision`, counts too: the author described one version, and another can differ in any field. std-mlx-audio, for example, can keep `family` in both cases. It loads a `model_path` checkpoint through the preset's model family (`_artifacts.py`, lines 404 to 405), and it refuses a checkpoint or revision of another family (`engine.py`, lines 540 to 548).
- **The default narrows nothing.** The default hook returns the declared sections unchanged, and the core reads an engine that does not subclass `EngineBase` the same way. That is true of every upper bound, but it is not enough for two fields. An engine whose configuration can run inference on other weights MUST drop the card fields as described above. An engine whose configuration selects a device MUST narrow `devices` to the devices that configuration can run on; a choice made at load time, such as `device="auto"`, keeps every device it can fall back to. Without that, the preflight would report `cpu` as holding for an instance configured for `cuda`. Section 13 explains why the design does not add a declarative alternative.

### 4.3 Estimates

An estimate is a number that holds only under conditions. A memory estimate is an object with four fields:

- `device`: the device token (section 4.4) whose memory it counts. `cpu` counts system memory, `cuda` or `rocm` the GPU's memory, and `metal` the memory an Apple GPU shares with the CPU.
- `bytes`: the memory inference needs on that device, in bytes, the loaded weights included.
- `conditions`: the configuration values the estimate assumes, as a mapping from configuration field to value, such as `{"device": "cuda", "compute_type": "float16"}`. The `device` key here is the configuration's field, not the token (section 4.4 lists both senses).
- `basis`: `measured`, the peak use the author observed, or `derived`, computed from facts such as the parameter count and the precision. A derived estimate covers only what the computation includes, which is usually the weights alone.

An estimate is for display. Nothing gates on it: not engine construction, not the preflight's verdict. Issue #8 already calls the memory estimate "display-only, never used for gating". An estimate applies to a configuration only when every condition matches it.

### 4.4 Vocabularies

Device, platform, and precision values are tokens from a standard list, with `x_<vendor>_<name>` for one outside it. They follow the grammar of the artifact section's vocabularies: lowercase, matching `^[a-z][a-z0-9_]{0,63}$` (`protocol.md`, line 876). A family value follows the same grammar and has no standard list. A reader keeps a token it does not know and treats it as a condition it cannot check.

- **Device tokens** name the kind of processor inference runs on, together with the vendor software that programs it. The standard tokens are `cpu`, `cuda` (NVIDIA GPUs), `rocm` (AMD GPUs), and `metal` (GPUs programmed through Apple's Metal). They are not values of the configuration's `device`. An engine maps its own values onto them: PyTorch's `mps` and MLX's GPU are both `metal`, and PyTorch on an AMD GPU is `rocm` although PyTorch calls that device `cuda`.
- **Platform tokens** are `<os>_<arch>`, with `linux`, `macos`, or `windows`, and an architecture such as `x86_64` or `arm64`. A reader names the platform of the running interpreter the same way and normalizes what Python reports: `aarch64` becomes `arm64`, and `AMD64` becomes `x86_64`. Under Rosetta, Apple's translator that runs x86_64 programs on Apple silicon, an x86_64 interpreter reports `x86_64`, which is the platform its native libraries are built for.
- **Family tokens** name the architecture the weights belong to, as the upstream project names it: `whisper`, `qwen3_asr`, `parakeet`. A derived model keeps its parent's family: `distil-large-v3` is `whisper`.

### 4.5 The engine's processes and the external service

Two sections draw the same line. The engine's processes are the process that holds the engine instance and every process the engine starts. An external service is any other process that inference sends requests to, on this machine or another: a vLLM server the operator runs, or a cloud API. "External" means outside the engine, as it does in the artifact lifecycle's external actions (`src/standard_asr/contract/artifacts.py`, line 37).

This line is a decision of this design (section 12). The artifact design draws its boundary by "lifecycle ownership, not process topology" (`model-management.md`, line 136). For weights it reaches a similar result: a service "the engine can only call" is out of the lifecycle, "even when that service runs on loopback" (lines 140 to 142).

## 5. `model_card`: the identity of the default weights

**Rule:** the section describes the weights the preset loads by default, as published at their source. Section 4.2 says when a configured instance drops it.

| Field | Type | Meaning |
| --- | --- | --- |
| `family` | token | The family token of the weights (section 4.4), such as `whisper` or `qwen3_asr`. |
| `parameter_count` | integer | The number of parameters, counted before any quantization. |
| `weights_license` | string | The license of the weights at their source, as an SPDX license expression: a standard short name such as `MIT` or `Apache-2.0`, or `LicenseRef-<name>` for a license outside the SPDX list. |
| `source` | string | The HTTPS URL of the page where the default weights are published, with no user name or password in it. |
| `precision` | token | How most of the parameters are stored: `float32`, `float16`, `bfloat16`, `int8`, or `int4`. |

Every field is optional (section 4.1).

**Why each definition is narrow:**

- **Counted before quantization.** Hugging Face reports 938,008,576 parameters for `Qwen/Qwen3-ASR-0.6B` and 782,426,112 for its 4-bit conversion `mlx-community/Qwen3-ASR-0.6B-4bit`, so a number read from a quantized checkpoint is not the model's count. A preset's name is not a count either: the model behind `0.6b` has 938 million parameters.
- **The license at the source.** The same weights can carry different licenses at different sources: `openai/whisper-large-v3` states Apache-2.0, and Systran's conversion, which std-faster-whisper loads, states MIT. The card states the license of the source it names.
- **Stored precision only.** A setting that changes the precision of the computation at load time, such as `compute_type`, does not change `precision`. The memory estimate's conditions name that setting instead.

The card does not list languages: `BaseProperties` owns the languages an engine accepts, and two lists would drift. Training data, evaluation results, and error rates stay out of this design. Until a later design defines them, a plugin can carry them in an `x_<vendor>_<name>` section.

## 6. `hardware`: what inference needs on this machine

**Rule:** like the artifact section, every field is an upper bound over the configurations the preset supports.

| Field | Type | Meaning |
| --- | --- | --- |
| `applicable` | boolean | Whether the engine's processes run the recognizer (section 4.5). Required when the section is present. When it is `false`, the other fields must be absent. |
| `devices` | list of device tokens | Every device inference can run on, under some supported configuration. |
| `platforms` | list of platform tokens | Every platform the engine can run on, as far as the author knows. A platform left out is one where the author knows the engine does not run. |
| `memory` | list of estimates | Memory estimates (section 4.3). |

An empty `devices` or `platforms` list is invalid. An author who does not know leaves the field out, which reads as unknown.

What lets an author list a platform: the plugin installs and loads there, for at least one Python version it supports. The list is an upper bound, so it has to cover every platform where the engine can run. An author who has not tested a platform lists it when the plugin's native dependencies publish wheels for it; section 9 shows that check for std-faster-whisper.

`applicable` answers question 4. An engine that only calls an external service runs no recognizer, so local hardware facts do not apply to it, even when the service runs on the same machine. `applicable` is not a locality label: it says nothing about the network, audio, or charges, which section 7 owns.

Devices and platforms are separate lists, so a list cannot say that one library builds CUDA support for Linux on x86_64 but not on arm64. The preflight then reports such a combination as unknown, never as holding (section 8). A map from platform to devices would say it, at the cost of a nested structure every author fills; section 13 lists it.

## 7. `external_service`: network, audio, and money

**Rule:** three booleans, each an upper bound over the configurations the preset supports, with its own definition. Each describes what the engine's processes do (section 4.5). Where the answer depends on an external service the engine cannot see into, such as whether it charges, the engine cannot rule the behavior out, so the field stays `true`. A loopback address in the configuration does not rule out `uses_network` either: an HTTP client that reads `HTTP_PROXY` sends even a request for `127.0.0.1` to the proxy unless `NO_PROXY` exempts it, and HTTPX 0.28.1 does.

| Field | True when, under some supported configuration, inference |
| --- | --- |
| `uses_network` | connects to another host. A connection to the loopback address does not count. Downloads during inference do not count either, because `artifacts.may_acquire_during_inference` declares them. |
| `sends_audio` | sends audio, anything computed from it, or a URL to fetch it from, to an external service (section 4.5), on this machine or another. |
| `may_charge` | can make a third party charge for the use, such as a fee per request or per minute of audio. The operator's own hardware or hosting costs do not count. |

Audio the application hands over as a URL is fetched by the core or by the engine. That fetch is the application's choice of host, so it counts toward none of the three.

An engine that sends nothing anywhere declares all three `false`. That is the common case for a local engine, and it is still a claim, so the author writes it.

## 8. What the preflight can conclude (item 14)

The preflight reads `effective_metadata` of the configured instance and compares it with what the machine shows. Each condition gets one of three verdicts: **holds**, **not met**, or **unknown**. The preflight never turns unknown into holds.

- **Platform.** Holds when the interpreter's platform token is in `platforms`. Not met when the list exists without it. Unknown when the list is unknown.
- **Devices.** `cpu` holds on every machine. The platform decides some other devices. `metal` holds on `macos_arm64` and is not met on Linux and Windows. It is unknown on `macos_x86_64`, because whether a library can use an Intel Mac's GPU depends on the library, and the preflight cannot find out without loading it; no first-party plugin declares that combination. `cuda` and `rocm` are not met on macOS. Every other verdict is unknown, unless item 14 finds evidence that starts no accelerator runtime, such as a driver file the operating system exposes. The preflight gives a verdict for each device in the effective list and does not conclude from them that inference runs: the roadmap leaves that claim to a real run (item 23).
- **Memory.** Information, not a verdict. The preflight shows each estimate, marks the ones whose every condition matches the configuration, and shows the machine's total memory.
- **The external service.** Facts about the configuration, shown as they are. They are not conditions of the machine.
- **Hardware that does not apply.** With `applicable = false`, the preflight reports that the engine's declaration says nothing about the hardware the service runs on, even when that service runs on this machine.

Dependency conditions, such as whether the engine's native library loads, belong to the preflight too. No section declares them. Item 14 decides how to check them without starting an accelerator runtime.

## 9. The first-party plugins as examples

Each example is the class-level declaration of one preset, from facts in the plugin or at the weights' source. Where no source states a value, the example leaves the field out, which is the point of section 4.1.

**std-faster-whisper, `faster-whisper/large-v3`:**

```json
{
  "model_card": {
    "family": "whisper",
    "parameter_count": 1543490560,
    "weights_license": "MIT",
    "source": "https://huggingface.co/Systran/faster-whisper-large-v3",
    "precision": "float16"
  },
  "hardware": {
    "applicable": true,
    "devices": ["cpu", "cuda"],
    "platforms": ["linux_x86_64", "linux_arm64", "macos_arm64", "macos_x86_64", "windows_x86_64"]
  },
  "external_service": {"uses_network": false, "sends_audio": false, "may_charge": false}
}
```

The preset loads `Systran/faster-whisper-large-v3`, the repository `faster_whisper.utils._MODELS` maps `large-v3` to. Its card states `license: mit` and float16 storage. The parameter count is that of `openai/whisper-large-v3`, which the conversion keeps. The platforms are those where every native dependency in the plugin's lock at ef23a00 publishes a wheel for at least one Python version the plugin supports. Those dependencies are CTranslate2 4.8.0, the inference library faster-whisper runs on, onnxruntime, PyAV, and tokenizers. On `macos_x86_64` that is Python 3.10 only, because the onnxruntime 1.26.0 the lock resolves for Python 3.11 and later has no wheel there. The platform is listed because the list is an upper bound (section 6); whether this interpreter can load the plugin is a dependency condition of the preflight (section 8). With `model_path` or `revision` set, the effective card is unknown. With `device="cpu"` or `device="cuda"`, the effective devices are that one device.

**std-mlx-audio, `mlx-audio/qwen3-asr-0.6b`:** `devices: ["metal"]` and `platforms: ["macos_arm64"]`, as the plugin's README states ("Apple Silicon only", line 22). All three `external_service` fields are `false`. The card has `family: "qwen3_asr"`, `parameter_count: 938008576`, `weights_license: "Apache-2.0"`, `precision: "int4"`, and the source `https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit` (`engine.py`, line 986), whose configuration states 4-bit quantization.

**std-qwen3-asr, `qwen3-asr/1.7b`:** `hardware: {"applicable": false}` and `external_service: {"uses_network": true, "sends_audio": true, "may_charge": true}`. All three are true for every preset, the vLLM ones included, because any preset can be configured to DashScope. An instance configured for vLLM at a loopback address keeps all three. The audio still leaves its processes. The plugin cannot know whether the service behind the address charges. And its HTTPX client follows `HTTP_PROXY` (`backends/vllm.py`, line 153), so even a loopback request can go to another host. The card describes the weights the preset asks the vLLM server for by name, `Qwen/Qwen3-ASR-1.7B`. The operator who runs that server is trusted (`AGENTS.md`, "Trust model"), so the card stands under vLLM, and it becomes unknown under DashScope.

The preset's reach to DashScope is a question this design leaves to the plugin. Configured that way, a preset that names open weights sends that name and the audio to a hosted service. Whether DashScope rejects the name or serves other weights under it is not checked here.

## 10. Versioning and compliance

The sections land after the core package's v0.2.0 release (item 8), which freezes protocol 0.2, as optional sections in a 0.2.x protocol patch, each with an entry in `PROTOCOL_FEATURE_MINIMUMS` (`src/standard_asr/contract/protocol_version.py`, line 34; AR.1). A reader older than the producer keeps them as unknown sections, which `DeclaredEngineMetadata` already does, and compliance does not report them as extensions without a namespace (`src/standard_asr/compliance.py`, lines 2148 to 2167).

Compliance can check the static half:

- the types, the token grammar, and that a section with `applicable = false` has no other hardware fields;
- that `effective_metadata` of a zero-configuration instance does not widen the declaration;
- that a preset class does not inherit a card or memory estimates from the class of another preset. A preset class inherits `declared_metadata` from its parent class, and std-faster-whisper's base class is itself the `large-v3` preset, so a subclass that does not declare its own would silently describe the `large-v3` weights. A base class that no entry point registers can still share one declaration with its subclasses (`protocol.md`, line 868). Equal content is no signal: two presets can honestly declare only `{"family": "whisper"}`.

Compliance cannot check that a declaration is true. That needs real inference on the declared hardware, which item 23 (runtime compliance) and item 24 (#8, part B) own.

## 11. Terms

`TERMINOLOGY.md` gains rows in the same change as the contract (item 13). Each row names its section of `declared_metadata` as the authority.

- **model card:** the identity and the source of a preset's default weights, the `model_card` section. It is narrower than the model card of a Hugging Face repository, which also describes training, evaluation, and intended use.
- **hardware declaration:** the `hardware` section, under the name issue #8 gives it.
- **device:** two senses, both in use. (1) The configuration's `device`, a value the engine passes to its native library, such as `mps`. (2) A device token of the `hardware` section, such as `metal` (section 4.4).
- **external service:** a process that inference sends requests to and the engine does not start, on this machine or another (section 4.5). "External" means outside the engine, as in the artifact lifecycle's external actions. It is narrower than *provider*, the far side of any integration, which an in-process library can be: an external service is a provider that runs as its own process. The section `external_service` is named after it.

## 12. Decisions for the maintainer

1. **The section names** `model_card`, `hardware`, and `external_service`. The third is the one to weigh: it groups three facts that a call to an external service usually causes, and it is not a locality label, because each fact keeps its own definition. Alternatives: three top-level sections, or a name such as `data_and_billing`.
2. **`effective_metadata` as a final property of `EngineBase`** that returns only the new sections and validates a protected hook (recommended): named like `effective_capabilities`, enforced like `artifact_status()`. The alternative is a method.
3. **No languages in the card** (recommended). #8 lists languages among the card's facts. `BaseProperties` already owns them, so this design keeps that one authority.
4. **Stored precision in the card.** #8, part A, puts `quantization` in the hardware declaration. This design treats the stored precision as part of the weights' identity, and the precision of the computation as configuration that estimates name.
5. **The boundary of the engine's processes** for `sends_audio` and `hardware.applicable` (section 4.5). It is new: the artifact design draws its boundary by lifecycle ownership instead.
6. **`may_charge` cannot narrow for a service the engine cannot see into.** Even an operator who runs vLLM on this machine sees `may_charge: true` for every std-qwen3-asr instance. The plugin cannot know whether the service behind the address charges (section 7). Recommended: keep the honest `true` in the standard. A plugin may add a setting through which the operator states that the service does not bill, and narrow on it, because the operator is trusted (`AGENTS.md`, "Trust model").
7. **No refusal at construction.** #8 suggests that the core raise a structured error when an engine is constructed on hardware it does not declare. This design recommends only the preflight's report. `platforms` is a bound the author draws from their knowledge, and a bound drawn too narrow would then block a machine where the engine works, with no way around it short of a plugin release. An engine still raises its own clear error when it cannot load.

## 13. Alternatives not taken

- **A locality field** (`local`, `cloud`): rejected again, for the reasons in `model-management.md`, section 4. `hardware.applicable` and the three service facts state, one fact at a time, what a locality field would have implied.
- **Required sections or fields:** a required field in an optional section invites invented values (section 4.1).
- **Gating on memory estimates:** an estimate holds only under its conditions, and #8 already rules it display-only.
- **Probing accelerators in the preflight:** loading a vendor's native code can start an accelerator runtime, which item 14 forbids.
- **A map from platform to devices:** more precise than two lists (section 6), but every author fills a nested structure for a case the preflight already reports as unknown.
- **A declarative list of the settings that replace the weights**, such as `model_card.replaced_by = ["model_path"]`, which would let `EngineBase` drop the card without an override. It ties the card to configuration field names, and it fails the same way the override does when an author forgets it. The override is the mechanism the other effective views already use.

## 14. Follow-ups outside this design

- Item 13: the contract types, their projection in `GET /v1/metadata/{model}` and `show`, `effective_metadata` with its validation, the compliance checks, the terminology rows, and the sections the first-party plugins declare. Each card lands with the override that drops it, so no release describes replaced weights with the default card.
- Item 14: the preflight.
- std-qwen3-asr: decide whether an open-weight preset may reach DashScope (section 9).
- Item 24 (#8, part B): verify the hardware declarations on real hardware in CI.
