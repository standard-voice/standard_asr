---
title: Inference Artifacts
---

# Inference artifacts

The Standard ASR artifact lifecycle lets an application inspect and acquire
persistent files or installed assets that an engine needs for inference. The
contract describes lifecycle ownership, not whether inference is local or
remote.

Every supported engine exposes both lifecycle methods. An engine on an
unsupported protocol line might not have them, so call
`require_artifact_protocol(engine)`
(exported from `standard_asr`) before looking either method up. The guard
raises a typed `ProtocolCompatibilityError` instead of an `AttributeError`.
`EngineBase` supplies guarded methods that raise the same error on such an
engine.

Use `artifact_status()` to inspect one configured engine without downloading,
loading weights, initializing an accelerator, or running inference:

```python
from standard_asr import ArtifactContext, RuntimeParams

context = ArtifactContext(params=RuntimeParams(language="en"))
report = engine.artifact_status(context)

print(report.readiness)
for requirement in report.requirements:
    print(requirement.artifact_id, requirement.state)
```

`ArtifactReport.applicable` states whether the selected context has an
inference-artifact lifecycle. A non-applicable report is not a synonym for a
remote engine. It only means that Standard ASR has no persistent artifacts to
manage for that context.

An omitted context mode prefers batch and otherwise uses the engine's only
declared inference mode. An explicit mode that the engine does not support
raises `ValueError`; status does not invent a dependency closure for an
execution path that cannot run.

## Requirement state

Each `ArtifactRequirement` reports one logical dependency in the resolved
context. Its state is one of the standard values below, or a forward-compatible
extension value:

| State | Meaning |
| --- | --- |
| `ready` | The engine has evidence that inference can resolve the requirement without acquiring a new persistent artifact. |
| `missing` | No usable artifact set is present. |
| `incomplete` | Some expected content is present, but the requirement is incomplete. |
| `corrupt` | Present content has a known integrity or layout failure. |
| `unknown` | A cheap, side-effect-free inspection cannot establish the state. |

Aggregate `readiness` considers only requirements with
`required_for_inference=True`. An optional missing requirement does not make
inference unavailable.

`can_acquire_now`, `may_acquire_during_inference`, and
`source_is_mutable` are independent facts. For example, a ready artifact can
still use a mutable source, and an engine can support both explicit acquisition
and implicit acquisition during its first inference call.

## Explicit acquisition

Call `acquire_artifacts()` to materialize every non-ready requirement that the
configured engine can acquire now:

```python
report = engine.acquire_artifacts(context)
```

This operation does not call a transcription endpoint. It is separate from
`prepare()`: acquisition makes persistent artifacts available, while
`prepare()` performs optional process-local warm-up.

After the native hook returns, the core queries status again. Every attempted
logical `artifact_id` must still appear and must be `ready`. The closure can add
newly discovered requirements, but it cannot replace a target silently. A
missing target is an engine-contract error; a target that remains `unknown` or
otherwise non-ready is a failed acquisition, even when it is optional.

Pass `refresh=True` to request source re-resolution for mutable references:

```python
report = engine.acquire_artifacts(context, refresh=True)
```

An immutable revision, digest, operator path, or installed asset is a refresh
no-op. Refresh widens the target set rather than narrowing it: on top of the
non-ready requirements a plain `pull` acquires, it adds every unblocked mutable
requirement, `ready` ones included. A ready floating reference is exactly what a
refresh exists to re-resolve, and it can never be acquired now, so a target rule
that skipped it would make `--refresh` a no-op. A mutable requirement that
carries any blocker stays out of the hook.

When downloads are disabled, the presence of any mutable requirement rejects the
refresh before blocker filtering. This ordering avoids claiming freshness
without the network metadata request and raises `ArtifactAcquisitionError` with
`reason="downloads_disabled"`.

An acquisition blocker describes why work cannot start. A known operator step,
such as accepting terms, authenticating, requesting access, or providing files,
uses `action_required` and includes one or more `ArtifactAction` values. An
engine that can only trigger acquisition during inference uses `unsupported`
with no fabricated action.

## Progress

The optional progress callback receives ordered `ArtifactProgress` values. The
core serializes callback delivery, validates each event, and does not treat a
callback exception as cancellation:

```python
def on_progress(event):
    print(event.phase, event.completed_units, event.total_units)


report = engine.acquire_artifacts(context, progress=on_progress)
```

If the callback fails, acquisition and final status inspection continue. A
successful operation then raises `ArtifactProgressCallbackError` with the final
report attached.

## Error handling

- `ProtocolCompatibilityError` means the installed engine and this core do
  not share a protocol line -- the engine is older or newer than the core
  supports. The message names the direction and the fix.
- `ArtifactStatusError` means side-effect-free status inspection failed.
- `ArtifactUnavailableError` means inference or warm-up cannot proceed because
  required artifacts are unavailable.
- `ArtifactAcquisitionError` means explicit or implicit acquisition is blocked
  or failed. Read its `reason`, `report`, `required_actions`, and
  `retriable_after` fields.
- `ArtifactProgressCallbackError` means acquisition succeeded but its observer
  failed.

Applications should use the structured fields and must not parse exception
messages.

## Engine-author contract

Every engine authors a static upper bound:

```python
from typing import ClassVar

from standard_asr.engine import (
    ArtifactDeclaration,
    DeclaredEngineMetadata,
    EngineBase,
    NO_ARTIFACT_LIFECYCLE,
)


class MyEngine(EngineBase):
    declared_metadata: ClassVar[DeclaredEngineMetadata] = DeclaredEngineMetadata(
        artifacts=ArtifactDeclaration(
            applicable=True,
            supports_explicit_acquisition=True,
            may_acquire_during_inference=True,
        )
    )
```

The configured instance can narrow these values. Implement
`_artifact_requirements(context)` for side-effect-free native inspection and
`_acquire_artifacts(context, requirements, refresh, progress)` for explicit
acquisition. The public `EngineBase` methods own context gating, aggregate
readiness, progress isolation, static-to-dynamic subset checks, and the final
status query.

An engine with no inference-artifact lifecycle uses
`DeclaredEngineMetadata(artifacts=NO_ARTIFACT_LIFECYCLE)`. An engine whose
only safe acquisition path is real inference declares explicit acquisition as
false and possible inference acquisition as true; it does not transcribe fake
audio from either lifecycle method.
