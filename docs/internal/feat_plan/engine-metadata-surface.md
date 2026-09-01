> **Superseded numbering.** This record predates the protocol re-base: read
> its "protocol 1.1" as the artifact-lifecycle generation, now **protocol
> 0.2** on the pre-stable 0.x line. Spec AR.1
> (docs/content/specification/protocol.md) is authoritative.
>
> **Superseded behavior.** The version-rule paragraph below ("Protocol 1.0
> engines remain discoverable and transcribable") describes a transition
> tolerance that was deleted with the re-base: within protocol major 0 the
> minor is the breaking axis, so the preceding generation is NOT
> transcribable -- every gated surface rejects it with
> ``ProtocolCompatibilityError``. Only import-free discovery still lists
> such an engine. AR.1 governs.

# Engine metadata surface decision

Status: approved with the model artifact lifecycle design on 2026-08-26.

## Decision

Protocol 1.1 has four class-readable declaration surfaces with different jobs:

| Surface | Lifetime | Consumer | Meaning |
| --- | --- | --- | --- |
| `properties` | Engine class | Discovery and runtime | Static identity and I/O boundaries. |
| `declared_capabilities` | Engine class | Gating and negotiation | Hierarchical feature support queried by `supports()`. |
| `declared_metadata` | Engine class | Selected-model tooling and wire metadata | Typed lifecycle and operational facts that do not gate runtime parameters. |
| `config_type` | Engine class | Construction tooling and settings UIs | Typed init-config schema. |

`declared_metadata` is one forward-compatible typed aggregate. Its root accepts
unknown JSON-valued sibling sections, while each known section remains a closed
model. The first required section is `ArtifactDeclaration`; future hardware and
catalog work from issues #8 and #19 must add sibling sections instead of
creating new class variables or forcing unrelated facts into capabilities.

## Why lifecycle facts do not enter capabilities

Capability-tree admission is not defined by whether a node gates a parameter;
the tree already contains informational behavior nodes. The boundary here is
lifetime and consumer: capabilities describe inference behavior and are queried
through `supports()`, while artifact declarations describe management behavior
consumed by selected-model tooling. Putting acquisition in capabilities would
make `supports()` negotiate an operation outside inference.

This preserves the capability-kinds decision without repeating its rejected
argument. It also honors the streaming-cadence constraint that whichever
metadata family lands first must leave one coherent home for adjacent families.

## Wire and discovery projection

The in-process aggregate has a corresponding per-model wire projection at
`GET /v1/metadata/{model}`. `standard-asr show` renders the same canonical JSON.
Both operations resolve only the selected plugin and use the metadata fault
boundary.

Bulk `standard-asr list` and `GET /v1/models` remain entry-point-only. They do
not resolve engine classes, import every plugin, instantiate engines, or scan
artifact stores. This avoids heavyweight imports, conflicting native
dependencies, and a single broken plugin denying discovery of every other
model.

## Version rule

Protocol 1.1 engines must author `DeclaredEngineMetadata` and its `artifacts`
section. A plugin-owned base class can provide a shared declaration to its own
subclasses. Inheriting the transition placeholder from core `EngineBase` is not
authorship and fails compliance. Protocol 1.0 engines remain discoverable and
transcribable, but lifecycle tooling raises or reports
`ProtocolCompatibilityError` before reading the new surface.

## Related decisions

- [`capability-kinds.md`](./capability-kinds.md)
- [`streaming-cadence-and-tuning.md`](./streaming-cadence-and-tuning.md)
- [`model-management.md`](./model-management.md)
- Issues #8 and #19 for future metadata siblings
