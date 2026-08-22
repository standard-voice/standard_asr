# Capability Kinds (Design Case)

**Status:** Design settled 2026-08-08. This document closes issue #45. It also
removes the block on issues #37, #8, and #19. The decisions change the engine
protocol and the wire schema. There is no code yet. Each decision tells you
what to build.

> Issue #45 calls this topic the "capability ontology". This document uses the
> word **kind**, because `kind` is the name of the new field.

## 1. Problem statement

The capability tree uses one node type and one query for four different kinds
of statement:

- **Requestable** — the caller can ask for the feature.
- **Behavior** — the engine does this. The caller cannot ask for it or refuse it.
- **Limit** — a boundary on a requestable feature, for example a maximum term count.
- **Observation** — a measurement that changes with the machine or the environment.

A consumer cannot tell one kind from another. `DiarizationCap.always_on` and
`StreamingCapabilities.emits_partials` are both `FlagCap` nodes. They have the
same type and the same fields. Their meanings are opposite.

A `True` value on `emits_partials` gives the caller a fact about the engine. A
`True` value on `always_on` tells the caller that the engine applies
diarization to every request. The source calls this a **semantic inversion**.
It records the inversion in prose, in `contract/capabilities.py` at lines 470
to 474. Prose does not reach a machine.

The result is two defects:

1. `supports("streaming.re_segments") == True` does not mean the caller can ask
   for re-segmentation. A user interface cannot tell a request control from a
   disclosure.
2. Each new field family repeats the special case. Issue #37 adds fields that
   are requestable **and** behavioral. Issues #8 and #19 add observations,
   which is a fourth kind.

## 2. Context — what exists today

The gating layer holds the only operational definition of "requestable". It is
the `_GATED_PARAMS` table in `runtime/gating.py`:

| `RuntimeParams` field | Capability path |
| --------------------- | --------------- |
| `language` | `language.runtime_override` |
| `word_timestamps` | `word_timestamps` |
| `diarization` | `diarization` |
| `phrase_hints` | `guidance.phrase_hints` |
| `prompt` | `guidance.prompt` |

A drift guard in the same module proves that every `RuntimeParams` field is
either in this table or in an explicit exempt set. One exempt field is itself
requestable: `candidate_languages` has a `RuntimeParams` field, but
`language.effective_candidate_languages` owns its gating, so `gate_params()`
leaves it alone. The requestable set is therefore the five table rows plus
`language.candidate_languages`. It is exact. The tree does not mark it:
every one of these paths resolves to a declared node, but no field on the
node says that the caller can request it.

Every other declared **leaf** node is a behavior: `streaming_input`,
`streaming_output`, `self_resamples`, `emits_partials`, `re_segments`,
`word_stability`, `reconnect`, `finality_level`, `timestamps`,
`guidance.mutable_mid_stream`, and `diarization.always_on`.

The constraint submodels are a separate case. `DiarizationConstraints` and its
siblings hold the limits on a requestable feature. They are queryable paths,
because `iter_queryable_paths()` yields them. They are not `_CapNode`
subclasses, so a `kind` field on `_CapNode` alone does not reach them. D4
settles this.

## 3. Decisions (2026-08-08)

**D1 — There are four kinds: `requestable`, `behavior`, `limit`, and
`observation`.** These names are final. Use them in the code, the spec, and the
documents.

**D2 — Observations stay out of the capability tree.** A capability
declaration is static and comes from the engine class. An observation depends
on the host machine, and two machines give different answers. The two ideas
have different lifetimes, so one tree must not hold both. Hardware facts (#8)
and model-card facts (#19) get their own surface.

This decision alone unblocks #8 and #19. Neither issue needs a change to the
capability tree. Both can start now.

**D3 — The kind must be data on the wire, not only a Python method.** The
standard targets clients in other languages. A generated JavaScript client
reads the JSON Schema, and it never sees a Python method. If the kind lives
only in a Python API, every non-Python client keeps the current defect. The
kind must therefore appear in `canonical_json()` **and** in the generated JSON
Schema. The schema is the artifact a non-Python client reads, so the schema is
the contract this decision is about.

**D4 — Add a `kind` field to the node. Do not move the nodes.** A move of
behaviors into a `behavior.*` subtree breaks every path in every consumer, in
the spec, and in the tests. The move adds no information that the `kind` field
does not already give. Cost is high and value is zero, so the tree shape stays.

Each capability leaf node gains one field. The shared alias gives the domain,
and each concrete class pins **one** value:

```python
CapabilityKind = Literal["requestable", "behavior", "limit"]


class FlagCap(_FlagLikeNode):
    kind: Literal["behavior"] = "behavior"


class WordTimestampsCap(_FlagLikeNode):
    kind: Literal["requestable"] = "requestable"
```

Do not give a node the shared three-value type. That type states the domain
only. It lets a node hold a valid but wrong value, such as a behavior node that
declares `kind="requestable"`. A single-value `Literal` per class makes the
wrong value a validation error, and it renders as a `const` in the generated
schema, so a generated client sees the one pinned value.

The `const` alone does not make the property required. The field carries a
default, and pydantic keeps a defaulted field out of the schema's `required`
list in both schema modes. Set
`json_schema_serialization_defaults_required=True` on the shared model config
and generate the wire schema in serialization mode. The wire document always
carries `kind`, because `canonical_json()` dumps defaults, so that schema
describes the wire truthfully. Item 4 tests the result.

Pinning the kind per class exposes one place where the class inventory is too
coarse today: `language.runtime_override` is requestable — it is the
`language` row of `_GATED_PARAMS` — but the field is typed `FlagCap`, and
`FlagCap` pins `behavior`. Give the node its own class before the pin lands:

```python
class RuntimeOverrideCap(_FlagLikeNode):
    kind: Literal["requestable"] = "requestable"
```

Every other `FlagCap` site is a behavior, so `FlagCap` keeps the `behavior`
pin. `CandidateLanguagesCap` already has its own class; it pins `requestable`.
The item 5 check exists to catch exactly this mismatch; the class split
removes the mismatch before the check can fire.

The domain holds three values, not the four kinds of D1. `observation` is never
a node kind, because D2 keeps observations out of the tree. A reader who finds
`observation` on a node has found a defect.

Two further scope rules:

- **Constraint submodels carry `kind="limit"`.** The four constraint classes
  have no shared base today. Each one derives directly from `_JsonExtraModel`,
  in the same way as `_CapNode` and `_Container`. Add a `_ConstraintNode` base
  that mirrors `_CapNode`, and put the field there.
- **Containers do not carry a kind.** `BatchCapabilities`,
  `StreamingCapabilities`, and the other `_Container` types group nodes and
  declare nothing. `can_request()` returns `False` for a container path, in the
  same way as for a behavior.

`iter_queryable_paths()` yields container paths too, so **not** every queryable
path carries a kind. The rule is narrower: a path carries `kind` when it
resolves to a capability node or to a constraint node. A container renders in
`canonical_json()` and in the schema with no `kind` property, exactly as it does
today. A client reads the absence as "this path groups nodes; it declares
nothing".

**D5 — Add `can_request(path)`. Keep `supports(path)` unchanged.**
`supports(path)` keeps its current meaning and its fail-closed rule, so no
existing caller breaks. The new query answers the question a user interface
needs:

```python
def can_request(self, dot_path: str) -> bool:
    """Return True when the caller can ask for the feature at dot_path."""
```

`can_request(path)` returns `True` only when the node is supported **and** its
kind is `requestable`. For all other nodes it returns `False`.

**D6 — `always_on` stops being a special case.** The node gets
`kind="behavior"`. Rewrite the prose note in `contract/capabilities.py`; do not
delete it. The `kind` field replaces one sentence of that note, which is the
sentence about the semantic inversion. The rest of the note carries meaning
that no field states.

Keep this meaning in the rewritten note:

- `always_on=True` means the engine applies diarization to every request.
- The engine may emit speaker labels that the caller never asked for.
- `always_on` may be supported only when `supported` is `True`. A model
  validator already enforces this rule.

**D7 — A requestable feature and a behavior are two nodes, never one.**
Issue #37 needs punctuation and ITN as caller requests, and it also needs to
disclose an engine that always applies them. Do not overload one node with both
meanings, because that recreates the `always_on` problem. Use the pattern that
`diarization` already shows:

- `text_processing.punctuation` with `kind="requestable"`.
- `text_processing.punctuation.always_on` with `kind="behavior"`.

A behavior child node needs the same invariant that `DiarizationCap` already
holds: the child may be supported only when its parent requestable feature is
supported. Add the validator and the tests with the new nodes. Without the
rule, an engine can declare that it always applies punctuation that it does not
support.

## 4. Effect on the blocked issues

| Issue | State after this document |
| ----- | ------------------------- |
| #45 | Closed. D1 gives the kinds. D4 gives the mechanism. |
| #37 | Unblocked. D7 gives the node shape for punctuation and ITN. |
| #8 | Unblocked. D2 keeps hardware facts out of the tree. No tree change is necessary. |
| #19 | Unblocked. D2 applies to model-card facts in the same way. |

## 5. Remaining implementation items

These items are implementation work, and they do not block the issues above:

1. Add a `_ConstraintNode` base for the four constraint classes. Add the `kind`
   field to `_CapNode` and to `_ConstraintNode`. Set the value on each node
   class. Give `language.runtime_override` its own `RuntimeOverrideCap` class,
   as D4 sets out.
2. Add `can_request()` to `DeclaredCapabilities`. Return `False` for a
   container path and for an absent path.
3. Render `kind` in `canonical_json()`, and update the two-layer isomorphism
   test set.
4. Assert that the generated JSON Schema carries a **required** `kind` property
   on every concrete capability node and constraint node, and that its value is
   the one value that class pins. A domain-only check passes a node that
   declares the wrong kind, so it is not enough. D3 rests on the schema, so a
   test must hold the schema correct. The `canonical_json()` test in item 3
   does not cover it.

   The assertion holds only for a serialization-mode schema with
   `json_schema_serialization_defaults_required=True` set, as D4 sets out. A
   validation-mode schema leaves a defaulted `kind` out of `required`.
5. Assert in the compliance suite that every entry in `_GATED_PARAMS` resolves
   to a node with `kind="requestable"`.

   The second column of `_GATED_PARAMS` holds a **mode-relative suffix**, not a
   full path. `gate_params()` resolves it as `f"{mode}.{cap_suffix}"`. The
   assertion must therefore qualify each suffix with `batch` and with
   `streaming`, and it must skip a mode the engine does not declare. A raw
   lookup of `language.runtime_override` resolves nothing.
6. Assert the **reverse** direction as well: every node with
   `kind="requestable"` appears in `_GATED_PARAMS`, or in a new explicit
   exemption set beside it.

   Item 5 alone leaves a hole. A new requestable node with no gating entry makes
   `can_request()` answer `True` while the gating layer never checks the
   request. The request then reaches the engine ungated, and an engine that does
   not support it ignores it in silence. That is the cardinal sin.

   `runtime/gating.py` already holds the same guard on the other axis. Its
   import-time assertion proves that every `RuntimeParams` field is in
   `_GATED_PARAMS` or in `_UNGATED_PORTABLE_FIELDS`. Mirror that pattern, with
   the same loud failure and a written reason for each exemption.

   The exemption set starts with one real entry: `language.candidate_languages`
   is requestable, and `language.effective_candidate_languages` owns its
   gating — see `_UNGATED_PORTABLE_FIELDS` and the written reason it carries.

   The reverse check compares mode-relative suffixes, the same representation
   item 5 resolves. For each declared mode, walk the requestable nodes under
   that mode domain, strip the leading `{mode}.` from each path, and require
   the suffix in `_GATED_PARAMS` or in the exemption set. The exemption set
   holds mode-relative suffixes too, so one entry covers both modes.
7. Update `docs/spec/specification.md` and `docs/spec/server.md` for the new
   field and the new query.
8. Rewrite the `always_on` note in `contract/capabilities.py`, as D6 sets out.
   Do not delete it.

Items 5 and 6 are the important pair. Only together do they make the gating
table and the tree prove each other. Item 5 alone proves one direction, and one
direction is not enough.
