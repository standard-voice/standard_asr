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
either in this table or in an explicit exempt set. The requestable set is
therefore already exact. It is simply not visible in the capability tree.

Every other declared node is a behavior: `streaming_input`,
`streaming_output`, `self_resamples`, `emits_partials`, `re_segments`,
`word_stability`, `reconnect`, `finality_level`, `timestamps`, and
`diarization.always_on`.

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
kind must therefore appear in `canonical_json()`.

**D4 — Add a `kind` field to the node. Do not move the nodes.** A move of
behaviors into a `behavior.*` subtree breaks every path in every consumer, in
the spec, and in the tests. The move adds no information that the `kind` field
does not already give. Cost is high and value is zero, so the tree shape stays.

Each capability node gains one field:

```python
kind: Literal["requestable", "behavior", "limit"]
```

Each node class declares its own value, and an engine author never sets it.

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
`kind="behavior"`. The prose inversion note in `contract/capabilities.py` is
then wrong, because the field states the same fact. Delete the note when the
field lands. A node with `kind="behavior"` always reads as a disclosure, so
there is no inversion left to explain.

**D7 — A requestable feature and a behavior are two nodes, never one.** Issue
#37 needs punctuation and ITN as caller requests, and it also needs to disclose
an engine that always applies them. Do not overload one node with both
meanings, because that recreates the `always_on` problem. Use the pattern that
`diarization` already shows:

- `text_processing.punctuation` with `kind="requestable"`.
- `text_processing.punctuation.always_on` with `kind="behavior"`.

## 4. Effect on the blocked issues

| Issue | State after this document |
| ----- | ------------------------- |
| #45 | Closed. D1 gives the kinds. D4 gives the mechanism. |
| #37 | Unblocked. D7 gives the node shape for punctuation and ITN. |
| #8 | Unblocked. D2 keeps hardware facts out of the tree. No tree change is necessary. |
| #19 | Unblocked. D2 applies to model-card facts in the same way. |

## 5. Remaining implementation items

These items are implementation work, and they do not block the issues above:

1. Add the `kind` field to `_CapNode` and set the value on each node class.
2. Add `can_request()` to `DeclaredCapabilities`.
3. Render `kind` in `canonical_json()`, and update the two-layer isomorphism
   test set.
4. Assert in the compliance suite that every path in `_GATED_PARAMS` resolves
   to a node with `kind="requestable"`. This makes the gating table and the
   tree prove each other.
5. Update `docs/spec/specification.md` and `docs/spec/server.md` for the new
   field and the new query.
6. Delete the inversion note in `contract/capabilities.py`.

Item 4 is the important one. It removes the chance that the two definitions of
"requestable" drift apart.
