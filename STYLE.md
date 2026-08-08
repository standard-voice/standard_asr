<!-- SPDX-FileCopyrightText: 2026 Standard Voice Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Writing standard

This file defines how to write English prose in Standard ASR. It adapts
[ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/) for a
software project. Every contributor and every AI agent that works in this
repository must follow it. `AGENTS.md` points here; the term rules live in
[`TERMINOLOGY.md`](TERMINOLOGY.md).

The goal is one thing: a reader understands the behavior from the prose alone,
with the least possible room for misunderstanding. Prose is part of the
contract, so we hold it to the same rigor as the code.

## Why an adapted standard

ASD-STE100 was built for aircraft maintenance manuals. Most of its rules improve
software prose too: short sentences, active voice, one word for one meaning, no
idioms. A few rules do not fit software, so this file states, for each rule
area, whether we **keep**, **adapt**, or **drop** it. Where a rule conflicts
with a code fact (an identifier, a domain term, a cross-reference role), the code
fact wins.

## Scope: what this governs

This standard governs English **prose**:

- docstrings (module, class, function, attribute);
- user-facing runtime strings (see the tier test below);
- English Markdown docs listed in the `mkdocs.yml` `nav`;
- internal `#` comments (clarity tier only — see below);
- new `CHANGELOG.md` entries from now on.

This standard does **not** govern, and must never change:

- **Code identifiers** — names are the contract. A prose rule never renames a
  symbol. American spelling applies to prose, not to a third-party or standard
  library name (write "the task is canceled" in prose; keep the symbol
  `CancelledError`).
- **Chinese documents** — `docs/spec/specification.md` (the normative spec),
  `docs/design-notes/`, `docs/research/`, `docs/work_doc/`, and other Chinese
  files. Do not translate or edit them. Read them as the source of truth (see
  "Fact-check every meaning change" below).
- **reStructuredText roles and code spans** — `:class:`, `:func:`, `:meth:`,
  `:mod:`, `:data:`, double-backtick code spans, and `::` literal blocks stay
  verbatim. Never reflow or "simplify" the text inside them.
- **SPDX headers**, license text, and historical `CHANGELOG.md` entries.

## The two tiers

The standard has two tiers. Use one mechanical test to pick the tier.

A text is **user-facing (full standard)** if it can reach a person who does not
read the source at that moment. This includes:

- the string argument to `raise ...Error(...)`, `logger.*`, `warnings.warn`,
  and `print`;
- `argparse` `help=`, `description=`, and `epilog=`;
- FastAPI and Pydantic text: `Field(description=...)`, `FastAPI(title=...)`,
  route and OpenAPI docstrings, and WebSocket `{"message": ...}` frames;
- **every docstring**, public or private (mkdocstrings renders them, and
  consumers read them);
- Markdown docs in the `mkdocs.yml` `nav`.

A text is **internal (clarity tier)** if it is a `#` comment that explains
rationale to a maintainer. The clarity tier keeps the full standard's word and
term rules but **drops the sentence-length limit**, because a comment sometimes
needs a long sentence to record a subtle reason. Do not shorten rationale to hit
a word count. Keep the "why".

## The rules

Each rule area below is marked **KEEP** (apply as written), **ADAPT** (apply the
changed form), or **DROP** (does not apply here).

| Area | Ruling for this repo |
| --- | --- |
| **One word, one meaning** | **ADAPT.** Use the approved term for each concept from `TERMINOLOGY.md`. Do not use a forbidden synonym. Do not use one word for two meanings. |
| **Approved technical names** | **ADAPT.** The domain's `-ing` names are approved names, not banned gerunds: *streaming, gating, negotiation, diarization, resampling, coalescing, superseding*. Use them. |
| **Noun clusters ≤ 3 words** | **ADAPT (soft cap).** Keep new noun clusters to three words. Hyphenate a multi-word modifier so it reads as one unit ("frozen-prefix boundary"). Established compound API terms are exempt. Never break an identifier to meet the count. |
| **Verb tense** | **ADAPT.** Prefer the simple present and the imperative. Use the past tense for a past event. Avoid the perfect and progressive tenses where the simple tense is clear. |
| **Active voice** | **KEEP.** Write active voice. Use passive voice only when the actor is genuinely the runtime and naming it adds nothing ("The request is rejected with 422."). |
| **Sentence length** | **ADAPT / tier-split.** Keep a user-facing sentence short — about 20 words for an instruction, about 25 for a description. Keep the docstring **summary line** short. **DROP the hard limit** for `#` comments and for docstring body paragraphs that carry contract nuance. |
| **Imperative for instructions** | **KEEP.** Write instructions and `hint=` fields as commands ("Install the server extra: `pip install 'standard-asr[server]'`."). |
| **Articles** | **KEEP.** Use *a*, *an*, and *the* in prose. A telegraphic one-liner is allowed only in an `argparse` `help=` string, where the convention omits them ("List discovered models."). |
| **No slang or idioms** | **KEEP, with a carve-out.** Remove idioms and figurative phrases. Keep only a term of art the codebase already defines. |
| **Warnings and cautions** | **KEEP.** State the hazard first, then the action. Keep the `level`/`code`/`message` shape for diagnostics. |
| **Punctuation and symbols** | **KEEP.** No emoji in any shipped text, ever. The CLI uses ASCII markers (`[OK]`, `[FAIL]`, `[WARN]`, `[INFO]`) for this reason. Keep code spans and roles verbatim. |
| **American spelling** | **KEEP.** See `TERMINOLOGY.md`. |

## Five tension resolutions

These are the points where a strict reading of ASD-STE100 meets a software fact.
Apply the resolution as written.

1. **`-ing` domain terms are approved names.** ASD-STE100 restricts most `-ing`
   words. Our subsystem names (streaming, gating, negotiation, diarization,
   resampling, coalescing, superseding) are approved technical names. They are
   module names and spec terms. Use them without hesitation.

2. **Cross-reference roles and code spans are verbatim.** Never reflow, reword,
   or re-case text inside a `:role:` or a `` `code span` ``. A rewrite works
   around them, never through them.

3. **RFC-2119 keywords depend on the reader.** Keep "MUST", "MUST NOT", "SHOULD",
   and "MAY" in uppercase when the sentence states a rule to an **engine
   author** (for example, a streaming event-construction error, or a docstring
   that cites the spec). Soften to a plain verb when the sentence answers an
   **application developer** who made an ordinary call mistake (for example,
   "candidate_languages MUST NOT contain 'auto'" becomes "candidate_languages
   cannot contain 'auto'"). The rule: state a spec obligation to the party who
   can break the spec; speak plainly to the party who made a normal mistake.

4. **A prose rule never changes an identifier.** American spelling and term
   rules apply to prose only. The prose says "canceled"; the symbol stays
   `CancelledError`. The prose says "normalize"; a third-party symbol keeps its
   own spelling.

5. **American spelling.** Use `normalize`, `behavior`, `initialize`,
   `serialize`, `analyze`, `color`, and `canceled`/`canceling` in prose. The
   only exception is a verbatim identifier (resolution 4).

## Fact-check every meaning change

This is the core rule for correctness. It applies to a rename **and** to a
rewritten message.

A rewrite **may** change what a text means. It **should**, where the original is
wrong, ambiguous, or misleading. Removing wrong or confusing copy is a goal of
this standard, not a risk to avoid. But any change that alters the conveyed
meaning must pass two gates, and you confirm both with more investigation than
feels necessary:

1. **Necessity.** There is a real reason to change. The original is wrong,
   ambiguous, or misleading. A pure style pass that keeps a correct meaning
   needs no such reason.
2. **Correctness.** The new text is accurate. You checked it against the
   authoritative spec (`docs/spec/specification.md`) and the real context: the
   surrounding code, the actual runtime behavior, and the related design notes.

Where the original is ambiguous, find the true intended meaning first, then
write it. Never guess. Never encode a wrong meaning to gain a shorter sentence.
Many names and statements are intentional and correct — prefer to clarify them
rather than change them.

## Terminology

Every domain term follows [`TERMINOLOGY.md`](TERMINOLOGY.md): one canonical term
per concept, a short definition, the approved usage, and the forbidden synonyms.
The controlled code vocabularies (the `DIAG_*` codes, the compliance codes, the
enums, and the `Literal` sets) have a single source of truth in the code;
`TERMINOLOGY.md` points to it and never copies it.

## Checklist before you commit prose

- Each concept uses its canonical term from `TERMINOLOGY.md`.
- Spelling is American; no British forms in prose.
- No emoji anywhere.
- User-facing sentences are short and active.
- Instructions and hints are imperative.
- Every meaning change is necessary and fact-checked against the spec.
- Code spans, roles, and identifiers are unchanged.
- `uv run ruff check` passes (pydocstyle included).
