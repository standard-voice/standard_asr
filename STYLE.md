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

- docstrings (module, class, function, attribute), in `src/` and in `tests/`;
- user-facing runtime strings (see the tier test below);
- English Markdown under `docs/`, published or not (`docs/legacy/` is
  historical and exempt);
- English Markdown at the repository root: `README.md`, `CONTRIBUTING.md`,
  `AGENTS.md`, `RELEASING.md`, this file, and `TERMINOLOGY.md`. Working notes
  under `work/` are exempt;
- internal `#` comments (clarity tier only — see below);
- test prose: docstrings, comments, and assertion labels in `tests/`;
- new `CHANGELOG.md` entries from now on.

`README.md`, `docs/index.md`, and the `AGENTS.md` preamble are the project's
front door. They are governed for **accuracy and terminology** like everything
else, but they may use a wider register than reference prose: a longer
sentence, a rhetorical structure, or an established figure of speech ("USB-C
for ASR", "the cardinal sin") is acceptable there when it carries a true claim.
Reference prose keeps the no-idiom rule as written.

This standard does **not** govern, and must never change:

- **Code identifiers** — names are the contract. A prose rule never renames a
  symbol. American spelling applies to prose, not to a third-party or standard
  library name (write "the task is canceled" in prose; keep the symbol
  `CancelledError`).
- **Chinese documents** — `docs/spec/specification.md` (the normative spec),
  `docs/design-notes/`, `docs/research/`, `docs/work_doc/`, and other Chinese
  files. Do not translate or edit them. Read `docs/spec/specification.md` as
  the source of truth; the other Chinese documents are background context,
  ranked by the authority order in "Fact-check every meaning change" below.
- **reStructuredText roles and code spans** — `:class:`, `:func:`, `:meth:`,
  `:mod:`, `:data:`, double-backtick code spans, and `::` literal blocks stay
  verbatim. Never reflow or "simplify" the text inside them.
- **SPDX headers**, license text, and historical `CHANGELOG.md` entries.

## The two tiers

The standard has two tiers. Apply these three steps **in order** and stop at the
first that matches. They are ordered because one string can satisfy the first
two — a `#` comment inside a doctest matches both — and step 3's own list names
text the first two steps already claim; the first match wins.

1. **Verbatim (no tier applies).** Text inside a `:role:`, a `` `code span` ``,
   a `::` literal block, a Markdown fenced code block, or a doctest example —
   its `>>>`/`...` lines and its expected output — is never rewritten. Work
   around it, never through it.
2. **Internal (clarity tier).** A `#` comment. The clarity tier is the **full
   standard with only the sentence-length limit removed**,
   because a comment sometimes needs a long sentence to record a subtle reason.
   Do not shorten rationale to hit a word count. Keep the "why". A directive
   comment (`# noqa: ...`, `# pragma: no cover`, a section marker) is not prose
   and is governed by nothing here.
3. **User-facing (full standard).** Everything else in scope. This is the
   default, so an unlisted case falls here rather than escaping the standard.
   Among what it covers:
   - every string that can reach a person who is not reading the source —
     whichever way it is written: an argument to `raise ...Error(...)`,
     `logger.*`, `warnings.warn`, or `print`; a `hint=` or `param=` field; and
     a module-level constant that such a call later emits;
   - `argparse` `help=`, `description=`, and `epilog=`;
   - FastAPI and Pydantic text: `Field(description=...)`, `FastAPI(title=...)`,
     route and OpenAPI docstrings, and WebSocket `{"message": ...}` frames;
   - **every docstring**, public or private (mkdocstrings renders the public
     ones, and consumers read them all in the source);
   - every governed Markdown file.

**One length exception inside tier 3.** A docstring **body** paragraph that
states a contract obligation, a precedence rule, or an error-ownership boundary
is exempt from the sentence-length cap; nuance beats brevity there. The
docstring **summary line** is never exempt.

## The rules

Each rule area below is marked **KEEP** (apply as written), **ADAPT** (apply the
changed form), or **DROP** (does not apply here).

| Area | Ruling for this repo |
| --- | --- |
| **One word, one meaning** | **ADAPT.** Use the approved term for each concept from `TERMINOLOGY.md`, and never a forbidden synonym. A few words carry more than one meaning in this domain (*model*, *frame*, *adapter*, *provider*); `TERMINOLOGY.md` names every sense in use. Use a listed sense, make the sense clear from the sentence, and never introduce a sense the table does not list. |
| **Approved technical names** | **ADAPT.** The domain's `-ing` names are approved names, not banned gerunds: *streaming, gating, negotiation, diarization, resampling, coalescing, superseding*. Use them. |
| **Noun clusters ≤ 3 words** | **ADAPT (soft cap).** Keep new noun clusters to three words. Hyphenate a multi-word modifier so it reads as one unit ("frozen-prefix boundary"). Established compound API terms are exempt. Never break an identifier to meet the count. |
| **Verb tense** | **ADAPT.** Prefer the simple present and the imperative. Use the past tense for a past event. Avoid the perfect and progressive tenses where the simple tense is clear. |
| **Active voice** | **KEEP.** Write active voice. Use passive voice only when the actor is genuinely the runtime and naming it adds nothing ("The request is rejected with 422."). |
| **Sentence length** | **ADAPT / tier-split.** Keep a user-facing sentence short — about 20 words for an instruction, about 25 for a description. Keep the docstring **summary line** short. For the dropped limits, see "The two tiers". |
| **Imperative for instructions** | **KEEP.** Write instructions and `hint=` fields as commands ("Install the server extra: `pip install 'standard-asr[server]'`."). An instruction must name an action the reader can actually take: if the library offers no way to do the thing, say what happens instead. |
| **Articles** | **KEEP.** Use *a*, *an*, and *the* in prose. A telegraphic one-liner is allowed only where the convention omits them: an `argparse` `help=` string and a pydantic `Field(description=...)` ("List discovered models."). |
| **No slang or idioms** | **KEEP, with a carve-out.** Remove idioms and figurative phrases. Keep only a term of art the codebase already defines. |
| **Warnings and cautions** | **KEEP.** State the hazard first, then the action. Keep the `level`/`code`/`message` shape for diagnostics. |
| **Punctuation and symbols** | **KEEP.** No emoji or pictographs in any shipped text, ever. Typographic and mathematical symbols (→, ⇒, ⊆, §, ±) are allowed in docs, docstrings, and comments, but **not** in a runtime string: a message can be logged to a console that is not UTF-8, so keep `raise`, `logger.*`, and wire text ASCII. The CLI's status markers are ASCII for this reason (`[OK]`, `[FAIL]`, `[WARN]`, `[INFO]`). Keep code spans and roles verbatim. |
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
   The audience is the party the **obligation addresses**, not everyone who may
   read the sentence — a rendered docstring has many readers, and its MUST
   still addresses the engine author. Where the obligation addresses an
   **operator** or an **end user**, use a plain verb — an uppercase keyword
   only helps a reader who can act on the spec.

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

A rewrite is allowed to change what a text means, and it should where the
original is wrong, ambiguous, or misleading. Removing wrong or confusing copy is
a goal of this standard, not a risk to avoid. (The modal verbs in this section
are ordinary English, not RFC-2119 keywords; resolution 3 reserves the uppercase
forms for that.)

A change that alters the conveyed meaning is allowed only when both gates below
hold, and **you state both in the commit message**. A gate you cannot write down
is a gate you have not passed.

1. **Defect.** Name what is wrong with the original: it states a fact the code
   contradicts, it reads two ways, or it misleads about behavior. Quote the
   original. If you cannot name a defect, do not change the meaning — a pure
   style pass keeps the meaning and needs no gate.
2. **Authority.** Cite the source that establishes the new text, by path and
   line. Use whichever applies, in this order: the normative spec documents
   where they speak — `docs/spec/specification.md`, and the English
   `docs/spec/` pages for the surfaces they contract (the server wire API, the
   CLI, the download policy); otherwise the code path, a
   test that pins the behavior, or a design note. Much of the toolchain — an
   exit code, an `argparse` help string, a CLI marker — has no spec text; there
   the code and its tests are the authority.

Where the original is ambiguous, find the true intended meaning first, then
write it. Never guess. Never encode a wrong meaning to gain a shorter sentence.
Many names and statements are intentional and correct — prefer to clarify them
rather than change them. When no authority settles the point, leave the text
alone and open an issue: an unresolved question is cheaper than a confident
error.

## Terminology

Every domain term follows [`TERMINOLOGY.md`](TERMINOLOGY.md): one canonical term
per concept, a short definition, the approved usage, and the forbidden synonyms.
The controlled code vocabularies (the `DIAG_*` codes, the compliance codes, the
enums, and the `Literal` sets) have a single source of truth in the code;
`TERMINOLOGY.md` points to it and does not duplicate it (its one deliberate
excerpt, the two `level` scales, is quoted for contrast and moves with the
code).

## Checklist before you commit prose

- Each concept uses its canonical term from `TERMINOLOGY.md`.
- Spelling is American; no British forms in prose.
- No emoji anywhere; no non-ASCII symbol in a runtime string.
- User-facing sentences are short and active.
- Instructions and hints are imperative, and name an action the reader can take.
- Every meaning change states its defect and its authority in the commit message.
- Every claim about the code was checked against the code, not remembered.
- Code spans, roles, and identifiers are unchanged.
- `uv run ruff check` passes (pydocstyle included).
