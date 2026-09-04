<!-- SPDX-FileCopyrightText: 2026 Standard Voice Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Writing standard

This file defines how to write English prose in Standard ASR. The baseline is the Google developer documentation style guide, adopted here as the **pinned copy vendored in this repository** (`.vale/styles/Google`, errata-ai/Google v0.7.1) rather than as the live website. This file holds the scope, the tier system, the repo's deltas from that baseline, and the fact-check gate for meaning changes. Every contributor and every AI agent that works in this repository must follow it. `AGENTS.md` points here; the term rules live in [`TERMINOLOGY.md`](TERMINOLOGY.md).

Precedence: where this file speaks, it wins; where it is silent, the pinned baseline applies (its rules are listed in the appendix, so no network access is needed to know what it asks for); where both are silent, match the surrounding prose. Where any prose rule conflicts with a code fact (an identifier, a domain term, a cross-reference role), the code fact wins.

The baseline is pinned on purpose. A live URL cannot be a contract source: the [published guide](https://developers.google.com/style/) can change under us, so the same commit would mean one thing today and another next month, while CI kept enforcing the version it vendored. Treat the published guide as explanatory reading — helpful for the reasoning behind a rule, never the authority for whether prose is compliant. Upgrading the baseline is a deliberate commit: re-vendor the package, record the version here and in `.vale/styles/README.md`, and update the appendix (a test enforces that it matches).

The goal is one thing: a reader understands the behavior from the prose alone, with the least possible room for misunderstanding. Prose is part of the contract, so we hold it to the same rigor as the code.

## Why this baseline

The Google guide is written for developer documentation — API references, code samples, error messages — so it answers the questions this repo actually has (headings, lists, code font, link text, UI text). Its guide text is public under CC BY 4.0 and its Vale implementation is MIT, so the baseline can be vendored and read in place — an agent follows a rule it can open, not one it has to remember or fetch (see the appendix for the full list). That [Vale](https://vale.sh) implementation is maintained, so most of the baseline is enforced mechanically rather than by review (see "Enforcement" below). A few rules of ASD-STE100, the standard this repo adapted first, survive as deltas because they serve a contract-grade voice better than the baseline does; they are listed below, not implied.

## Scope: what this governs

This standard governs English **prose**:

- docstrings (module, class, function, attribute), in `src/` and in `tests/`;
- user-facing runtime strings (see the tier test below);
- English Markdown under `docs/`. The tree says which is which: `docs/content/` is the published documentation and is fully governed; `docs/internal/` holds Chinese documents, historical pages, and working notes, and `docs/site/` is the documentation application's own code tree — both are exempt, like `work/`, except the English the application renders on its own pages, which the front-door paragraph below covers;
- English Markdown at the repository root: `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `RELEASING.md`, this file, and `TERMINOLOGY.md`. Working notes under `work/` are exempt;
- internal `#` comments (clarity tier only — see below), in any language the repository uses. The Python ones are linted; the shell, workflow, and config ones are review-owned (see "Enforcement" for why the gate cannot read them);
- test prose: docstrings, comments, and assertion labels in `tests/`;
- new `CHANGELOG.md` entries from now on (by review: the mechanical gate cannot separate a new entry from the pre-standard history, so `CHANGELOG.md` stays outside `scripts/vale.sh`);
- writing that goes on GitHub: issue bodies and comments, pull request descriptions, and the prose in a commit message. Review owns these, because Vale cannot read GitHub. A tracking issue is often the first thing a reader sees.

Scope is not coverage. This standard governs every item above; the mechanical gate reaches a subset of them, and "Enforcement" states exactly which. Prose outside the gate's reach is not less governed — it is governed by review, and a reviewer is owed a precise list of what they own rather than a false sense that a green gate covered it.

`README.md`, `docs/content/index.md`, the `AGENTS.md` preamble, and the home page the documentation application renders from `docs/site/app/(home)/page.tsx` are the project's front door. They are governed for **accuracy and terminology** like everything else, but they may use a wider register than reference prose: a longer sentence, a rhetorical structure, or an established figure of speech ("USB-C for ASR", "the cardinal sin") is acceptable there when it carries a true claim. Reference prose keeps the no-idiom rule as written. The home page is the one front-door surface Vale cannot read, because it is TypeScript, so review owns it: its code samples, feature copy, and page metadata make claims about the API, and a claim there is checked against the code like any other.

This standard does **not** govern, and must never change:

- **Code identifiers** — names are the contract. A prose rule never renames a symbol. American spelling applies to prose, not to a third-party or standard library name (write "the task is canceled" in prose; keep the symbol `CancelledError`).
- **Chinese documents** — `docs/content/specification/protocol.md` (the normative spec) and the Chinese files under `docs/internal/` (`design-notes/`, `research/`, `work_doc/`, `misc.md`). Do not translate or restyle them; they change when their content changes. Read `docs/content/specification/protocol.md` as the source of truth; the other Chinese documents are background context, ranked by the authority order in "Fact-check every meaning change" below.
- **reStructuredText roles and code spans** — `:class:`, `:func:`, `:meth:`, `:mod:`, `:data:`, double-backtick code spans, and `::` literal blocks stay verbatim. Never reflow or "simplify" the text inside them.
- **SPDX headers**, license text, and historical `CHANGELOG.md` entries.

## The two tiers

The standard has two tiers. Apply these three steps **in order** and stop at the first that matches. They are ordered because one string can satisfy the first two — a `#` comment inside a doctest matches both — and step 3's own list names text the first two steps already claim; the first match wins.

1. **Verbatim (no tier applies).** Text inside a `:role:`, a `` `code span` ``, a `::` literal block, a Markdown fenced code block, or a doctest example — its `>>>`/`...` lines and its expected output — is never rewritten. Work around it, never through it.
2. **Internal (clarity tier).** A `#` comment. The clarity tier is the **full standard with only the sentence-length limit removed**, because a comment sometimes needs a long sentence to record a subtle reason. Do not shorten rationale to hit a word count. Keep the "why". A directive comment (`# noqa: ...`, `# pragma: no cover`, a section marker) is not prose and is governed by nothing here.
3. **User-facing (full standard).** Everything else in scope. This is the default, so an unlisted case falls here rather than escaping the standard. Among what it covers:
   - every string that can reach a person who is not reading the source — whichever way it is written: an argument to `raise ...Error(...)`, `logger.*`, `warnings.warn`, or `print`; a `hint=` or `param=` field; and a module-level constant that such a call later emits;
   - `argparse` `help=`, `description=`, and `epilog=`;
   - FastAPI and Pydantic text: `Field(description=...)`, `FastAPI(title=...)`, route and OpenAPI docstrings, and WebSocket `{"message": ...}` frames;
   - **every docstring**, public or private (the site's API reference renders the public ones, and consumers read them all in the source);
   - every governed Markdown file.

**One length exception inside tier 3.** A docstring **body** paragraph that states a contract obligation, a precedence rule, or an error-ownership boundary is exempt from the sentence-length cap; nuance beats brevity there. The docstring **summary line** is never exempt.

## Deltas from the Google guide

Each delta below overrides the baseline. The mechanically checkable ones map one-to-one to Google rules disabled in `.vale.ini`; do not re-enable such a rule there without deleting its delta here, and do not add a delta here without tuning Vale to match. `.vale.ini` also carries a second kind of disable that is NOT a style delta and is documented inline there instead: mechanical adaptations of the checker itself — document-shaped rules turned off for docstrings (`Google.Headings`, `Google.HeadingPunctuation`), extraction artifacts (`Vale.Repetition`, `Google.DateFormat`), the `Vale.Spelling` replacement (`StandardASR.Spelling`), and doctest syntax read as prose (`Google.Ellipses` for `plugins/discovery.py`). Those change how the tool reads the text, never what good prose is.

### Additions the baseline does not have

- **One word, one meaning.** Use the approved term for each concept from `TERMINOLOGY.md`, and never a forbidden synonym. A few words carry more than one meaning in this domain (*model*, *frame*, *adapter*, *provider*); `TERMINOLOGY.md` names every sense in use. Use a listed sense, make the sense clear from the sentence, and never introduce a sense the table does not list.
- **Define it, or do not use it.** Write for a reader who knows software but does not know this project. Every term must be one of three things: a word that reader already knows, a term `TERMINOLOGY.md` defines, or a name that appears in the code. Explain anything else where you first use it, or pick a plainer word. Inventing a phrase is the usual way this goes wrong. "The gate is the enforcement topology for every selected-engine surface" sounds precise and tells the reader nothing; "the core checks the protocol version in the same way everywhere" says the thing. No checker catches this, so review owns it.
- **Sentence-length targets.** Keep a user-facing sentence short — about 20 words for an instruction, about 25 for a description. Keep the docstring **summary line** short. The dropped limits are in "The two tiers" above. (A numbered target is retained from ASD-STE100 because an agent can act on a number; "keep it short" alone drifts.)
- **Hazard first.** A warning or caution states the hazard, then the action. Keep the `level`/`code`/`message` shape for diagnostics.
- **Actionable instructions.** Instructions and `hint=` fields are imperative and must name an action the reader can actually take: if the library offers no way to do the thing, say what happens instead.
- **Articles in prose.** Use *a*, *an*, and *the*. A telegraphic one-liner is allowed only where the convention omits them: an `argparse` `help=` string and a pydantic `Field(description=...)` ("List discovered models.").
- **Noun clusters.** Keep new noun clusters to three words (soft cap). Hyphenate a multi-word modifier so it reads as one unit ("frozen-prefix boundary"). Established compound API terms are exempt. Never break an identifier to meet the count.
- **ASCII runtime strings.** No emoji or pictographs in any shipped text, ever. Typographic and mathematical symbols (→, ⇒, ⊆, §, ±, ×) are allowed in docs, docstrings, and comments, but **not** in a runtime string: a message can be logged to a console that is not UTF-8, so keep `raise`, `logger.*`, and wire text ASCII. The CLI's status markers are ASCII for this reason (`[OK]`, `[FAIL]`, `[WARN]`, `[INFO]`).

  "Pictograph" is drawn by block, not by taste, and the **whole dingbats and miscellaneous-symbols range is banned** — check marks, ballot marks, stars, and the dingbat multiplication marks included. Several of those are not emoji by Unicode's own property, so the ban is deliberately wider than "emoji": a dingbat imitation of an operator has a real counterpart, and the real one is the correct character anyway. Write × (U+00D7), never the dingbat multiplication mark at U+2715; for a status column, use the ASCII markers above rather than a check-and-cross pair. The permitted symbols live in the arrow, math, and general-punctuation ranges, which the rule leaves alone. (This paragraph names the banned characters by code point for the same reason: the rule stays live here, so quoting one literally would fail the gate.)
- **RFC-2119 keywords by audience.** Keep "MUST", "MUST NOT", "SHOULD", and "MAY" uppercase when the sentence states a rule to an **engine author** (for example, a streaming event-construction error, or a docstring that cites the spec); the Google guide's lowercase "must" applies everywhere else. Soften to a plain verb when the sentence answers an **application developer** who made an ordinary call mistake ("candidate_languages MUST NOT contain 'auto'" becomes "candidate_languages cannot contain 'auto'"). The rule: state a spec obligation to the party who can break the spec; speak plainly to the party who made a normal mistake. The audience is the party the **obligation addresses**, not everyone who may read the sentence — a rendered docstring has many readers, and its MUST still addresses the engine author. Where the obligation addresses an **operator** or an **end user**, use a plain verb.
- **No column wrap in Markdown.** Do not break lines inside a paragraph at a column width. Let the editor wrap. If a paragraph looks too long as one line, split the paragraph organically. Do not reflow a paragraph you are not otherwise editing. Docstrings and comments live in code and keep the code's line length.

### Divergences where this repo overrides the baseline

- **Spaced em dashes.** Write `word — word`, not `word—word`. House style throughout. (Vale: `Google.EmDash` off.)
- **Logical quoting.** Punctuation goes outside the closing quote when the quoted text is an exact string, value, or message — which in this repo is nearly always. Never move a period inside quotes at the cost of misquoting a literal. (Vale: `Google.Quotes` off.)
- **Passive voice for the runtime actor.** Write active voice; use passive only when the actor is genuinely the runtime and naming it adds nothing ("The request is rejected with 422."). (Vale: `Google.Passive` off.)
- **Semicolons and parentheticals.** A semicolon may join tightly coupled clauses, and a parenthetical may carry contract nuance. Prefer short sentences first; do not delete nuance to satisfy a rhythm rule. (Vale: `Google.Semicolons`, `Google.Parens` off.)
- **Uncontracted verbs.** Prefer "is not" over "isn't" in reference and contract prose; the baseline prefers contractions for warmth, which is not this repo's register. Front-door surfaces may contract. (Vale: `Google.Contractions` off.)
- **No first-use acronym expansion for ubiquitous terms.** API, CLI, HTTP, JSON, PCM, URL, WAV, and peers need no expansion; `TERMINOLOGY.md` and the controlled code vocabularies govern domain terms. Expand a genuinely obscure acronym on first use. (Vale: `Google.Acronyms` off.)
- **Project voice.** "We" is allowed in the project-voice files (`README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `RELEASING.md`, `docs/content/index.md`, mission/goals/advisories); reference prose addresses the reader as "you" and avoids "we". (Vale: `Google.We` off for those files.)
- **Approved `-ing` names.** The domain's `-ing` subsystem names — *streaming, gating, negotiation, diarization, resampling, coalescing, superseding* — are approved technical names. Use them without hesitation; never "fix" them.
- **Precise terms kept against the Google word list.** The project vocabulary (`.vale/styles/config/vocabularies/StandardASR/accept.txt`) exempts words the baseline's word list would rewrite but that are precise or canonical here: *application* (protocol vocabulary; "app" is not), *file path* (distinct from a URL path and a `data:` URI), *disable/disabled* (a mechanical state), *abort*, *terminate*, and *kill* (three distinct failure semantics; "stop" is weaker than any of them), *above* (a position in a source file, or a numeric comparison), *touch* (the file-system sense), *cloud* (product names and "cloud storage"), *best* and *guarantee* ("best-effort" is canonical vocabulary; a stability guarantee is a protocol commitment, not marketing), *latest* ("latest wins" is exact coalescing semantics), and *sees*/*tells* (information-flow verbs for a code actor). Marketing puffery stays banned — by review, not by word list.
- **First person inside quotation marks.** A quoted first-person clause voices a stakeholder's perspective ("my code drove the session incorrectly", "use my own default cache") and is allowed; the documentation's own voice stays second person. (Vale: `Google.FirstPerson` off — its `i` token also misreads a loop index.)
- **Capitals after label colons.** A run-in bold label (`**Pre-1.0:** Minor releases may ...`), a goal ID (`G.1: Establish a ...`), and a stakeholder lead-in (`**Plugin authors**: Learn how ...`) capitalize the first word after the colon; a colon inside an ordinary sentence still introduces a lowercase word, by review. (Vale: `Google.Colons` off.)
- **Split test comments.** In `tests/`, one sentence may span two comments that bracket the code under assertion (`# The invalid name was reported...` above it, `# ...and the valid engine's checks still ran` below), keeping each claim attached to the exact line it verifies. (Vale: `Google.Ellipses` off in `tests/`; elided code in any comment still belongs in a code span. The same rule is off for `plugins/discovery.py`, whose doctest `...` lines are tier-1 verbatim syntax that Vale's docstring view reads as prose.)
- **Numbered section headings.** Spec pages and step-by-step guides number their headings ("## 3. REST endpoints"); the number is a stable cross-reference anchor (§3, §4.2). (Vale: `Google.HeadingPunctuation` off for those files.)

### Spelling

Use American spelling in prose: `normalize`, `behavior`, `initialize`, `serialize`, `analyze`, `color`, `canceled`/`canceling` (see `TERMINOLOGY.md`, "Spelling"). The only exception is a verbatim identifier: the prose says "canceled"; the symbol stays `CancelledError`. A third-party symbol keeps its own spelling.

## Fact-check every meaning change

This is the core rule for correctness. It applies to a rename **and** to a rewritten message.

A meaning change is one of two things, and what separates them is whether an authority already settles the point, not whether the edit touches a text or the code. Where an authority exists — the protocol, the code, a test — the change is a fact-check, and the two gates below apply to it: a text that describes the behavior wrongly is corrected to match, and so is a behavior that violates the protocol, with the same defect and authority in the commit message. Where no authority settles the point, the change is a design decision: a new behavior, a deliberate change to a settled one, or a rule about how to work in this repository, in this file or any other. A decision has no older source to cite, so its commit message states the decision and its reason, and it changes the authority itself — the spec, the code, or the rule — so that every text describing it follows. A behavior that contradicts the protocol is a defect until the protocol is changed with it; landing the code does not make it the authority. The rest of this section is about the fact-check.

A rewrite is allowed to change what a text means, and it should where the original is wrong, ambiguous, or misleading. Removing wrong or confusing copy is a goal of this standard, not a risk to avoid. (The modal verbs in this section are ordinary English, not RFC-2119 keywords; the delta above reserves the uppercase forms for that.)

A fact-check that alters the conveyed meaning is allowed only when both gates below hold, and **you state both in the commit message**. A gate you cannot write down is a gate you have not passed.

1. **Defect.** Name what is wrong with the original: it states a fact the code contradicts, it reads two ways, or it misleads about behavior. Quote the original. If you cannot name a defect, do not change the meaning — a pure style pass keeps the meaning and needs no gate.
2. **Authority.** Cite the source that establishes the new text, by path and line. Use whichever applies, in this order: the normative spec documents where they speak — `docs/content/specification/protocol.md`, and the English `docs/content/specification/` pages for the surfaces they contract (the server wire API, the CLI, the download policy); otherwise the code path, a test that pins the behavior, or a design note. Much of the toolchain — an exit code, an `argparse` help string, a CLI marker — has no spec text; there the code and its tests are the authority.

Where the original is ambiguous, find the true intended meaning first, then write it. Never guess. Never encode a wrong meaning to gain a shorter sentence. Many names and statements are intentional and correct — prefer to clarify them rather than change them. When no authority settles the point, leave the text alone and open an issue: an unresolved question is cheaper than a confident error.

## Terminology

Every domain term follows [`TERMINOLOGY.md`](TERMINOLOGY.md): one canonical term per concept, a short definition, the approved usage, and the forbidden synonyms. The controlled code vocabularies (the `DIAG_*` codes, the compliance codes, the enums, and the `Literal` sets) have a single source of truth in the code; `TERMINOLOGY.md` points to it and does not duplicate it (its one deliberate excerpt, the two `level` scales, is quoted for contrast and moves with the code).

## Enforcement

Three layers, weakest claim first:

- **`uv run ruff check`** enforces docstring structure (pydocstyle, Google convention) in `src/`. Tests and docs sample code are exempt from the structure rules (`pyproject.toml` per-file-ignores: a docstring is not forced onto every test function); their prose stays governed by this standard, checked by Vale and review like everything else.
- **`scripts/vale.sh`** lints the prose itself — Markdown plus the comments and docstrings in `src/` and `tests/` — against the vendored Google package and the `StandardASR` style (the mechanizable subset of `TERMINOLOGY.md`). The full run, warnings and suggestions included, is kept at zero, and the CI gate enforces exactly that: `scripts/vale.sh --gate` fails on any alert at any level, and `scripts/vale.sh --selfcheck` proves the gate composition (config, exemption glob, target list) still flags a planted violation in every target, root documents included, by mirroring the target layout into a temporary directory rather than writing to the working tree. Three extraction gaps are known and disclosed — prose that Vale never sees and review must own. Vale skips a module docstring that follows the SPDX header; it never reads Python string literals; and it skips attribute docstrings (the bare string under an assignment, as on an enum member). Module docstrings, tier-3 runtime strings, and attribute docstrings were each swept manually when the gate landed.

  The gate's corpus is Markdown plus `src/` and `tests/` Python (`scripts/vale.sh`, `TARGETS`). Prose in **shell, workflow, and config comments is in scope and review-owned**, not linted: Vale ships a comment extractor for Python but not for those formats, so mapping them would read their code as prose (measured on this repo: `esac`, `fi`, `printf`, and `pipefail` come back as misspellings, drowning the two real defects in the same run). Widening the corpus therefore waits for real extractor support; until then a reviewer owns those comments, and the sweep that landed this gate covered them once by hand. Two detector gaps are accepted rather than worked around: the serial comma is required (the baseline agrees), but `Google.OxfordComma` is off because its pattern cannot tell a two-item pair or an appositive from a list; and `Google.Spacing` is off for Python files because a dotted exception name in a Google-style `Raises:` key (`pydantic.ValidationError:`) is bare by docstring convention and reads to the rule as a missing sentence space. Review owns both.
- **Review** carries everything a regular expression cannot see: the right actor, a claim matching the code, the meaning-change gate above. Vale passing is not prose passing.

## Checklist before you commit prose

This list names every rule in this file, one line each, so that it is the whole pre-read; the section a line comes from is the ruling text when a line is unclear or disputed.

- Text inside a code span, a role, a fenced block, or a doctest is unchanged ("The two tiers").
- A `#` comment keeps its reason, even in a long sentence; a directive comment is not prose ("The two tiers").
- Each concept uses its canonical term from `TERMINOLOGY.md`.
- Every term is one the reader knows, one `TERMINOLOGY.md` defines, or one the sentence explains.
- Spelling is American; no British forms in prose.
- No emoji anywhere; no non-ASCII symbol in a runtime string.
- User-facing sentences are short and active; prose uses articles, and only an `argparse` help string or a `Field(description=...)` may be telegraphic.
- A new noun cluster has at most three words, hyphenated to read as one unit.
- Instructions and hints are imperative, and name an action the reader can take.
- A warning or caution states the hazard, then the action.
- MUST and SHOULD stay uppercase only where the sentence states a rule to the party who can break the spec ("Deltas", RFC-2119 keywords by audience).
- A Markdown paragraph is one source line; a paragraph you are not editing keeps its line.
- House style: spaced em dashes; punctuation outside a quoted literal; "is not" over "isn't" in reference prose; "we" only in the project-voice files; a capital after a run-in label's colon ("Divergences").
- Every change an existing authority settles states its defect and that authority in the commit message; a design decision states its reason and changes the authority itself.
- Every claim about the code was checked against the code, not remembered.
- Code spans, roles, and identifiers are unchanged.
- `uv run ruff check` passes (pydocstyle included).
- `scripts/vale.sh --gate` passes (it fails on any alert at any level).

## Appendix: the pinned baseline, rule by rule

The vendored package (`.vale/styles/Google`, errata-ai/Google v0.7.1) is the mechanized form of the baseline this standard adopts. It is listed here so that a contributor — or an agent with no network access — can see what the baseline asks for without leaving the repository, and so that an upgrade that adds or drops a rule cannot pass unnoticed: `tests/test_style_baseline.py` fails when this table and the vendored package disagree.

A rule marked **Off** is a deliberate house delta, explained in "Deltas from the Google guide" above and in `.vale.ini`; the standard still governs the underlying question, by review. **Off in Python only** marks a checker adaptation: the rule assumes a document and misreads docstrings. **Minus named files** means a per-file exception applies on top, for a reason `.vale.ini` records inline at that section. Note that the baseline is only the mechanized floor — this file wins wherever the two differ, and the rules in `.vale/styles/StandardASR` add what `TERMINOLOGY.md` requires on top.

| Rule | What it asks for | Status here |
| --- | --- | --- |
| `AMPM` | Write clock times as `9:00 AM`, with a space before it. | Enforced |
| `Acronyms` | Expand an unfamiliar acronym on first use. | Off — house delta |
| `Anthropomorphism` | Do not give software human qualities (a service does not `think` or `want`). | Enforced |
| `Colons` | Start lowercase after a colon. | Off — house delta |
| `Contractions` | Prefer contractions in user-facing prose. | Off — house delta |
| `DateFormat` | Write dates as `July 31, 2016`. | Off in Python only |
| `Ellipses` | Avoid ellipses in prose. | Enforced, minus named files |
| `EmDash` | Set em dashes tight, with no surrounding spaces. | Off — house delta |
| `ExcessiveClaims` | Drop unverifiable claims (`effortless`, `painless`). | Enforced |
| `Exclamation` | No exclamation points. | Enforced |
| `FirstPerson` | Avoid first-person singular pronouns. | Off — house delta |
| `Gender` | Do not use a gendered pronoun as the neutral one. | Enforced |
| `GenderBias` | Use gender-neutral role nouns. | Enforced |
| `HeadingPunctuation` | No period at the end of a heading. | Off in Python only, minus named files |
| `Headings` | Use sentence-style capitalization in headings. | Off in Python only |
| `Jargon` | Avoid jargon the audience may not share. | Enforced |
| `Latin` | Write `for example` and `that is`, not the Latin abbreviations. | Enforced |
| `LyHyphens` | No hyphen after an adverb ending in *-ly*. | Enforced |
| `OptionalPlurals` | No parenthesized plurals (`file(s)`). | Enforced |
| `Ordinal` | Spell out ordinals in text. | Enforced |
| `OxfordComma` | Use the serial comma. | Off — house delta |
| `Parens` | Use parentheses judiciously. | Off — house delta |
| `Passive` | Prefer active voice, naming the actor. | Off — house delta |
| `Periods` | No periods inside an acronym. | Enforced |
| `Quotes` | Put commas and periods inside quotation marks. | Off — house delta |
| `Ranges` | Write a numeric range without `from` or `between`. | Enforced |
| `Semicolons` | Use semicolons judiciously. | Off — house delta |
| `Slang` | No internet slang abbreviations. | Enforced |
| `Spacing` | One space after sentence-ending punctuation. | Off in Python only |
| `Spelling` | Use American spelling. | Enforced |
| `Timeless` | Avoid time-bound words (`currently`, `new`). | Enforced |
| `Units` | No-break space between a number and its unit. | Enforced |
| `We` | Avoid first-person plural. | Enforced, minus named files |
| `Will` | Prefer the present tense over `will`. | Enforced |
| `WordList` | Use the guide's preferred term. | Enforced |
| `WordListCase` | Use the guide's capitalization for a term. | Enforced |
