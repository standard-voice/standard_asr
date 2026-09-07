<!-- SPDX-FileCopyrightText: The Standard ASR Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Writing standard

Write so a reader understands the behavior and can act on it without reading the implementation. Your reader knows software and has never seen this project. Give them the facts they need, in the order they need them.

Accuracy comes before style: no writing preference justifies changing a fact, hiding a limit, or respelling an identifier. When a writing rule meets a code fact, the code fact wins.

This file governs English writing here and wins wherever it differs from the baseline, the vendored `errata-ai/Google` Vale package under `.vale/styles/Google`, which checks part of the Google developer documentation style guide. The appendix lists every rule in that package. Where neither settles a choice, match what the surrounding text does.

## Scope

This standard covers:

- Docstrings, comments, and assertion messages in `src/` and `tests/`.
- English comments in the rest of the repository, including scripts, workflows, and configuration files.
- Text the library or the toolchain shows a person: errors, logs, warnings, diagnostics, CLI help, schema descriptions, server messages. What matters is where a string ends up, not where it is written, so an argument to `raise`, a `Field(description=...)`, and a WebSocket message frame all count.
- English Markdown under `docs/`, except `docs/internal/` and `docs/site/`.
- The root documents `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `RELEASING.md`, `TERMINOLOGY.md`, this file, and new `CHANGELOG.md` entries.
- What you write for this project on GitHub: issues and their comments, pull request descriptions, and commit messages.

`README.md`, `docs/content/index.md`, the part of `AGENTS.md` above its first `##` section, and the home page rendered from `docs/site/app/(home)/page.tsx` are the project's **front-door surfaces**: the first thing a reader meets. They may use a longer sentence and a settled figure of speech such as "USB-C for ASR", and the claim underneath still has to be true. The site is its own program with its own rules ([`docs/site/README.md`](docs/site/README.md)): this standard reaches the English it renders, not its code, and review is what checks that.

Outside this standard: Chinese documents, the working notes under `docs/internal/` and `work/`, license text, SPDX headers, and the `CHANGELOG.md` entries written before it. Keep a Chinese document in Chinese unless someone approves the translation; a wrong translation of a specification is worse than a late one. The writing rules in [`AGENTS.md`](AGENTS.md#writing) still cover every line you add to one.

Scope and tool coverage are different questions. "Enforcement" says what each tool reads. Text no tool reaches is governed all the same.

## Three tiers

Read the three tiers in order and stop at the first that matches.

1. **Exact text.** Identifiers, literal values, cross-reference targets such as a `:class:` role, code spans, fenced blocks, and the lines of an example — the `>>>` and `...` lines and the expected output — keep their exact characters, and a wording pass never touches them. A `#` comment inside an example is the one exception: tier 2 claims it. Correct a wrong example as a deliberate change: run it again and update what its output shows. If a check flags words inside an example, fix the code the example runs; never edit expected output to quiet a checker.
2. **Comments.** A comment gives the reason, the constraint, or the consequence a reader cannot see in the code beside it. Do not restate the code, and keep the comment next to what it explains. A tool directive such as `# noqa` or `# pragma: no cover` keeps the exact syntax the tool needs; words you add around it are an ordinary comment. A comment inside an example is a comment too: edit it here, then run the example again.
3. **Everything else.** The rest of this file applies in full. This tier is the default, so anything the first two do not claim lands here: every docstring, public or private, every string a person can see, and every governed Markdown file.

## Write for the reader

### Lead with the fact the reader came for

Start with the behavior, the result, or the action, then the conditions and the explanation that make it usable. Put an exception next to the rule it qualifies.

Name the actor and what it does: the application supplies a parameter, the runtime checks it, the engine transcribes the audio. Use the passive voice when the actor is unknown, already named, or beside the point, and never invent one just to avoid it.

Keep a sentence to one idea. Past about 25 words, look for a place to split it, and keep the long sentence only when splitting it would separate a condition from what it qualifies. Write ordinary grammar, articles included. Two surfaces may drop articles and write a fragment, because they are labels: an `argparse` `help=` string and a pydantic `Field(description=...)`, as in "List discovered models."

### Use terms the reader can place

**One word, one meaning.** Use the term [`TERMINOLOGY.md`](TERMINOLOGY.md) gives for each concept, and none of the synonyms it forbids. Some words carry several senses here; that file lists each one, so use a listed sense and write the sentence so the reader can tell which. The controlled code vocabularies, such as the `DIAG_*` codes, live in the code that `TERMINOLOGY.md` points at. Do not rename a concept for variety.

Every term is one your reader knows, one `TERMINOLOGY.md` defines, or a name in the code. Explain anything else where it first appears, or pick a plainer word for it. Naming is not explaining: say what a code name does the first time a reader meets it. Expand an acronym your reader may not know, and leave the common ones alone.

Keep the exact domain word even where a general word list prefers another one; the accepted vocabulary is `.vale/styles/config/vocabularies/StandardASR/accept.txt`, and a word the checker accepts is still not a reason to make a vague or inflated claim.

Unpack a pile of nouns when the relationship between them is unclear; three nouns in a row is the soft cap. "The limit on the size of an audio frame" is easier to read than "the audio frame size limit constraint". A hyphen can tie a modifier together, but it cannot rescue an invented phrase. Keep a settled term such as *streaming* or *diarization* when that is the concept you mean.

### Cut filler, not meaning

Delete a sentence that announces an explanation, praises the design, or repeats what you just said. Replace a vague benefit with the behavior behind it. Do not manufacture jargon or slogans to make an ordinary fact sound big: "the gate is the enforcement topology for every selected-engine surface" hides the actor and the action, where "the core checks the protocol version the same way everywhere" says the thing.

Keep every qualification that affects correctness: a limit, an uncertainty, a failure. A shorter sentence that drops one is worse than the long one. Skip jokes, cultural references, and idioms in an instruction, and keep the tone level: do not blame the reader or call an ordinary error a disaster.

Use a paragraph for connected reasoning, a list for steps or parallel items, and a table for a comparison. Do not turn every sentence into a bullet.

Do not wrap prose at a column width; let the editor wrap it. A docstring and a comment live in code and keep the code's line length.

## What each surface needs

### Docstrings

Follow the Google docstring structure that `pyproject.toml` configures. Open with a short summary of what the thing does, then give what a caller needs to use it correctly: the inputs, the return value, the constraints, the side effects, and the errors. Say who owns a decision, and which rule wins, where a caller has to choose between them. Do not repeat a type annotation unless the sentence adds to it.

Describe the promise, not the steps that implement it. A private docstring tells a maintainer what they need. A test docstring says what behavior the test pins; a test needs no docstring just to fill a template.

### Messages and instructions

An error says what failed and names the input or the operation it failed on, and adds the remedy when one exists and the recipient can reach it. An instruction, a `hint=` field included, names one concrete action, usually with an imperative verb. Never invent a recovery step: when nothing can be done, say what happens instead.

A warning says what can go wrong, or what was lost, then what to do about it, in a tone that matches the real risk. Keep a structured diagnostic's fields and codes as they are: a better message never justifies changing the data model.

Uppercase the RFC 2119 words `MUST`, `MUST NOT`, `SHOULD`, and `MAY` only in a sentence that states or quotes a rule of the specification, and make the obligated party plain. Uppercase does not make an ordinary error clearer: for a caller who made a mistake, the validator's `candidate_languages cannot contain 'auto'.` (`effective_candidate_languages`, in `src/standard_asr/contract/language.py`) is the right tone.

### GitHub writing

Open an issue or a pull request description with the problem and the behavior it produces, and give a reader who was not in the discussion enough to reproduce or check it. Say what you verified, and where that check stops. Leave out the history of your drafts unless it explains a decision. `AGENTS.md` holds the commit rules: one logical change per commit, with a concise imperative subject.

## Spelling and characters

### Spelling

Use American spelling: `normalize`, `behavior`, `canceled` (`TERMINOLOGY.md`, "Spelling"). Never respell an identifier: the sentence says "canceled" and the symbol stays `CancelledError`. A third-party name keeps its own.

### ASCII runtime strings

Keep the fixed text the library and the toolchain emit in ASCII, so a message survives a console that cannot decode Unicode. The CLI status markers are `[OK]`, `[FAIL]`, `[WARN]`, and `[INFO]`. Data passing through is untouched: a transcript, a path, or a value quoted back to the user keeps whatever characters it has. Show such a value in a code span, which the checker does not read inside.

Use no emoji and no pictograph in anything we write or ship. Typographic and mathematical symbols such as →, ⊆, §, ±, and × are fine in documentation, docstrings, and comments. `.vale/styles/StandardASR/Emoji.yml` holds the exact ranges the checker rejects, and it goes wider than emoji: three whole blocks of the Unicode table are out (U+2600 to U+27BF, U+2B00 to U+2BFF, U+1F000 to U+1FAFF), a plain dingbat check mark or multiplication mark (U+2715) included. Write the real multiplication sign (U+00D7) instead of the dingbat, and use the ASCII markers above for a column of passes and failures.

## Check facts and changes in meaning

Check every claim about the code against the code, and take each fact from the source that owns it:

- The pages under `docs/content/specification/` state what the protocol, the wire API, the CLI, and the download policy require.
- A type signature and its docstring state what the Python API promises.
- The implementation shows what the library does; a test is evidence for the behavior it exercises.
- `TERMINOLOGY.md` owns the terms, and points at the code that owns each controlled vocabulary.

When two disagree, do not settle it by editing whichever is easiest to change. Work out which is wrong: the implementation breaks a requirement, the description is wrong, or the design needs to change. Behavior that contradicts the protocol is a defect until the protocol changes with it; landing the code does not make it the authority. A design note explains intent and never overrides the published contract.

Where an authority settles the change, name the defect in the commit message and cite that authority by path and line, quoting the original words where that makes the defect clear. Where none does, the change is a decision: state it and its reason, change the source that owns the fact, and carry the change into every description of it.

A wording change that keeps the meaning needs no such note. Resolve an ambiguity that affects behavior before you rewrite the claim, and never guess at a contract to make a sentence read better. Where the decision is not yours, say what the open question is, leave that text alone, and make the edits that do not depend on it; open an issue when the question outlives the change.

## Deltas from the Google guide

A delta is a rule where this file overrides the baseline. Each one below names the Google rule it switches off in `.vale.ini`. Do not switch a rule back on without deleting its delta, and do not add a delta without changing `.vale.ini`.

- **Spaced em dashes.** Write `word — word`, not `word—word`. (`Google.EmDash` off.)
- **Logical quoting.** Punctuation goes outside the closing quote when the quoted text is an exact string, value, or message, which here it nearly always is. Never misquote a literal to move a period inside. (`Google.Quotes` off.)
- **Passive voice where it serves.** The baseline asks for active voice everywhere; here the question is whether naming the actor adds anything. "The request is rejected with 422" is fine. (`Google.Passive` off.)
- **Semicolons and parentheticals.** A semicolon may join two tightly coupled clauses, and a parenthetical may carry a nuance of the contract. Reach for a short sentence first, and never drop a nuance for rhythm. (`Google.Semicolons`, `Google.Parens` off.)
- **Uncontracted verbs.** Write "is not", not "isn't": a negation in a contract has to be impossible to miss, and the baseline's contractions are warmer than a reference should sound. The front-door surfaces named in "Scope" may contract. (`Google.Contractions` off.)
- **No first-use expansion for a common acronym.** API, CLI, HTTP, JSON, PCM, URL, WAV, and their peers need none. Expand one that is genuinely obscure. (`Google.Acronyms` off.)
- **First person inside quotation marks.** A quoted first-person clause voices a stakeholder ("my code drove the session incorrectly"). The document's own voice stays second person. The rule cannot come back in any case: its `i` token also matches a loop index. (`Google.FirstPerson` off.)
- **Capitals after a label colon.** A run-in bold label (`**Pre-1.0:** Minor releases`), a goal identifier (`G.1: Establish`), and a docstring section key (`Args:`) capitalize the next word. A colon inside an ordinary sentence still takes a lowercase word, and review owns that half. (`Google.Colons` off.)
- **Project voice.** "We" belongs to the project-voice documents: `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `RELEASING.md`, `TERMINOLOGY.md`, this file, `docs/content/index.md`, the mission and goals pages, and `docs/compatibility-advisories.md`. Everywhere else, say "you" and name the actor. (`Google.We` off for those files.)
- **Numbered section headings.** A specification page and a step-by-step guide number their headings ("## 3. REST endpoints"), because the number is a stable cross-reference. The exemption is per file: number a heading only in a file `.vale.ini` lists, adding the file there first if it belongs, and keep an existing number when you edit around it. (`Google.HeadingPunctuation` off for those files.)
- **Split test comments.** In `tests/`, one sentence may span two comments that bracket the code under assertion, so each half sits on the line it verifies. Elided code inside a comment still belongs in a code span. (`Google.Ellipses` off in `tests/`, and in `plugins/discovery.py`, whose doctest continuation lines the checker reads as sentences.)

## Enforcement

Run `prek run --all-files` and `uv run pytest` before you call a change done (`AGENTS.md`, and [`CONTRIBUTING.md`](CONTRIBUTING.md#git-hooks-prek)). The test suite is a push hook, so `prek run` does not run it, and neither the checks nor the tests replace review.

- `uv run ruff check` checks docstring structure, with pydocstyle under the Google convention. `pyproject.toml` exempts `tests/` and the sample code under `docs/` from the structure rules; what they say is governed all the same.
- `scripts/vale.sh --gate` runs the vendored Google rules, the `StandardASR` rules, and Vale's own, and fails on any alert at any level, a suggestion included. The target and exemption arrays in that script are the one statement of what Vale reads.
- `scripts/vale.sh --selfcheck` plants a violation in a temporary copy of the layout and proves the gate still catches it. Run it whenever you change the Vale configuration, a rule, or the target list.
- `uv run pytest tests/test_style_baseline.py` checks that the appendix lists the vendored rules with the status `.vale.ini` gives each one.

Vale reads the governed Markdown plus the comments and docstrings it can extract from `src/` and `tests/`. It cannot see a module docstring that follows the SPDX header, an attribute docstring, or any other Python string literal, and it does not read the site's rendered English, the comments in other file formats, or anything on GitHub. All of that belongs to review, and so does every `CHANGELOG.md` entry: the whole file sits outside the gate, because the gate cannot tell a new entry from the history written before this standard.

`.vale.ini` switches a rule off for one of two reasons, and says which: a delta declared above, or a workaround for a checker that misreads what it checks. A workaround leaves the writing rule standing: the serial comma is still required though `Google.OxfordComma` cannot tell a list from a pair, and a sentence still takes one space after its period though `Google.Spacing` misreads a Google-style `Raises:` key. Change this file, `.vale.ini`, and the appendix together. Fix a false positive at the narrowest scope that works, and never damage a correct sentence to quiet a checker.

Review carries what no regular expression can: the right actor, a claim that matches the code, the rules above about meaning, and every surface the tools cannot reach. A clean run of the checks establishes none of it.

## Checklist before you commit

Read the sections you need before you write. Use this on what you wrote.

- Does the reader get the useful fact first, and every term they need to place it?
- Did anything vague, decorative, or invented get in?
- Is every claim about the code checked against the code, and every limit, condition, and literal still there?
- Do comments explain reasons, docstrings explain promises, errors name what failed, and warnings state the risk first?
- Do the conventions hold: American spelling, ASCII in emitted text, house punctuation, "we" only in a project-voice document, no column wrap in Markdown?
- Is `MUST` or `SHOULD` uppercase only where the sentence states a rule of the specification, with the obligated party named?
- Does the commit message name the defect and its authority, or state the decision and its reason?
- Did `prek run --all-files` pass, along with the tests the change needs?

## Appendix: the pinned baseline, rule by rule

The baseline is the vendored `errata-ai/Google` Vale package, v0.7.1, under `.vale/styles/Google`. It checks part of the Google developer documentation style guide, not the whole of it, and the published website is background reading, never the authority. The table is here so a contributor, or an agent with no network access, can read the baseline without leaving the repository.

**Off** means the checker is disabled everywhere, as a delta declared above or because it misreads what it checks; `.vale.ini` gives the reason in each case. **Off in Python only** and **minus named files** mark narrower exceptions. A disabled checker does not retire the writing rule, and `.vale/styles/StandardASR` adds what `TERMINOLOGY.md` requires on top. The accepted vocabulary subtracts too: `accept.txt` holds `sees` and `tells`, the only two words `Anthropomorphism` looks for, so that checker catches nothing here; it also holds `best`, `guarantee`, and `latest`. Review owns every word the vocabulary lets through.

Upgrade the vendored package in its own commit: re-vendor it, record the version in `.vale/styles/README.md` and here, and update this table. `tests/test_style_baseline.py` fails when the table and the package disagree on a rule or a status; only review can check that a summary is true.

| Rule | What it asks for | Status here |
| --- | --- | --- |
| `AMPM` | Write clock times as `9:00 AM`, with a space before it. | Enforced |
| `Acronyms` | Expand an unfamiliar acronym on first use. | Off |
| `Anthropomorphism` | Do not give software human qualities; the copy checks only `sees` and `tells`. | Enforced |
| `Colons` | Start lowercase after a colon. | Off |
| `Contractions` | Prefer a contraction to the spelled-out form. | Off |
| `DateFormat` | Write dates as `July 31, 2016`. | Off in Python only |
| `Ellipses` | Avoid an ellipsis in a sentence. | Enforced, minus named files |
| `EmDash` | Set em dashes tight, with no surrounding spaces. | Off |
| `ExcessiveClaims` | Drop an unverifiable claim: `best`, `simplest`, `fastest`, `guarantee`. | Enforced |
| `Exclamation` | No exclamation points. | Enforced |
| `FirstPerson` | Avoid first-person singular pronouns. | Off |
| `Gender` | Do not use a gendered pronoun as the neutral one. | Enforced |
| `GenderBias` | Use gender-neutral role nouns. | Enforced |
| `HeadingPunctuation` | No period at the end of a heading. | Off in Python only, minus named files |
| `Headings` | Use sentence-style capitalization in headings. | Off in Python only |
| `Jargon` | Avoid jargon the audience may not share. | Enforced |
| `Latin` | Write `for example` and `that is`, not the Latin abbreviations. | Enforced |
| `LyHyphens` | No hyphen after an adverb ending in *-ly*. | Enforced |
| `OptionalPlurals` | No parenthesized plurals (`file(s)`). | Enforced |
| `Ordinal` | Spell out ordinals in text. | Enforced |
| `OxfordComma` | Use the serial comma. | Off |
| `Parens` | Use parentheses judiciously. | Off |
| `Passive` | Prefer active voice, naming the actor. | Off |
| `Periods` | No periods inside an acronym. | Enforced |
| `Quotes` | Put commas and periods inside quotation marks. | Off |
| `Ranges` | Do not mix range forms: no `from` or `between` before a hyphenated range. | Enforced |
| `Semicolons` | Use semicolons judiciously. | Off |
| `Slang` | No internet slang abbreviations. | Enforced |
| `Spacing` | One space after sentence-ending punctuation. | Off in Python only |
| `Spelling` | Use American spelling. | Enforced |
| `Timeless` | Avoid a time-bound word: `currently`, `latest`, `soon`. | Enforced |
| `Units` | No-break space between a number and its unit. | Enforced |
| `We` | Avoid first-person plural. | Enforced, minus named files |
| `Will` | Prefer the present tense over `will`. | Enforced |
| `WordList` | Use the guide's preferred term. | Enforced |
| `WordListCase` | Use the guide's capitalization for a term. | Enforced |
