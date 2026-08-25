# Standard ASR

Standard ASR is a **Python library that defines and enforces a universal interface protocol for ASR (speech-to-text) inference**. Think USB-C for speech recognition, or what the OpenAI Chat Completion API did for LLMs: once a protocol becomes the common language, any new engine that adopts it is **instantly usable by every application in the ecosystem** — and any application that speaks it can use **any compliant engine** without changing a line of code.

**What this repo contains:**
- A **runtime library** (`standard-asr`): audio input negotiation & conversion, capability discovery & gating, structured diagnostics, streaming session management, plugin discovery via entry points.
- A **toolchain**: CLI, FastAPI server (expose any engine over HTTP/WS), compliance test suite.
- **No ASR models.** Each engine is a separate pip-installable plugin package (for example, `std-faster-whisper`, `std-openai`) that implements the standard interface. Standard ASR discovers installed plugins automatically.

**What this repo does NOT contain:** speech recognition code, model weights, or training. We build the bridge, not the endpoints.

**We are in the pre-release stage.** Always choose the long-term optimal design over backwards compatibility.

## Stakeholders — consider all three in every decision

- **App developers** (primary users): one stable interface for all engines. No vendor lock-in. Zero-config discovery.
- **ASR engine authors**: low barrier to publish a compliant plugin. Implement one interface → get CLI, reference server, compliance tests for free — and your engine is instantly compatible with every Standard ASR application, no per-app integration needed. Focus on models, not plumbing.
- **End users**: choose the best ASR for their language or domain — install a plugin, use it immediately, no app changes needed.

## Philosophy

- **Code is the contract.** Public API signatures, types, and docstrings are promises. A developer should understand behavior from the code alone. Every name is a design decision.
- **DX above all.** Optimize for the app developer. Zero-config, zero-surprise, zero-ambiguity. Battery-included where it helps (audio loading, SRT/VTT renderers), but keep heavy deps optional (`[audio]`, `[server]`).
- **Explicit > implicit.** Silent wrong results are the cardinal sin. When in doubt, fail loudly or emit a structured diagnostic — never silently degrade. When DX convenience and explicitness conflict, **correctness wins** (a loud error the developer can fix beats a silent wrong transcript).
- **Standard-library rigor.** This is infrastructure others build on for 10 years. Types complete, boundaries sharp, error paths explicit, no implicit behavior.
- **Security by default.** Credentials use `SecretStr`. URLs validated (HTTPS, no SSRF). Unsafe options require explicit opt-in.
- **Trust model: plugins are trusted code. The security layer defends against ACCIDENTS, not adversaries.** An installed engine plugin runs arbitrary in-process code — it can already read `os.environ` and open sockets, so no in-library boundary can contain a malicious plugin, and we do not build one. What we DO defend against is the honest mistake: a mis-pasted credential echoed back by pydantic's `input_value=...`, a secret riding inside an exception message into a log or a wire response. The whole defense is three cheap rules — scrub validation-error echoes, wrap credentials in `SecretStr`, never format a raw exception chain into operator logs or client responses. **Hard budget:** no proofs about third-party code, no introspection of pydantic/CPython internals, no machinery against hostile metaclasses / forged markings / subverted C slots — those attackers don't need our log path. A defense that needs its own review rounds to stay correct is a defect source, not a defense. Reviewers: do not file findings that assume an adversarial plugin author; they are out of scope by this definition.

## Rules

- Python 3.10+. Cross-platform (macOS, Windows, Linux). A platform- or
  hardware-specific feature must be optional: the core runs everywhere, with a
  graceful fallback or a clear error where the extra is absent.
- Modern typing syntax: `str | None` and built-in generics (`list[int]`), never
  `typing.Optional` / `typing.List`. PEP 8 naming: `snake_case` functions and
  variables, `PascalCase` classes, descriptive names.
- `uv` for deps. Pydantic v2 for data models. FastAPI for server. A new
  dependency needs a compatible license, active maintenance, and scrutiny of
  its supply chain; prefer the standard library and existing dependencies
  (`CONTRIBUTING.md`, "Dependency policy").
- `ruff` + `pyright` strict + `pytest` with 100% coverage target.
- `ruff` rule `NPY201` enabled. CI tests against numpy 1.26 AND latest 2.x.
- Google-style docstrings (English): summary, args, returns, raises.
- English for all code, comments, logs. `logging` module — no `print`.
- Prose follows [`STYLE.md`](STYLE.md) (the Google developer documentation
  style guide as the baseline, plus this repo's deltas) and
  [`TERMINOLOGY.md`](TERMINOLOGY.md) (canonical terms, American spelling). **Read
  both before you edit a docstring, a user-facing string, or a Markdown file.**
  Tier: text inside a role, a code span, a `::` literal block, a fenced code
  block, or a doctest is verbatim; a `#` comment is the clarity tier (the full
  standard minus the sentence-length cap); **everything else is the full
  standard** — every docstring, every string that can reach a user, and every
  governed Markdown file (`STYLE.md` defines the scope; this file is in it). No
  emoji, and no non-ASCII symbol in a runtime string. A meaning change must
  state its **defect** and its **authority** (spec section, code path, test, or
  design note — by path and line) in the commit message; prefer clarifying an
  intentional name over changing it. Check every claim about the code against
  the code — a remembered fact is not a fact.
- SPDX license header on every `.py` file:
  ```python
  # SPDX-FileCopyrightText: 2026 Standard Voice Contributors
  # SPDX-License-Identifier: Apache-2.0
  ```
- Documentation lives in `docs/`. `docs/content/` is the published site
  content: every Markdown file under it ships, a page needs a frontmatter
  `title`, sibling links use explicit relative paths (`./other-page.md`),
  and raw HTML fails the build. `docs/internal/` never ships. `docs/site/`
  is the site application (Node and pnpm; oxlint and oxfmt gate its code,
  and lychee link-checks the export in CI) — Python development never
  needs it. Read `docs/site/README.md` before changing the site.
- Commits: imperative mood, concise. One logical change per commit.
