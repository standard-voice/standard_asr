# Standard ASR — Copilot instructions

The canonical guidance for this repository is **[`AGENTS.md`](../AGENTS.md)**.
Read it first. Prose rules are in **[`STYLE.md`](../STYLE.md)**; terminology is
in **[`TERMINOLOGY.md`](../TERMINOLOGY.md)**. This file repeats only the rules
that a code assistant most often breaks. When this file and `AGENTS.md` differ,
`AGENTS.md` wins.

## What this project is

Standard ASR is a Python library that defines and enforces a universal interface
protocol for ASR (speech-to-text) inference. It ships **no** ASR models. Each
engine is a separate pip-installable plugin. The runtime discovers installed
plugins, negotiates audio, gates parameters against declared capabilities, and
returns a constant-shape result. Python 3.10 and later. Cross-platform (macOS,
Windows, Linux).

## Non-negotiable rules

- **Prose:** follow `STYLE.md` (adapted ASD-STE100) and `TERMINOLOGY.md`
  (canonical terms). Use **American spelling**.
- **No emoji** in any shipped text — messages, logs, docstrings, or docs. The
  CLI uses ASCII markers (`[OK]`, `[FAIL]`, `[WARN]`, `[INFO]`).
- **Docstrings:** Google style, English. Include a summary, then `Args:`,
  `Returns:`, and `Raises:` where they apply.
- **Logging:** use the `logging` module. Never use `print` for library output.
- **Type hints:** modern syntax. Use `str | None` and `list[int]`, not
  `Optional[str]` or `List[int]`. Every signature is fully typed.
- **Naming (PEP 8):** `snake_case` for variables, functions, and modules;
  `PascalCase` for classes. A name is a design decision — choose it with care.
- **A meaning change is gated.** A rename or a reworded message must be
  necessary and fact-checked against the spec (`docs/spec/specification.md`) and
  the real behavior. Many names are intentional; prefer to clarify them.
- **Checks:** all code must pass `uv run ruff format`, `uv run ruff check`
  (pydocstyle included), and `uv run pyright`.
- **Dependencies:** prefer the standard library and existing dependencies. Use
  `uv add`/`uv remove`/`uv run`, not `pip`.
- **Cross-platform:** core logic runs on macOS, Windows, and Linux. A
  platform-specific or hardware-specific feature is optional, with a graceful
  fallback or a clear error.
