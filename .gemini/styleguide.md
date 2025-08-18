`version: 2025.08.10-1`

# standard asr


---


## 1. Core Project Context

This is a standard ASR (Automatic Speech Recognition) library that provides a simple interface for interacting with various ASR models. It supports many popular ASR models and allows developers to easily transcribe audio into text.

This project supports Python 3.10 and above. It is designed to be easy to use and integrate into existing applications.


  - **Key Principles:**
      - **Offline-Ready:** Core functionality MUST work without an internet connection.
      - **Separation of Concerns:** Strict frontend-backend separation.
      - **Clean code:** Clean, testable, maintainable code, follows best practices of python 3.10+ and does not write deprecated code.

Some key files and directories:

```
docs/                   # Documentation files
.github/               # GitHub configuration files
src/standard_asr/          # Core ASR library code
pyproject.toml       # Project metadata and dependencies
README.md            # Project overview and instructions
uv.lock              # Dependency lock file
```


## 1. Overarching Coding Philosophy

  - **Simplicity and Readability:** Write code that is simple, clear, and easy to understand. Avoid unnecessary complexity or premature optimization. Follow the Zen of Python.
  - **Single Responsibility:** Each function, class, and module should do one thing and do it well.
  - **Performance-Aware:** Be mindful of performance. Avoid blocking operations in async contexts. Use efficient data structures and algorithms where it matters.
  - **Adherence to Best Practices**: Write clean, testable, and robust code that follows modern Python 3.10+ idioms. Adhere to the best practices of our core libraries (FastAPI, Pydantic v2).

## 3. Detailed Coding Standards

### 3.1. Formatting & Linting (Ruff)

  - All Python code **MUST** be formatted with `uv run ruff format`.
  - All Python code **MUST** pass `uv run ruff check` without errors.
  - Import statements should be grouped by standard library, third-party, and local modules and sorted alphabetically (PEP 8).

### 3.2. Naming Conventions (PEP 8)

  - Use `snake_case` for all variables, functions, methods, and module names.
  - Use `PascalCase` for class names.
  - Choose descriptive names. Avoid single-letter names except for loop counters or well-known initialisms.

### 3.3. Type Hints (CRITICAL)

  - Target Python 3.10+. Use modern type hint syntax.
  - **DO:** Use `|` for unions (e.g., `str | None`).
  - **DON'T:** Use `Optional` from `typing` (e.g., `Optional[str]`).
  - **DO:** Use built-in generics (e.g., `list[int]`, `dict[str, float]`).
  - **DON'T:** Use capitalized types from `typing` (e.g., `List[int]`, `Dict[str, float]`).
  - All function and method signatures (arguments and return values) **MUST** have accurate type hints. If third party libraries made it impossible to fix type errors, suppress the type checker.

### 3.4. Docstrings & Comments (CRITICAL)

  - All public modules, functions, classes, and methods **MUST** have a docstring in English.
  - Use the **Google Python Style** for docstrings.
  - Docstrings **MUST** include:
    1.  Summary.
    2.  `Args:` section describing each parameter, its type, and its purpose.
    3.  `Returns:` section describing the return value, its type, and its meaning.
    4.  (Optional but encouraged) `Raises:` section for any exceptions thrown.
  - All other code comments must also be in English.

### 3.5. Logging

  - Use `logging` module for all informational or error output.
  - Log messages should be in English, clear, and informative. Use emoji when appropriate.

## 4. Architectural Principles

### 4.1. Dependency Management

  - First, try to solve the problem using the Python standard library or existing project dependencies defined in `pyproject.toml`.
  - If a new dependency is required, it must have a compatible license and be well-maintained.
  - Use `uv add`, `uv remove`, `uv run` instead of pip to manage dependencies. If user uses conda, install uv with pip then.
  - After adding a new dependency, in addition to `pyproject.toml`, you must add the dependency to `requirements.txt` as well.

### 4.2. Cross-Platform Compatibility

  - All core logic **MUST** run on macOS, Windows, and Linux.
  - If a feature is platform-specific (e.g., uses a Windows-only API) or hardware-specific (e.g., CUDA), it **MUST** be an optional component. The application should start and run core features even if that component is not available. Use graceful fallbacks or clear error messages.
