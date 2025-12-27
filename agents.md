`version: 2025.09.1-1`

# standard asr

## 1. Core Project Context

This is a standard ASR (Automatic Speech Recognition) library that provides a simple interface for interacting with various ASR models. It supports many popular ASR models and allows developers to easily transcribe audio into text.

This project supports Python 3.10 and above. It is designed to be easy to use and integrate into existing applications.


**Key Principles:**
  - **Clean code:** Clean, testable, maintainable code, follows best practices of python 3.10+ and does not write deprecated code.
  - **Follow guiding principles and mission:** Strictly follows design philosophies, goals, and missions stated in `docs/mission.md`, `docs/misc.md`, and `docs/goals.md`. Use these as the guide and 验收清单 in designing and code reviewing stage. DX, elegancy and the stated mission and goals are the top priorities in this project.

Some key files and directories:

```
docs/                   # Documentation files
.github/               # GitHub configuration files, inlcuding workflows
src/standard_asr/          # Core library code
pyproject.toml       # Project metadata and dependencies
README.md            # Project overview and instructions
```

## 1. Overarching Coding Philosophy

**Adherence to Best Practices**: Write clean, testable, and robust code with proper design patterns that follows modern Python 3.10+ idioms. Adhere to the best practices of our core libraries (FastAPI, Pydantic v2).

Tech Stack
- astral uv: `uv add`, `uv remove`, `uv run`
- fastapi, pydantic v2, ruff, pyright strict, pytest with 100% test cov.

Code style
- Google python style docstring in English for everything.
- Docstrings **MUST** include summary, args, returns, and raises.
- Use English for all comments and logs.
- Use `logging` module.
- Use Python standard library or existing project dependencies defined in `pyproject.toml`.

All core logic **MUST** runs on macOS, Windows, and Linux.

Documentation
- write clear, verbose, comprehensive documentation in `docs`. Always consider mission, goals, stakeholders, and the characteristics and goals of the people who will read our documentation.

Comprehensive tests

Everything we do are for the mission stated in `docs/mission.md` and `docs/goals.md`. If some of the documented things conflict with each other, ASK ME.


## 工作方式

先写核心验收文档 `work/criteria.md`，设计方案 `work/plan.md`，根据验收文档审查设计方案，再写 todo `work/todo.csv`，然后把所有 todo 完成。todo 中要包含所有任务，比如文档，所有的 implementation 任务，完成某个 feat 任务之后的代码审查，测试，和完成所有任务之后最后的 criteria 验收，所有改动要与 mission 和 goals 高度对齐。把东西写成带 tick box 的 todo 项目，方便追踪进度。

在合适的时候，往 git 中提交改动。遵守 conventional commit 规范。如果在 git changes 中发现并非由你做出的改动，分析那些改动之后，分批进行提交。

在提交之前，总是审核代码，运行测试，linter, type checker，审核 todo 完成状况，并且再看看 todo 是否需要更新或添加更多 todo。最终提交时，分批提交，让 commit 原子化。

**总是主动更新 todo.csv，并把 todo.csv 中的所有任务都做完**

