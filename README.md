# Standard ASR

[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Static Badge](https://img.shields.io/badge/Join%20Chat-Zulip?style=flat&logo=zulip&label=Zulip&color=blue&link=https%3A%2F%2Fstandard-voice.zulipchat.com)](https://standard-voice.zulipchat.com)


> ⚠️⚠️⚠️ Standard ASR is still work in progress!! Breaking changes may be introduced at any moment!!
> 
> For production use, please wait until `v1.0.0` release, where we will be stabilizing the APIs and enforce migration policy when breaking changes do happen. We strictly follow semantic versioning.
>
> Please test out standard library and give us feedback or your opinion. Let's shape the future of ASR library together!

![standard_asr_concept](docs/assets/concept.jpg)


## Introduction

**Standard ASR** is a universal protocol that standardizes how applications interact with Automatic Speech Recognition (ASR) engines.

> **Think of it as USB for speech recognition** — a common interface that lets any application work with any ASR engine, seamlessly.

### Why Standard ASR?

**Write once, run with any ASR engine.** Instead of writing custom integration code for every ASR library, you write it once against the Standard ASR protocol. Your application automatically works with any compliant ASR engine — today's and tomorrow's.

**ASR engines as plug-and-play modules.** With a standard protocol, ASR models become true plugins. Your application stays completely agnostic of which engine it uses. Each engine declares its capabilities, and your app adapts dynamically. Support the latest models on day one — without changing a single line of code.

**Future-proof your application.** The AI landscape evolves rapidly. New engines emerge, others become obsolete. Standard ASR ensures your application survives these changes effortlessly.

### Who Benefits?

**For ASR engine developers & researchers:**
- Focus on what matters — building better models — not boilerplate code
- Get CLI, Web API, and testing tools for free by implementing one interface
- Ship professional, production-ready packages with minimal engineering effort
- Reach every application in the Standard ASR ecosystem instantly

**For application developers:**
- Write ASR integration code once; it works with all compliant engines
- Zero vendor lock-in — switch providers without rewriting business logic
- Escape dependency hell with clean, isolated plugin architecture
- Automatic model discovery — your app adapts to whatever the user installs

**For end users:**
- Access cutting-edge models faster as integration barriers disappear
- Choose any ASR engine that fits your language or domain — not what developers picked for you

## Entrypoint Quickstart

Standard ASR discovers compliant plugins through the ``standard_asr.models``
entrypoint group. Each plugin exposes one or more model presets using keys like
``<engine_id>/<model_name>``. A tiny demo plugin ships in ``cookbook/std_dummy_asr``
so you can try the workflow without extra dependencies:

```bash
uv run uv pip install -e cookbook/std_dummy_asr
uv run standard-asr models list
uv run standard-asr compliance entrypoints
uv run python cookbook/sample_client.py
```

The sample client will discover the installed model, instantiate it, and print a
synthetic transcript. Use this flow as a template when building your own plugin.

## Python Usage

```python
from standard_asr import discover_models
import numpy as np

registry = discover_models()
asr = registry.create("dummy/echo")
audio = np.zeros(16_000, dtype=np.float32)
result = asr.transcribe(audio)
print(result.text)
```

## CLI Quick Usage

```bash
standard-asr models list
standard-asr transcribe dummy/echo path/to/audio.wav
```

## FastAPI Server (Optional)

```bash
pip install "standard-asr[server]"
standard-asr serve --host 0.0.0.0 --port 8000
```

See `docs/spec/api.md` for API details.



---


- Strictly follows semantic versioning
- Pydantic v2 to model ASR's settings
- Fully async support
- pytest (strict mode passed)
- use logging


# A: 核心目标:
- A.1: 做 ASR 推理领域的 usb 标准: 提供通用的接口，让 ASR 推理开发者，ASR 使用者，能有一个共同的标准，互相沟通。
- A.2: 提供测试套件和周边工具，让 ASR 库开发者更好的开发好用，稳健，工程化的库
- A.3: 适配过 Standard ASR 标准的代码，应该可以免配置直接跑任何 Standard ASR compliant 的模型。如果有额外配置项，需要用 pydantic 暴露出去，让 WebUI 和 GUI 和数据库能动态生成配置。


Standard ASR:
- 不是一个包含所有 ASR 模型或所有 ASR 模型的 adapter 的库。Standard ASR 只定义协议和接口并提供配套工具。与具体 ASR 引擎的适配，应该由 ASR 引擎的开发者 (或者 ASR 引擎的 fork) 来完成。

许多 ASR 包尝试去做一个工具箱，把所有 ASR 模型都塞进去。我们尝试做 ASR 模型和应用开发者的桥梁，提供共同的语言来相互交流。

为什么不把所有 ASR 模型都塞到 standard asr 中，让开发者自己选要用什么模型？
- It doesn't scale. Many ASR models are licensed under GPL or AGPL or resctrictive license. Many ASR engines also have conflicting dependencies. In addition, centralized solutions like this creates a giant maintainence overhead, whereas standard asr gives the maintainence back to the ASR developers.
- One of our core goals is to give the right to choose ASR model to the end user. There should be no code changes or maintainers attention required for applications to use new ASR models.


---

# Faq

> But why do we need to support different ASR engines in our application? Why not just support whisper?

- Different language have different SOTA ASR models. Whisper may be strong in some language and not in others.
- GPU acceleration support varies across platforms.
- AI world evolves fast... SOTA will be refreshed.

With Standard ASR, write once, forget about it. Countless ASR engines are automatically supported.

# Contribution
Please review `CONTRIBUTING.md` file before you make your contribution.

# Communication
We use **Zulip** for development communication:
- https://standard-voice.zulipchat.com

# License

This project is licensed under the Apache 2.0 License. Please checkout [LICENSE](./LICENSE) for more details.
