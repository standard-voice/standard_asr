# 在开发过程中使用 AI

AI 开发早已无法抵挡，但让 AI 写出出色的，可维护性高的代码依旧存在挑战。

下面提供一些能让 AI 写更符合我们项目的代码的指南。

## repomix
Standard ASR 核心代码量不多，所以比起用 RAG 或 context engineering 手段压缩上下文 (cursor, github copilot)，还不如把整个仓库代码都丢到 LLM 的上下文中。


有个叫 [repomix](https://repomix.com/) 的工具可以把整个项目的所有代码变成 LLM-Ready 的纯文本。

```sh
repomix
```

只包含文档正文和 Standard ASR 库的代码。
```sh
repomix --ignore "LICENSE,.gitignore,.github/copilot-instructions.md,NOTICE,docs/stylesheets,mkdocs.yml,.cache"
```

只包含文档正文
```sh
repomix --include "docs" --ignore "docs/stylesheets"
```
