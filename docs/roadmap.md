# 项目规划与路线图 (Planning and Roadmap)

Standard ASR 项目将遵循明确的阶段性规划，以确保稳健发展和社区参与。

![stages-of-development](./assets/stages.png)

> ⚠️ **注意**: 本项目目前仍处于早期开发阶段。在 `v1.0.0` 版本发布之前，API 可能会发生重大变化。我们严格遵循语义化版本控制。

## R.1: Draft Stage (MVP 阶段)

此阶段的目标是完成一个最小可用产品（MVP），并进行初步的社区测试。

### R.1.1: 核心功能原型 (Working Prototype)
核心标准
- [x] 输入类型确定 (np array)
- [x] 输入类型的 utils
- [x] 将 properties 变成静态属性
- [] 设计并实现 feature flag
- [ ] Option 参数
- [ ] 设计并实现 TranscriptionResults

- [ ] 完成 `Properties` 系统的设计与实现，特别是语言代码标准化 (还有 validator 相关的测试)。
- [ ] 完成基于 Pydantic 的 `Config` 系统设计。
- [ ] 添加流式支持
- [ ] 测试: standard compliance test suite
- [ ] 填充项目元数据
- [ ] 完成 test pypi 发布，构建完整的打包发布 pipeline

工具链
- [ ] 实现插件发现机制 (entrypoint)，发现插件函数 和中央工厂
- [ ] CLI
- [ ] Web API 工具

### R.1.2: 自动化流程 (CI/CD Automation)
- [x] 配置 actions: Linters, Type Checkers, ~~Code quality scan, security scan~~ (公开之后才行)。
- [x] 设置自动化测试（action, 多 Python 版本）。
- [ ] 配置 pre-commits (typer checker, linter)
- [ ] 配置 test coverage，实现 100% 覆盖。
- [ ] 建立自动生成 Changelog，自动化版本号，自动发布 (release-please)，和发布到 PyPI 的流程。

### R.1.3: 基础文档 (Foundational Documentation)
- [ ] 关于页面，如何成为一份子
- [ ] **贡献指南 (CONTRIBUTING.md)**: 明确贡献流程、代码风格和许可协议。
- [ ] Code of conduct
- [ ] **开发者文档**:
    - **应用开发者指南**: 如何在项目中使用符合规范的 ASR 库。
        - 输入音频的类型: 工具函数概述
    - **ASR 开发者指南 (Cookbook)**: 如何将现有的 ASR 库适配为 Standard ASR 插件。
        - 输入音频的类型
- [ ] **快速入门 (Quick Start)**: 提供简单易懂的上手教程。

### **R.1.4: 社区建设 (Community Building)
- [ ] 建立交流论坛（如 Zulip Channel）。
- [ ] 文档化社区和新人指南
- [ ] standard asr compliant badge
- [ ] 支持 standard asr 的 asr 和 项目展示页面。

## R.2: Beta Stage (生态扩展阶段)

在 MVP 稳定后，我们将专注于扩大生态系统和完善工具链。

- [ ] 多语言 SDK

- **R.2.1: 完善工具链**: 稳定并增强 CLI、Web API 服务器等周边工具的功能。
- **R.2.2: 扩展官方插件**: 适配更多主流的开源 ASR 模型，提供官方维护的插件包。
- **R.2.3: 社区贡献**: 鼓励并支持社区开发者贡献他们自己的 ASR 插件。
- **R.2.4: 收集反馈**: 积极与早期用户沟通，收集反馈并迭代改进核心 API 和工具。

## R.3: Stable Stage (稳定版发布)

- **R.3.1: API 稳定**: 发布 `v1.0.0` 版本，稳定核心 API。此后的任何破坏性变更都将遵循语义化版本规范，并提供清晰的迁移指南。
- **R.3.2: 长期支持**: 为稳定版本提供长期的维护和支持。
