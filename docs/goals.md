# 项目目标 (Project Goals)

为了实现我们的使命和哲学，我们设定了以下具体、可执行的项目目标：

### G.1: 建立通用接口 (Establish a Universal Interface)

- **G.1.1: 标准化核心接口**: 定义一个纯粹的、几乎零依赖的核心抽象接口（`StandardASR`），规范化 `transcribe` / `transcribe_async`（批量）与 `start_transcription`（流式）等核心方法的行为。
- **G.1.2: 音频输入协商与恒定输出**: 定义统一的音频输入类型体系（`AudioInput` 判别联合：本地路径 / 编码字节 / 波形数组 / 可拉取 URL / base64 / 云存储 URI）。引擎在 Properties 中声明其接受的输入形态与采样率边界，标准层按确定性协商矩阵在两者之间转换：可转换则转换，并对任何有损步骤发出结构化 diagnostic；不可转换则显式失败，绝不静默降级。输出为恒定 schema 的 `TranscriptionResult`——返回类型不随任何参数变形。（normative 定义见 `docs/spec/specification.md` §音频输入与采样率、§结果模型。）
- **G.1.3: 属性、能力与配置声明机制**: 强制要求所有 ASR 引擎声明三类机器可读元数据，供调用者与 UI 动态查询和适配：
    - **Properties（静态身份）**: 引擎固有的 I/O 边界——接受的输入形态、采样率边界、`selectable_languages`（遵循 IETF BCP 47）等；
    - **Capabilities（功能能力）**: 层级化能力树（见 G.1.4）；
    - **Config（初始化配置）**: Pydantic 模型。设备选择（CPU/GPU）等「相关才适用」的标准字段经标准 config mixin（如 `DeviceConfigMixin`）声明——字段出现在 config 模型中即表示适用，auto-UI 据此渲染。硬件可用性取决于宿主环境与安装变体（如 torch 的 CPU/CUDA 构建），属部署与配置关注点，标准层无法静态核验，故**不**作为 Properties 静态声明。
- **G.1.4: 可选功能标准化**: 为流式输入/输出、word-level timestamps、说话人分离等高级功能定义标准化的接口和返回格式，并通过层级化 Capabilities 树声明支持情况（fail-closed：缺失即不支持；`engine.supports()` 点路径查询是唯一标准查询方式）。
- **G.1.5: 统一流式语义**: 把不同引擎差异巨大的实时转写行为——可改写的 interim、两遍重打分对已发段的合并/拆分、逐 token 不可变前沿、VAD 切段后只发段末结果——统一为一套事件协议（`partial` / `final` / `supersede` / `progress` / `done` / `error`）+ 段生命周期 + 显式稳定性保证（`stable_until` 冻结前缀、`final`/`closed` 两级终态），使实时字幕、语音助手等应用代码跨引擎不变。

### G.2: 提供开发者工具套件 (Provide a Developer Toolkit)

- **G.2.1: 一键合规验证**: 提供标准的合规测试套件，让 ASR 开发者以一条命令验证其实现是否符合 Standard ASR 规范（entry point 元数据、能力声明不变量、参数门控、流式事件序列、sync 桥行为）。合规套件应与运行时共用同一套校验逻辑，保证合规判定与运行时行为不漂移。
- **G.2.2: 开箱即用的周边工具**: 为任何符合标准的 ASR 库自动提供实用工具，包括：
    - **命令行接口 (CLI)**: 用于发现已装引擎、快速测试文件或启动麦克风进行识别。
    - **Web API 服务**: 内置 FastAPI 服务器，一键将任何 ASR 库封装为 HTTP / WebSocket 服务。
    - **模型管理工具**: 辅助用户下载和管理 ASR 模型（lazy-loading 与下载策略见 `docs/spec/download-policy.md`）。
- **G.2.3: Boilerplate 模板**: 提供一个标准的项目模板，让 ASR 开发者可以快速启动一个符合规范的插件项目。

### G.3: 实现动态与零配置运行 (Enable Dynamic and Zero-Config Operation)

- **G.3.1: 动态配置生成**: 所有引擎的初始化参数和推理参数都必须通过 Pydantic 模型进行定义和暴露（含 JSON Schema 输出与凭证字段的 secret 标记）。这使得 WebUI、GUI 或其他系统能够自动生成配置界面，实现免配置或动态配置。
- **G.3.2: 插件自动发现**: 应用开发者只需安装 `standard-asr` 核心库和所需的 ASR 插件包。核心库通过 entry-point 机制自动发现所有已安装的、符合规范的 ASR 引擎。

### G.4: 构建可扩展的插件化生态 (Build an Extensible Plugin Ecosystem)

- **G.4.1: 核心与实现分离**: `standard-asr` 包本身是一个近零依赖的纯框架（核心仅 `numpy` + `pydantic`）。每个 ASR 引擎的支持都由一个独立的插件包（如 `std-faster-whisper`）来提供。
- **G.4.2: 依赖与许可证隔离（诚实边界）**: 插件化架构把每个引擎的依赖与许可证（如 AGPL/GPL）隔离在独立的 pip 包中——应用按协议与成本需求自由选择插件，许可证责任边界清晰。它**不能消除**同一环境内的硬性依赖冲突（如 numpy 1.x 与 2.x 插件无法共存于单进程——这是 Python 事实，非架构可解）。对此标准提供两层方案：`standard-asr doctor` 做只读冲突诊断（spec §DEP.5）；进程隔离逃生舱（subprocess + UDS + `shared_memory`，spec §DEP.4）作为硬冲突下应用代码不变的解法。
- **G.4.3: 插件生态目录**: 维护一个公开的兼容引擎目录（已知插件、能力摘要、许可证），让应用开发者与最终用户能够发现「现在有哪些引擎可用」。这是生态飞轮的展示窗口，也是「用户自由选择引擎」承诺的入口。

### G.5: 跨语言 Wire 协议 (Cross-Language Wire Protocol)

- **G.5.1: Wire 契约一等化**: 将 HTTP / WebSocket 契约（现 `docs/spec/server.md`）演进为**独立版本化的跨语言规范**——拥有自己的 schema、兼容性规则与语言无关的一致性测试；Python FastAPI server 是其参考实现而非定义本身。非 Python 应用经此协议获得与 Python 应用同构的能力：模型发现、能力查询、批量转写与流式事件流。
- **G.5.2: 两层同构**: 进程内 Python 协议与 wire 协议保持同构——同一能力模型（canonical JSON）、同一结果 schema、同一事件流语义。任何一层的协议演进都必须在另一层有对应表达，防止两层漂移成两个标准。
