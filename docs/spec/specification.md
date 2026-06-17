
# Standard ASR 设计规范

> 本文件是 Standard ASR 的**当前权威规范**。
> 所有章节标题带「— NORMATIVE」,其中的 **MUST / MUST NOT / SHOULD** 按 RFC 2119 解读。
> 经过 2026-06 的差距分析、设计、多轮独立审查定稿。
>
> **早期设计笔记**（2025–2026 初的工作草稿,使用旧模型）已移至 → `设计/legacy/pre-normative-设计笔记-2025-2026初.md`,不再包含在本文件中。

---

# 音频输入与采样率 (Audio Input & Sample Rate) — NORMATIVE

> **本节定义**：应用如何向引擎传递音频、标准层如何在应用提供的音频形态与引擎接受的形态之间进行协商与转换、以及采样率的声明与重采样责任。本节取代 idea_docs 中「所有引擎只吃 np.float32」的旧契约。
> **另见**：[§能力系统](#能力系统-capabilities--normative)（Capabilities 树结构与 Properties 边界规则）、[§流式协议](#流式协议-streaming--normative)（流式音频输入的 `audio_format` 与裸 PCM 帧）、[§Init Config](#init-config-baseconfig--normative)（初始化配置）。
> **组织**：概述 → 术语 → 声明与参数 → 行为（规范）→ 示例 → 附注与理由。

## 1. 概述（为什么需要这套机制）

不同 ASR 引擎对音频输入的要求差别很大。有的引擎（如 OpenAI）只接受文件上传；有的（如 Faster Whisper、Sherpa-ONNX）直接消费内存中的 NumPy 数组；有的云服务（如 ElevenLabs、AWS）要求应用提供一个可公开访问的 URL，由服务端自行拉取音频。采样率同样复杂：大多数模型原生运行在 16 kHz，但 OpenAI Realtime 硬性要求 24 kHz，电话场景的专用模型运行在 8 kHz。

本节定义了一套统一的音频输入类型体系、一个确定性的协商与转换矩阵、以及围绕采样率的声明与重采样规则，使应用代码在任何引擎上都能以相同的方式传递音频。

## 2. 术语

| 术语 | 含义 |
|---|---|
| `AudioInput` | 标准定义的**判别联合 (discriminated union)**，是应用传给 `transcribe(audio, …)` 的音频参数类型。包含六种变体。判别依据是显式的类型标签，**不是**对字符串内容的嗅探。 |
| `InputKind` | 引擎在 Properties 中声明自己接受哪些音频形态的封闭枚举：`"array"` / `"encoded_bytes"` / `"encoded_file"` / `"fetchable_url"` / `"storage_uri"`。 |
| `AudioStorageUri` | provider 云存储 URI（`s3://`、`gs://`、`oss://`、`abfs://` 等）。语义="引擎用自己的云 SDK 凭证解析此对象"。与 `AudioUrl` 区分：它**不是** HTTPS 可公开拉取的 URL，**不经过**标准的 SSRF 校验器（标准既不拉取也不能在无引擎凭证下访问云存储）。 |
| 协商 (negotiation) | 标准层在应用提供的形态与引擎接受的形态之间寻找匹配或最小成本转换路径的过程。 |
| 透传 (passthrough) | 应用提供的形态恰好是引擎接受的形态之一，标准层不做任何变换。 |
| diagnostic | 标准层在执行有损转换、假定参数等非理想路径时，返回给应用的结构化通知。 |
| canonical 格式 | 标准规定的基准音频表示：数组 = `float32` 单声道 `[-1, 1]`；wire/流式 = 16-bit signed LE PCM；默认 16 kHz。 |
| `[audio]` extra | `pip install standard-asr[audio]` 安装的可选依赖组，提供压缩格式解码和高质量重采样器。 |
| SSRF | Server-Side Request Forgery——攻击者利用服务端 URL 拉取访问内网资源。本节对 `AudioUrl` 的安全策略正是防范此风险。 |

## 3. 声明与参数

### 3.1 音频输入类型 `AudioInput`

| 变体 | 构造参数 | 含义 | 自描述采样率 |
|---|---|---|---|
| `AudioPath(value: str \| PathLike)` | 本地文件路径 | 磁盘上的音频文件 | 是（文件头） |
| `AudioBytes(data: bytes, container: str \| None = None)` | 编码音频字节 | 内存中的编码音频，`container` 为可选格式提示 | 是（文件头） |
| `AudioArray(samples: NDArray, sample_rate: int \| None = None)` | 波形数组 | 已解码的原始波形 | 否（除非给 `sample_rate`） |
| `AudioUrl(value: str)` | 远程 URL | 语义="服务端可拉取"；安全约束见 R5 | 是（远端头） |
| `AudioBase64(value: str)` | base64 / data-URI | 编码为 base64 的音频 | 是 |
| `AudioStorageUri(value: str)` | provider 云存储 URI | 引擎用自己的云 SDK 凭证解析（`s3://`/`gs://`/`oss://`/`abfs://` 等，scheme 白名单校验）；构造期校验 scheme，**不**经 SSRF 校验（见 R5.2） | 是（远端/服务端） |

**便利强制转换 (coercion)**：

| 裸类型 | 转换目标 | 说明 |
|---|---|---|
| `str` | `AudioPath` | **永远**视为本地路径。URL/base64/storage-URI 须显式用 `AudioUrl`/`AudioBase64`/`AudioStorageUri`。 |
| `bytes` | `AudioBytes` | — |
| `(ndarray, int)` | `AudioArray(samples, sample_rate)` | 元组第二元素为采样率 |
| `ndarray` | `AudioArray(samples, sample_rate=None)` | 未提供采样率，由 R6 的 strict/best_effort 决定 |

**流式音频是另一套表示**：流式通过 `start_transcription(audio_format=…)` 声明格式，`send_audio(chunk)` 喂裸 PCM 帧。batch 的 `AudioInput` 不用于流式逐块。唯一交叠：「整段输入+流式输出」（OpenAI SSE）中 `start_transcription` 的初始整段参数仍是 `AudioInput`（见 [§流式协议](#流式协议-streaming--normative)）。

### 3.2 引擎声明 —— Properties

| Property | 类型 | 必填 | 含义 |
|---|---|---|---|
| `accepted_input` | `set[InputKind]` | 是 | 引擎接受的音频形态集合 |
| `native_sample_rate` | `int` | 是 | 模型原生采样率（通常 16000 或 8000） |
| `accepted_sample_rates` | `list[int] \| SampleRateRange \| "any"` | 是 | 引擎接受的采样率：离散列表、连续区间 `{min,max}`、或 `"any"` |
| `required_input_sample_rate` | `int \| None` | 否 | 线路协议硬性要求的采样率（如 OpenAI Realtime 24000） |
| `max_file_size` | `int \| None` | 否 | 最大文件/数据大小（字节，如 OpenAI 25 MB） |
| `max_audio_duration` | `float \| None` | 否 | 最大音频时长（秒） |
| `wire_encodings` | `list[str] \| None` | 否 | 流式 `audio_format` 会话允许的 wire 编码白名单（如 `["pcm_s16le", "mulaw"]`）。缺席语义见下。 |
| `description` | `str \| None` | 否 | 仅**展示性**的人类可读描述。MUST NOT 承载影响协商/门控的机器可读数据。 |

仅行为性 `self_resamples: bool` 放 Capabilities（见 [§能力系统 R7](#能力系统-capabilities--normative)）。引擎身份字段（`engine_id`/`model_name`/`protocol_version`）的 normative 定义见 [§Init Config IC.2](#init-config-baseconfig--normative)。

**无 blanket metadata（normative）。** Properties **不含**自由 `extra: dict` 元数据口袋——与 §C 对能力树「砍掉 blanket `metadata`」（无已知用例、鼓励非结构化信息、破坏机器可读性）的决策对称。引擎需表达的机器可读信息 MUST 走结构化通道：标准字段（additive-minor 提案）或能力树的 `x_<vendor>_*` 实验命名空间；纯展示文本走 `description`。这避免在 wire 层形成无 schema、不可移植的私有并行声明通道。

**`accepted_sample_rates` 三种形态（normative）。** 引擎按其真实 I/O 边界三选一声明：
- **离散列表** `list[int]`：穷举接受的采样率（本地引擎常见）。
- **连续区间** `SampleRateRange { min, max }`：接受 `[min, max]` 闭区间内**任意**采样率（如 AWS Transcribe 的 `8000–48000`，研究 4）。`min`/`max` MUST > 0 且 `min ≤ max`；wire 序列化为 `{"min":8000,"max":48000}`。区间型引擎 MUST 用此形态而非退化为 `"any"`（过宽，丧失调用前判定）或穷举几个点（过窄，区间内合法率被无谓重采样有损）。
- **`"any"`**：接受任意采样率（如自行重采样的引擎）。

**声明卫生（normative）。** 列表形态：**重复**采样率是声明错误，MUST 在 Properties 声明期拒绝（与 `wire_encodings`、语言列表的重复拒绝同款契约）；列表为空 MUST 拒绝（无意义；如需"接受任意"用 `"any"`）；每个值 MUST > 0。

**采样率隶属判定（normative）。** 「输入采样率 ∈ `accepted_sample_rates`」对三形态的统一判定：`"any"`→恒真；列表→`rate ∈ list`；区间→`min ≤ rate ≤ max`（闭区间）。R7 的重采样决策、可达性不变量、流式会话率校验**均**用此统一判定（实现 `sample_rate_accepted`），任一调用点不得各行其是。重采样目标选择：列表取最近成员（偏好不升采样）；区间取输入**钳制进** `[min, max]`（最近可达点）。

**`wire_encodings` 缺席语义（normative）。** 当引擎声明 `wire_encodings` 为具体列表时，会话建立 MUST fail-closed 拒绝列表外的 `audio_format.encoding`（实现：`EngineBase.ensure_stream_format_supported`）——一个引擎从未声明的编码 MUST NOT 被当作 PCM 误帧而静默错误转写。当 `wire_encodings` 为 `None` 时语义为「**unconstrained**」：标准层无法校验编码，遂跳过编码检查并信任引擎接受任意编码（典型为自管 wire 格式的适配器，仅经无参 `start_transcription()` 开会话、从不收 `audio_format`）。此处对编码是 **fail-open**——是对自管格式引擎的有意让步，而非能力系统 fail-closed 默认的例外（采样率与声道仍无条件 fail-closed）。补偿：合规套件对**声明了 `streaming_input` 却未声明 `wire_encodings`** 的引擎发 **warning**（漏声明会在 `audio_format` 会话上打开静默误帧窗口；自管格式引擎可忽略此 warning）。`pcm_s16le` 是 canonical wire 编码（[§AI 术语](#音频输入与采样率-audio-input--sample-rate--normative)：16-bit signed LE PCM）。

## 4. 行为（规范）

**R1 — 类型判别。** `AudioInput` 的判别 MUST 基于显式类型标签，MUST NOT 基于字符串内容嗅探。

**R2 — 引擎声明。** 每个引擎 MUST 在 Properties 中声明 `accepted_input`。

**R3 — 协商与转换矩阵。** 有直接匹配时透传；无匹配走最低成本转换；无路径抛 `IncompatibleAudioInputError(provided, accepted, hint)`。

| 应用提供 ↓ ╲ 引擎接受 → | `array` | `encoded_file/bytes` | `fetchable_url` | `storage_uri` |
|---|---|---|---|---|
| `AudioPath` | decode→array（需 `[audio]`） | 读文件→bytes，透传 | **FAIL** | **FAIL** |
| `AudioBytes` | decode→array（需 `[audio]`） | 透传 | **FAIL** | **FAIL** |
| `AudioArray` | 透传（+ 采样率 R6–R8） | encode→WAV bytes（R4） | **FAIL** | **FAIL** |
| `AudioUrl` | **v1: FAIL**（R5）；v2: fetch→decode | 透传给接受 URL 引擎 | 透传 | **FAIL** |
| `AudioBase64` | b64decode→decode→array | b64decode→bytes | **FAIL** | **FAIL** |
| `AudioStorageUri` | **FAIL**（R5.2） | **FAIL**（R5.2） | **FAIL**（R5.2） | 透传（零转换） |

- 有损单元格 MUST 发 `audio_conversion` diagnostic（统一 `Diagnostic` 形状：`level`/`code`/`message`/`param?`/`provided?`/`effective?`——from/to/lossy 等细节进 `message` 与 `provided`/`effective`，不是独立字段）。
- 协商 MUST 支持调用前判定：`can_accept(provided, accepted) -> bool` / `negotiate(provided, accepted) -> ConversionPlan | NoViablePath`，并体现在引擎 card / 文档。
- **死角**（本地数据 + 只接受 `fetchable_url`，或 storage-URI + 不接受 `storage_uri`）：MUST fail-explicit。标准 MUST NOT 做 upload-broker，且 MUST NOT 在无引擎凭证下拉取云存储。
- **内存源 → 只接受 `encoded_file`（不接受 `encoded_bytes`）的引擎**：实现选择 **FAIL** 而非落临时文件——数组 encode 输出 `BytesIO`、bytes/base64 解码为内存字节，标准 MUST NOT 为投递而写临时文件（临时文件有泄漏/只读 FS/TOCTOU 风险，见 R4 理由）。此处行为以**代码为准**：上表 `AudioBytes`/`AudioBase64` × `encoded_file/bytes` 写作"透传"是针对接受 `encoded_bytes` 的引擎；当引擎**仅**接受 `encoded_file` 时，协商返回 `NoViablePath`（fail-explicit），比字面矩阵更严。

**R4 — 数组→编码文件 encoder。** 当 `AudioArray` 遇到只接受文件的引擎时：输出 MUST 为内存 `BytesIO`（MUST NOT 落磁盘）；canonical 编码 = WAV/16-bit PCM LE/mono；多声道 MUST 降混+diagnostic；float32→int16 有损 MUST 发 diagnostic；编码后 MUST 预检 `max_file_size`，超限抛清晰本地错误。

> **canonical 量化约定（normative，钉死跨语言一致性）**：float32→int16 的量化 MUST 为：先 clip 到 `[-1.0, 1.0]`，再 `round_half(sample × 32767)`（四舍五入到最近整数，**MUST NOT** 向零截断——截断使量化误差上界从 0.5 LSB 翻倍到 1 LSB 并引入向零偏置），写为 little-endian int16。非有限样本（NaN/±Inf）在 cast 前 MUST 被消毒为 `0`/`±1`（int16 无法表示 NaN，cast 是未定义行为），且该消毒 MUST 发 `non_finite_audio` diagnostic（与数组直达路径同 code，使编码路径与数组路径对该输入的可观测性一致）。解码侧的反向缩放为 `÷32768`；编解码往返因此有一个 `32767/32768`（≈ −0.00027 dB）的有意衰减，可接受。钉死此约定是为了让 wire 协议的其他语言实现产生逐字节一致的 PCM，避免一致性测试出现 ±1 LSB 噪声。

**R5 — `AudioUrl` 安全策略。**
- **引擎/云自取**：转发前 MUST 校验 HTTPS-only + 默认拒绝私网/环回/link-local IP 段（RFC1918、127/8、169.254/16、::1、fc00::/7；可显式 opt-in）+ 重定向上限+每跳重校验。**opt-in 入口**：私网拒绝的显式放宽是 **Init Config 级**部署开关 `allow_private_urls: bool`（`BaseConfig`，默认 `False`），由标准转写管线（`EngineBase._prepare_audio`）透传给校验器；HTTPS 要求**不**可放宽。它属部署属性而非请求属性，故归 Init Config 而非 `RuntimeParams`（不随请求漂移），且与 `strict` 同列 `_ENV_EXCLUDED_FIELDS`——环境变量 MUST NOT 静默放宽该安全策略。
- **标准自取**：**v1 MUST NOT 实现**（SSRF 高危）。`AudioUrl` v1 仅作透传。
- v2 若开放：MUST 额外 DNS pin + 流式读取硬上限 + 超时。
- **v1 校验边界（明示，非静默缺口）**：v1 标准层**只校验调用方给出的那个初始 URL**——解析其 host、核验 HTTPS 与所有解析到的 IP 非私网（实现：`validate_fetchable_url`），随后把**字面 URL**透传给引擎；标准层**自身不拉取、不发起任何请求、因此也不存在需要逐跳重校验的重定向链**。上文第一条里的"重定向上限+每跳重校验"是**引擎/云端 fetcher**的责任（它在 fetch 时会独立地再次解析域名——故 resolve-time IP 检查对 DNS rebinding 仅是 advisory，不是硬保证）。也就是说：**v1 里跟随并逐跳重校验重定向是引擎侧的义务，不是标准层的**；待 v2 标准开放自取时，per-hop 重校验才落到标准层（连同上一条的 DNS pin / 读取上限 / 超时）。

**R5.2 — `AudioStorageUri` 安全模型（独立于 R5）。** provider 云存储 URI（`s3://`/`gs://`/`oss://`/`abfs://` 等）由**引擎**用自己的云 SDK 凭证解析，标准既不拉取也不能在无凭证下访问，因此：
- 构造期 MUST 校验 scheme 落在 provider 存储 scheme 白名单内（小且 extensible-by-constant，无运行时注册表）；MUST 拒绝 `file://`、`http(s)://`、空值、未知 scheme，报清晰错误。
- MUST NOT 经过 R5 的 HTTPS / public-IP SSRF 校验器（storage URI 不是 HTTPS-fetchable，标准无 SSRF 攻击面）。
- 协商：仅当引擎接受 `storage_uri` 时透传（零转换）；其余一律 **FAIL**（标准不是 upload-broker）。

**R6 — 采样率。** canonical = 16 kHz mono。裸数组无采样率时：strict MUST 抛错（"pass AudioArray(samples, sample_rate)"）；best_effort MAY 假定 16k 但 MUST 每次发 `assumed_sample_rate` diagnostic。**绝不静默假定。** `sample_rate`：batch 选填；bare-PCM streaming 必填（会话锁定）；header-bearing buffered 输入（OpenAI SSE）自描述豁免。

**R7 — 重采样责任。** `accepted_sample_rates` **始终权威**，不论 `self_resamples`：输入 ∉ accepted 且 ≠ `required_input_sample_rate` → 标准 MUST 重采样；无可达目标 = 定义错误，MUST NOT 静默透传。8 kHz 电话 = 独立原生模型，MUST 经 entrypoint preset 选择，MUST NOT 升采样原生率输入。24 kHz realtime = `required_input_sample_rate`，标准重采样；流式缺 `[audio]` 时 MUST 在会话建立时报错。**可达性不变量**：`required_input_sample_rate`（若设）MUST 被 `accepted_sample_rates` 接受（当后者为具体列表**或区间**时，即非 `"any"`；按上文统一隶属判定）——重采样目标必须可达；同理 `native_sample_rate` MUST 被 `accepted_sample_rates` 接受（非 `"any"` 时）——否则引擎自身的原生率输入会被静默重采样（如 8 kHz 电话模型被升采样到 16k），而 8 kHz 是独立原生模型而非低采样率变体。两条不变量均在 Properties **声明期**即校验（`BaseProperties`），而非延迟到会话建立。

**R7.1 — `required_input_sample_rate` 对 batch 路径绝对权威（normative）。** 对 batch 的数组与编码（array / encoded）路径：当 `required_input_sample_rate` 已设且**输入采样率 ≠ required** 时，标准 **MUST** 重采样到 `required`，**不论** `accepted_sample_rates` 为 `"any"` 抑或其具体列表已含输入采样率。即「输入 ∈ accepted 但 ≠ required」这一可构造组合（如 `accepted_sample_rates=[16000, 24000]` + `required_input_sample_rate=24000` + 输入 16000）下，**MUST** 重采样到 24000 而 **MUST NOT** 透传 16000——硬性线路要求（如 OpenAI Realtime「Always 24000」）必须被精确满足。此条把参考实现既有行为升格为 normative，消除 R7 字面只覆盖「∉ accepted 且 ≠ required」时其它实现可能透传的歧义；流式侧 `ensure_stream_format_supported` 的 required 绝对优先与此同构（见下方 v1 实现说明 ①）。

**R7.2 — 重采样目标率的选择顺序（normative）。** 当 R7 判定必须重采样时，标准 **MUST** 按以下顺序选择目标采样率（首个适用者胜）：① `required_input_sample_rate`（若设）；② 否则 `native_sample_rate`。R7 的两条可达性不变量保证：经引擎管线（`accepted_sample_rates` 为具体列表或区间时 required 与 native 均被 accepted 接受）①②**必然可达**，故无需第三级。仅当调用者**绕过 Properties 不变量直接调用** `execute_plan`（声明不满足不变量）时，实现 MAY 退到一个防御性兜底：在 accepted 中选与源采样率**绝对距离最近**者，并在等距时**偏好不升采样**（`≤ source`）的目标（区间则把源钳入 `[min, max]`；源采样率未知时选 `min(accepted)` / 区间下界以最小化无谓升采样）。此兜底是实现细节而非 normative 要求，独立实现可自选等价的 fail-loud 策略。把目标选择写入规范，避免其它实现选 `accepted[0]` 或 `max(accepted)` 导致同一输入跨实现得到不同质量的重采样结果。

> **v1 实现说明**：v1 不在标准层重采样**流式裸帧**（流式引擎自行处理 wire 帧），因此上文"流式缺 `[audio]` 时会话建立报错"针对的是**未来**标准层流式重采样落地后的路径。v1 的会话建立校验（`EngineBase.ensure_stream_format_supported`）做**三项** fail-closed：①**wire 采样率可达性**——`required_input_sample_rate`（若设）MUST 强等（**含 `accepted_sample_rates="any"` + `required` 的可构造组合**），否则 wire rate MUST 被 `accepted_sample_rates` 接受（非 `"any"` 时，按统一隶属判定——列表成员或区间内）；②**mono-only**——`audio_format.channels` MUST = `1`（与 [§ST 3.1](#流式协议-streaming--normative):"v1 增量 wire 输入仅支持单声道"一致）；③**wire 编码**——当引擎声明 `wire_encodings`（具体列表）时拒绝列表外编码（`wire_encodings=None` 为 unconstrained，编码 fail-open，缺席语义见 [§AI 3.2](#音频输入与采样率-audio-input--sample-rate--normative)）。采样率拒绝是「标准层流式重采样」deferred 路径落地前的 v1 替代行为（届时该拒绝转为重采样）。批量 `transcribe` 路径的重采样与 `[audio]` 行为按本条全量生效。

**R8 — 重采样质量与许可证。** fallback MUST 抗混叠（FFT-based，MUST NOT 裸线性/抽取）；用了 fallback MUST 发 `resampled_with=fallback` diagnostic。许可证：SHOULD clean-room FFT；MUST NOT vendoring SoX/soxr(LGPL/GPL) 或 2016 前 libsamplerate；soxr 仅作 `[audio]` 依赖。

**R9 — 内存与大媒体。** `AudioBytes`/`AudioArray` = 装得下内存的形态；大媒体 SHOULD 用 `AudioPath`/`AudioUrl`；标准读文件/URL 给文件型引擎时 SHOULD 流式不全缓冲。

**R10 — 新增 Properties。** `accepted_input`/`max_file_size`/`max_audio_duration`/`native_sample_rate`/`accepted_sample_rates`/`required_input_sample_rate`/`wire_encodings` MUST 落在 §C 的层级化模型内。关于 `supported_input_formats`（容器格式协商）：v1 由 `accepted_input` + encoder 容器选择间接覆盖，后续可 additive 补充。`max_audio_duration` **强制点**：标准仅在输入**已解码为数组**（时长可测）时校验并 fail-loud；编码透传（file/bytes passthrough）MUST NOT 为测时长而强制全解码（见 R9），改由 `max_file_size` 作本地护栏 + 引擎自行兜底。

## 5. 示例

### 示例 A：本地数组 → OpenAI（只接受文件上传）

引擎：`accepted_input={"encoded_file","encoded_bytes"}`，`max_file_size=26214400`（25 MB）。应用传入 `AudioArray(samples=my_array, sample_rate=16000)`。

1. 协商：矩阵 `AudioArray` × `encoded_file/bytes` → R4 encode。
2. 执行：`BytesIO`；float32→16-bit PCM LE（有损，发 diagnostic）；mono。
3. 预检：12 MB < 25 MB，通过。
4. 结果：WAV bytes 传给 OpenAI + `audio_conversion` diagnostic。

### 示例 B：本地文件 → Faster Whisper（只接受数组）

引擎：`accepted_input={"array"}`，`self_resamples=true`，`accepted_sample_rates="any"`。应用传入 `AudioPath("/data/interview.mp3")`。

1. 协商：矩阵 `AudioPath` × `array` → decode（需 `[audio]`）。
2. 执行：解码 mp3→float32 数组，文件头读出 44100 Hz。
3. 重采样：`self_resamples=true` 且 `"any"` → 透传。引擎自行降至 16k。
4. 结果：float32 数组传给引擎 + decode diagnostic。

### 示例 C：AudioUrl → 只接受数组的引擎（v1 FAIL）

引擎：`accepted_input={"array"}`。应用传入 `AudioUrl("https://example.com/audio.wav")`。

1. 协商：矩阵 `AudioUrl` × `array` → v1: FAIL。
2. 结果：抛 `IncompatibleAudioInputError`，hint="请用 AudioPath 提供本地文件"。

### 示例 D：AudioUrl → ElevenLabs（接受 URL）

引擎：`accepted_input={"encoded_file","encoded_bytes","fetchable_url"}`。应用传入 `AudioUrl("https://storage.example.com/audio.flac")`。

1. 协商：透传。R5.1 校验 HTTPS + 非私网 IP，通过。
2. 结果：URL 直接传给 ElevenLabs，零转换。

### 示例 E：24 kHz realtime 重采样

引擎：`required_input_sample_rate=24000`，`accepted_sample_rates=[24000]`，`self_resamples=false`。流式会话 `audio_format.sample_rate=16000`。

1. 16000 ∉ `[24000]` → 标准 MUST 重采样 16k→24k（R7）。
2. 检查 `[audio]` 已装；若未装，会话建立时报错（R7）。
3. 结果：24 kHz PCM 帧传给引擎 + `resampled_with` diagnostic。

## 6. 附注与理由

- **为何用判别联合而非字符串嗅探**：一个名为 `https%3A//...` 的本地文件、一个 base64 恰好以 `/` 开头的字符串，都会导致嗅探误判。显式类型标签消除所有歧义，对 IDE 类型提示也更友好。
- **为何裸 `str` 只映射 `AudioPath`**：最常见且最安全的语义是文件路径。如果允许裸 str 被推断为 URL，恶意输入可能触发 SSRF。URL/base64 强制显式类型是一道安全防线。
- **为何 v1 禁止标准自取 URL**：SSRF 是 v1 最不值得冒的风险。"接受 URL 就透传；不接受就报错"比安全实现一个 URL fetcher 简单得多。
- **为何用独立的 `AudioStorageUri` 变体而非放宽 `AudioUrl`**：两者安全模型根本不同。`AudioUrl` 是 HTTPS 可公开拉取的 URL，标准的 SSRF 校验器（HTTPS-only + 拒私网 IP）正是为它而设。provider 云存储 URI（`s3://`/`gs://`/…）则由**引擎用自己的云 SDK 凭证**解析——标准无法、也不应在无凭证下拉取它，对它跑 public-IP SSRF 校验既无意义又会硬拒掉整类合法云引擎（AWS Transcribe batch 强绑 S3 URI、Google STT v2 仅接受 `gs://`）。把它放进 `AudioUrl` 会迫使一个变体承载两套互斥的安全语义；独立变体让"标准侧无 SSRF 面、引擎侧凭证解析"这一事实在类型与协商矩阵中显式可见，且 scheme 白名单在构造期即 fail-loud。
- **为何 encoder 输出到 BytesIO**：临时文件有泄漏风险、在只读 FS 上失败、有 TOCTOU 竞态。BytesIO 在进程内完成全部操作。
- **为何 canonical = WAV/16-bit PCM LE**：最简单的无压缩容器 + 行业标准量化深度 + 主流 CPU 字节序 = 最大兼容性。
- **为何采样率放 Properties**：采样率是引擎的静态 I/O 边界，不随功能/模式改变。唯一例外 `self_resamples`（行为性）归 Capabilities。
- **为何 8 kHz 走 preset**：8 kHz 模型是完全不同的模型（训练数据、声学特征不同），不是"同一模型的低采样率版"。阿里/Google 均明确警告升采样掉精度。
- **重采样许可证**：soxr (LGPL) 不能 vendoring 进 Apache-2.0 核心。内置 FFT 实现（算法不可版权化）确保许可证干净。
- **为何 HTTP server 不预解码**：工具链 server 收到上传后 MUST 以 `AudioBytes`（multipart）/`AudioBase64`（JSON）形态喂入引擎自身的标准协商，MUST NOT 自行预解码/归一到 16 kHz。否则会破坏按引擎的采样率要求（R7，如 8 kHz 电话被静默升采样、24 kHz realtime 无法协商），也会让只接受 encoded/url 的云引擎无法被 server 暴露（违背 G.2.2"封装任何引擎"）。base64 仅作传输编码，由协商层解码。


---
---

# 语言选择 (Language Selection) — NORMATIVE

> **本节定义**：应用如何指定转写语言、引擎如何声明所支持的语言、以及未指定时如何解析出最终生效的语言。本节将已定稿的语言系统以本规范的 Capabilities 结构完整重述，并修正了若干边界情形。
> **另见**：[§能力系统](#能力系统-capabilities--normative)（Capabilities 树结构）、[§Runtime 参数](#runtime-参数-runtime-parameters--normative)（参数如何传入 `transcribe`）、`设计笔记和决策/1 language code 设计.md`（设计背景与决策记录）。
> **组织**：概述 → 术语 → 声明与参数 → 行为（规范）→ 示例 → 附注与理由。

## 1. 概述（为什么需要这套机制）

不同 ASR 引擎处理语言的方式差别很大，标准需要用一套接口同时覆盖所有情形：有的引擎**初始化即锁定**一种语言；有的**每次转写都可切换**；有的能**自动识别**（业界称 LID——只需传入 `auto`，引擎自行判断音频所属语言）；有的在自动识别的基础上还能用**候选语言列表 (candidate languages)** 缩小识别范围、提高准确率（例如"这段音频只可能是中文或英文"）。以下按"术语 → 声明 → 行为"逐步说明。

## 2. 术语

| 术语 | 含义 |
|---|---|
| `auto` | 标准保留字（**不是** BCP-47 标签），意为"让引擎自行识别语言"。可出现在 `selectable_languages` 与 `default_language` 中；**不得**出现在候选语言列表里——候选的语义是"在这些语言中选择"，而 `auto` 的语义是"自行识别"，二者互斥（normative 要求见 R3 步 4）。 |
| `selectable_languages` | 应用**可显式指定**的语言集合（BCP-47 标签 + 可选的 `auto`）。UI 的语言下拉框应据此生成。属 Properties（静态身份）。 |
| `detectable_languages` | `auto` 模式下引擎**可能识别出**的语言集合。可与 `selectable_languages` 不同——有的引擎能识别多种语言，但不接受显式指定，只能通过 `auto` 自动识别。属 Properties；用于 UI 展示、文档生成及候选语言校验。 |
| `default_language` | 初始化时设定的默认语言。有两个作用：① 对于不支持运行时切换语言的引擎，这就是其固定语言；② 对于支持的引擎，这是"请求未指定语言时"的回退值。何时必填见 §4 R1。 |
| 候选语言<br>(candidate languages) | 一个**有序列表**，仅当 `effective_language` = `auto` 且引擎支持该功能时生效。语义为**偏好顺序 + 允许集合 (allowlist)**：引擎将识别范围限制在此列表内，并尽量优先列表中靠前的语言。 |

## 3. 声明与参数

**引擎声明 —— Properties（静态身份）：**

| Property | 类型 | 必填 | 说明 |
|---|---|---|---|
| `selectable_languages` | `list[BCP-47 \| "auto"]` | 是 | 应用可指定的语言集合 |
| `detectable_languages` | `list[BCP-47]` | 支持 `auto` 时必填 | `auto` 模式下可能识别的语言；候选语言的校验依据 |

**声明卫生（normative）。** 两个语言列表均在 Properties **声明期**规范化并校验（`BaseProperties`）：每个 BCP-47 标签 MUST 先**归一化**（大小写/分隔符），归一化后才与保留字 `auto` 比较——`"AUTO"`/`"Auto"` 等大小写写法 MUST 被识别为保留字 `auto`，**不得**被当作合法 BCP-47 标签放行（否则会绕过"`auto` 不得出现在 `detectable_languages`"这条 §2 约束）。两个列表均为**集合**语义：归一化后**重复**（含 `"en-US"`/`"EN-US"` 这类仅大小写不同的写法）是声明错误，MUST 拒绝（与 `wire_encodings`、`accepted_sample_rates` 的重复拒绝同款契约），报出冲突的原始写法。顺序保留（驱动 UI 下拉框）。

**引擎声明 —— Capabilities**（按 `<mode>` = `batch` / `streaming` **分别**声明；两种模式的能力可以不同，例如 batch 支持候选语言而 streaming 不支持）：

| 能力路径 | 节点类型 | 含义 |
|---|---|---|
| `capabilities.<mode>.language.runtime_override` | flag `{supported}` | 是否允许单次请求通过 `language` 参数覆盖 `default_language`（`false` = 初始化后锁定） |
| `capabilities.<mode>.language.candidate_languages` | bounded `{supported, constraints:{max}}` | 是否支持候选语言；`max` = 数量上限（一般 ≤ 4--5）。**原 Properties 中的 `max_candidate_languages` 已移至此处**（见 [§能力系统 C.6](#能力系统-capabilities--normative)） |

**应用传入的参数：**

| 参数 | 位置 | 类型 | 默认 | 依赖能力 | 说明 |
|---|---|---|---|---|---|
| `default_language` | Init Config | `BCP-47 \| "auto"` | 见 §4 R1 | — | 默认语言 |
| `default_candidate_languages` | Init Config | `list[BCP-47] \| None` | `None` | `candidate_languages` | 默认候选语言列表 |
| `language` | Runtime | `BCP-47 \| "auto" \| None` | `None` | `runtime_override` | 覆盖本次请求的语言 |
| `candidate_languages` | Runtime | `list[BCP-47] \| None` | `None` | `candidate_languages` | 覆盖本次请求的候选语言列表 |

传入**不被支持**的参数时，按全局 **strict / best_effort** 策略处理：strict 模式抛出 `UnsupportedFeatureError`；best_effort 模式忽略该参数并返回结构化 diagnostic（包含哪个参数被忽略、原因、以及最终生效的值）。

## 4. 行为（规范）

**R1 — `default_language` 必填规则。** 只要引擎暴露语言轴（即 `selectable_languages` 有定义，哪怕只含一种语言或仅含 `["auto"]`），Init Config 就 **MUST** 提供 `default_language`（值可以是该唯一语言或 `auto`），且其值 MUST ∈ `selectable_languages`。完全没有语言概念的引擎（不暴露 `selectable_languages`）不受此要求约束，其 `effective_language` 按 R2 步 3 短路为 `None`。

**R2 — `effective_language` 解析**（每次转写时执行）：
1. 若 `<mode>.language.runtime_override.supported` 为真 **且** 本次请求传入了 `language` → 使用该 `language`。
2. 否则，若引擎有语言轴 → 使用 `default_language`（R1 保证其存在）。
3. 否则 → `None`。

> **回退最终值诊断（best_effort）**：当本次请求传入了 `language` 但引擎不支持 `runtime_override`（即走步 2 回退到 `default_language`）时，§Runtime 参数 R2 的门控会先把该 `language` 当作"不支持的标准集参数"丢弃（strict 抛错；best_effort 丢弃 + `unsupported_parameter_ignored`，其 `effective=None`）。但引擎**实际**用 `default_language` 转写——为兑现 best_effort 诊断"必须报告最终值"的契约（门控层无从得知 `default_language`），标准层 SHOULD 在此回退发生时**补发**一条诊断（`code="language_fell_back"`，`provided`=被丢弃的请求语言、`effective`=实际生效的 `default_language`）。strict 模式不到达此处（门控已先抛错）。

**R3 — `effective_candidate_languages` 解析**：
1. 若 `effective_language` ≠ `auto` → `None`（候选语言仅在自动识别模式下有意义）。
2. 否则，确定**生效列表**：优先使用本次请求的 `candidate_languages`；若未提供，则使用 `default_candidate_languages`。两者皆未提供（无可约束的列表）→ `None`，**不发任何 diagnostic**（无候选传入即无物可忽略）。
3. 否则（确有生效列表），若 `<mode>.language.candidate_languages.supported` 为假 → `None` + 一条 diagnostic（"候选语言被忽略：当前引擎/当前模式不支持此功能"，`provided` = 被忽略的列表、`effective` = `None`）。该 diagnostic **仅在步 2 确实拿到非空列表时发出**——这样「`default_language=auto` 且不支持候选语言」这一常见引擎形态（多数本地 Whisper 系）在普通请求上不会被注入一条虚假 warning。
4. 否则（支持且有生效列表），对结果列表执行校验：**去重但保序**；**禁止包含 `auto`**；每个值 MUST ∈ `detectable_languages`；长度 MUST ≤ `…candidate_languages.constraints.max`（超出时：strict 模式抛错；best_effort 模式截断 + diagnostic）。

> **strict / best_effort 边界（与 [§Runtime 参数 R3](#runtime-参数-runtime-parameters--normative) 的"不支持即报错"张力的澄清）**：这里三类情况分属两套错误模型，**不要混淆**：
> - **功能不支持**（步 3，`candidate_languages.supported=false` 且确有候选列表）：这是一个**不支持的标准集参数**，但 R3 步 3 明确把它降为 `None` + diagnostic——**永不抛错**，**独立于 strict/best_effort**。它是 Runtime §R2"不支持参数 strict 抛错"的一个**显式 carve-out**：候选语言不支持时静默忽略并诚实诊断，比为一个纯优化项硬失败更合理。**前提**：确有候选语言被传入（步 2 拿到非空列表）；什么都没传时步 2 直接短路为 `None`，不发 diagnostic。
> - **值非法**（步 4 的"malformed BCP-47 标签"或包含保留字 `auto`）：这是**调用方的代码 bug**（如把 `"english"`/`"auto"` 当候选传入），MUST **始终抛 `ValueError`**，**独立于 strict/best_effort**——与 §Runtime R3 的 `provider_params` 错误"始终抛、不被 strict/best_effort 吞掉"同源。
> - **值合法但不可达**（步 4 的"non-detectable"或"超 `constraints.max`"）：这才走 §R2 的 strict/best_effort 策略——strict 抛错，best_effort 丢弃/截断 + diagnostic。
>
> 实现：`standard_asr.language.effective_candidate_languages`（其 `Raises` 文档逐条对应上述三类）。

**R4 — 运行时 `language` 的可选性判定用 RFC 4647 lookup（细化标签接受 + 引擎归约义务）。** 运行时 `language`（即 R2 解析出的 `effective_language`，`auto` 除外）对 `selectable_languages` 的成员判定 MUST 采用 [RFC 4647 §3.4 的 "Lookup"](https://www.rfc-editor.org/rfc/rfc4647#section-3.4) 回退算法：请求标签**或其逐级去尾得到的任一前缀**命中 `selectable_languages` 即视为可选。去尾时 singleton（单字符）子标签 MUST 与其前一子标签一并去除（如 `zh-x-foo` 直接退到 `zh`）。这让引擎只声明主语言子标签（`en`）即可接受其地区/脚本细化（`en-US`、`zh-Hant`）。**引擎归约义务**：当请求以细化标签命中时，标准层把**完整的请求标签**原样交给引擎，引擎 MUST 自行归约到它能处理的粒度（实现见 std-faster-whisper 插件：`normalize_bcp47(resolved).split("-")[0]`）。标准层 SHOULD 在以细化命中（前缀匹配而非精确匹配）时发一条 informational diagnostic（`code="language_refinement_accepted"`），使调用方可见该标签是经归约接受而非精确成员；不改变任何值。

> **与 `default_language` / `candidate_languages` 的不对称（明示，非缺陷）**：`default_language`（R1：MUST ∈ `selectable_languages`）与候选语言（R3 步 4：每个值 MUST ∈ `detectable_languages`）仍采用**精确成员**判定，**不**走 RFC 4647 lookup。理由：二者是**引擎作者对自己声明集的一致性自检**（声明期/配置期校验，故障域是「引擎声明是否自洽」），而运行时 `language` 是**应用的请求**（故障域是「用户请求能否被服务」）——放宽请求侧的匹配以提升 DX、同时保持声明侧的严格自检，是有意的设计，不要求三处同则。实现：`EngineBase._selectable_match`（运行时 lookup）vs `_validate_language_config`（`default_language` 精确）vs `language.effective_candidate_languages`（候选精确）。

## 5. 示例

引擎 `batch` 模式声明：`runtime_override.supported=true`、`candidate_languages.supported=true`、`constraints.max=3`、`detectable_languages=[en, zh-Hans, ja, ko]`。初始化配置：`default_language="auto"`。某次请求传入：`language="auto"`、`candidate_languages=["ja","en","ja"]`。

- **`effective_language`**：R2 步 1——`runtime_override.supported` 为真且传入了 `language="auto"` → 结果为 `auto`。
- **`effective_candidate_languages`**：R3 步 1——`effective_language` 为 `auto`，通过 → 步 2——`candidate_languages.supported` 为真，通过 → 步 3——本次请求提供了 `candidate_languages`，使用请求列表 → 步 4——去重保序得 `["ja","en"]`，两者均 ∈ `detectable_languages`，长度 2 ≤ 3 → 最终结果为 **`["ja","en"]`**（自动识别时优先考虑日语，其次英语）。

## 6. 附注与理由

- **R1 为何如此严格（修复 totality 漏洞）**：R2 步 2 会回退到 `default_language`。旧规则"仅多语言或支持 runtime override 的引擎才必填"在**单语言**引擎上留有漏洞——该引擎既非多语言、又不支持 runtime override，按旧规则无需提供 `default_language`，但 R2 步 2 仍会读取它 → 行为未定义。改为"有语言轴就必填"，以保证 R2 为全函数（total function）。
- **`max_candidate_languages` 归位**：从 Properties 移入 `capabilities.<mode>.language.candidate_languages.constraints.max`——"只有在某功能被支持时才有意义的上限"应与该功能定义在一起（[§能力系统 C.6](#能力系统-capabilities--normative) 的边界规则）。旧 Properties 中的表述以本节为准。


---
---

# 能力系统 (Capabilities) — NORMATIVE

> **本节定义**：引擎如何声明自身支持的功能集合（能力）、应用如何查询这些能力、以及标准如何保证能力系统在版本演进中的向前兼容性。本节将已定稿的统一层级化能力模型以完整的规范格式重述，取代此前三套并存系统（`supports_*` 布尔字段、`FeatureFlag` 枚举、`feat_flag` 字典）的旧设计。
> **另见**：[§语言选择](#语言选择-language-selection--normative)（能力系统在语言功能上的具体应用）、[§Runtime 参数](#runtime-参数-runtime-parameters--normative)（参数如何受能力门控）、`设计笔记和决策/6 核心设计决策 2026-06-06.md` D5（设计背景与决策记录）。
> **组织**：概述 → 术语 → 声明与参数 → 行为（规范）→ 示例 → 附注与理由。

## 1. 概述（为什么需要能力系统）

不同的 ASR 引擎支持的功能差异巨大：有的引擎能在 batch 模式下提供词级时间戳，却在 streaming 模式下不支持；有的支持候选语言列表，另一些完全不支持；有的能发射 partial 结果，有的只给 final。应用开发者需要一种统一的方式来**发现**引擎到底支持什么、**查询**某项具体功能是否可用、并在功能不可用时获得**一致的行为**（拒绝或降级）。

能力系统解决的核心问题是：**让引擎以结构化、机器可读的方式声明自身功能，让应用以统一的 API 查询这些功能，并让标准层据此进行一致的参数门控。**

## 2. 术语

| 术语 | 含义 |
|---|---|
| Capabilities（能力） | 引擎支持的功能集合。以层级化树结构表达，按 mode 域分组。每个叶节点携带 `supported` 信息，告知应用该功能是否可用。 |
| mode 域 (mode domain) | 能力树的顶层分区。当前标准定义两个封闭的 mode 域：`batch`（对应 `transcribe`）和 `streaming`（对应 `start_transcription`）。`job` 保留待 major 版本扩展。引擎不支持某 mode 时，省略该域。 |
| DeclaredCapabilities（声明能力） | 引擎在**类级别 (ClassVar)** 静态声明的能力全集。无需实例化引擎、无需鉴权即可发现。`standard-asr show`、注册表、UI 生成、REST `GET .../capabilities` 均读取此值。 |
| effective_capabilities（生效能力） | 引擎**实例化后**，根据实际配置可能**收窄**的能力子集。例如，引擎声明支持 `word_timestamps`，但用户未配 `forced_aligner`，则运行时不可用。**不变量 `effective ⊆ declared`**（只能关、不能凭空开）。合规测试强制校验此子集关系。 |
| 引擎全局能力 | 不绑定在任何 mode 域内的正交能力。放在能力树顶层，如 `streaming_input`、`streaming_output`。 |
| 节点原型 (archetype) | 能力树叶节点的三种固定形态：flag、bounded、enum/mode（详见 §3.3）。 |
| 点路径 (dot-path) | 用于 `engine.supports()` 查询的字符串，以 `.` 分隔层级。从 mode 域或顶层正交能力起始，**不带** `capabilities.` 前缀。如 `"batch.word_timestamps"`、`"streaming.guidance.phrase_hints"`。 |
| fail-closed | 能力系统的默认安全策略：任何未声明的能力键一律视为不支持，而非报错或假定支持。 |
| `x_*` 命名空间 | 实验性能力的保留前缀，格式 `x_<vendor>_<feature>`。显式标记为非标准，遵循与标准能力相同的门控规则。 |

## 3. 声明与参数

### 3.1 两层能力模型：Declared / Effective

| 层 | 类型 | 生命周期 | 用途 | 约束 |
|---|---|---|---|---|
| `DeclaredCapabilities` | ClassVar（类级别静态量） | 免实例化、免鉴权即可读取 | 发现与展示 | MUST 是 class-level 静态；MUST NOT 依赖运行时配置 |
| `effective_capabilities` | 实例属性（可选） | `__init__(config)` 后产生 | 运行时门控 | `effective ⊆ declared`（只能关闭已声明的能力）|

`--no-instantiate` 发现路径（CLI `show`、注册表查询）只读 `DeclaredCapabilities`。

### 3.2 层级结构与 mode 域

能力树的顶层是**封闭的 mode 域**（`batch` / `streaming`），引擎不支持某 mode 即省略（§4 R1 fail-closed：省略 = 不支持）。mode 内按功能分组；不绑定 mode 的引擎全局能力放顶层。**mode 域与引擎全局能力 MUST 显式区分。**

```yaml
capabilities:

  # ── batch mode ──
  batch:
    language:
      runtime_override:     { supported }
      candidate_languages:  { supported, constraints: { max } }
    word_timestamps:        { supported, granularities: [word, segment, char] }
    guidance:
      prompt:               { supported, constraints: { max_tokens? } }
      phrase_hints:         { supported, constraints: { max_terms, max_chars_per_term, max_words_per_term } }
    diarization:            { supported, constraints: { max_speakers? } }   # v1 多为 false

  # ── streaming mode ──
  streaming:
    language:               { ... }              # MAY 与 batch 不同
    word_timestamps:        { supported, granularities: [word, segment, char] }
    guidance:                                    # MAY 与 batch 不同（限额/可变性）
      prompt:               { supported, constraints: { max_tokens? } }
      phrase_hints:         { supported, constraints: { max_terms, max_chars_per_term, max_words_per_term } }
      mutable_mid_stream:   { supported }        # 中途可变 guidance（§RT 3.3；流式专有；默认 false=会话锁定）
    emits_partials:         { supported }
    re_segments:            { supported }        # 是否可能发 supersede 事件
    word_stability:         { supported }        # 是否提供有意义的 stable_until
    reconnect:              { mode: seamless | lossy | unsupported }
    finality_level:         { mode: final | closed }
    timestamps:             { mode: native_frame_aligned | post_align | none }

  # ── 引擎全局正交能力 ──
  streaming_input:          { supported }
  streaming_output:         { supported }
  self_resamples:           { supported }   # 唯一行为性能力（§AI 3.2）；仅信息性，R7 仍以 accepted_sample_rates 为权威
```

同一功能（如 `language`、`guidance`）在 `batch` 和 `streaming` 下**分别声明**——同一引擎在不同模式下的能力可以不同。

### 3.3 节点原型

每个叶节点采用三种固定原型之一。所有原型都可派生出统一的 `.supported` 布尔值，使 strict/best_effort 门控对所有节点一致：

| 原型 | 结构 | `.supported` 派生 | 适用场景 |
|---|---|---|---|
| **flag** | `{ supported: bool }` | 直接取值 | 简单支持/不支持，如 `emits_partials` |
| **bounded** | `{ supported: bool, constraints: { ... } }` | 直接取值 | 支持且有标准层机器可校验的限额，如 `candidate_languages` |
| **enum/mode** | `{ mode: Literal[...] }` | `supported := mode not in {"none", "unsupported"}` | 多种互斥实现级别，如 `reconnect` |

`constraints` 专用于标准层**机器可校验**的限额。自由描述性信息 MUST NOT 塞入节点；标准**无 blanket `metadata`**（已废弃）。

**空枚举语义（normative）。** bounded 节点的**枚举型**限额（如 `word_timestamps.granularities`）取空列表时语义为「**未列举 = 不约束（全部）**」，而非「不提供任何值」。但对标准 typed 节点这是**不可表示**态：典型如 `WordTimestampsCap` 的校验器强制 `supported=true` 时 `granularities` MUST 非空（消除「supported 却未列举」的歧义——静默兑现一个未列举的粒度是 cardinal sin）。因此「空 = 全部」仅对 JSON 来源的 raw `x_*`/dict 节点（无 typed 校验器）可达，并仅用于收窄比较（`covers()`）：从「空（全部）」收窄到任意具体子集均合法。运行时参数门控**不**依赖此语义——支持的 `word_timestamps` 总是列举其 granularities，故请求值 MUST ∈ 已列举集合。

**enum/mode 强度序与 token 唯一性（normative）。** enum/mode 节点的合法收窄由**强度序**定义（如 `reconnect`：`seamless` > `lossy` > `unsupported`；`timestamps`：`native_frame_aligned` > `post_align` > `none`；`finality_level`：`closed` > `final`）——`effective` 的 mode MUST 是 `declared` 的同级或更弱级别，否则即放宽（违反 `effective ⊆ declared`）。该强度序按**裸 token 全局键控**（实现 `_MODE_REDUCTIONS`），故**不变量**：同一 major 内**新增**的标准 enum/mode 节点 MUST NOT 复用既有 token 而赋予**不同**的强度序（当前三族 token 互不相交）。未登记于强度序表的 token（含 `x_*` 实验节点的自定 token、未来未知 token）在收窄比较中 **fail-closed**（任何变化一律视为放宽拒绝）。

## 4. 行为（规范）

**R1 — 缺失即不支持 (fail-closed)。** 应用 MUST 将能力树中**缺失的键**视为**不支持**。省略整个 `streaming` 域 = 不支持 streaming。

**R2 — 容忍解析未知键。** 能力容器 MUST 宽容解析未知键（忽略并继续，不报错）。这使新版引擎能被旧版应用安全解析。

> R1 和 R2 是一对**不对称规则**：缺失键 → 安全假定不支持；多出的未知键 → 安全忽略。共同保证向前兼容。

**R3 — additive-within-major。** 同一 major 版本内，能力键只能**新增**，不能修改语义或删除（改/删 ⇒ major 升级）。**无 per-feature `version`**——兼容由协议大版本号统管。

**R4 — 实验能力 `x_*`。** 保留 `x_<vendor>_<feature>` 命名空间供实验性能力；门控规则与标准能力相同。提升为标准能力时 MUST 去掉 `x_` 前缀（遵循 RFC 6648）。

**R5 — `engine.supports()` 点路径查询。** 应用查询能力的**唯一**标准方式。缺失路径返回 `False`（R1 的体现）。应用 MUST NOT 手动遍历能力树或捕获缺键异常。

**R6 — canonical JSON。** `capabilities` 在 Python 侧是 typed pydantic 树；REST `GET .../capabilities` 暴露 canonical JSON，每个节点（叶节点 + **存在的容器/mode 域**）带（派生的）`supported` 字段。enum/mode 节点的 `supported` 由服务端注入（跨语言统一探针），实现见 `DeclaredCapabilities.canonical_json()`。结构契约（跨语言客户端 MUST 可依赖）：**根对象本身不带 `supported`**（它是所有 mode 的容器，非能力）；**缺席的 mode 域序列化为 `null`**（fail-closed，等价于不支持）；`constraints` 等限额子模型非能力节点，**不注入** `supported`（仅保留其限额字段）。

**R7 — Capabilities 与 Properties 边界（定死一处家）。**

| 判定标准 | 归属 | 示例 |
|---|---|---|
| 只有当特性 X 被支持时才有意义的限额 | X 的能力节点 `constraints` | `batch.language.candidate_languages.constraints.max` |
| 引擎固有 I/O 边界值 | Properties | `accepted_sample_rates`、`max_file_size` |

据此 **`max_candidate_languages` 已从 Properties 移入** `capabilities.<mode>.language.candidate_languages.constraints.max`。

## 5. 示例

**查询 streaming 模式是否支持 phrase_hints 引导：**

假设引擎声明中 `streaming.guidance` 下没有 `phrase_hints` 键：

```python
engine.supports("streaming.guidance.phrase_hints")   # → False（R1 fail-closed）
engine.supports("batch.guidance.phrase_hints")        # → True
engine.supports("streaming_input")                    # → True（顶层正交能力）
```

如果应用在 streaming 下传入 `phrase_hints`，strict 抛 `UnsupportedFeatureError`；best_effort 忽略并返回 diagnostic。

**effective 能力收窄：** 引擎声明 `batch.word_timestamps.supported=true`，但用户未配 `forced_aligner` → `effective_capabilities` 中该项为 `false`。DeclaredCapabilities 不变（CLI/注册表仍显示"支持"），运行时门控拒绝该请求。子集不变量保持。

## 6. 附注与理由

- **废弃三套旧系统**：`supports_*`（无分域、无约束）、`FeatureFlag`（无法表达 batch/streaming 差异）、`feat_flag`（per-feature version 无用——协议兼容是大版本全有或全无）。统一层级树是三者的超集。
- **fail-closed 而非 fail-open**：声明不完整的引擎不会被误认为支持所有功能。应用可放心依据 `engine.supports()` 做决策。
- **边界规则(R7)**：消除「同一限额两个真相源」的歧义——限额与所约束的功能始终同住。
- **砍掉 blanket `metadata`**：无已知用例、鼓励非结构化信息、破坏机器可读性。需要时可通过 additive-minor 以结构化字段添加。


---
---

# Runtime 参数 (Runtime Parameters) — NORMATIVE

> **本节定义**：应用在每次转写请求中可以传入哪些参数（可移植标准集 + 引擎特有逃生舱）、引擎如何校验和响应这些参数、以及 `guidance` 引导家族的共享契约与扩展机制。
> **另见**：[§能力系统](#能力系统-capabilities--normative)（`engine.supports()` 点查）、[§语言选择](#语言选择-language-selection--normative)（`language`/`candidate_languages` 解析）、[§流式协议](#流式协议-streaming--normative)（流式参数冻结）、[§Init Config](#init-config-baseconfig--normative)（init/runtime 边界）。
> **组织**：概述 → 术语 → 声明与参数 → 行为（规范）→ 示例 → 附注与理由。
> **取代**：idea_docs `spec/options.md` 的子类化方案。

## 1. 概述（可移植性与灵活性的张力）

"Runtime 参数"是应用在调用 `transcribe` 或 `start_transcription` 时、随每次请求传入的设置——比如语言、是否需要词级时间戳、或一段引导识别的提示文本。

不同引擎暴露的旋钮千差万别。如果标准允许引擎自由添加字段（旧方案：子类化 `BaseTranscribeOptions`），可移植性从根上被打破。但如果只暴露固定字段，又会扼杀高级功能。

Standard ASR 的解法是**双层设计**：封闭的**可移植标准集**（跨引擎验证过的标准字段，由 capability 门控）+ 受控的 **`provider_params`**（引擎特有的 typed 逃生舱，显式标注为锁定特定引擎）。在此之上，`guidance` 引导家族用**扁平字段 + 共享契约**统一不同引擎的引导能力。

## 2. 术语

| 术语 | 含义 |
|---|---|
| `RuntimeParams` | 每次请求的参数容器。**封闭的** pydantic 模型（`extra="forbid"`, `frozen=True`）。ASR 作者不得新增顶层字段。 |
| `provider_params` | `RuntimeParams` 中唯一的引擎特有槽位。typed pydantic 模型（`extra="forbid"`），承载不可移植旋钮。使用即锁定该引擎。 |
| `guidance` 家族 | 可移植标准集中"引导识别"的一组扁平字段。v1 含 `prompt` 和 `phrase_hints`，共享一套行为契约。 |
| channel | `guidance` 家族中的单个字段。每个在能力树有节点 `capabilities.<mode>.guidance.<channel>`。 |
| strict / best_effort | 全局处理策略。strict：不支持的标准集参数抛 `UnsupportedFeatureError`。best_effort：忽略+返回结构化 diagnostic（哪个参数被忽略/为什么/最终值）。 |

## 3. 声明与参数

### 3.1 v1 可移植标准集

| 字段 | 类型 | 默认 | Capability 路径 | 含义 |
|---|---|---|---|---|
| `language` | `str \| None` | `None` | `<mode>.language.runtime_override` | 本次语言（BCP-47 / `"auto"`），覆盖 `default_language`。解析见 [§语言选择 R2](#语言选择-language-selection--normative)。 |
| `candidate_languages` | `list[str] \| None` | `None` | `<mode>.language.candidate_languages` | 候选语言列表（仅 `auto` 下有意义）。解析见 [§语言选择 R3](#语言选择-language-selection--normative)。 |
| `word_timestamps` | `WordTimestampGranularity \| None` | `None` | `<mode>.word_timestamps` | 词级时间戳。**枚举**（`word \| segment \| char`），非 bool。 |
| `prompt` | `str \| None` | `None` | `<mode>.guidance.prompt` | 自由文本软提示（§3.3）。 |
| `phrase_hints` | `list[str] \| None` | `None` | `<mode>.guidance.phrase_hints` | 词条 boost 集（§3.3）。 |
| `on_unsupported` | `Literal["fail", "degrade_to_prompt"]` | `"fail"` | （无——策略指令） | `guidance` 降级策略（§3.3 opt-in 降级）。**无 capability 路径**：它是控制*遇到不支持的 guidance channel 时是否降级*的策略字段，不被能力门控。`"fail"` = **不降级**（按 strict/best_effort 走标准门控：strict 抛错、best_effort 丢弃+diagnostic）——它**不**强制本次请求整体失败，最终行为由全局 strict/best_effort 决定；`"degrade_to_prompt"` = 单向降级 rich→prompt。它是 `RuntimeParams` 顶层字段且**随 wire 可移植集传输**（`WireRuntimeParams`）。 |

### 3.2 `provider_params` 逃生舱

| 字段 | 类型 | 说明 |
|---|---|---|
| `provider_params` | `<EngineParams> \| None` | 引擎发布的 typed pydantic 模型（`extra="forbid"`）。传错引擎的 params 模型 = 校验错误（swap 安全）。 |

**类型匹配是精确的（normative）**：swap 安全用**精确类型**核对（`type(provided) is <EngineParams>`），**不是** `isinstance`。`isinstance` 会静默接受 `<EngineParams>` 的**子类**——而同 vendor 引擎家族用继承复用参数模型是自然写法（`EngineBParams(EngineAParams)`），于是把 B 的 params 传给引擎 A 会通过，B 独有的旋钮被静默忽略（正是头号大罪）。因此**每个引擎 MUST 发布独立的终端 params 类型**；继承不是声明跨引擎兼容的通道。引擎 MUST NOT 把裸基类 `ProviderParams` 声明为其 `provider_params` 类型（裸基类无字段、对任何 params 都放行，使 swap 安全归零——合规套件 SHOULD 对此报 error）；调用方传裸基类实例（或被强转成裸基类的 mapping）在构造期即被 `RuntimeParams` 拒绝。

要点：错误**始终抛异常**（独立于 best_effort——代码契约，非能力协商）；schema MUST 作 **JSON Schema** 暴露（`GET .../params-schema`，可移植契约是 JSON Schema 非 Python 类）；auto-UI MUST 隔离标注"engine-specific：用了锁定 {engine}"并默认折叠；治理：≥N 独立引擎语义等价 → minor 版本提升为标准集（单向）。

### 3.3 `guidance` 引导家族

**扁平字段**直接在 `RuntimeParams` 上（`params.prompt`/`params.phrase_hints`），不嵌套子对象。对应 capability 在 `capabilities.<mode>.guidance.<channel>` 下。

**v1 channels:**

| Channel | 类型 | Capability 约束 | 映射 |
|---|---|---|---|
| `prompt` | `str \| None` | `{supported, max_tokens?}` | OpenAI `prompt` / Whisper `initial_prompt` / Qwen `context` / 火山 `context` |
| `phrase_hints` | `list[str] \| None` | `{supported, max_terms, max_chars_per_term, max_words_per_term}`（**per-mode**，batch/streaming 限额可不同） | ElevenLabs `keyterms` / faster-whisper `hotwords` / Tencent/FunASR hotwords |

**共享契约**：每 channel = optional · best-effort · 非绑定 · **正极性** · capability 协商 · **永不静默降级**。

**`prompt.max_tokens` 计数（normative）**：标准层无引擎 tokenizer，故对 `max_tokens` 的强制用一个**保守的、脚本感知的近似**——空白分词数 **加上** 每个**无空格脚本**（CJK / 假名 / 谚文 / 泰文 …）码点各计 1——**不是**引擎的精确 token 数。纯空白词数会把无空格长提示（如下 §5.3 Qwen3 `context`→`prompt`）坍缩成 ~1 token，从而漏过它实际超出的 `max_tokens`；CJK 项消除该漏判。**「无空格脚本」按 Unicode Script 属性界定**（normative）：凡 Script 为 CJK 表意文字、假名（Hiragana/Katakana，**含**半角片假名 U+FF66–FF9F、片假名音标扩展、Kana Supplement/Extended）、谚文（Hangul，**含** NFD 分解出的组合 Jamo U+1100–11FF 与兼容 Jamo）、泰文/老挝文/高棉文/缅甸文、彝文的码点均计入——逐 block 取主干子集是不允许的（半角片假名、NFD 谚文等是真实书写形态而非长尾）。**保证的诚实范围**：该近似相对「空白分词 + 无空格脚本逐码点」的计数永不少计；但相对引擎真实的 **subword（BPE）tokenizer**，长拉丁词 / URL / 长数字串（我们计 1，BPE 可能计 6–17）**可能少计**，超预算的此类 prompt 可能溜过门控。因此作者声明 `max_tokens` 时 SHOULD 在引擎硬上限之下预留余量（headroom），而非贴线声明；标准对声明值 MUST NOT 超出（strict 抛错，best_effort 截断+diagnostic）。

**null 语义**：`None`=未请求；`[]`=请求但空（显式"无 hints"）；capability `supported=true`=引擎能 honor。

**`prompt` 截断契约（normative）**：当 best_effort 因超 `max_tokens` 而截断 `prompt` 时，结果 MUST 是原文的一个**前缀**（"丢弃尾部"），**保留原文空白**——不得把保留部分的换行/连续空白重排为单空格（多行提示保留其换行；降级路径用 `\n` 连接原 prompt 与框定 hints 的结构同样保留）。截断点 MUST NOT 落在**组合字符序列**（基码点 + 其后续组合标记，如泰文基辅音 U+0E01 与元音符 U+0E31）内部，以免留下基字符被切去标记的半成品字位（判定用 Unicode general category `M*`，而非 canonical combining class——后者对许多组合标记为 0）。

**`phrase_hints` 词条（normative）**：每个词条 MUST 非空且非纯空白（空/纯空白词条无 boost 语义，且会击穿降级存活判定——`"" ∈ 任意串` 恒真——使标准层误报降级成功）；`RuntimeParams` / wire 在构造期拒绝含空白词条的列表（用 `[]` 表达"无 hints"）。

**空哨兵的门控行为（normative）**：list channel 的 `[]`（请求但空）**MUST NOT 参与能力门控**——它不计为一次请求，不被门控、不降级、不报 unsupported diagnostic，标准层原样透传（引擎即便不支持该 channel 也只会收到 `[]`，等同未请求）。这是为了让"用户清空了一个多选框"与"用户没碰它"在不支持的引擎上行为一致（最小惊讶）。与之相对，str channel 的空串 `""`（`prompt=""`）**是一个真实值**，按非空 prompt 一样门控（不支持 channel：strict 抛错、best_effort 丢弃+diagnostic；支持则计入 `max_tokens`）——str 没有独立的"请求但空"哨兵，`""` 即"请求了一个空提示"，显式优于隐式。独立实现 MUST 遵循此归属，否则同一请求跨实现行为分叉。

**未知 channel 不对称**：引擎收到未声明 channel = fail-closed（strict raise / best_effort diagnostic）；app 收到未知 capability 键 = 容忍忽略。

**opt-in 降级**：`on_unsupported="degrade_to_prompt"` 启用单向降级（rich→prompt，框定文本）；默认 fail-closed（直接序列化短语进 prompt 掉精度，禁自动）；每次降级 MUST 发 diagnostic。

**明确不进 `guidance`**：verbatim/disfluency/标点/ITN → 标准三态 flag reserve（§4 R6）；profanity-mask/entity-redaction/格式化 → 结果/格式模型；系统指令 → 出 ASR 范围；domain → preset；`bias_resource`(注册词表) → Init Config。

**扩展**：新 channel = additive-minor；`x_<vendor>_<channel>` 实验 + 提升去前缀（RFC 6648）；无 per-channel version；channel 名不复用。

**流式**：guidance MAY 进 `capabilities.streaming.guidance.*`；中途可变性由 flag 节点 `capabilities.streaming.guidance.mutable_mid_stream`（`{ supported }`，§C 3.3）声明，经 `engine.supports("streaming.guidance.mutable_mid_stream")` 查询。该 flag **流式专有**（batch guidance 无此节点，单 shot 谈不上中途可变）。默认 `supported=false`=会话锁定（与 fail-closed 一致）；v1 **保留**该声明位但不承诺 `update_guidance()` 方法——合规套件不要求任何运行时行为，`covers()` 的集合包含自动拒绝 `declared=false→effective=true` 放宽。

## 4. 行为（规范）

**R1 — 请求类型封闭。** `RuntimeParams` 是封闭类型（`extra="forbid"`）。ASR 作者 MUST NOT 新增顶层字段。合规测试强制。

**R2 — strict / best_effort。** 不支持的标准集参数：strict 抛 `UnsupportedFeatureError`；best_effort 忽略+结构化 diagnostic。`provider_params` 错误不走此策略（R3）。

**R3 — `provider_params` 错误模型。** 未知键/类型错/越界 MUST 始终抛 `InvalidProviderParamError`，独立于 strict/best_effort。校验顺序：先 `provider_params`（快失败），再标准集门控；二者 MUST NOT 互相吞掉。

**R4 — `guidance` 共享契约。** 每 channel MUST 遵守：optional / best-effort / 非绑定 / 正极性 / capability 协商 / 永不静默降级。未知 channel 不对称（引擎 fail-closed / app 容忍）。降级 opt-in+单向+diagnostic。

**R5 — 流式参数冻结。** 流式会话中 `RuntimeParams` 在 `start_transcription` 时锁定，MUST NOT 中途修改（`mutable_mid_stream` 除外，见 §3.3）。

**R6 — 识别行为三态 flag（v1 占位）。** verbatim/disfluency/punctuation/ITN/profanity-filter 未来作标准三态（`unset | on | off`，capability 门控），不进 `guidance`。v1 先走 `provider_params`。

**R7 — 批量运行时失败错误契约。** `transcribe` / `transcribe_async` 在引擎执行期（模型推理、网络调用、SDK）发生失败时 MUST 抛 `TranscriptionError`，并以 `raise TranscriptionError(...) from exc` 保留原始异常为 `__cause__`——使应用得以**跨引擎**用单一类型捕获运行时失败，而非各引擎各抛其原生异常（`RuntimeError` / SDK 异常 / `requests.HTTPError`…）。这是流式 [§6.2](#流式协议-streaming--normative) `error` 事件 `engine_error` 码的批量对应物：流式把引擎逃逸异常包装为 `engine_error` 事件，批量把它包装为 `TranscriptionError`。它表示**引擎/运行时故障**，与调用方可修的错误（`ConfigError` / `UnsupportedFeatureError` / `InvalidProviderParamError` / `AudioProcessingError`）属不同故障域，server MUST 映射为 5xx（而非 4xx）。约束作用于引擎模板钩子 `_transcribe`（`EngineBase` 为批量管线提供包装位点）；合规套件无法静态核验引擎是否包装，故本契约由规范 + 模板 + 文档约束，不进运行时强制门控。

### 5.1 OpenAI：prompt + temperature

```python
result = engine.transcribe(
    audio,
    params=RuntimeParams(
        language="en",
        prompt="This is a meeting about the Q3 budget review.",
        provider_params=OpenAIParams(temperature=0.0),  # 引擎特有
    ),
)
```

`prompt` → 标准集（可移植）；`temperature` → `provider_params`（不可移植，换引擎会被捕获）。

### 5.2 ElevenLabs：keyterms → phrase_hints

```python
result = engine.transcribe(
    audio,
    params=RuntimeParams(
        phrase_hints=["Anthropic", "Claude", "Standard ASR"],
    ),
)
# 适配器内部：phrase_hints → ElevenLabs keyterms
```

### 5.3 Qwen3：context → prompt

```python
result = engine.transcribe(
    audio,
    params=RuntimeParams(
        prompt="前文提到了量子计算和超导材料的最新进展。",
    ),
)
# 适配器内部：prompt → Qwen3 context
```

### 5.4 provider_params swap 安全

```python
# 从引擎 A 切到 B，忘了改 provider_params：
result = engine_b.transcribe(
    audio,
    params=RuntimeParams(
        provider_params=EngineAParams(beam_size=5),  # 类型错！
    ),
)
# → 立即抛 InvalidProviderParamError，不静默忽略
# （精确类型核对：即便 EngineAParams 恰是 EngineBParams 的父/子类，也照样抛——
#   继承不豁免 swap 安全，否则 A 独有旋钮会被引擎 B 静默忽略。）
```

合规套件提供 swap 安全探针 `check_provider_params_swap_safety`：以一个**外来** `provider_params` 子类（私有于套件、永不与引擎声明类型重合）调 `transcribe`，断言无论 strict / best_effort 都抛 `InvalidProviderParamError`。因 R3 规定 `provider_params` 先于音频解码校验（快失败），该探针在触达模型前即返回，**无计费副作用**——抓的是绕过 `EngineBase` 模板又忘了校验的引擎。

## 6. 附注与理由

- **为何拆 prompt 为 prompt + phrase_hints**：Whisper `initial_prompt`(free-text) ≠ ElevenLabs `keyterms`(phrase boost)。单一 `prompt` 字段让同样的值在不同引擎上做不同的事——假可移植。拆开后语义精确，capability 和 constraints 分别门控。
- **为何 provider_params always-raise**：best_effort 为能力协商设计；`provider_params` 错误是代码 bug（如忘了改 params 就换引擎）。对 bug 快失败，不静默继续。
- **为何扁平字段而非子对象**：子对象增嵌套无新信息；typed-item list 有 oneof null 歧义+丢可发现性。扁平最简——IDE 补全即可发现 channel。
- **路由出 guidance 的理由**：识别行为开关有确定可检验效果（开/关），不是建议性引导；格式化/脱敏是输出后处理；系统指令会让 namespace 退化为指令垃圾桶。
- **reserve 候选**：`context` 独立 channel（v1 并入 prompt）；`phrase_hints` 权重（先 provider_params）；`pronunciation_hints`；`negative_phrases`（须先加 polarity 轴）；音频载荷类（音频示例/few-shot → 独立家族）。


---
---

# 结果模型 (Transcription Result) — NORMATIVE

> **本节定义**：`transcribe` 和流式会话返回的转写结果的统一数据结构——顶层 `TranscriptionResult`、其子模型 `Segment` / `Word`、多通道与说话人分离的表示方式、以及格式渲染（SRT/VTT）的职责归属。
> **另见**：[§能力系统](#能力系统-capabilities--normative)（capability 决定 optional 字段是否被填充）、[§流式协议](#流式协议-streaming--normative)（`TranscriptionEvent` 与事件流中的 Segment/Word 共享）、[§Runtime 参数](#runtime-参数-runtime-parameters--normative)（`word_timestamps` 等参数如何影响结果）。
> **组织**：概述 → 术语 → 声明与参数 → 行为（规范）→ 示例 → 附注与理由。
> **取代**：idea_docs `spec/results.md`。

## TR.1 `TranscriptionResult`（恒定 schema）
```
TranscriptionResult:
  text: str                              # 必填；完整转写
  detected_language: str | None          # BCP-47；auto 模式回传
  language_confidence: float | None      # 0-1
  duration: float | None                 # 秒
  segments: list[Segment] | None
  words: list[Word] | None               # 扁平词级（也可嵌 segment 内）
  channels: list[ChannelResult] | None   # 多通道分离（TR.4）
  diagnostics: list[Diagnostic]          # best_effort / 转换 / 降级 诊断
  metadata: dict[str, Any]               # 标准化元信息
  extra: dict[str, Any]                  # 引擎特定/实验（含 provider 渲染格式）
```
- **返回类型恒定**：`response_format` 不把返回变字符串；多通道不把顶层换成 `transcripts[]`。
- **null 规则（消歧）**：capability 声明「是否支持」；字段 `None`=**未请求/不适用**；`[]`=**请求但空**（如静音）。app 判「不支持」看 **capability**，不看字段 null。

## TR.2 `Segment` / `Word`（流批共享子模型）
```
Segment: start:float  end:float  text:str
         words:list[Word]|None  speaker:str|None  channel:int|None
         avg_logprob/no_speech_prob/…:float|None  extra:dict
Word:    start:float  end:float  text:str
         probability:float|None  speaker:str|None  channel:int|None  extra:dict
```
- **时间单位 MUST = float 秒，原点 = 提交音频的第一个采样（音频时间 t=0）**，与 §ST 同一原点。**每通道内**跨段单调；多通道时不同通道的段 `[start, end]` **允许重叠**（双声道同时说话），顶层 segments 按 `start` 稳定排序、`start` 相同时按 `channel` 排。适配器把 ms / protobuf-duration / ticks 转入。
  > **排序是引擎义务，非构造期强制（明示，与 TR.4 不对称）**：该 `(start, channel)` 排序与每通道单调性是**引擎/适配器的义务**，由合规套件校验（§G.2.1 合规与运行时共用校验逻辑），但**不**在 `TranscriptionResult` 构造期强制——`StreamReducer` 对无时间戳引擎合法地保留到达顺序、且仅按 `start` 排（无 channel tie-break），严格的 `(start, channel)` 构造校验会误拒合法归约结果。与 TR.4 的构造期强制（见下）不对称是有意的：TR.4 拒绝的是**不可表示的歧义形状**（无合法生产者），而违反 TR.2 排序的乱序 segments 是合法可表示的中间产物。渲染器在自身边界防御性重排。
- `probability ∈ [0,1]`；若引擎给 logprob，**另立字段**，不与 probability 混。
- **流批共享**：`TranscriptionEvent.segment/.words`（D10）MUST 用**同一** `Segment`/`Word`；流式专属字段（`stable_until` 等）加在**事件包装层**，不污染共享子模型。
- **`session.result() -> TranscriptionResult`**：流式会话可归约为最终结果（反映 `final`；late `closed` 重格式化可更新它）。

## TR.3 时间戳粒度
`word_timestamps` 枚举 `word|segment|char`；char 级 reserve（additive）。

**声明语义（normative）**：`capabilities.<mode>.word_timestamps.granularities` 声明的是引擎**能诚实交付**哪些粒度的时间戳，**不是**上游 API 是否有同名开关。引擎 MUST 声明它能服务的**每一个**粒度——包括零成本恒真的那些。多数模型每次转写都恒带 per-segment `start`/`end`：此种引擎 MUST 声明 `segment`，即便上游没有独立的 "segment 模式" 参数；否则标准层会把最便宜、恒可满足的 `segment` 请求当作"不支持"硬拒（strict）或丢弃（best_effort），制造**假不兼容**。声明后映射 MUST 按粒度精确：仅 `word` 触发词级（forced-alignment）计算，`segment` 请求 MUST NOT 回填未请求的词级数据（`words=None`=未请求，§TR.1 null 规则）。

## TR.4 多通道（恒定 shape，非顶层 `transcripts[]`）
- 顶层 `text`/`segments`/`words` **始终是全通道、说话人/通道无关的完整转写**——多通道时是**按时间合并所有通道**（不是 channel-0-only；使「无视 channels」安全无损）。
- `channels: list[ChannelResult] | None`：`None`=未做通道分离（常态）；present=每 `ChannelResult{channel:int, text, segments, words}`，**每通道一条**。
- 顶层 `Segment/Word.channel` 可选，给单遍 app 看 provenance（纳入 Google `channelTag` / EL `channel_index`）。
- **不变量**：present 时顶层可由 `channels` 按时间推导，两视图不冲突。**构造期强制**（不可表示的非法形状 MUST 在模型构造时被拒绝）：
  - `channels` 中任一通道携带 `segments`/`words` 而顶层对应字段为 `None` ——否则"无视 channels 安全无损"的承诺被静默打破（渲染器会把仅 channels 的结果坍缩成单条无时间 cue）。
  - `channels` 中 `channel` 索引**重复**——TR.4 语义是「每通道一条」，重复使顶层合并歧义、且按 channel 建字典的消费方静默丢一半数据。

## TR.5 说话人（v1 reserve shape，feature 延后）
diarization 特性 v1 多不支持，但 **shape 现在就预留**（additive-safe）：`Segment.speaker`（**权威**）+ `Word.speaker`；**不**加顶层 `speakers[]` roster（YAGNI，需要时 additive）。

## TR.6 SRT/VTT 等格式（核心渲染，非返回类型）
- 禁 `response_format`→字符串。核心库提供 **`to_srt(result) -> str` / `to_vtt(result) -> str`**（基于恒定 `segments`），每个 compliant 引擎一键可得（**强于现状**：现状只有部分引擎给）。
- **cue 文本消毒（格式正确性，normative）**：渲染器 MUST 保证 cue payload 既不能伪造/破坏 cue 结构，也不能被静默丢弃。
  - **行终止符归一**：先把 `\r\n` 与裸 `\r` 归一为 `\n`，再折叠空行——裸 `\r` 在 WebVTT 与多数 SRT 解析器中均是行终止符，不归一则 `\r\r` 可绕过空行折叠伪造 cue。
  - **WebVTT 实体转义**：`to_vtt` MUST 按 W3C WebVTT cue-text 文法转义 `&`→`&amp;`、`<`→`&lt;`、`>`→`&gt;`（先 `&` 后 `<`/`>`，避免二次转义）。裸 `<` 会开启 cue-span tag、被浏览器 tokenizer 消费至下一个 `>`，使尖括号内文本（如引擎泄漏的 `<unk>`/`<|...|>` token、口述数学）在字幕中**静默消失**——正中「静默错误结果是头号大罪」。转义 `>` 同时使 payload 中的 `-->` 不再可能被读作 cue timing。
  - **SRT 不转义**：SRT 无字符引用机制，`to_srt` MUST NOT 套用实体转义（否则把字面 `&amp;`/`&lt;` 显示给用户）；`&` 与尖括号原样透传，下游若需中和标签应在渲染前对转写文本处理。
- **`segments` 缺失时的回退（normative，消除跨实现未定义行为）**：基于 §TR.1 null 规则——`segments is None`（未请求/不适用）且 `text` 非空 → 合成一条覆盖全文的 cue：`[0, duration]`，`duration` 未知（如归约流）时用固定 `[0, 3s]`（播放器静默丢零时长 cue，故回退 cue MUST 非零时长）。`segments == []`（请求但空，如静音）→ 零 cue，绝不杜撰。其他语言 SDK MUST 采用同一回退以保「同结果同渲染」。
- provider 渲染的高保真格式仅作 **`result.extra["provider_formats"]["srt"]`** 透传，显式非可移植、非推荐路径。
- 据此 `response_format`/`additional_formats` 退出可移植 runtime 集（渲染是事后，非参数）。

## TR.7 演进规则
新结果字段 MUST optional + `None`/空默认；现有字段类型/名在 major 内冻结；`extra → 一等` 提升 additive。**实体/脱敏（GAP-27）等 niche 家族走扩展命名空间，MUST NOT 进顶层**——顶层只放普适字段（text/language/duration/segments/words + 可选 channel/speaker）。


---
---

# 依赖与兼容 (Dependencies) — NORMATIVE

> 定稿。对应 D4。

## DEP.1 核心依赖
核心 = `pydantic` + `numpy`（仅此）。numpy 用 **interpreter-conditional 下界、无硬上限、只用长期稳定 API 子集**：
```
numpy>=1.26; python_version <  "3.13"
numpy>=2.1;  python_version >= "3.13"
```
（1.24/1.26 无 cp313 wheel；Python 3.13+ 起需 numpy 2.x。）**无上限 cap**（遵 numpy 下游指南 / SPEC-0）。

## DEP.2 稳定子集强制
核心只用 numpy 1&2 行为一致的 API；**`clip`/`astype` 等有行为变化的点 MUST 防御**（编码路径**先 `clip` 再 cast`**；禁 `copy=False`，用 `np.asarray`）。

CI MUST 守住 numpy 1.x↔2.x 的兼容面,通过以下并行通道(实现见 `.github/workflows/`,策略见 `CONTRIBUTING.md`「Dependency policy」):
- **ruff NPY201** —— 静态拦截 numpy 2.0 移除/改名的 API。
- **warnings-as-errors** —— `pytest` `filterwarnings=["error", …]`,把 numpy(及其他)的 deprecation 升级为失败(取代旧的逐 job `-W error`)。
- **锁定通道**(`--locked`,py3.10–3.14):跑提交的 `uv.lock`,即当前 numpy 2.x。
- **下界通道**(`--resolution lowest-direct`,py3.10):贴 `numpy>=1.26` 下界跑,守住 numpy 1.x 兼容面。
- **numpy floor 通道**(py3.13 钉 `numpy==2.1.*`):守住 `numpy>=2.1` 这一 interpreter-conditional 下界(下界通道在 3.10 上不会触及它)。
- **每日 canary**(两轴):`latest` 稳定 + `prerelease`(`uv lock --upgrade [--prerelease allow]`)。prerelease 轴是旧 **numpy-nightly canary 的后继**,提前捕捉 NEP 50 等尚未发布的上游行为变更;非 PR 门禁,失败仅开追踪 issue。

> 旧表述「numpy 1.26 与最新 2.x 双测 + numpy-nightly canary lane」由上述锁定/下界/numpy-floor/canary 四通道等价替代(对应 D1/D4 与依赖管理规格)。

## DEP.3 不强制 numpy 2+
标准固定 **numpy-float32-ndarray 类型**，不固定版本；不排除仍绑 numpy 1 的引擎（如 FunASR）。

## DEP.4 硬冲突 = 进程隔离
插件-插件 numpy1-vs-2 **无法在单进程共存**（Python 事实，非设计可消除）。逃生舱：**subprocess + UDS + `shared_memory`**（轻于 FastAPI；避免文本序列化、保零拷贝）为首选；FastAPI server 留给真·远程/跨语言。MUST 定义 `engine_id → endpoint` 寻址；提供薄 **isolation shim**（fast-follow）使 app 代码不变。

## DEP.5 `standard-asr doctor`
只读诊断：枚举已装插件 entrypoint，读各自 `Requires-Dist`，按**运行解释器**求值环境标记（PEP 508 marker；只取标记成立或缺失的行），算 numpy 版本区间交集，空交集报冲突 + 一行补救（含「3.13 上 `numpy<2` 无 wheel」）。**不做** resolve/install。

**v1 范围（诚实声明）**：doctor **只精确诊断 `numpy`**——它是标准本身唯一的共享原生依赖（DEP.1），其 1.x↔2.x 是干净的 C-ABI 断层、且冲突完整编码在版本区间里，故版本区间交集可判定。其余共享原生库的冲突在 v1 **明确未覆盖（known-uncovered）**，因为其冲突模型与 numpy 根本不同、无法用同一套版本区间交集判定：
- **torch**：冲突是 CUDA 构建*变体*（`cpu`/`cu118`/`cu121`…），**不**体现在版本号 specifier 里。
- **onnxruntime vs onnxruntime-gpu**：是包*身份*冲突（两个不同分发包），非版本区间冲突。

把 numpy 的版本交集逻辑泛化到上述库会给出**自信而错误**的诊断（本工具的基数罪），故 v1 不做。对这些库的硬冲突，依 **DEP.4** 的通用进程隔离（subprocess + UDS + `shared_memory`）逃生舱处理。未来若要精确诊断，需为每类库引入其特有的冲突模型，而非复用 numpy 的版本交集。


---
---

# Init Config (BaseConfig) — NORMATIVE

> 定稿。对应 D7。pydantic v2，UI-discoverable。

## IC.1 结构
`BaseConfig` = 判别器 `engine` + 「相关才用」可选标准字段（IC.5 mixin）+ 引擎声明字段。

## IC.2 判别器（解碰撞 + 解身份混淆）
`engine` MUST = **entrypoint 派生的 `engine_id`**（registry 唯一、PEP503 规范化），作者**不手写**；跨插件路由是 **registry 查找**，**不是**宇宙级 pydantic `Union`（开放世界无法枚举）。发现层 MUST 检测重复 `engine_id`（两分发包归一同名）→ fail-loud / 标记 shadowed。

**两层 fail-loud（normative）。** 标记与报告分两层，互补：发现层把碰撞记入 `ModelRegistry.shadowed_engine_ids` 并在默认（非 strict）发现时 `logger.warning`、strict 发现时抛 `EntrypointValidationError`（实现：`discover_models`）；**合规套件 MUST 把每个 shadowed `engine_id` 报告为 error**（实现：`check_entrypoints`，code `engine_id_collision`），且 MUST 在默认运行中也报告它——碰撞使 `config.engine` 路由依赖安装顺序，是协议级身份混乱，不能仅以一行日志放行。无效 entrypoint 名（strict 发现会抛 `EntrypointValidationError`）同样 MUST 由合规套件转为 error issue 而非异常：合规检查永远返回报告（`check_entrypoints` 承诺 `Raises: None`），并随后以宽松模式重新发现，使同环境内合法引擎仍被检查。

## IC.3 凭证安全（normative）
- 凭证字段 MUST 用 **`SecretStr`**，且 MUST 是 `BaseConfig` 的**顶层标量字段**（mask `repr`/`str`/默认 `model_dump`）。secret-field 标记 MUST NOT 出现在嵌套 submodel 或 secret 容器（如 `list[SecretStr]`）上：脱敏管线（类定义期强制、空白保留 validator、`public_dump` 按名遮蔽）只覆盖顶层标量字段，嵌套/容器 secret 会静默泄漏明文——参考实现在**类定义期 fail-loud 拒绝**这两种形态（`__pydantic_init_subclass__`），指引把凭证提升为顶层标量。
- 序列化：`/v1/models`、持久化、telemetry 用**脱敏 dump**（参考实现 `BaseConfig.public_dump()`，亦是默认 `model_dump`/`model_dump_json` 行为）；仅显式调用**专门的 reveal-dump API**（参考实现 `BaseConfig.reveal_dump()`——独立命名方法，比布尔开关更易 grep 审计）在进程内调引擎 SDK 时材料化明文。该明文结果 MUST NEVER 被日志/持久化/下发 `/v1/models`/telemetry。
- secret-field 标记（`json_schema_extra`）→ auto-UI 渲染 password / write-only；REST POST 收、GET 不回。
- **密（`api_key`/token） vs 端点路由（`base_url`/`region`/`org_id`，非密）分两类**：后者可日志/UI/序列化。

## IC.4 env 回退（normative）
- **`STANDARD_ASR_<NORMENGINE>__<NORMFIELD>`**（引擎段与字段段之间用**双下划线** `__` 分隔）。normalization = 大写、把每个非字母数字**连续段**折成**单个** `_`（非每字符一个 `_`），故任一段不含 `__`，使 `__` 边界唯一可解析。**为何双下划线**：单下划线分隔下 `(engine="openai", field="api_key")` 与 `(engine="openai-api", field="key")` 都归一为 `STANDARD_ASR_OPENAI_API_KEY`，引擎边界不可恢复——两个引擎可能静默读到彼此凭证。**碰撞检测**：单 config 类内拒绝两字段归一同名（跨引擎碰撞已由 `__` 边界根治）。
- 一约定 per 字段名（引擎 native 名如 EL `xi-api-key` 在适配器映射到标准名）。**覆盖范围**：env 回退覆盖 config 的**所有字段**——标准 mixin 字段（凭证/端点路由/device/语言/download root）**与**引擎声明字段（如 `beam_size`、`model_path`）均获得对应 env 入口，这是有意的全表 DX。**仅排除三个 fail-loud 安全/身份字段**：`engine`（entrypoint 派生身份，绝不由 env 设）、`strict` 与 `allow_private_urls`（env 不得静默翻转 best_effort 或放宽 SSRF 守卫）。
- **复合（非标量）字段的 env 值先按 JSON 解析**：env 变量恒为裸字符串，无法强转为 `list`/`dict` 等（如标准字段 `default_candidate_languages: list[str]`），故对复合注解字段先 `json.loads`，**解析失败保留原串**让构造期**响亮失败**（绝不静默丢值）。标量字段（含凭证 `SecretStr`、`Path`）**原样透传**、不被重解释。
- 优先级：**显式 config > env > （必填缺失）报错**。「显式」= 该字段名**作为键出现**在显式入参中——显式传入的 `None` **是一个值**、压过 env（规则是「显式即胜」，非「显式非 None 才胜」）；包装层若以 `None` 默认透传可选 kwargs，需先剔除 `None` 键才能让 env 回退生效。多账户：保留 profile 段 hook（v1 不实现）。

## IC.5 适用性谓词（applicability —— 跨 §C/§AI/IC 同一规则）
可选标准字段用 **capability-bearing config mixin**（`DeviceConfigMixin`/`LanguageConfigMixin`/`DownloadConfigMixin`…）：**字段出现在模型里 ⇒ 适用**——auto-UI 据此渲染正确表单，无需逐字段隐藏。「缺失 ⇒ 不适用」；「present-with-default ⇒ 适用-默认」。

## IC.6 `default_language`（解与语言设计的矛盾）
**适用 ⇒ 必填或标准默认**：引擎声明语言能力（非 trivial `selectable_languages` 或 `supports_runtime_language_override`）则 MUST 供 `default_language`（可为 `auto`）——保 `effective_language` 算法 total。D7 **不**放松已定稿语言设计（取代原「对所有引擎非强制」表述）。

## IC.7 init / runtime 边界
init = 实例存续期固定、属安装/部署选择（权重/路径、device、凭证、batch size、aligner 装配、默认语言）；runtime = 每请求可变。**Tie-breaker：能按请求变 ⇒ runtime（`provider_params`），不进 init**（即便引擎也接受构造期传）。模型选择 = **entrypoint preset**，非 init `model` 字段。

## IC.8 多 artifact
nested 引擎声明 submodel（按 model-family）+ 标准 **artifact 路径解析 helper**（相对 cache-dir、存在性、可选 checksum）；标准**不**标准化 bundle 形状。

## IC.9 lazy 纯度不变量
`__init__` 捕获 config MUST 纯——无 FS 创建 / 路径探测 / GPU init / 网络。cache-dir、凭证仅在 `_ensure_model_loaded` 材料化，受 `allow_downloads()` 门控。download/cache 走 `DownloadConfigMixin`（`download_root` + 优先级：显式 > `STANDARD_ASR_MODEL_DIR` > 库默认 HF cache > `~/.cache/standard-asr`）。

## IC.10 `bias_resource` 归这里
注册词表/模型句柄（Aliyun `vocabulary_id`、Tencent `HotwordId`…）= 引擎声明 init 字段（账户级资源）；如需 per-request 选择，薄 `provider_params` 旋钮（资源**身份**仍在 init）。

## IC.11 `prepare()` 预热钩子（optional，normative）
**可选**的实例方法，供 `standard-asr prepare` 与生产/CI 预热（download-policy §4）显式触发权重下载/加载，把 IC.9 的 lazy 副作用从首次转写挪到一个无计费、无转写的调用点。契约：
- **签名 MUST 为零参同步方法** `def prepare(self) -> None`：**MUST NOT** 是 coroutine function（`async def`）——否则零参调用只得到未 await 的 coroutine，工具链会误报"预热完成"（静默假成功）。`EngineBase` 提供默认 **no-op** 实现；声明语义不同的同名 `prepare` 即违规。
- **幂等**：重复调用安全（内部应短路到已加载的模型，复用 `_ensure_model_loaded` 之类的守卫）。
- **MUST 自查 `runtime.allow_downloads()`**：禁止下载且权重缺失时 MUST 抛 `DiscoveryError`（与 IC.9 / download-policy §2 同一下载门控义务），**绝不**静默跳过或以真实转写代跑（云引擎会被计费）。
- **无钩子 = reported no-op**：未覆盖 `prepare` 的引擎（落到 `EngineBase` 默认实现）视为"无需预热"，工具链报告 no-op 而非失败。
- 合规：`compliance` 套件检查 `prepare` 存在时**零参且非 coroutine function**；CLI 对 coroutine function 显式报错（不静默）。


---
---

# 流式协议 (Streaming) — NORMATIVE

> **本节定义**：Standard ASR 引擎如何提供实时（流式）转写——应用如何开启流式会话、如何喂入和接收音频与结果、结果事件的格式与修订规则、以及连接中断时如何恢复。
> **另见**：[§能力系统](#能力系统-capabilities--normative)（Capabilities 树结构）、[§结果模型](#结果模型-transcription-result--normative)（Segment/Word 定义）、[§音频输入](#音频输入与采样率-audio-input--sample-rate--normative)（输入类型）。
> **组织**：概述 → 术语 → 接口与能力 → 事件模型 → 段生命周期 → 生命周期与健壮性 → 示例 → 附注与理由 → 能力清单 → v1 ship vs defer。
> **取代**：idea_docs `spec/streaming.md`。

---

## 1. 概述（流式转写是什么、为什么复杂）

"流式"指**应用在说话的同时就能看到转写结果**——不用等整段录完。这在实时字幕、语音助手、电话客服等场景下是刚需。

但不同引擎的流式做法差别极大（调查覆盖了 30+ 引擎），标准要用一套接口同时覆盖它们。主要分歧：

- **输入方式**：有的引擎能在说话的同时**逐块接收音频**（如 ElevenLabs、Qwen3-ASR）；有的要求**先把整段音频上传完**，再流式返回结果（如 OpenAI Audio API `stream=true`）。
- **结果修订**：有的引擎吐出的中间结果**可能被推翻**（Google/AWS/ElevenLabs）；有的**一旦吐出就不改**（Kyutai STT、Voxtral Realtime）。
- **分段边界**：有的引擎按语句端点自动切段；有的（两遍重打分引擎如 WeNet）甚至会**事后合并或拆分段**。

本节把这些统一为一套**事件模型 + 段生命周期 + 能力声明**。

---

## 2. 术语

| 术语 | 含义 |
|---|---|
| `TranscriptionSession` | 一次流式转写会话。通过 `start_transcription(...)` 开启；应用在会话上喂入音频和接收事件。 |
| `segment` / `segment_id` | 一段连贯的转写文本（通常对应一句话或一段发言）。每段由引擎或适配器分配一个**稳定 id**（字符串），用于在事件流中追踪、更新和最终锁定该段。 |
| `partial` 事件 | 引擎对某段的**当前最佳猜测**。partial 的文本可能随着更多音频到来而变化（下一个 partial 会携带该段的完整当前文本，覆盖之前的）。 |
| `final` 事件 | 引擎**不再因新音频改变**该段文本。表示一个语句/段落的转写已确定。 |
| `supersede` 事件 | 引擎用一组新段**替换**一组旧段（用于两遍重打分等场景，详见 §5）。是**核心事件**，每个 compliant 应用都 MUST 处理。 |
| `stable_until` | 一个非负整数，标明 `text` 的前多少个 **codepoint** 已冻结、不会再变（`text[:stable_until]` 即冻结前缀）。适配器 SHOULD 使该值落在字素簇 (grapheme cluster) 边界上。简单应用可忽略它；语音助手用它判断"前缀中哪些字已安全可以行动"。 |
| `audio_processed_until` | 浮点数，表示引擎已处理到的音频时间点（秒），原点 = 本次会话的第一个音频采样。 |

---

## 3. 接口与能力

### 3.1 两个方法（批量 vs 流式，返回类型不同）

标准有两个入口，**返回类型恒定、不会因某个 flag 变形**：

| 方法 | 何时使用 | 返回 |
|---|---|---|
| `transcribe(audio, params)` | 整段音频、等全部转写完 | `TranscriptionResult` |
| `start_transcription(…)` | 任何需要"流式输出"的场景 | `TranscriptionSession` |

`start_transcription` 的签名（修复验证 C-1：增量输入与整段输入共存）：

```python
start_transcription(
    *,                                         # 全部 keyword-only：三个同型可选参数防位置混淆
    audio_format: AudioFormat | None = None,   # 增量喂入时：编码 + 采样率 + 声道，会话锁定
    params: RuntimeParams | None = None,
    audio: AudioInputLike | None = None,       # 整段输入时（如 OpenAI SSE）：直接传完整音频
    deadlines: StreamDeadlines | None = None,  # 应用侧 deadline 覆盖（§6.1），逐字段可选
) -> TranscriptionSession
```

- **keyword-only（`*`）**：`audio_format` / `params` / `audio` 三个可选参数同型易位置混淆，故签名全部 keyword-only——跨实现的可移植代码 MUST 用关键字实参调用。
- **`audio` 接受 coercion**：`audio` 的类型是 `AudioInputLike`，与 `transcribe` 的 `audio` 同规则接受裸值强制转换（裸 `str` = 本地路径、`bytes` = 编码字节、`ndarray` / `(ndarray, sample_rate)` = 波形），由标准层 `coerce_audio_input` 归一为 `AudioInput`。

- **增量喂入**（ElevenLabs realtime、Qwen3 streaming）：传 `audio_format`，之后用 `send_audio(chunk)` 逐块喂裸 PCM 帧。
- **`deadlines`**：应用对会话终止 deadline（`done_timeout` / `max_idle` / `max_session_seconds`，语义见 §6.1）的逐字段覆盖。优先级 MUST 为：应用显式设置 > 适配器构造时选择 > 标准默认；由标准层模板在适配器构造会话**之后**统一施加（不依赖适配器转发，杜绝静默丢失）。未显式设置的字段不受影响。
- **整段输入 + 流式输出**（OpenAI Audio SSE）：传 `audio`（一个完整的 `AudioInput`，如文件路径或编码字节），引擎一次收完后流式返回结果。
- `audio_format` 与 `audio` **互斥**；同时传 MUST 报错。**两者皆缺是合法的**：引擎可在适配器内部自管 wire 格式（如固定协议格式的引擎），此时 `start_transcription()` 不带任何参数即开启增量会话——标准层只在两者**同时出现**时报错。**无参调用的能力门控**：无参开启的是增量（自管 wire 格式）会话，语义上即 §3.2 的 `streaming_input` 能力，故标准层 MUST 对无参调用施加与 `audio_format` 路径**相同**的 `streaming_input` 门控——仅声明 `streaming_output` 的引擎即使实现了流式钩子，无参调用也 MUST fail-closed 抛 `UnsupportedFeatureError`（缺失即不支持，[§能力系统 R1](#能力系统-capabilities--normative)），而非交回一个无法喂入的增量会话（实现：`EngineBase.start_transcription`）。
- **v1 增量 wire 输入仅支持单声道（mono-only）**：`audio_format.channels` MUST = `1`。与批量 `transcribe` 路径不同，标准层**不处理**增量 wire 帧（它们被直接转发给流式引擎），因此**无法**像批量那样对多声道做降混；声明 `channels != 1` 的会话 MUST 在建立时 fail-closed 报错（实现：`EngineBase.ensure_stream_format_supported`）。如需多声道，调用方 MUST 自行在客户端降混到 mono 再喂入。多声道 wire 输入是未来能力（与 §AI R7 的"标准层流式重采样"同属 deferred 路径）。

### 3.2 两个正交能力轴

流式能力由两个**互相独立**的布尔 capability 描述：

| Capability | 含义 | 示例 |
|---|---|---|
| `streaming_input` | 引擎能否在说话的同时**逐块接收**音频、并据此影响转写 | ElevenLabs realtime ✓、Qwen3 vLLM ✓、OpenAI Audio API ✗ |
| `streaming_output` | 引擎能否在**全部输入到达之前**就开始返回结果（partial 或 final） | 上述三个都 ✓ |

**注意一个容易搞混的点**：OpenAI Audio API 的 `stream=true` 需要先上传完整文件（`streaming_input=false`），但它**会**在转写完成前就开始返回 delta 事件（`streaming_output=true`）。

另一个重要的子能力：**`emits_partials`**——引擎是否会发出 `partial` 事件。`streaming_output=true` 且 `emits_partials=false` 的引擎只在每段结束时发一个 `final`，不发中间结果。这覆盖了"流式 VAD 切段 + 离线识别每段"的模式（如 FireRedASR2S + SenseVoice）。

### 3.3 全双工与喂入方式

会话是**全双工**的——喂入音频和接收结果可以**同时进行、互不阻塞**：

```python
async with engine.start_transcription(audio_format=mic_format) as session:
    # 方式 A：托管喂入——session 自己管喂入的生命周期和异常
    session.feed(microphone_source)

    # 然后独立地接收事件
    async for event in session:
        if event.type == "partial":
            show_live(event.segment_id, event.text)
        elif event.type == "final":
            commit_segment(event.segment_id, event.text)
```

也可以手动喂入（`send_audio(chunk)` / `end_audio()`），但**二者不可混用**：

- 使用 `session.feed(source)` 后，任何手动调用 `send_audio` 或 `end_audio` MUST 引发 `InvalidSessionUseError`。（`feed` 在 source 耗尽时自动调 `end_audio`。）
- 使用手动方式后，`feed` 同样 MUST 引发 `InvalidSessionUseError`。
- `feed()` 重复调用 MUST 引发 `InvalidSessionUseError`（一个会话至多拥有一个托管源）。

**异常类型语义**：上述混用 / 重复 feed 都是对**仍然存活**的会话的用法错误，故抛 `InvalidSessionUseError`（`StandardASRError` + `ValueError` 混入，与 `ConfigError` 同型），**而非** `StreamClosedError`——后者专指真正的生命周期关闭（`send_audio` 在 `end_audio()` 之后、或终态事件之后，见 §6.1）。调用者据此可区分「我的代码用错了会话」与「会话已结束」。

**`feed()` 的输入类型**：`feed` 接受**字节块的可迭代对象**（`Iterable[bytes]` 或 `AsyncIterable[bytes]`）或单个 `bytes` / `bytearray` 块。

- **拒绝 `str`**：`str` 满足 `Iterable[str]`，把**文件路径**误传给 `feed` 是高概率笔误；标准 MUST 在调用点抛 `TypeError`（而非逐字符消费或在适配器深处化为费解的 `engine_error`），提示「整段文件走 `start_transcription(audio=...)`（§3.1），增量输入是裸 PCM 字节块」。
- **接受 `AsyncIterable`（非仅 `AsyncIterator`）**：只实现 `__aiter__` 的自定义异步音频源（最 Pythonic 的形态）MUST 被接受并经 `aiter()` 归一为异步迭代器消费，而非落入同步分支以 `input_source_error` 失败。

标准优先使用 **async**；sync 版由标准统一封装（§6.5），ASR 引擎开发者**只需实现 async 版**。

### 3.4 `segment_id` 的生成规则

- 引擎的原始协议如果自带 id（如 AWS `ResultId`、OpenAI diarized 的 `segment_id`），适配器 MUST 使用它。
- 如果引擎不提供 id（如 OpenAI 非 diarized 的 SSE 只有一个连续文本流），适配器 MUST 按确定性规则合成（如 `"seg-0"`、`"seg-1"`…），保证**同一引擎的不同运行、给同样音频，产生相同的 id 序列**。
- **段之间独立**：一个新的 `segment_id` 可以在前一个段的 `final` 或 `closed` **之后**开始发 `partial`——这是云 WebSocket 引擎的标准模式（interim→commit→新段）。

---

## 4. 事件模型

### 4.1 事件类型

每个事件是一个 `TranscriptionEvent`，包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | `"partial" \| "final" \| "supersede" \| "progress" \| "done" \| "error"` | 事件类型 |
| `segment_id` | `str \| None` | 所属段的稳定 id（`done` 和部分 `error`/`progress` 可为 `None`） |
| `text` | `str \| None` | 该段的当前完整文本（`partial`/`final` 必有） |
| `stable_until` | `int \| None` | 已冻结的 codepoint 数量（`text[:stable_until]` = 冻结前缀；见 §4.2） |
| `words` | `list[Word] \| None` | 词级细节（可选，与 [§结果模型](#结果模型-transcription-result--normative) 共享同一 `Word` 定义） |
| `start` / `end` | `float \| None` | 段的起止时间（秒，原点 = 会话第一个音频采样）|
| `audio_processed_until` | `float \| None` | 引擎已处理到的音频时间点（§4.4） |
| `old_ids` / `new_ids` | `list[str]` | 仅 `supersede` 事件使用（§5） |
| `detected_language` | `str \| None` | 引擎检测到的语言（BCP-47，非 `auto`；与结果模型同规则校验）。是 §6.3 重连连续性承诺的载体——重连前后 MUST 保持一致 |
| `code` / `recoverable` / `retriable_after` | | 仅 `error` 事件使用（§7.2） |
| `reconnect` / `gap_start` / `gap_end` | | 仅 `progress` 的重连通知使用（§7.3） |
| `extra` | `dict[str, Any]` | 引擎特定/实验数据槽位（默认 `{}`）。语义与 [结果模型](#结果模型-transcription-result--normative) 的 `extra` 一致：**引擎特定、不可移植**，可移植应用代码 MUST NOT 依赖其中的键。提供给适配器透传原生协议的额外信息（如 token 级置信度、自定义元数据），而无需扩展封闭的标准字段集 |

**`extra` 的 wire 转发/剥离规则**（两层同构，G.5.2）：

- **非 `error` 事件**：`extra` MUST 原样透传上 wire——`extra` 是引擎扩展槽位（与 `Segment`/`Word` 的 `extra` 惯例一致），其他语言的客户端按本表即可解释收到的键。
- **`error` 事件**：server MUST 在出栈前清空 `extra`（置为 `{}`）后再发给客户端。标准层把人类可读的诊断字符串存于 `extra["detail"]`（`engine_error` catch-all 下即 `str(exc)`，可能含文件路径 / 上游 URL / 凭证片段），故不得转发给（未认证的）客户端；安全的结构化字段（`code` / `recoverable` / `retriable_after` / `segment_id` 及 gap/reconnect 字段）保留。被剥离的 detail 由 server 记入操作日志（详见 [server.md §4.2](server.md)）。

每种事件的含义：

- **`partial`**：引擎对该段的**当前最佳猜测**。`text` = 该段的完整当前文本（不是增量/delta——追加式引擎如 OpenAI 由适配器在内部累积 delta，对外总是发完整文本）。下一个 partial 会覆盖上一个。
- **`final`**：该段的转写**不再因新音频而改变**。它可能仍被 `supersede`（两遍重打分场景）或被 `closed` 事件更新（后处理标点/ITN 修正）。
- **`supersede`**：用一组新段替换一组旧段。详见 §5。
- **`progress`**：时间推进 / 心跳 / 重连通知，不改变任何段的文本。
- **`done`**：整个会话结束，不再有任何事件。
- **`error`**：出错。

### 4.2 稳定前缀（`stable_until`）

这是流式模型最核心的概念之一，但**简单应用可以完全忽略它**。

**直觉**：引擎边听边写时，**前面的字越来越确定，最后面几个字还不稳定**。`stable_until` 就是标明"前面多少个字已经冻结、不会再变"的那条线。

**精确定义**：`stable_until` 是一个非负整数，表示 `text` 中前多少个 **codepoint** 已经被冻结。Python 中 `text[:stable_until]` 即冻结前缀——codepoint 是 Python 字符串的原生索引单位，直接可用、零依赖。

**组合字符不变量**：`stable_until` **MUST NOT** 切开 Unicode combining character sequence——即 `text[stable_until]`（如果存在）的 `unicodedata.combining()` MUST 为 0。这保证切点不会落在印度系 matra、阿拉伯语变音符等组合标记的中间，**用 stdlib `unicodedata` 即可验证，零第三方依赖**。标准 SHOULD 提供 `validate_stable_until(text, stable_until) -> bool` helper，合规测试 MUST 校验此不变量。

> 实际上，适配器从引擎拿到的稳定边界通常是 word/token 级的，映射到 `text` 中的位置天然满足此不变量，不需要额外处理。

适配器 SHOULD 进一步使 `stable_until` 落在**字素簇 (grapheme cluster) 边界**上（覆盖 emoji ZWJ 等 combining 以外的多 codepoint 序列），但这是 SHOULD 级建议，不强制。完整 UAX#29 字素簇支持留作 v2 如有需求时的 additive 扩展。

**规则**：
- `stable_until` 在同一段内 **MUST 只增不减**（冻结的前缀永不回退）。
  - **`closed` 终态修正的豁免（见 §5.3 / §5.4）**：此不可变规则约束的是**进行中的识别**——只要识别仍在继续，已冻结前缀 MUST NOT 被后续 `partial`/`final`/`supersede` 改写（实现：标准层 `_LifecycleGuard` 抑制改写并发 `frozen_prefix_rewritten` diagnostic）。**唯一例外**是段进入终态时的 `closed` 事件：它 MAY 对已冻结文本做一次后处理改写（补标点 / ITN / 大小写），这不是"识别回退"而是终态定稿（§5.3、§5.4）。应用收到 `closed` 时 MUST 用新文本**替换**显示，而非追加。**该豁免显式覆盖单调钳制本身**：后处理改写可能**缩短**文本（ITN "twenty twenty" → "2020"），`closed` 事件的 `stable_until` 因此 MAY 小于此前的冻结前沿；标准层 MUST NOT 把它向上钳回（那会产生 `stable_until > len(text)` 的非法线值），只 MAY 修复结构边界（`0 ≤ stable_until ≤ len(text)`、不切组合字符）。合规套件 MUST NOT 因合法的 `closed` 收缩判错。
- 适配器 MUST **保守地**设 `stable_until`——宁可偏小（少冻结几个字），不可偏大（声称冻结了实际可能再变的字）。
- 引擎没有 `right_context`（前瞻窗口）或时间戳信息时（如 Qwen3-ASR streaming），MUST 报 `stable_until=0`——表示没有冻结任何字。相应地，`word_stability` capability 应声明为 `false`。
- **简单应用**：可以无视 `stable_until`，只用 `partial` 显示、`final` 提交。
- **语音助手**：读 `text[:stable_until]` 作为"可安全行动的前缀"。

### 4.3 累积/replace 归一化

所有 `partial` 和 `final` 事件的 `text` 字段 MUST 是**该段的完整当前文本**（累积/replace 语义），而不是 delta/增量。

这是一条**对适配器的硬性要求**。有些引擎（如 OpenAI Audio SSE）原始协议发的是追加式的 text delta——适配器 MUST 在内部累积这些 delta，然后对外发累积后的完整文本。原因：所有引擎都可以从完整文本无损地推出 delta（做差即可），但反过来——从 delta 推出完整文本——在有些引擎（那些会回退/修改旧文本的引擎）上做不到。选择更通用的表示，确保应用代码在所有引擎上都能一致工作。

### 4.4 音频时间游标与心跳

每个事件可以携带 `audio_processed_until`（浮点秒数），表示引擎**已经处理到**的音频时间点。

- 原点 = 本次会话的**第一个音频采样**的时刻（音频时间 t=0），与 [§结果模型](#结果模型-transcription-result--normative) 中 `Segment.start/end` 的原点相同。
- MUST **单调递增**（不回退）；跨重连窗口期保持旧值（见 §7.3）。
- **`progress` 事件**可以只携带 `audio_processed_until` 而不改变任何段的文本——用于心跳、表示"引擎在等更多证据"（DSM 架构的 padding token 场景）、或通知重连。
- **心跳是 MAY，不是义务**。标准层的活性兜底不依赖心跳——`done_timeout` 由「事件**或**音频消费」共同重置（§6.1），纯静音期的会话存活不需要适配器做任何事。业界（Deepgram / AWS / Google 等）一致将流活性锚定在**音频流**而非转写事件流上，没有任何原生协议或官方 SDK 要求接收侧心跳节拍（docs/research/5）。
- 在两种情形下适配器 **SHOULD** 发 `progress`：
  - 原生协议在接收侧本来就有活动信号（Deepgram 的空 Results、Azure 的 NoMatch、Speechmatics 的 ack 等）时，SHOULD 将其镜像为携带真实游标的 `progress`——免费的活性信息不应被适配器吞掉；
  - 适配器处于**长时间无事件的真实工作期**——整段输入会话的长静默计算（无增量消费信号可作锚点，见 §6.1）、feed 暂停期间适配器主动向引擎发原生 keepalive、重连尝试耗时较长（§6.3）——时，SHOULD 周期性发 `progress` 反映该真实活动。
- `progress` 心跳 MUST NOT 携带捏造的 `audio_processed_until`（引擎并未实际处理到的时间点）；没有可靠游标就不携带该字段。标准层 MUST NOT 替引擎合成心跳——事件流上的每个事件都来自适配器对引擎真实行为的翻译，这是协议诚实性的底线。

---

## 5. 段生命周期

### 5.1 状态机

每个 `segment_id` 有一个生命周期，按以下规则转移：

| 当前状态 | 合法事件 | 转移到 |
|---|---|---|
| `open`（刚出现的新段） | `partial`(同 id) | `open`（停留） |
| `open` | `final`(同 id) | `final` |
| `open` | `closed`(同 id，`finality=closed`) | `closed`（无需先经过普通 `final`——一步定稿合法） |
| `open` | `supersede`(该 id 在 `old_ids` 中) | `superseded` |
| `final` | `supersede`(该 id 在 `old_ids` 中) | `superseded` |
| `final` | `closed`(同 id，`finality=closed`) | `closed` |
| `closed` | —（终态） | `closed` |
| `superseded` | —（终态） | `superseded` |

**不合法的转移**（MUST NOT 发生，适配器 MUST 抑制）：
- 同一 id 的 `partial` 在 `final` 之后。
- 同一 id 的 `partial`/`final` 在 `closed` 之后。
- `closed` 段出现在任何 `supersede` 的 `old_ids` 中。

### 5.2 `supersede`（段替换）

两遍重打分引擎（如 WeNet U2++）的第二遍可能改变段的文本甚至段的边界（合并两段、拆分一段）。`supersede` 就是为此设计的：

```
supersede(old_ids=["seg-3","seg-4"], new_ids=["seg-5"])
```

意思是"原来的 seg-3 和 seg-4 作废，用新的 seg-5 替代"。之后 seg-5 会像正常段一样收到 `partial` → `final`。

**核心 reduce（每个 compliant 应用 MUST 实现）**：

```python
if event.type == "partial":
    segments[event.segment_id] = event.text       # 显示
elif event.type == "final":
    segments[event.segment_id] = event.text       # 提交/替换
elif event.type == "supersede":
    for old_id in event.old_ids:
        del segments[old_id]                       # 删除旧段
    # new_ids 的内容会随后通过 partial/final 事件到达
```

**`supersede` 是核心事件（非可选）**——即使 `re_segments` capability 为 `false`（引擎承诺不发 supersede），应用代码也 MUST 包含上面的 reduce 逻辑。这样无论切换到任何引擎都安全。

**规则与不变量**：
- `old_ids` 与 `new_ids` **MUST 无交集**——一个被替换的 id 不会被复用；id 一旦出现在 `old_ids` 中即退休。
- **顺序语义**：`old_ids` 与 `new_ids` 都 MUST 按**阅读（时间）顺序**排列——这是下面"冻结前缀保留"规则做拼接（concatenation）的前提。
- `new_ids` 中的段可能先以 `partial` 到达（不一定立刻是 `final`）——应用的 reduce 应在 `new_ids` 的第一个事件到达时就开始渲染新段文本。
- **排序**：`supersede` 事件 MUST 在其 `new_ids` 的任何 `partial`/`final` 之前投递；`old_ids` 中的 id 必须在之前已被**宣告**过——「宣告」= 收到过至少一个 `partial` / `final`，**或**作为更早一次 `supersede` 的 `new_ids` 被引入（链式 supersede `A→B`、`B→C` 中，`B` 即使从未收到 partial/final 也算已宣告）。
- **冻结前缀保留（拼接覆盖规则）**：`supersede` 操作 MUST 保留已冻结的文本。设 **F_old** = 被替换的旧段（按 `old_ids` 顺序）各自冻结前缀 `text[:stable_until]` 的拼接；**F_new** = 新段（按 `new_ids` 顺序）各自当前冻结前缀的拼接（随新段后续 `partial`/`final` 不断冻结更多文本而增长）。**不变量**：F_old 与 F_new MUST 在其公共前缀上一致——任何一方都 MUST NOT 改写另一方。换言之，**用户已经看到并"确信不变"的文字，在段被替换后仍然不变**。
  - 这条规则**统一覆盖** 1→1、多→1（合并）、1→多（拆分）、多→多 各种基数；1→1 只是 n=m=1 的退化情形，无需特殊处理。（例：旧段冻结前缀是"你好世界"，无论新分段是单个 seg("你好世界", `stable_until`≥4) 还是拆成 seg("你好", su≥2)+seg("世界…", su≥2)，拼接后都必须以"你好世界"开头。）
  - **方向不对称**：**改写/分歧方向** MUST **及早（eagerly）**检查——一旦某个新段冻结了文本，就把当前的 F_new 与 F_old 在公共前缀上比较，分歧即拒绝（这是"用户看到的字被改写"的根本性错误方向）。「拒绝」作用于**整个触发事件**（含其未冻结尾部）而非仅冻结部分——只伤害不合规适配器，且保证被拒事件不会以半改写状态泄出。而"新分段冻结的文本严格少于 F_old"是**保守安全方向**（新分段只是还没把全部文本重新冻结回来），允许暂时留待后续事件补齐，至多记一条软诊断、不强制拒绝。这样实现复杂度有界（无需判定"何时所有重叠新段都已关闭"）。该"至多一条软诊断"在实现中是：会话到达终态（或合规重放结束）时，若某个 supersede 的 F_new 仍严格短于 F_old，标准层发一条 **`info` 级 `supersede_obligation_unfulfilled`** diagnostic（点名受影响的 `new_ids`），表示未重新冻结的尾巴被从 lineage 中丢弃——它**不是 error、不拒绝**任何事件，supersede 依旧成立（实现：`TranscriptionSession.finalize`）。
- `re_segments` capability：`false` 表示引擎承诺不发 `supersede`（finals 只增不改）；`true` 表示可能发。
- **lineage 是 set-to-set（v1 已知限制）**：`old_ids`/`new_ids` 表达 re-segmentation 的**基数**（哪些退休、哪些出现），但**不**承载 per-old→per-new 的逐对映射——merge+split（多→多）时无法判定某个新段具体源自哪个旧段。规范不要求逐对映射；冻结前缀保留（上）按**拼接**的 F_old/F_new 校验而非逐对。逐对 edit-ops/diff 是 §10 deferred 方向（additive-later）。

### 5.3 两级终态

- **`final`**：该段的文本**不再因新音频而改变**——但仍可能被 `supersede`（两遍重打分）或被 `closed` 事件原位修正（后处理标点/ITN/大小写修正）。
- **`closed`**：该段**彻底不可变**——连后处理修正都不会再有。`closed` 段 MUST NOT 出现在任何后续 `supersede` 的 `old_ids` 中。

引擎通过 `finality_level` capability 声明它能保证到哪一级。**默认保守**：若引擎无法确认 `final` 后是否还会有后处理修正（如 ElevenLabs 的 committed 段是否会被重格式化目前未明确），MUST 声明 `final`、MUST NOT 声明 `closed`。

`closed` 事件的格式：对同一 `segment_id` 再发一次 `final`，携带 `finality="closed"` 标记（文本可能因后处理而有变化——例如补了标点）。此后该 id 进入终态。

### 5.4 `closed` 与 `re_segments=false` 的交互

当引擎声明 `re_segments=false`（不发 `supersede`，finals 只增不改）时，`closed` 事件仍然可能**原位修改文本**（例如加标点），这不算"重分段"——段的 id 和边界不变，只是内容被后处理修正。应用在收到 `closed` 事件时应当用新文本**替换**（非追加）已显示的内容。

---

## 6. 生命周期与健壮性

### 6.1 方法调用规则

| 调用 | 条件 | 行为 |
|---|---|---|
| `send_audio(chunk)` 在 `end_audio()` 之后 | — | MUST raise `StreamClosedError`（生命周期关闭） |
| `send_audio(chunk)` 在终态事件之后 | — | MUST raise `StreamClosedError`（生命周期关闭） |
| `send_audio` / `end_audio` 在 `feed()` 之后 | feed 模式已声明 | MUST raise `InvalidSessionUseError`（混用） |
| `feed()` 在手动输入之后 | 手动模式已声明 | MUST raise `InvalidSessionUseError`（混用） |
| `feed()` 重复调用 | — | MUST raise `InvalidSessionUseError`（至多一个托管源） |
| `feed(str)` | — | MUST raise `TypeError`（见 §3.3：文件路径请用 `start_transcription(audio=...)`） |
| `end_audio()` 重复调用 | 手动模式 | 幂等（不报错，不重发） |
| `end_audio()` 重复调用 | `feed()` 模式 | MUST raise `InvalidSessionUseError`（`feed` 自己管 `end_audio`，手动调是混用） |
| 第二次迭代会话（再次 `__aiter__` / 并发 `async for`） | — | MUST raise `InvalidSessionUseError`（单消费者契约，见下） |
| `done` 不到达 | 管线不活动超过 `done_timeout` | MUST 发 `error(code="done_timeout")`（活性兜底，定义见下） |

**单消费者契约（single-consumer）**。一个会话有**唯一的事件流、唯一的消费者**：事件存活于一个共享的有界缓冲区，第二个并发迭代器会与第一个**竞争**取事件，把事件流静默地拆分到两者之间（各自只见到任意子集，谁的 `result` 都不再等于完整事件流，破坏 stream == result 不变量），并重置按迭代计时的 deadline 锚点。这两者都是对仍然存活会话的用法错误，故第二次 `__aiter__` MUST **大声报错**（`InvalidSessionUseError`，与混用 / 重复 feed 同类，而非 `StreamClosedError`）而非交回一个竞争迭代器。sync 桥（`SyncSession`）内部只取一次迭代器，天然单消费者，不受影响。

**活性与终止保证（liveness）**。会话有三道相互独立、各管一事的 deadline。设计原则：**默认配置 MUST NOT 误杀合法的静音直播会话**（用户停顿 30 秒、会议冷场几分钟期间，VAD 型引擎在接收侧通常什么都不发——这是合法行为，不是故障），同时**挂死的管线 MUST 在有限时间内合成终态**（迭代器不会在应用无能为力的情况下永挂）。

- **`done_timeout`（默认 300s）——管线不活动兜底。** 计时锚点是**最近一次管线活动** = max(最近收到的任意事件, 适配器最近一次经 `audio_chunks` 消费喂入的音频块)。超过 `done_timeout` 无任何活动 → 标准层合成 `error(code="done_timeout")` 终态。三个推论：
  - **静音存活**：直播会话中应用持续喂入（静音也是音频；业界引擎本就要求持续送音频否则服务端先断——Deepgram 10s / AWS 15s，docs/research/5），适配器持续消费即持续重置锚点——引擎接收侧静默任意久都不会被误杀，且无需适配器做任何事。
  - **`done` 必达**：`end_audio()` 之后无音频可消费，锚点冻结，本 deadline 退化为对引擎「冲刷并送达 `done`」窗口的硬界限——这是该 deadline 名字的由来。
  - **整段输入会话**（`audio=...`）没有增量消费信号，锚点只有事件；长静默计算的适配器因此 SHOULD 周期性发 `progress`（§4.4）。
  - 应用 MAY 经 `deadlines` 显式设为 `None` 关闭（显式 opt-out）；默认 MUST 为有限值。本 deadline 是**防挂死兜底，不是引擎活性检测**——传输层失败的及时上报是适配器义务（见下），不靠这里。
- **`max_idle`（默认禁用）——内容停滞检测，opt-in。** 超过该时长没有**内容事件**（`partial` / `final` / `supersede`；心跳与音频消费**不**重置）→ 合成 `error(code="stream_stalled")`。静音是直播会话的正常状态，故 MUST NOT 默认启用；适用于「持续有语音」可被预期的场景，或检测「持续消费音频却永不产出内容」的病态引擎。
- **`max_session_seconds`（默认禁用）——绝对墙钟上限，opt-in。** 到达即合成 `error(code="session_timeout")`。计时原点 MUST 是**会话建立时刻**（`async with` 进入 / `__aenter__`），而非首次迭代（首个 `__anext__`）——会话建立到开始迭代之间的时间**计入**上限；中断迭代后重新迭代 MUST NOT 重置该原点（否则上限可被无意地无限续期，成本/资源控制承诺落空）。**这是三道 deadline 中唯一的会话级（绝对）上限**：`done_timeout` / `max_idle` 是活动相对的，按迭代重新计时是正确语义；`max_session_seconds` 则锚定会话本身（实现：`TranscriptionSession.__aenter__` 记录原点，`_iterate` 共享）。

**适配器活性义务**：传输层 / 引擎 SDK 报告的连接失败（断连、握手失败、原生超时、服务端关闭码）MUST 及时翻译为 `error` 事件投递进会话——可恢复的标 `recoverable=true`（配合 §6.3 重连），不可恢复的即终态；MUST NOT 静默吞掉。引擎活性检测的正确位置是传输层（WS ping/pong 等）+ 适配器的诚实上报，业界全部官方 SDK 均如此分层，无一在接收侧跑「无结果即判死」的定时器（docs/research/5）。原生协议有「无音频即断连」策略的引擎（Deepgram 10s、AWS 15s、ElevenLabs 等），适配器 SHOULD 负责原生 keepalive（或按该引擎惯例持续转发静音帧），使应用合法的喂入节奏不被引擎侧闲置策略误杀。

### 6.2 错误事件

`error` 事件通过事件流投递（不是从 `send_audio` 抛出的异常）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | `str` | 错误码（标准码集 + 引擎可扩展） |
| `recoverable` | `bool` | `true` = 引擎可能恢复、会话可继续；`false` = 终态 |
| `retriable_after` | `float \| None` | 若可重试，建议等待多少秒 |

`progress` 事件在任何非终态下合法——包括 `recoverable=true` 的错误之后（用于通知恢复进度）。

**标准错误码集（标准层自身会合成的码，应用 SHOULD 识别）**：

| code | 含义 | recoverable |
|---|---|---|
| `done_timeout` | 管线不活动（无事件且无音频消费）超过 `done_timeout`（默认 300s），标准层合成终态（§6.1 活性兜底） | `false` |
| `stream_stalled` | 超过 `max_idle` 未收到内容事件（心跳与音频消费不算；默认禁用，opt-in），标准层合成终态 | `false` |
| `session_timeout` | 超过 `max_session_seconds` 绝对墙钟上限（默认禁用，opt-in），标准层合成终态 | `false` |
| `backpressure` | 发送侧有界缓冲区溢出（消费太慢），标准层合成终态（§6.4） | `false` |
| `input_source_error` | `feed()` 的音频源自身抛错 | `false` |
| `engine_error` | 适配器 `_produce` 逃逸的未分类异常 | `false` |
| `content_lost` | 重连缝隙真实丢失音频的保真度警告（§6.3）；**非终态** | `true` |

引擎可扩展自有错误码；与标准码冲突的语义 MUST NOT 复用上表名字。

### 6.3 重连（透明、但诚实）

许多引擎有会话时间上限（Google STT ≈ 5 分钟、ElevenLabs 有 `session_time_limit_exceeded`）。标准层应**在底层自动重连**，让应用无需关心这些限制。但重连**不保证无损**——标准的承诺是**「会话存活」而非「应用完全感知不到」**。

具体规则：

- **`segment_id`、时间戳、检测到的语言** 在重连前后 MUST 保持连续。
- 跨 lossy 缝隙 MUST 发一条 `progress` 事件，携带 `reconnect=true`、`gap_start`、`gap_end`。
- session 拥有一个**有界的滚动音频缓冲区**（用于重连后向引擎重喂最近音频）。

**按音频源分类**（live mic 不可回放）：

| 音频源类型 | 行为 |
|---|---|
| **可回放**（文件/数组/服务端有缓冲的引擎） | 重连后从缓冲区重喂，缝隙在内部弥合；对应用几乎透明。 |
| **不可回放**（live mic 等实时源） | 重连期间到达的音频如果超出了缓冲区容量，**会真丢**。**丢失由适配器判定**——与 `segment_id`/时间戳/语言连续性同理（见上，皆为适配器责任），由适配器据其 gap 窗口与 replay 覆盖情况决定本次重连是否真丢（实现：`note_reconnect(..., content_lost=True)`）。**结构化事件由标准发出**：当适配器判定真丢时，标准 MUST 在 `progress` 事件后**紧跟一条** `error(code="content_lost", recoverable=true)` 事件；这是非终态 fidelity warning：会话继续存活，gap 被诚实报告，后续事件继续流动直到正常 `done` 或真正的终态 `error`。不可回放源的缓冲区只覆盖"已捕获但尚未确认处理"的窗口。 |

`reconnect` capability：引擎声明 `seamless`（无损，仅 stateless / 服务端有状态引擎可声明）或 `lossy`（可能有缝隙，DSM 等有状态本地模型 MUST 声明 `lossy`）或 `unsupported`。

### 6.4 背压

当事件消费方处理速度慢于产生速度时：

- **`partial` 事件**：按 `segment_id` 合并——只保留该段最新的 partial（保留最大 `audio_processed_until`）。**合并 MUST 被同 segment 的 `final`/`closed`/`supersede` 作废**——如果 partial 尚未投递但该段已进入终态或被替换，该 partial MUST 丢弃（避免复活已替换的段）。
- **`final`、`supersede`、`done`、`error` 事件**：**永不丢弃、永不重排**。
- 发送侧有界缓冲区，溢出发 `error`。
- sync 桥的事件队列（§6.5）也遵守同样的背压规则。
- **诊断通道同样有界。** 标准层 `_LifecycleGuard` 的 lifecycle-suppression 诊断（抑制的非法转移、钳制的 `stable_until` / 音频游标）经 `session.diagnostics()` 暴露——一个每事件都触发钳制的轻微越界引擎（如游标持续抖动）在数小时直播会话中会无界累积诊断，这与会话其余资源（事件缓冲、音频队列、滚动音频缓冲区均有界）的哲学相悖。故诊断列表 MUST 有上限：达到上限后，标准层 MUST NOT 再保留逐条诊断，而是聚合为单条尾部 `diagnostics_truncated` 汇总（按 code 计数）——上限被命中这一事实 MUST 被诚实上报，绝不静默丢弃（实现：`_LifecycleGuard`，默认上限 `DEFAULT_MAX_GUARD_DIAGNOSTICS`）。

### 6.5 sync 桥

标准统一提供 sync→async 桥（一个后台线程跑事件循环，session 拥有、在 `__exit__`/`close()` 拆除），ASR 引擎开发者**只需实现 async 版**。

**适配器契约**（MUST 遵守，否则 sync 桥会死锁或泄漏）：
- 所有绑定到事件循环的资源（WebSocket 连接等）MUST 在 `__aenter__` 中创建。
- MUST NOT 触碰当前线程的 ambient event loop（不调 `asyncio.get_event_loop()`）。
- 生产/消费使用标准提供的线程安全原语。

合规测试包含：从外部线程驱动 async adapter，验证不死锁、不泄漏。

---

## 7. 示例

### 7.1 最简场景：显示实时字幕

```python
async with engine.start_transcription(audio_format=mic_format) as session:
    session.feed(microphone)

    async for event in session:
        if event.type == "partial":
            update_caption(event.segment_id, event.text)
        elif event.type == "final":
            finalize_caption(event.segment_id, event.text)
        elif event.type == "supersede":
            for old_id in event.old_ids:
                remove_caption(old_id)
```

### 7.2 语音助手：利用稳定前缀

```python
async for event in session:
    if event.type == "partial" and event.stable_until and event.stable_until > 0:
        frozen_prefix = grapheme_clusters(event.text)[:event.stable_until]
        maybe_start_responding_to(frozen_prefix)
```

### 7.3 OpenAI Audio SSE（整段输入 + 流式输出）

```python
async with engine.start_transcription(audio=AudioPath("meeting.mp3")) as session:
    async for event in session:       # partial 事件在转写完成前就会到达
        if event.type == "partial":
            show_progress(event.text)
        elif event.type == "final":
            save_transcript(event.text)
```

---

## 8. 附注与理由

- **为什么 `supersede` 是核心事件而非可选？** 如果 `supersede` 只是可选的高级功能、简单应用可以忽略它，那么当简单应用切换到一个会发 `supersede` 的引擎时，它的 reduce 就是错的（旧段没被删、新段凭空出现 → 文本重复/矛盾）。把 `supersede` 设为核心意味着每个应用的 reduce 都天然能处理段替换，无论引擎是否实际使用。代价 = 应用代码多 3 行（`del segments[old_id]`）；收益 = 切引擎永远安全。
- **为什么要用累积/replace 而不是 delta？** delta 更小，但只有在引擎永不修改已发文本时才有效。很多引擎（Google/AWS/ElevenLabs/Qwen3/WeNet）会修改已发文本。选累积 = 适用于所有引擎；delta 的应用可以用两次累积文本做差得到。
- **为什么 `stable_until` 用 codepoint 而非字素簇？** codepoint 是 Python 字符串的原生索引单位——`text[:stable_until]` 直接可用、零依赖。实际上适配器从引擎拿到的稳定边界是 word/token 级的,映射到 `text` 中的位置天然就是字素簇边界,不存在"切到组合字符中间"的现实场景。标准用 SHOULD 建议适配器落在字素簇边界,而非用 MUST 强制标准层维护 UAX#29 状态机——这与标准不做音频解码(交给 `[audio]`)、不做 URL fetch(交给引擎)是同一哲学:标准定义语义,不承担不必要的实现。
- **整段输入为何需要 `start_transcription` 的 `audio` 参数（验证 C-1）？** OpenAI Audio SSE 需要一次上传完整文件然后流式收结果。如果只有 `audio_format` + `send_audio(chunk)`，应用就得把 mp3 文件假装成 PCM 帧来喂——但 mp3 不是 PCM（§AI.1 明确禁止混淆编码容器与裸 PCM 帧）。增加 `audio` 参数让整段输入走正确的 `AudioInput` 路径。

---

## 9. 能力清单（v1 流式 Capabilities）

以下 capability 节点住在 `capabilities.streaming.*`（参见 [§能力系统](#能力系统-capabilities--normative)）：

| Capability 路径 | 节点类型 | 含义 |
|---|---|---|
| `streaming_input` | flag `{supported}` | 引擎全局；能否增量喂入 |
| `streaming_output` | flag `{supported}` | 引擎全局；能否增量返回结果 |
| `streaming.emits_partials` | flag `{supported}` | 是否发 partial 事件（false = 只发段末 final） |
| `streaming.re_segments` | flag `{supported}` | 是否可能发 supersede |
| `streaming.word_stability` | flag `{supported}` | 是否提供有意义的 `stable_until` |
| `streaming.reconnect` | enum `{mode: seamless\|lossy\|unsupported}` | 重连能力 |
| `streaming.finality_level` | enum `{mode: final\|closed}` | 能保证到哪级终态 |
| `streaming.timestamps` | enum `{mode: native_frame_aligned\|post_align\|none}` | 流式时间戳来源 |
| `streaming.guidance.*` | 同 §R.4 | 流式引导（可与 batch 限额不同） |
| `streaming.language.*` | 同 §LANG.3 | 流式语言能力（可与 batch 不同） |

---

## 10. v1 ship vs defer

**v1 包含**：`partial`/`final`/`supersede`/`progress`/`done`/`error` + 稳定 `segment_id` + 保守 `stable_until`(codepoints, SHOULD 字素簇边界) + `end_audio` + 两级终态标志 + 正交 input/output 能力 + `reconnect(lossy,gap)` + session 拥有 pump + 标准 sync 桥 + 音频时间游标。验证可驱动三个基准（OpenAI SSE / ElevenLabs realtime / Qwen3 vLLM）。

**defer（additive-later）**：运行时 `target_latency` 调整（v1 仅构造期固定）；`update_guidance()` 中途改引导（v1 保留 `mutable_mid_stream` 能力标志但不承诺方法）；revision 的 edit-ops/diff；无缝 DSM 重连（v1 声明 lossy）；多通道流式展开。


