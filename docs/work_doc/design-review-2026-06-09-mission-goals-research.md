# 设计审查：当前设计 vs Mission / Goals / 调查研究（含产品形态评估）

审查日期：2026-06-09
审查范围：`docs/spec/`（specification.md、cli.md、server.md、download-policy.md）、`src/standard_asr/` 实现、`docs/goals.md`、`docs/mission.md`、`docs/research/1 大規模 ASR 調查 2026-05-09.md`、`docs/research/3 streaming 补充调查 2026-06-05.md`、README.md。

---

## 0. 总评

**设计与调查研究的对齐度非常高，是本项目目前最强的资产。** 调查 #1 列出的 8 条「不可调和分歧」每一条都有明确的设计回应；调查 #3 的 6 个最高优先行动项有 5 个落地、1 个显式 defer 并记录在规范 §10。规范与实现一致性好（独立代码审查确认），且实现比规范更严格之处均有文字记录。

**主要结论**：
1. 当前产品形态（Python 进程内协议 + entry-point 插件生态 + 工具链）对 v1 是**正确的选择**，理由见 §3。
2. 但 mission 声称服务非 Python 开发者，而规范的 normative 核心是 Python 形状的——**wire protocol（HTTP/WS 契约）应升格为一等、独立版本化的跨语言规范**，这是通往「真正的标准」最重要的一步。
3. 最大的功能覆盖缺口是 **job（提交/查询/回调）模式**：调查 #1 中 AWS / Azure batch / 阿里 / 腾讯 / 火山全部是 job 型，v1 完全无法诚实表达（详见 §1.3-G1）。
4. 调查 #1 明确建议「把 OpenAI-compatible 作为一等映射层」，目前两个方向（server 提供 OpenAI 兼容端点；通用 OpenAI-compatible client 插件）都未做——这是**最便宜的采用杠杆**。
5. `docs/goals.md` 已落后于规范演进（G.1.2 的"输入为 numpy 16kHz"旧契约已被音频协商机制取代；G.4.2 对依赖冲突的解决程度过度声称），需要回写更新。

---

## 1. 设计 vs 调查研究

### 1.1 调查 #1「八大不可调和分歧」逐条对照

| # | 分歧（调查 #1 §4） | 设计回应 | 状态 |
|---|---|---|---|
| 1 | 输入形态不可统一（multipart / 云存储 URI / numpy / URL） | `AudioInput` 判别联合（6 变体）+ `InputKind` 声明 + 确定性协商矩阵 + `AudioStorageUri` 独立安全模型 | ✅ 完整覆盖 |
| 2 | 流式语义不同（SSE delta / gRPC 双向 / 自定义 WS / 库内增量） | 统一事件模型（6 类事件）+ `streaming_input` ⟂ `streaming_output` 正交轴 + 适配器累积/replace 归一化（ST §4.3） | ✅ 完整覆盖 |
| 3 | partial revision 语义不同 | `stable_until` 冻结前缀 + `final`/`closed` 两级终态 + `supersede` | ✅ 完整覆盖 |
| 4 | 语言码不可直接共用 | BCP-47 统一 + `selectable`/`detectable` 分离 + `auto` 保留字 + 适配器映射；provider 模型码（腾讯 `16k_zh`）经 entrypoint preset | ✅ 完整覆盖 |
| 5 | 时间单位不统一 | MUST float 秒、原点 = 音频 t=0，适配器换算 | ✅ 完整覆盖 |
| 6 | 返回 shape 随 flag 改变 | TR.1 恒定 schema；SRT/VTT 是渲染函数（`to_srt`/`to_vtt`）非返回类型；多通道不换顶层形状（TR.4） | ✅ 完整覆盖，直接反制 NeMo `return_hypotheses` / OpenAI `response_format` 反模式 |
| 7 | 热词/引导不是简单 boolean | `guidance` 家族（`prompt` + `phrase_hints` 拆分）+ `bias_resource` 归 Init Config（IC.10）+ provider_params 兜底 | ✅ 完整覆盖 |
| 8 | job 生命周期差异大（submit/query/webhook） | **`job` mode 域仅保留名字，无任何设计** | ❌ 最大缺口，见 §1.3-G1 |

### 1.2 调查 #3「六个最高优先行动项」落实状况

| # | 行动项（调查 #3 §4） | 状态 |
|---|---|---|
| 1 | re-segmentation：`supersede/merge/split` 段生命周期 | ✅ `supersede(old_ids→new_ids)` + 冻结前缀拼接保留不变量（ST §5.2，含方向不对称的及早检查）——设计比调查要求的更精细 |
| 2 | `stable_until` 左侧绝对不可变 + adapter 保守设定写成硬约束 | ✅ ST §4.2 + `_LifecycleGuard` 基座层防御 + `validate_stable_until()` |
| 3 | final 分级为 `final` / `closed`，provider 声明保证级别 | ✅ `finality_level` capability + `closed` 事件语义（ST §5.3/5.4） |
| 4 | 独立于 `stable_until` 的音频时间游标 + padding/心跳 | ✅ `audio_processed_until` + `progress` 事件（DSM 场景显式覆盖） |
| 5 | revision 事件可选携带 edit-ops/diff | ⏸ 显式 defer（ST §10 + §5.2 lineage set-to-set 已知限制）——可接受，已诚实记录 |
| 6a | `no-interim streaming` 档 | ✅ `emits_partials=false` |
| 6b | `reconnect: seamless/lossy/unsupported` | ✅ `ReconnectCap` + lossy gap 报告 + `content_lost` 事件（比调查要求更完整） |
| 6c | `target_latency` / `right_context` 会话级可配 | ❌ **完全缺席**——连构造期的标准参数都没有，见 §1.3-G5 |

其余调查 #3 细化点：DSM immutable 前沿（`stable_until` 理想退化情形）✅；Moonshine provisional/finalized ✅（同模型）；Canary 逐 chunk 重预测 ✅（保守 `stable_until` 硬约束）；流式时间戳来源三态（`native_frame_aligned/post_align/none`）✅ 照单全收。

### 1.3 缺口清单（按战略影响排序）

**G1 — job / 异步批量模式缺位（高）。**
调查 #1 中 AWS Transcribe、Azure batch、阿里录音文件、腾讯 `CreateRecTask`、火山 submit/query 全是 job 型——这是商业云 ASR 的**主流形态**，且与 `storage_uri` 输入（已支持）天然成对。v1 的 `transcribe()` 是同步阻塞调用；适配器理论上可在内部轮询，但 5–12 小时音频、72 小时结果保存期的场景在阻塞调用里无法诚实表达（进程必须活着等结果）。
另外，规范说「`job` 保留待 **major** 版本扩展」——这个前提值得质疑：capabilities R1（缺失即不支持）+ R2（容忍未知键）的 fail-closed 设计，恰恰使**新增一个 mode 域是 additive 安全的**（旧应用看到未知 `job` 域会安全地视为不支持）。job 完全可以作为 minor additive 落地。
**建议**：pre-1.0 至少做出设计决定（接口形状：`submit() -> JobHandle` / `poll()` / `result()`，或 `transcribe_job()`），哪怕实现 defer；并把「major 才能加 mode 域」的说法修正为「mode 域可 additive 新增」。

**G2 — OpenAI-compatible 一等映射层未做（高，采用杠杆）。**
调查 #1 OpenAI 章节的标准化启示原话：「标准需把 "OpenAI-compatible" 作为一等映射层」。两个方向都缺：
- **入站**（server 侧）：`POST /v1/audio/transcriptions` 兼容端点 → 现存海量 OpenAI-audio 客户端**零改动**即可用任何 Standard ASR 引擎。这是最便宜的应用侧采用通道。
- **出站**（插件侧）：一个通用 `std-openai-compatible` client 插件（base_url 可配）→ 一次性覆盖 OpenAI / Groq / GLM / Fish 等兼容或半兼容服务。注意调查的警告：兼容通常只是子集（Groq 无 SRT/streaming），capability 声明要按最小公分母或按 preset 区分。
**建议**：入站兼容端点作为 server 的 v1.x 功能排期；出站通用插件作为官方 cookbook 插件。

**G3 — 三态识别行为旗标全部推迟（中）。**
调查 #1 收敛设计第 8 条：标点 / ITN / 脏词过滤 / disfluency 是「常见后处理开关，默认值差异大，需要三态 unset/on/off」。规范 R6 只占位，v1 全走 `provider_params`——意味着这些**高频常用旋钮在 v1 不可移植**（换引擎必须改代码）。这与「possibility of silent behavior difference across engines」直接相关：同一应用在 A 引擎默认带标点、在 B 引擎默认不带，应用无从声明意图。
**建议**：把 `punctuation` / `itn` 两个最高频旗标提前到 v1.x minor（三态 + capability 门控），其余维持占位。

**G4 — diarization 不对称（中）。**
能力树有 `diarization` 节点（含 `max_speakers`），结果模型有 `Segment.speaker`/`Word.speaker`，但 `RuntimeParams` **没有可移植的请求旋钮**——应用无法可移植地"打开" diarization。capability 可声明、结果可表达、请求不可达，三角不闭合。
**建议**：v1.x 增加 `diarization: bool | None`（或 `num_speakers` hint）运行时参数；在此之前在规范里写明「v1 经 provider_params 请求」。

**G5 — 会话级延迟/前瞻参数缺席（中低）。**
调查 #3 §3.7：Nemotron 4 档延迟、Voxtral 80ms–2.4s 连续可配、Canary waitk/alignatt——「目标延迟/右上下文应是会话级可配参数」。当前连构造期的标准参数都没有（只能走引擎 init config / provider_params）。ST §10 把「运行时中途调整」defer 是合理的，但**构造期标准参数**（`target_latency_ms`，provider may ignore）是 additive 低成本项。
**建议**：作为 `start_transcription` 的可选标准参数加入 v1.x 计划。

**G6 — 翻译任务未置语（低）。**
调查覆盖了 OpenAI `/translations`、Whisper translate task、Canary AST，规范对 `task=translate` 完全沉默。沉默会被解读为 undefined 而非 out-of-scope。
**建议**：在规范加一行 explicit non-goal（或保留 `task` 命名空间），一句话即可。

**G7 — 实体检测/脱敏、情绪等扩展字段（低）。**
ElevenLabs entities/redaction、腾讯 EmotionType 等。TR.7 已声明走扩展命名空间不进顶层——方向正确，无需行动，仅记录。

---

## 2. 设计 vs goals.md / mission.md

### 2.1 目标达成矩阵

| 目标 | 状态 | 说明 |
|---|---|---|
| G.1.1 标准化核心接口 | ✅ | `StandardASR` Protocol + `EngineBase` 模板方法 + `transcribe/transcribe_async/start_transcription` |
| G.1.2 定义数据格式 | ✅ 但**目标文本已过时** | 目标写「输入为 numpy.array、16kHz」——规范已明确「取代所有引擎只吃 np.float32 的旧契约」，演进为 AudioInput 协商。设计是对的，**goals.md 需要回写** |
| G.1.3 Properties 声明 | ✅（一处出入） | 语言 BCP-47 ✅；但目标提到的「硬件支持（CPU/GPU）」元数据在 `BaseProperties` 中不存在（只有 config 侧 `DeviceConfigMixin`）。要么补 Properties 字段，要么改 goals 文字 |
| G.1.4 可选功能标准化 | ✅ | 能力树 + 流式协议 + word_timestamps 枚举 + diarization shape 预留（请求旋钮缺口见 G4） |
| G.2.1 自动化测试套件 | ⚠️ 部分 | `compliance.py` 覆盖 entrypoints / streaming param gating / sync bridge；但规范多处承诺的合规校验（事件序列重放、冻结前缀保留、`effective ⊆ declared`）散落在运行时防御层，**面向引擎作者的一键合规跑测（pytest 插件或 `compliance run <engine>`）尚未成形** |
| G.2.2 CLI / Web API / 模型管理 | ⚠️ 部分 | CLI ✅、server ✅、models cache/prepare ✅；**目标明确承诺的「启动麦克风进行识别」CLI 不存在**（CLI 完全没有流式入口） |
| G.2.3 Boilerplate 模板 | ⚠️ 部分 | cookbook 有 `std_dummy_asr`/`std_faster_whisper` 可抄，但无正式脚手架（copier/cookiecutter 模板） |
| G.3.1 动态配置生成 | ✅ | Pydantic 模型 + JSON Schema 暴露 + SecretStr/secret 标记 + applicability mixin（IC.5） |
| G.3.2 插件自动发现 | ✅ | entry-point group `standard_asr.models` + PEP-503 归一 + 冲突检测 |
| G.4.1 核心与实现分离 | ✅ | 核心仅 numpy+pydantic；extras 隔离 |
| G.4.2 解决依赖冲突 | ⚠️ **过度声称** | 插件化解决的是**打包与许可证**隔离；同一 venv 内 numpy1-vs-2 插件依然冲突（DEP.4 自己承认是「Python 事实」）。真正解法（subprocess + UDS + shared_memory 隔离 shim）是 fast-follow 未实现。doctor 只诊断不解决。**goals 文字应改为「隔离 + 诊断 + 进程级逃生舱」** |

### 2.2 Mission 一致性

mission（成为 ASR 推理标准、应用与模型完全解耦、双侧友好）与设计方向一致，且被调查**实证强化**了（碎片化问题真实存在，30+ 引擎、八大分歧）。两点张力：

1. **S.1 §2 明确把非 Python 开发者列为核心用户**（「我们工具链自带 API 端点和 SDK」），但 normative 规范是 Python 形状的；server.md 虽然写得像合约，但定位是「Python 库的附属工具」。SDK 一个都没有（可接受 pre-release，但 mission 承诺了）。→ 见 §3 建议。
2. **mission/goals 写于 streaming 设计之前**——如今流式协议（stable_until / supersede / 两级终态）是整个标准技术含量最高、差异化最强的部分，mission/goals 对它只有一句话（G.1.4）。文档应反映「流式语义统一」是核心价值主张之一。

---

## 3. 产品形态评估：是不是最佳形态？

### 3.1 候选形态对比

| 形态 | 代表 | 对本项目的适配度 |
|---|---|---|
| A. 纯纸面规范 + 一致性测试 | W3C 式 | ❌ 违背「DX above all」；没有杀手级参考实现的纸面标准无人采用；丢掉音频协商/工具链这些 battery-included 价值 |
| B. 单体聚合包 | 旧式 xxx-all 包 | ❌ 已被 README/mission 正确否决（维护瓶颈、依赖地狱、许可证）；调查中 FunASR 的「模型拼装器」形态也印证单体不可扩展 |
| C. 网关/代理优先（wire-first） | LiteLLM proxy | ⚠️ 语言无关是优点，但**丢掉进程内 numpy 零拷贝路径**——本地模型（faster-whisper/sherpa/NeMo）是 ASR 生态的半壁江山，也是相对云聚合器的核心差异化；且对最简单场景（Python 应用 + 本地模型）强加一层服务运维 |
| D. **当前形态**：Python 进程内协议 + entry-point 插件 + 工具链（CLI/server/合规） | 现状 | ✅ ASR 引擎事实上活在 Python；插件 entry-point 给了 wire 标准给不了的零配置发现；本地+云一套接口是独有卖点 |

**结论：D 是 v1 的正确形态。** 关键论据：(a) 调查里 17/30+ 引擎是 Python 本地库，wire-first 无法服务它们的零拷贝路径；(b) 「pip install 插件 → 自动发现」的 DX 是采用飞轮的核心，依赖 Python 包生态；(c) server 已经给了非 Python 应用一条路。

### 3.2 但形态需要两个演进（不换形态，补短板）

**E1 — wire protocol 升格为一等规范。**
现状：server.md 是「实现的合约」。目标：把 HTTP/WS 契约抽出为**独立版本化的跨语言规范**（自己的 schema、自己的兼容性规则、自己的（语言无关）一致性测试），Python server 降格为参考实现。这是 mission「成为标准」从「Python 生态标准」走向「ASR 推理标准」的必经之路——OpenAI Chat Completions 之所以成为标准，正因为它是 HTTP 形状而非某语言 SDK 形状。顺序建议：v1 先冻结 Python 协议 → v1.x 抽出 wire spec → 之后才谈多语言 SDK。

**E2 — OpenAI-compat 双向 shim（见 §1.3-G2）。** 入站兼容端点 + 出站通用插件，是采用成本最低、杠杆最大的两件事。

### 3.3 Mission/Goals 本身是否合适

- **Mission 合适，保留。**「成为 ASR 推理领域的标准」+ 双侧友好哲学 + 三层利益相关者分析都经受住了调查检验。建议仅做两处增强：① 明确「标准」分两层（进程内 Python 协议 = 本地引擎的零拷贝层；wire 协议 = 跨语言层），两层保持同构；② 把「流式语义统一」写进核心价值主张。
- **Goals 需要一轮回写**（goals.md 已是规范的下游而非上游）：
  - G.1.2 改为引用 AudioInput 协商（删除 numpy/16k 旧契约表述）；
  - G.1.3 的硬件元数据要么落 Properties 要么删句；
  - G.4.2 改为「打包/许可证隔离 + doctor 诊断 + 进程隔离逃生舱」的诚实表述；
  - 新增 G.1.5（或 G.5）：wire protocol 作为独立可版本化契约；
  - 新增：插件生态目录/注册页（标准的采用面需要一个「已有哪些引擎」的公开清单——这是生态飞轮的展示窗口）；
  - G.2.1 落实为「面向引擎作者的一键合规跑测 + 合规徽章」。
- **北极星 DX 指标建议**：「一个只会基础 Python 的引擎作者，从模板到通过合规测试的时间」（mission P.1.2 的可测量化）。核心库的 standard-library 严格度不应外溢为插件作者的门槛。

---

## 4. 行动清单（建议优先级）

| 优先级 | 行动 | 对应 |
|---|---|---|
| P0（pre-1.0 决定） | job mode 接口形状设计决定 + 修正「mode 域需 major」表述 | G1 |
| P0（文档债） | 回写 goals.md（G.1.2/G.1.3/G.4.2 + 新增 wire/生态目录条目）；mission 补流式价值主张 | §2.2/§3.3 |
| P1 | 面向引擎作者的一键合规跑测（事件序列重放校验纳入） | G.2.1 |
| P1 | OpenAI-compat：server 入站兼容端点 + cookbook 通用出站插件 | G2 |
| P1 | wire spec 升格路线图（先冻结 Python 协议） | E1 |
| P2 | `punctuation`/`itn` 三态旗标进标准集 | G3 |
| P2 | diarization 可移植请求旋钮 | G4 |
| P2 | CLI 麦克风/流式命令（goals 承诺） | G.2.2 |
| P2 | 插件脚手架模板（copier） | G.2.3 |
| P3 | `target_latency_ms` 构造期标准参数 | G5 |
| P3 | 规范一行 explicit non-goal：translate task | G6 |

## 5. README

已同步重写（见 README.md diff）。要点：修正原 streaming 示例**不含 supersede 的 reduce**（按规范 ST §5.2 属于不合规应用示例——README 自己教错）；改用 `session.feed()` 托管喂入；合并两段重复的「为什么不做单体包」论述；补 capability 查询与 SRT 渲染示例；修正若干语法错误（"projects still suffers" 等）；「first-party forks」改为更准确的「first-party adapter plugins」。
