
[[#设计]]
[[#问题与解决]]

# 设计:
可以查的百科词条和决策文档。

## Language code 标准

遵守 IETF BCP 47 标准的 language code。

对于不使用 BCP 47 标准的 ASR 引擎，用我们提供的转换器，把 BCP 47 转换成 ISO 639-1 或其他类型。

**支持 `auto` 语言。`auto` 是 Standard ASR 的保留字，不是 BCP-47 标签。** 如果使用 auto，就代表语言可以自动被识别。

> [!important] `auto` 的约束
> - `auto` 可以出现在 `selectable_languages` 和 `default_language` 中
> - `auto` **禁止**出现在 `candidate_languages` 中（语义不成立：auto 是指令，不是候选语言）
> - 在 streaming 中，auto 识别在流开始时锁定，不意味着可随时重新识别语言

>[!warning] BCP 47 支持更 general 的 zh 和 en 这种 code，我们需要在用户提供类似的 general language code 时提供预设的 fallback 逻辑。
>要注意 zh 能不能 fall 到 yue 的问题...


理论上，一个模型只能支持自己明确对外宣告的语言，所以不用太担心 language code 的问题。但实践上，标准化可以大幅增强自动化和可迁移性，比如自动选择合适的语言，让用户不用手动去设置语言。

不过... 标准化可能是个伪需求，因为手动设置 ASR 引擎很难避免。

### 降级映射策略
- **暂不做自动降级行为**。ASR 模型宣告什么 language code，调用者就必须传什么（exact match）。
- 未来可提供**工具函数**: 给定一个更宽泛的 language code（如 `zh`），用工具函数选取最合适的 `selectable_languages` 中的实际 language code（如 `zh-Hans`）。
- auto fallback 和 generalization 应该作为可选功能，未来推出。目前先做成明确宣称。




# 问题与解决

## 如何处理初始化语言参数和 runtime 语言参数的冲突?

### 问题描述:

ASR 模型支持不同语言。

language code 应该是模型初始化的参数，还是 runtime 的，也就是作为 transcribe 函数的参数，在每次音频传入时都可变的参数呢？

真的有模型要求初始化参数就有 language code 的嘛？

如果模型只支持一个语言，那好说，直接对外宣告这模型只支持一种语言就行。我们的标准可以让模型对外宣告自己支持什么语言。
但如果一个模型支持多种语言，但必须在模型初始化的时候作为参数传入，就很麻烦了。


### 解决方案: default language + runtime override

capability 宣告
- `supports_runtime_language_override`: 语言是否能在 runtime 可变 / 指定？

模型初始化 接受:
- `default_language` 

runtime 参数 接受
- `language`


对于 runtime 接受语言参数的模型
- default language 作为 fallback

对于初始化之后不能改语言的模型:
- `supports_runtime_language_override` 设为 false
- runtime 不接受 language 参数



## Candidate Languages
> 许多闭源 ASR 引擎，在自动识别语言的模式下，支持指定自动语言识别的范围。
> 在许多实现中，能够限制自动语言识别的选择范围。

如何设计 candidate languages?

candidate languages 的核心约束:
- 只在生效语言 (language) 设置成 auto 时有效
- 只在模型支持 candidate languages 功能 (`supports_candidate_languages` 为 true) 时有效
- candidate languages 中只能包含 `detectable_languages` 中列出的 language code。
- **`candidate_languages` 禁止包含 `"auto"`**（auto 是指令，不是候选语言）

如果语言不是 auto，或/且 模型不支持 candidate languages 功能时，传入的 default candidate languages 和 candidate languages 会被忽略。

### 列表顺序语义
candidate languages 是一个 python **有序列表**，语义为**偏好顺序 + allowlist**:
- 引擎/适配器**可以忽略顺序**，但必须把它当作 allowlist（允许集合）
- 如果底层服务支持 "preferred language / primary language"（如 Google Cloud Speech 的 primary + alternatives、AWS Transcribe 的 PreferredLanguage），**适配器应当用第一个元素作为 preferred/primary**（best effort）
- **不需要** `supports_candidate_language_order` 这样的 capability flag——不支持顺序的引擎只是忽略偏好信息，不会导致行为不确定

### 去重但保序
- 如果输入列表有重复元素，标准层去重但保留首次出现的顺序。

---

Init Config:
`default_candidate_languages`
- 预设的 candidate languages 列表。
- 只会在 auto 启用和模型支持此功能时产生作用。
- 传入的语言必须是 `detectable_languages` 之一
- 传入语言的数量不能超过 `max_candidate_languages` 的限制。每个模型不同，但尽量控制在 4 个以下。推荐写相关逻辑保证实际传入的 candidate language 小于 `max_candidate_languages` 定义的值。

Runtime Param:
`candidate_languages`: 
- 在 runtime param 中，可以用这个参数复写 default candidate language 设置的值。

Capability:
`supports_candidate_languages`: bool
- 模型宣告自身是否支持 candidate languages 功能

Properties:
`max_candidate_languages`: int
- Standard ASR 校验层允许传入的 candidate languages 数量上限。
- 不必完全等于底层引擎的硬限制: 对有硬限制的引擎设为硬限制值，对只有软建议的引擎设为官方建议值。
- 一般这个数字在 4-5 之间。

概念: effective candidate language: the version of candidate languages that actually apply
- 只在 transcribe 时生效的语言 (effective language) 为 auto 时有效
- 只在模型支持 candidate languages 功能 (`supports_candidate_languages` 为 true) 时有效
- runtime 设置的 candidate_language 总是优先于 `default_candidate_languages`。
