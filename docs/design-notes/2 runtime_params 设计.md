
>[!note] 之前叫 options，但不直觉。


runtime_parameters 的目的
- 允许应用在 runtime 插入每个 request 可能发生变化的设置
- 比如语言，上下文，其他 runtime 参数，具体能传什么东西由 ASR 作者决定，在声明之后提供 schema 让应用填入参数。

> runtime_params 应该避免让 ASR 开发者自定义字段，因为这样会变得 unswappable。
> init config 可以，因为 init config 用户应该自己调整设置项。

什么叫 "runtime"?
- runtime 指的是 "transcribe" 函数调用的那个时间点。是音频被传给 ASR 模型的时间点。
- 对于 streaming，runtime 指的是第一个音频块被传给 ASR 模型的节点。

在 stream 的过程中，不允许修改参数。

## Strict / Best_effort 策略

当 runtime 参数对应的 capability 被关闭时，标准提供两档处理策略:

### strict 模式（推荐用于本地开发）
- 传了不支持的参数 → 直接抛 `UnsupportedFeatureError`
- 目的: 在开发阶段尽早发现配置错误，避免"以为参数生效了但实际被忽略"

### best_effort 模式（推荐用于生产环境）
- 传了不支持的参数 → 继续运行，但返回**结构化 diagnostic**
- **Diagnostic 不只是日志**: 在返回值中包含结构化信息，让调用方可以写稳定逻辑
- Diagnostic 包含:
  - 哪个参数被忽略
  - 为什么（哪条 capability 不支持）
  - 最终 effective 的值是什么
  - 这是 warning 还是 error

### 为什么不能只有 "忽略 + log warning"?
日志不是 API 合约。日志可能被重定向、被采样、被聚合、被异步写入。更重要的是：日志文本无法被程序稳定解析。应用以为参数生效了但实际被忽略，会导致转写结果彻底错误（尤其在多语言音频里）。结构化 diagnostic 让调用方可以程序性地检测和处理这种情况。
