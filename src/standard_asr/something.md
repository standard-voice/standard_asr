

---
# 任务一: 完善 standard_asr_user 示例应用
把这个 standard_asr_user 做的极其完善和完整。把一个 application dev 可能需要做的事，比如使用 option，读取 metadata，传入 runtime 参数之类的，都使用一下。
- 如果我想指定语言，该如何？
- faster-whisper 如何向外传递自己支持的语言信息？
- 我们能否自动生成模型相关文档？
- registry.names() 真的列出了所有模型吗？还是只列出了本地有的模型？
- 我们是否存在一种机制，来快速的让人看到模型或 engine 的所有信息？要怎么做呢？

查看 standard_asr 的 agents.md，cookbook, mission, goals, README, 并深度研究其代码，了解 standard_asr 的设计理念和目标。我们需要在 standard_asr_user 中完整的体现出 standard_asr 的设计理念，目标和功能。

## 工作方式

先写核心验收文档 `work/criteria.md`，设计方案 `work/plan.md`，根据验收文档审查设计方案，再写 todo `work/todo.csv`，然后把所有 todo 完成。todo 中要包含所有任务，比如文档，所有的 implementation 任务，完成某个 feat 任务之后的代码审查，测试，和完成所有任务之后最后的 criteria 验收，所有改动要与 standard asr 的 mission 和 goals 高度对齐。把东西写成带 tick box 的 todo 项目，方便追踪进度。

在提交之前，总是审核代码，运行测试，linter, type checker，审核 todo 完成状况，并且再看看 todo 是否需要更新或添加更多 todo。最终提交时，分批提交，让 commit 原子化。

**总是主动更新 todo.csv，并把 todo.csv 中的所有任务都做完**


## side notes:
值得注意的是，standard asr 目前还在开发中。如果你在工作过程中发现 standard asr 需要改进的地方，请把这些改进建议写到 `work/standard_asr_improvements.md` 中。

