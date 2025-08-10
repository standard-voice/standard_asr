

## 模型下载位置:
- 如果是 None，就用包预设的下载位置。比如用 huggingface 的就下载到 huggingface 预设的位置。用 modelscope 的就丢到 modelscope 预设的位置。如果包预设没置顶，就丢到我们预设的下载位置。


## CLI 界面
- `asr` 直接启动麦克风
- `asr <filename>` 用已经安装的预设 asr 处理指定的档案
- `asr serve` 启动 api endpoint，serve 预设模型? 如果指定模型的话那就是指定模型
- `asr install ` 安装模型...?


## 开发者应该要能结构化的获得所有已安装 / 已加载的 asr 模块。


## Boiler plate
ASR 作者应该有个仓库，可以看我们的模版，往里面跟着 guide 改一改就能给自己的 ASR 做 standard asr 适配。如果是一个 AI，也应该要轻松的做好适配。

# AI helper
必须要写个 prompt，让 AI 能正确的使用 standard_asr 和适配 standard_asr



# badge
我们应该做个 badge，让使用我们库的项目可以贴 badge，让支持我们的 asr 项目也可以贴 badge。
