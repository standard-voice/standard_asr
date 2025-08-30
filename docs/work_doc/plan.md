～～`2025/06/29`～～

# Case
> 得有个懂 Python 最佳实践 (或者说有提高代码品味的意愿的人 + 懂最佳实践的 AI...)，去改变现在 asr / tts 库领域混乱的现状。
> 事实就是，现在 asr / tts 库乱七八糟，调用困难，测试困难。写一个使用本地 asr / tts 的项目，每个 asr / tts 引擎都得单独做适配。每个项目都得给每个引擎单独做适配。这很累，而且我深刻的理解这一点。
> 开源项目应该要能轻松的适配各种 asr / tts 引擎，毕竟从 Open-LLM-VTuber 就能知道，他们的调用方式大同小异。
> 

# motivation
- 剥离 Open-LLM-VTuber 的 ASR 模块，让 ASR 维护的分工程度更高。
- 将 standard_asr 本身变成一个零依赖的纯框架，每个 ASR 引擎的支持都由一个独立的插件包来提供，解决依赖冲突的问题。

## 解释
A.2 为了解决下面的问题
1. asr / tts 库，依赖混乱，互相之间存在冲突。做一个统一的，包含所有 asr / tts implementation 的工具库是不可能的 (因为 pyproject.toml 就算是 dep group 也会一起 resolve)
2. asr / tts 库协议千奇百怪，如果有 AGPL, GPL 协议的，我们就不能直接包含在我们的库中。设计成 plugin 系统，我们核心接口做成纯框架，纯接口，用户根据他们自己想使用的库和自己项目的协议，选择想要的库用。

```
https://g.co/gemini/share/ddf9ef577e40
```


# A: 核心目标:
- A.1: 做 ASR 推理领域的 usb 标准: 提供通用的接口，让 ASR 推理开发者，ASR 使用者，能有一个共同的标准，互相沟通。
- A.2: 提供测试套件和周边工具，让 ASR 库开发者更好的开发好用，稳健，工程化的库
- A.3: 适配过 Standard ASR 标准的代码，应该可以免配置直接跑任何 Standard ASR compliant 的模型。如果有额外配置项，需要用 pydantic 暴露出去，让 WebUI 和 GUI 和数据库能动态生成配置。


做一个不包含任何实际 ASR 实现的灵活的框架，让开发者可以快速适配大量 ASR 模型，让 ASR 库可以轻松的支持我们标准 (或是我们自己适配成插件)。

哲学:
- P.1: 调用者友好: 调用 Standard ASR 库的代码，应该要能接近 0 配置的调用几乎所有的 Standard ASR compliant 模型。开发者可以完全不知道用户会使用什么 ASR 引擎，开发者只需要知道用户使用的是 Standard ASR compliant 的 ASR 库即可。
- P.2: ASR 开发者友好: ASR 开发者和模型制作者，应该能在只懂最基础的 Python 的状况下，快速的适配 Standard ASR 标准。


## A.3 免配置运行
为了达到这个目标
- 核心配置，比如语言，可能要标准化
    - 为了解决不同 language code 的问题，我们可能要写语言标准的转换工具


# 还未整理的想法
应用开发者安装 standard_asr 和需要的插件库，然后用 standard_asr 的函数可以发现所有安装的 asr 库。
使用时，可以用我们的工厂获得 asr 实例，也可以直接 import 对应的库 (但不推荐)。

周边工具包括
- 模型下载器
- web ui
- 测试工具
- 

- 支持 web api 交互，自带 fastapi 服务器 (dep group)，可以用 stainless 生成 sdk 库
- 只要实现了 interface，就可以直接变成 ASR plugin
- 可以用库启动 ASR web 服务器，也可以用 python 代码调用。


ASR
- 支持 stream in, stream out (实时字幕输出) (可选)
- 支持 word level timestamp (可选)
- 可能不是所有 ASR 都支持所有功能，因此得做机制告诉 API 自己不支持某项功能

配置
- 需要自带 config schema，带 description

模型下载可能不能放在包里面，要放在公共目录下



格式
- input: np.array
- sample rate: 16khz
- output: 如果启用 word level timestamp，就是 dictionary。如果不启用，就是文本。

初始化参数 和 推理参数，都可以传入额外的参数
推理参数就是单次 request 可以选择要不要丢进去的东西

推理可选参数，由 asr 定义，写在 pydantic 模型中。 



开源社区 (standard_asr):
一个 organization，包含核心，和适配的 ASR。

适配我们标准的 ASR:
- pip 安装后，可以直接 drop-in 
- 提供 web interface，测试工具等等。


(asr_suite)
- ~~提供许多 asr 的一个 library，可以用 `pip install asr_suite[asr_opt]` 来快速安装 asr。~~ 把 asr 库全都放在一个 package 里作为可选项是不可能的，我们可以提供类似 ollama 这样的 cli 工具来快速安装和管理 asr 模块，但他们可能有许多注定不能被安装在一起。预设可能就是以隔离的 web 服务器安装在本地 (pipx 或 uvx?)


文档里面应该要有个地方让使用或支持我们标准的开发者们往里面写自己的项目。




版本更新时，server api，test 之类的东西可以是小版本更新，因为不影响


一个 ASR provider 可能会支持很多模型，我们要让他们能够提供模型预设，又或者是不同模型可能要用不同类型的设置项？这个要怎么设计？


注意到输入音频格式可能区别比较大，我们可能要... 有自动转换音频格式的 helper? 反正 properties 里面可能要写支持的格式，采样率等信息，另外做一些工作确保当音频格式不符时，可以报 warning 后自动用 ffmpeg 尝试转换。另外，如果音频根本不存在或无效，要有清晰的报错信息。


用 feature flag 的方式，让调用者知道这个 ASR 支持什么功能 (streaming, word level timestamp, speaker diarization, etc) 到底会回传什么东西。


asr 引擎加入新的自己的功能时，加到 extra 条目里面，如果未来标准化了，就迁移到标准做法。



# 对适配的 asr 库的要求

- 做好 migration 的准备: 初始化的 config 不能乱动，要遵守语义化版本号的规范，如果要删除某些项，先 deprecate，然后过几个版本删除，删除的版本应该要是大版本。


