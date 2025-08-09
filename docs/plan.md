`2025/06/29`



motivation
- 剥离 Open-LLM-VTuber 的 ASR 模块，让 ASR 维护的分工程度更高。

核心目标:
- 做 ASR 领域的 OpenAI Compatible API。
- 提供通用的接口，让 ASR 推理开发者，ASR 使用者，能有一个共同的标准。

一个 ASR 的 Interface，以同一套接口支持尽可能多的 ASR 接口，让开发者可以快速适配大量 ASR 模型。

- 支持 web api 交互，自带 fastapi 服务器
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
- 提供许多 asr 的一个 library，可以用 `pip install asr_suite[asr_opt]` 来快速安装 asr。




版本更新时，server api，test 之类的东西可以是小版本更新，因为不影响


