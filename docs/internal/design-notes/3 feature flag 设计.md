
初衷:
feature flag:
- A dictionary of standard ASR protocol optional features supported / not supported by this ASR engine.
- 表达这个 asr 引擎支持的 optional features 范围

- 就放在 asr properties 里面，作为一个 `feat_flag` 的一个 pydantic class，里面的 key 是当前版本支持的所有 feature，value 记录 feature 是否开启，相关版本号，还有相关信息之类的?


(实际实现会用 pydantic 做，这里只是个概念展示)
```yaml
feat_flag:
  feature1:
    supported: true
    version: "1.0.0"
    description: "Description of feature1"
    metadata: {}
  feature2:
    supported: false
    version: "1.2.0"
    description: "Description of feature2"
    metadata: {}
  input_streaming:
    supported: false
    version: "1.2.0"
    description: "Whether this ASR engine support receiving audio input via streaming"
    metadata: {}
  result_streaming:
    supported: true
    version: "1.2.0"
    description: "Whether this ASR engine support streaming ASR result chunks back to the client"
    metadata: {}
```

#### version:
- version 代表上一次这个 feature 定义被修改的版本号。是预设的值，开发者不用改。不，应该说开发者连看都不用看。我们项目本体遵守 semantic versioning，包可安装但协议不兼容的情况不会出现。所以这东西没办法用来判断兼容性，但似乎能给开发者提供更多信息? (另类 changelog)


这样设计 version 能有任何帮助吗？ 可以在 asr 版本号之外提供更多兼容性信息吗？是不是不行？ protocol 只要改，不是大版本号就得变吗？
我们应该设计这个 version 到 feature flag 里面吗？

#### description
就是 这个 feature flag 的 description

#### metadata
如果对应的 feature 有一些参数，需要 asr engine expose 给应用的，可以放在这边?
