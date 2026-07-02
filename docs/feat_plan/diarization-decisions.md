# Diarization 技術決策文檔

**狀態：** 設計案 (`diarization.md`) 經三輪獨立審查 (15 個 agents) + 實證驗證，
再經一輪獨立第四方複核（Round 4：全部承重聲明對照 shipped code 再驗證，
全數成立；並對照 mission/goals 審計）。本文檔已按 Round 4 修正——
含 wire 層代價補記（決策 1）、`always_on` 落點修正（決策 2）、
凍結 speaker 語義釘死與 3+4+5 耦合（決策 3–5）、驗證器加嚴（決策 6）、
重連安全預設（決策 7）、新增決策 8（粒度能力 defer）。
本文檔記錄所有**需要人類判斷**的技術分叉。每個決策包含完整上下文，
無需跳回其他文件。

---

## 背景

Standard ASR 要為 diarization (說話人識別) 加入標準化支持。
19/38 調查條目支持 diarization（17 個雲端/本地引擎 + 2 個 mlx-audio
模型級：VibeVoice、Granite 4.1-plus），但**每個引擎的激活方式、
標籤格式、能力範圍都不同** — 正是標準存在的理由。

設計案的核心結構已通過驗證（DiarizationRequest | None 在
RuntimeParams 上、DiarizationCap 在 batch/streaming 能力樹、
Segment.speaker + Word.speaker 結果形狀），但三輪審查發現了幾個
設計分叉需要決定。以下依重要性排序。

---

## 決策 1：`num_speakers` 是否進 v1 標準集

### 問題是什麼

`num_speakers` 是告訴引擎「這段音頻大約有幾個說話人」的提示，
13/17 支持 diarization 的引擎接受某種形式的 speaker count hint
（ElevenLabs `num_speakers`、Google `min/max_speaker_count`、
AWS `MaxSpeakerLabels` 等）。

問題是：是否在 v1 就把它放進標準可移植集 `DiarizationRequest.
num_speakers`，還是先讓各引擎用自己的 `provider_params` 承載。

### 為什麼重要

這個決定影響設計的複雜度——`num_speakers` 帶來的不只是一個欄位：

1. **需要一個 hint capability (`accepts_speaker_count_hint: bool` on
   DiarizationConstraints)**，因為 4/17 引擎不接受此提示。
   這個 bool 是 constraints 上的第一個布爾欄位，需要 `_node_narrows()`
   的小幅擴展來捕捉 `False → True` 放寬。

2. **需要 `model_copy` gating 模式**——`DiarizationRequest` 是 frozen
   的，gating 要去掉 `num_speakers` 必須建構一個新物件
   (`request.model_copy(update={"num_speakers": None})`)。這是現有
   gating 裡的全新模式（其他都是標量/列表直接設 None）。

3. **clamp 邊界**——`max_speakers` 是 `gt=0`（引擎可宣告 1），但
   `num_speakers` 是 `ge=2`（diarization 至少 2 人），clamping 到
   `max_speakers=1` 會產生非法值，且 `model_copy` 不重跑 validator。
   需要 clamp 下限為 2 的特殊處理。

4. **hint capability 本質上是空的**——三輪審查確認
   `accepts_speaker_count_hint=True` 只代表「引擎不會因為收到
   num_speakers 而報錯」。是否真的用了這個 hint 改善品質是**無法
   驗證的**（合規套件沒有真實多說話人音頻）。一個無法驗證的能力
   宣告實質上是空承諾。

### 方案

#### A. defer `num_speakers` 到 `provider_params`（推薦）

v1 的 `DiarizationRequest` 是一個**純 enable marker**——空物件，
唯一作用是 presence = 啟用 diarization：

```python
class DiarizationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # v1: no fields. Presence = enable.
```

各引擎的 count hint 走 `provider_params`：
```python
RuntimeParams(
    diarization=DiarizationRequest(),
    provider_params=ElevenLabsParams(num_speakers=3),
)
```

v1.1 或更晚，當有足夠 field evidence 確認 hint 語義收斂時，
`num_speakers` 畢業進標準集（additive-minor，不破壞現有代碼）。

**被刪掉的東西：** `accepts_speaker_count_hint` bool、`_node_narrows`
bool 分支、`model_copy` gating 模式、clamp-floor 邊界。

**對引擎作者的衝擊：** 13 個引擎把 count hint 放在自己的
`ProviderParams` 子類裡（它們本來就需要 `ProviderParams` 放
threshold / roles / model_id 等引擎特有旋鈕——多一個 `num_speakers`
不增加負擔）。

**Wire 層代價（Round 4 / R4-A 補記）：** `provider_params` 在 wire 層是
discover-only——`WireRuntimeParams` 以 `extra="forbid"` 拒絕它
（`contract/params.py:369-404`，設計如此，非疏漏）。defer 之後，
HTTP/WS 的跨語言客戶端**完全沒有** count hint 通道，直到 `num_speakers`
畢業。這是 defer 的最大隱性代價（G.5「wire 是一等公民」之下必須記錄）；
它是 `provider_params` 的系統性性質而非 diarization 特有，但**畢業的
觸發條件因此應包含「wire 客戶端的實際需求」**，而不只是引擎側語義收斂。

**相容性：** 當 `num_speakers` 將來畢業，它是 `DiarizationRequest`
上的 additive field：

```python
# 將來 v1.1
class DiarizationRequest(BaseModel):
    num_speakers: int | None = Field(default=None, ge=2)
```

現有 `DiarizationRequest()` 不受影響。Wire 的 `{"diarization": {}}`
不受影響。

#### B. v1 就 ship `num_speakers`（bool-gated）

保留 `num_speakers: int | None = Field(default=None, ge=2)` 和
`accepts_speaker_count_hint: bool` on `DiarizationConstraints`。

**代價：** 上述 4 點複雜度全部引入。

**收益：** 13/17 引擎的 speaker count hint 立即可移植——app 寫
`DiarizationRequest(num_speakers=3)` 就能跨引擎工作。

### 總體型態

| 維度 | A (defer) | B (ship) |
|---|---|---|
| 標準集欄位 | 0 | 1 (`num_speakers`) |
| 能力系統新增 | 0 | 1 (`accepts_speaker_count_hint` bool + `_node_narrows` 擴展) |
| gating 新模式 | 0 | 1 (`model_copy`) |
| 引擎作者負擔 | 各自 `ProviderParams` | 標準集 field 直接映射 |
| 可移植 count hint | ❌ 需知引擎 | ✅ 跨引擎 |
| wire 客戶端可用 hint (R4-A) | ❌ (`provider_params` 不可 wire 構造) | ✅ |
| hint 可驗證性 | N/A | 否 (vacuous) |
| 可回到另一方案 | ✅ additive | ❌ (已 ship 就不能刪) |

### 我的看法

**A (defer)。** 理由：
- `DiarizationRequest()` 作為純 enable marker 是最小且正確的 v1。
- `num_speakers` 的 hint 語義在引擎間不統一（exact count vs max vs
  hint），premature 標準化風險比 defer 大。
- 13 引擎的 `provider_params` 負擔近乎零（它們本來就需要
  `ProviderParams`），不是可移植性損失。
- defer 是可逆的 (additive graduation)；ship 是不可逆的 (刪是 breaking)。
- 三輪審查的最大複雜度集中點全在 hint 機制；刪掉它設計品質跳一級。

Round 4 複核後**維持 A**——hint 是精度優化而非功能，wire 層代價
（上文 R4-A）不足以翻盤，但必須作為已記錄的代價進入畢業判準。

---

## 決策 2：always-on diarization 引擎怎麼處理

### 問題是什麼

設計的結果模型契約是：
> `diarization=None`（未請求）→ `Segment.speaker = None`

但有些引擎**不能不 diarize**：

- **VibeVoice ASR**（Microsoft）—— 全 survey DER 最佳 (3.42%)，
  架構上是 joint ASR+diarization，LLM 在單一自回歸 pass 中同時
  生成 Speaker/Start/End/Content。沒有「不生成 Speaker」的選項，
  因為 Speaker token 和 Content token 交織在同一個輸出序列裡。

- **Rev.ai** —— 預設開啟 diarization（需 `skip_diarization` 關），
  但至少**可以**關。

- **火山引擎** —— `enable_speaker_info` 預設 true，但可設 false。

三者程度不同：Rev.ai/火山是「預設開但可關」，VibeVoice 是「架構
上不可關」。前兩者的 adapter 可以傳 skip/false 來關掉；VibeVoice
的 adapter 要不產出 speaker，唯一辦法是**解析完再丟掉**。

**已驗證的現實**（跑過真實代碼）：
- `std-mlx-audio` 的 `ModelBackend.to_result` 簽名是 `(native, *,
  duration, want_words)` —— **5 個 backend 全部沒有
  `diarization_requested` 旗標**。adapter 根本不知道 app 有沒有
  請求 diarization。
- VibeVoice 的 `GenericSttBackend` 今天已經 parse 了 diarization
  JSON 重建 text，然後把 `Speaker` 丟掉。

### 為什麼重要

如果設計堅持「未請求 = speaker 必為 None」，那 VibeVoice（全
survey 最佳 local diarizer）不能合規。adapter 必須：

1. 取得一個 `diarization_requested` 旗標（今天不存在）；
2. 若未請求，解析出 Speaker 後又丟掉它——**主動丟棄模型自然
   產出的有價值數據**。

這不只是浪費——strip 路徑是預設（多數呼叫不請求 diarization），
keep 路徑是例外，bug 最容易藏在 strip 路徑而且是靜默方向（漏
帶 speaker 沒人知道）。

反過來：如果設計允許 always-on 引擎在 `diarization=None` 時也
填 `Segment.speaker`，那 app 不能再用 `speaker is not None` 判定
「有沒有做過 diarization」——要改看 capability。但 capability 判定
其實**本來就是正確做法**（spec 的 null 規則明確說「判 supported
看 capability，不看 field null」，TR.1 line 628）。

### 方案

#### A. 引入 `always_on` 概念（推薦；落點經 Round 4 / R4-B 修正）

在 **`DiarizationCap` 節點本身**加一個裸 bool 欄位宣告
「我的 diarization 是不可關閉的」：

```python
class DiarizationCap(_FlagLikeNode):
    constraints: DiarizationConstraints = Field(default_factory=DiarizationConstraints)
    always_on: bool = Field(default=False)   # 裸 bool，直接在能力節點上

    @model_validator(mode="after")
    def _always_on_requires_supported(self) -> DiarizationCap:
        # 使矛盾態不可表示（一行 validator，無 Round-3 FlagCap 的同構代價）
        if self.always_on and not self.supported:
            raise ValueError("always_on=True requires supported=True.")
        return self
```

**為何不放 `DiarizationConstraints`（Round 4 修正原方案）：** spec §C 3.3
明文定義 `constraints` 專用於「標準層**機器可校驗的限額**」，而
`always_on` 不是限額、是**行為事實**——它的同類是 `self_resamples`
（spec 特意標註的「唯一行為性能力」，§AI 3.2 / §C R7 為其安置專門
說理）。塞進 constraints 會侵蝕 R7「定死一處家」的邊界規則。裸 bool
放節點上已對照真實機制驗證為**路徑層惰性**：`_derive_supported` 對
非節點值 fail-closed（`supports("….always_on")` 恒 `False`）、
`_iter_paths` 不產出路徑、`canonical_json` 原樣輸出為普通欄位——
無 Round-3 那類三面失同步的危險。

語義：
- `always_on=True` 的引擎 MAY 在 `diarization=None` 時填充
  `Segment.speaker`，這**不是違規**。
- **能關就必須關（R4-B 補）：** 可以關閉 diarization 的引擎（Rev.ai
  `skip_diarization`、火山 `enable_speaker_info=false`）MUST 在未請求時
  關閉。`always_on` 保留給**架構上不可關**的模型（VibeVoice），
  MUST NOT 為省 adapter 工夫而宣告（不可驗證，但與其他聲明誠實性
  要求同類）。
- **對 TR.1/TR.3 的具名豁免（R4-B 補）：** TR.3 對 word timestamps
  恰好禁止「回填未請求的數據」（`words=None`=未請求）。always-on 填充
  是對這條哲學的**有意、具名**豁免，spec MUST 明文寫出例外及理由，
  否則兩節互相矛盾。
- app 判「有沒有做過 diarization」用
  `engine.supports("<mode>.diarization")`，**不用** `speaker is
  not None`（與 TR.1 null 規則一致）。
- **App 無法拒收（R4-B 補）：** 想避免 speaker 標籤的應用（隱私場景）
  在 always-on 引擎上沒有標準關閉手段，v1 只能自行剝除
  （見 diarization.md §5.4 非目標）。
- adapter 要加 `diarization_requested` 旗標嗎？不需要。always-on
  引擎的 adapter 永遠填 speaker；普通引擎的 adapter 只在
  `diarization is not None` 時填 speaker。旗標隱含在 params 裡。

**整體型態：** 引擎宣告時選 `always_on=True/False`；app 不需要
特殊處理（capability 判定本就是 spec 的正確路徑）；adapter 不需要
strip 路徑。VibeVoice 的 adapter 把 `Speaker` 直接映射到
`Segment.speaker`，完事。

> Round 4 亦考慮過更輕的變體：完全不加欄位，只寫 normative MAY +
> 在填充未請求 speaker 時發一條 `info` 診斷
> （`code="unrequested_speaker_labels"`）——用**既有的**結構化診斷通道
> 而非新聲明欄位。因失去**轉寫前**可發現性而不作為主機制，但該診斷
> MAY 與旗標並存。

#### B. 要求 adapter strip speaker when not requested

字面遵守 `diarization=None → speaker=None`。

**代價：**
- VibeVoice adapter 必須取得 `diarization_requested` 旗標
  （改 `ModelBackend.to_result` 簽名，影響全部 5 個 backend）。
- strip 路徑是預設，keep 路徑是例外——bug 最容易藏在預設路徑。
- 主動丟棄高品質數據。

**收益：** `speaker is not None` 保持為「請求了 diarization」的
信號（但 spec 說不應該這樣判定）。

#### C. v1 不支持 always-on 引擎的 diarization

VibeVoice 繼續像今天一樣丟棄 Speaker、宣告零 diarization。

**代價：** 放棄全 survey DER 最佳的 local diarizer。問題只是推遲，
不是解決。

**收益：** v1 不用處理這個邊界。

### 總體型態

| 維度 | A (always_on) | B (strip) | C (不支持) |
|---|---|---|---|
| VibeVoice 可合規 | ✅ | ✅ (有成本) | ❌ |
| adapter 複雜度 | 低 (永遠填) | 高 (strip/keep 分支) | 零 (今天) |
| `speaker is not None` 信號 | ❌ 不可靠 | ✅ 可靠 | N/A |
| 正確判定方式 | capability (已是 spec 正道) | field null (spec 不推薦) | N/A |
| `to_result` 簽名改動 | 否 | 是 (全 5 backend) | 否 |
| 丟棄高品質數據 | 否 | 是 (strip 路徑) | 是 (今天) |
| 決策可逆性 | ❌ (always_on 加了不能刪) | ✅ (可改成 A) | ✅ (推遲) |

### 我的看法

**A (always_on)。** 理由：
- 它讓最佳 local diarizer 立即可用，零 adapter 複雜度。
- 它跟 spec 的 null 規則 (TR.1 line 628) 完全一致：field null 不等於
  不支持，看 capability 才對。
- B 的 strip 路徑是**主動丟棄已有數據**+在靜默方向留 bug 窗口，
  違反「explicit > implicit」。
- 代價（`speaker is not None` 不再是 diarization-requested 的信號）
  不大——spec 已經說不應該用 field null 判定；但這是對 TR.1/TR.3 的
  鬆動，MUST 以具名豁免寫進 spec（見上）。
- `always_on: bool` 作為 `DiarizationCap` 節點上的裸 bool（R4-B 修正：
  不放 `DiarizationConstraints`——constraints 是限額袋，行為事實的
  同類是 `self_resamples`），路徑層惰性已驗證，實現成本 = 一個欄位
  + 一行反矛盾 validator。

#### 修訂（2026-07-02，合併前復審）

合併前復審發現上面「裸 bool、路徑層惰性」的論證有兩個漏洞：

1. **「行為事實 vs 可請求能力」的區分是假的。** 能力樹裡本來就有大量
   非可請求的行為事實——`self_resamples`、`emits_partials`、
   `re_segments`、`word_stability`、`reconnect`、`finality_level`、
   `timestamps`。「always_on 是行為事實所以不該是可查詢路徑」不成立。
2. **表示不一致。** `self_resamples`（本文自己稱為 always_on 的「同類」）
   是**可查詢的 `FlagCap` 節點**，參與 `supports()`、
   `iter_supported_paths()`、`canonical_json()`（注入 `supported`）、
   `covers()`；而 always_on 卻是被這四者排除的裸 bool。同族事實、兩種
   表示。

**修訂後型態：** `always_on` 改為 `DiarizationCap` 的 `FlagCap` 子節點，
與 `self_resamples` 完全一致。**現在可用的：** `supports("<mode>.
diarization.always_on")` 是有效查詢路徑；聲明 supported 時出現在
`iter_supported_paths()`；`canonical_json()` 統一注入 `supported`；
`covers()` 以標準集合包含把「declared 未聲明 → effective 聲明」的漂移
當作違規拒絕（原裸 bool 無此保護）。矛盾態
`supported=false ∧ always_on 聲明 supported` 仍由構造期 validator 拒絕。
wire 形狀：`{"supported": true, "always_on": {"supported": true}}`；裸
bool 現在會被 pydantic 大聲拒絕（pre-release，無相容 shim）。**唯一存活
的區別**是語義反轉的文字說明：其他 flag 的 `true`＝「你可請求」，
always_on 的 `true`＝「強加給你」（未請求也可能出現 speaker 標籤）——
這是文字，不是表示層差異。原「路徑 fail-closed 是設計」「canonical
JSON 不注入 supported」的主張已作廢。

---

## 決策 3：streaming 中 `stable_until` 是否保護 speaker

### 問題是什麼

`stable_until` 是流式協議的核心概念——它標記 `text[:stable_until]`
已凍結，語音助手可以對凍結前綴**採取行動**（開始回應、路由等）。

但 `stable_until` 今天**只保護文字**。`_frozen_prefix_rewritten`
（streaming.py:1227-1245）只比較 `event.text[:prior_su]` 與存儲的
凍結文字。speaker 不在比較範圍內。

所以這個場景是合法的：

1. partial: `text="Hello", stable_until=5, speaker="A"`
   → 語音助手看到凍結的 "Hello" 來自 Speaker A，開始路由到 A。
2. 後續 partial: `text="Hello there", stable_until=5, speaker="B"`
   → 文字前綴 "Hello" 不變 → guard 放行
   → **speaker 從 A 靜默變成 B**
   → 語音助手已經路由到 A，現在是 B

這在不 diarize 的世界裡不存在（speaker 始終是 None）。一旦
diarization 上線，這就是一個**承重的靜默錯誤路徑**——而且
只在有真實多說話人的流式音頻上才會觸發，開發者測試時（通常
用單人音頻）不會碰到。

### 為什麼重要

兩個問題交織：

1. **對語音助手場景的正確性保證** —— spec §7.2 明確鼓勵讀
   `text[:stable_until]` 做行動決策。如果 speaker 可以在凍結
   區域內悄悄改，那行動決策的 speaker 維度是不可靠的。

2. **引擎行為的現實** —— Google STT、AWS Transcribe 在流式時
   確實會隨著更多音頻重新分配 speaker（initial clustering 不穩定，
   accumulate 更多 speaker embedding 後改善）。如果我們禁止凍結
   區域內的 speaker 改動，這些引擎的 adapter 需要：
   - 要麼延遲 speaker 分配直到引擎穩定（增加延遲）
   - 要麼 clamp 掉引擎的 speaker 修正（降低品質）
   - 要麼只在 `final` 時才附上 speaker（partial 永遠 speaker=None）

### 方案

#### A. 保護 frozen region 的 speaker（推薦）

在 `_LifecycleGuard` 旁加一個 `_frozen_speaker_rewritten` 守衛：
一旦 word 進入 segment 的 frozen prefix（`text[:stable_until]`），
其 speaker 在後續 partial/final 中 MUST NOT 改變（closed 終態
除外，與文字的 closed 豁免對稱）。

**引擎如何適應：** adapter 有三個策略：
- **延遲 speaker** —— partial 的 speaker 設 None 直到引擎穩定，
  final 再給。語音助手看到 `speaker=None` 就知道不能 route。
  這是最安全的策略。
- **保守 speaker** —— 一旦分配就不改，即使引擎內部重新聚類。
  犧牲品質換穩定。
- **native stable engines** —— Deepgram/ElevenLabs 的 streaming
  只在 final 給 speaker，不在 partial 給——天然合規。

**代價：** 一個新的守衛函數（~30 行，邏輯與 `_frozen_prefix_
rewritten` 對稱）。Adapter 不能盲目轉發 partial 的 speaker
（如果引擎重新分配了凍結區域的 speaker，adapter 必須 clamp
或延遲）。

**Round 4 (R4-C) 釘死的四個語義細節（「~30 行」只在此範圍內成立）：**

1. **只護 segment 級**（`event.speaker`）。詞級 speaker 的凍結追蹤
   需要逐事件對齊 words 列表，複雜度不同量級——v1 明確不做，
   詞級留作 adapter 義務。
2. **`None→X` 合法**（否則「延遲 speaker 到 final」——本決策自己
   推薦的 adapter 策略——直接違規）；**`X→None` 與 `X→Y` 同為違規**
   （撤回即改寫）。
3. **違規時抑制整個事件，不鉗制。** 鉗制（保留事件、還原舊 speaker）
   會把過時的歸屬繼續呈現——另一個方向的靜默錯誤結果；與
   `_frozen_prefix_rewritten` 行為對稱。抑制代價分兩檔：掉 in-flight
   `partial` 只是 ephemeral 損失（下一個 partial 重現該段）；但掉
   違規的普通 `final` 不是——歸約器只提交 final，該段永不提交，
   文本從 `session.result()` 永久缺失（只留 warning diagnostic）。
   合規出路：在 final 上復述已鎖定 speaker、延遲 speaker 到 final
   （`None→X` 合法）、或在守衛豁免的 `closed` 終態上定稿修正。
   副作用：盲轉發 Google/AWS partial 的不合規 adapter 會連續掉
   partial，若其 final 也帶改寫後的 speaker 則整段丟失——與 spec
   §5.2「只傷害不合規適配器」的既有立場一致，寫進 spec 即可。
4. **與決策 4、5 耦合，三者同進退：** 決策 4 的 carry-forward 只有
   在本決策禁止 `X→None` 時才安全（否則會把引擎刻意撤回的 speaker
   縫回事件流）；決策 5 的執行需要本 guard 順手引入的 per-segment
   最後已知 speaker 追蹤。

#### B. 不保護——文檔說明 `stable_until` 只覆蓋文字

在 spec §4.2 加一句：`stable_until` 的凍結保證只覆蓋 `text`，
不覆蓋 `speaker`。在 §7.2 語音助手指引中明確警告：
> "MUST NOT 對 partial 事件的 speaker 採取不可逆行動。只有
> `final` 的 speaker 是已提交的。"

**代價：** 語音助手的 speaker routing 只能等 `final`（不能用
凍結前綴的 speaker 提前路由）。如果引擎只在最後一刻才 `final`
（大多數引擎如此），這實質上意味著流式 diarization 的 speaker
路由延遲等於整個句子。

**收益：** 零實現成本。不限制引擎行為。Adapter 最簡單（直接
轉發引擎的 speaker，哪怕會變）。

### 總體型態

| 維度 | A (保護) | B (不保護) |
|---|---|---|
| 語音助手 speaker 路由 | ✅ 凍結後可靠 | ❌ 等 final 才行 |
| adapter 複雜度 | 中 (需 clamp/delay) | 低 (直接轉發) |
| 引擎品質 vs 穩定 | 犧牲品質換穩定 | 保留品質 |
| 實現成本 | ~30 行守衛 | ~2 行文檔 |
| 「correctness wins」符合度 | ✅ | ❌ |
| 流式 diarization 延遲 | 低 (凍結部分可路由) | 高 (等 final) |

### 我的看法

**A (保護)。** 理由：
- spec 的哲學是「correctness wins over DX」。speaker 在凍結
  區域內靜默改變是一個不可觀測的正確性違規——恰好是 cardinal sin
  的定義。
- 實務上，多數 streaming diarization 引擎（ElevenLabs、Deepgram）
  只在 final 才給 speaker——它們天然合規，零 adapter 成本。
- Google/AWS 的 partial speaker 重分配是已知行為，adapter 延遲
  speaker 到 final 就好——反正 app 拿到 partial speaker 也不
  敢 route（因為 partial 可能被推翻）。
- 守衛是對稱實現（與 `_frozen_prefix_rewritten` 同結構），維護
  成本低。

**但 B 也完全可行**——如果你認為 streaming diarization 的主要
場景是字幕（display-only, speaker 改了重畫就好）而非語音助手
路由，那不保護 speaker 是合理的。

---

## 決策 4（較小）：streaming backpressure partial-merge 中的 speaker

### 問題是什麼

backpressure 下，`_CoalescingBuffer.put()` (streaming.py:651)
用 `slot.event = event` 盲目替換整個事件。如果被合併掉的
partial 有 `speaker="A"`，新的 partial 恰好 `speaker=None`
（引擎還沒重新判定），speaker 就靜默丟失了。

**同一個 bug 今天已存在於 `detected_language`**——只是
diarization 上線後，它在 speaker 維度也會觸發。

### 方案

#### A. carry-forward last-non-null（推薦）

合併時保留最後一個非 null 的 speaker（和 detected_language）。
~5 行代碼改動。

#### B. 要求 adapter 在每個 partial 重複 speaker

不改合併器，adapter 的義務是每個 partial 都帶 speaker（即使
跟上一個一樣）。合規套件可測這一點。

**代價：** adapter 負擔高；不一致的 adapter 會靜默丟 speaker。

### 我的看法

**A。** 簡單、防禦性、跟 detected_language 一起修。adapter 最佳
實踐仍然 SHOULD 在每個 partial 重複 speaker，但 carry-forward
是兜底。

**前提（Round 4 / R4-C）：** carry-forward 的安全性依賴決策 3 禁止
`X→None`——若允許引擎刻意撤回 speaker，carry-forward 會把撤回的值
縫回事件流（合成引擎從未整體發出過的事件）。本決策與決策 3 必須
同時採納或同時推遲。合併範圍天然限於同一 `segment_id`
（coalescing 本就按段合併）。

---

## 決策 5（較小）：supersede 跨說話人合併

### 問題是什麼

`supersede` 把 `[seg-3(speaker A), seg-4(speaker B)]` 合成
`[seg-5]`。frozen-prefix obligation 保證了**文字**完整，但
seg-5 只有**一個** `speaker` 欄位——它選 A 還是 B？如果選 A，
speaker B 說的文字被靜默歸到 A。

set-to-set lineage（spec §5.2 line 1031）原理上無法保留 per-
speaker 歸屬（沒有 per-old → per-new 映射）。

### 方案

#### A. 禁止跨 speaker 合併（推薦）

normative：引擎 MUST NOT supersede 不同 speaker 的 segments
into a single new segment。合規測試強制。

**代價：** 限制兩遍重打分引擎的靈活性（WeNet U2++ 可能想
合併跨 speaker 段）。

**收益：** 完全消除靜默歸屬錯誤。

#### B. 保留 word-level speakers + segment speaker = None

允許合併，但要求 merged segment 攜帶 `words`（保留 per-word
speaker），segment 的 `speaker` 設 None（「混合，看 word
級別」）。

**代價：** 引擎必須在合併時保留 word 粒度，增加 adapter 負擔。

### 我的看法

**A。** 跨 speaker 合併本身就是語義上可疑的——一個 segment
（通常對應一句話）不應該包含兩個說話人的混合。禁止它
更接近語義正確性，且合規可測。

**執行機制（Round 4 / R4-C）：** normative MUST NOT + 合規測試為主；
運行時 guard 抑制為輔，且其副作用必須寫進 spec——被抑制的 supersede
使舊段在 reducer 中存活、新段隨後以全新段到達，歸約結果出現
**重複文本**（既有 guard 對非法 supersede 本來就是此行為，
「只傷害不合規適配器」）。執行需要 per-segment 最後已知 speaker 的
追蹤——正是決策 3 的 guard 順手維護的狀態，兩者一起實現。

---

## 決策 6（較小）：`Segment.speaker` 空字串驗證

### 問題是什麼

驗證確認 `Segment(speaker="")` 和 `speaker="   "` 都能構造，
是靜默壞數據。`""` 既不是 `None`（未判定）也不是真實標籤——
第三種未定義狀態。

`phrase_hints` 在 RuntimeParams 上早就有空字串驗證（因為
`"" ∈ any_string` 恆真會擊穿降級存活判定）。speaker 有相同危險。

### 方案

**唯一合理方案：** 加 `field_validator` 拒絕空、純空白、
**及前後帶空白**的 speaker（Round 4 / R4-F 加嚴：`"A "` 與 `"A"` 是
兩個不同字串 = 兩個不同說話人，一個 adapter 的 off-by-one 就靜默打破
within-result consistency；**拒絕而非歸一化**，與 phrase_hints 的
fail-loud 立場一致）on `Segment`、`Word`、`TranscriptionEvent`
（都要加，否則 malformed event.speaker 通過事件驗證後在 reducer 的
Segment 構造崩潰）。`None` 保持合法。

這是設計的明確 bug，不是 trade-off——D3 說「no changes needed」
是驗證後被推翻的。只需確認要不要在此 PR 一起修。

### 我的看法

**是，一起修。** ~15 行驗證器，防止一整類靜默壞數據。

---

## 決策 7（較小）：reconnect 後的 speaker 連續性

### 問題是什麼

spec §6.3 列舉了跨 reconnect 必須保持連續的三項：`segment_id`、
timestamps、`detected_language`。speaker 不在清單上。

Google STT 有 5 分鐘 session 上限。reconnect 後新 session 的
diarizer 從零開始聚類——同一個人可能從 `speaker_0` 變成
`speaker_1`。這對 app 來說是同一個 Standard ASR session，
同一個結果的 within-result consistency 被靜默打破。

### 方案

**唯一合理方案：** 把 `speaker` 加入 §6.3 的 continuity 清單。
adapter 必須維護跨 reconnect 的 label → speaker 映射。當
adapter 無法 re-map（引擎的新聚類結果與舊的無法對應）時，
MUST 發 diagnostic（與 `content_lost` 同類的 fidelity warning）。

**安全預設（Round 4 / R4-F 補充，把模糊義務變成機械規則）：**
盲聚類引擎重連後「無法映射」是**常態而非例外**（兩邊都是無身份
資訊的聚類）。spec 直接規定預設行為：重連後的新 cluster MUST NOT
復用重連前的標籤（除非 adapter 有身份證據，如 enrolled voices），
MUST 鑄造新標籤（`speaker_2`、`speaker_3`…）+ 發 diagnostic。
**過數說話人是安全方向；把不同的人靜默合併到同一標籤不是。**

### 我的看法

**加。** 不加的話 streaming diarization 在有 reconnect 的引擎上
（Google、ElevenLabs 都有 session 時限）靜默錯。鑄新標籤預設使
義務可執行、可合規測試。

---

## 決策 8（較小，Round 4 新增）：`DiarizationCap` 是否聲明歸屬粒度

### 問題是什麼

調查顯示 word:segment 歸屬粒度為 8:7（diarization.md §2.5；設計案
早期引用的「6:4」是未更新的舊數字，已修正）。是否在 `DiarizationCap`
加 `granularities`（仿 `WordTimestampsCap`），讓 app 預知結果形狀。
設計案原推薦「加」（跟隨 `WordTimestampsCap` 先例）。

### 我的看法（Round 4 反轉原推薦）

**defer，與 num_speakers 同批。** 理由：

- 決策 1 defer 之後 `DiarizationRequest` 是純 marker，app **無法請求**
  粒度——此能力淪為純資訊性（`WordTimestampsCap` 的粒度是可請求、
  可門控的，先例不成立）。
- G4 交互規則已規定實際填充行為（diarization 請求時 `Segment.speaker`
  恒填；`Word.speaker` 僅當 words 同時被請求時填）。
- 加它需要 `WordTimestampsCap` 式 supported⇒非空 validator，且令節點
  變 multi-archetype 混合體（Round 3 已指出）。
- additive，隨時可加。

---

## 總結

| 決策 | 推薦 | 不可逆？ | 可延遲到 v1.1？ |
|---|---|---|---|
| 1. num_speakers | defer（wire 代價已記錄, R4-A） | ✅ 可逆 (additive graduation) | 就是 defer |
| 2. always-on | `always_on` 為 `DiarizationCap` 的 `FlagCap` 子節點（可查詢；合併前復審由裸 bool 修訂） | ❌ 加了就在 | 可,但丟 VibeVoice |
| 3. frozen speaker | 保護（語義經 R4-C 釘死） | ❌ 加了就在 | 可,但須與 4、5 同批 |
| 4. partial-merge | carry-forward | ✅ 可逆 | 可,但須與 3、5 同批 |
| 5. supersede merge | 禁跨 speaker | ❌ 加了就在 | 可,但須與 3、4 同批 |
| 6. speaker 驗證 | 加（含前後空白, R4-F） | ❌ 加了就在 | 理論上可,但是明確 bug |
| 7. reconnect speaker | 加入連續性清單 + 鑄新標籤預設 | ❌ 加了就在 | 可 |
| 8. 粒度能力 | defer（Round 4 反轉） | ✅ 可逆 (additive) | 就是 defer |

**耦合關係（Round 4 / R4-C）：** 3+4+5 是一組——同時採納或同時推遲，
不可拆散（4 的安全性依賴 3 的 `X→None` 禁令；5 的執行依賴 3 引入的
per-segment speaker 追蹤）。

最保守的 v1 = 1(defer) + 2(always_on) + 6(驗證器) + 8(defer) +
核心 plumbing（`DiarizationRequest` marker、`TranscriptionEvent.speaker`、
StreamReducer 傳播 + **標準層統一合成**——batch 在
`EngineBase.transcribe` 後處理、streaming 在 reducer，單一釘死規則：
詞數多數決、平手取最早詞，見 diarization.md R4-D）。
3+4+5 這組與 7 可以在第一個真實 streaming diarization adapter 落地時
再進。但 2（always-on）和 6（驗證器）強烈建議不推遲——前者解鎖旗艦
diarizer，後者是明確 bug。
