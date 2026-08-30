# BBVS 项目规格

状态：实现基线  
版本：0.1  
最后更新：2026-08-30

## 1. 目标

BBVS 是一个模块化视频理解工作台。它将视频下载、媒体处理、关键帧、OCR、ASR、LLM/VLM 分析、检索问答和语言感知 PDF 报告组合为可独立运行、可检查、可缓存、可替换的阶段。

项目优先集成成熟的开源实现，不自行训练或重新实现 ASR、OCR、下载器和基础推理框架。模型相关能力通过稳定的数据模型和 adapter seam 与具体模型解耦。

## 2. 产品范围

### 2.1 已支持

- 使用 yt-dlp 下载单个在线视频、字幕和元数据。
- 按 `<视频ID>-<标题>` 自动归档运行目录。
- 使用 FFmpeg/ffprobe 探测媒体、提取音频和提取场景关键帧。
- 使用可替换的 ASR adapter 生成带时间戳与置信度的转录。
- 使用可替换的 OCR adapter 识别关键帧文本。
- 使用 OpenAI-compatible LLM/VLM endpoint 完成术语发现、低置信度 ASR 校验、视觉理解和分层语言感知总结。
- 将语音、屏幕文字、视觉分析和术语合并为统一时间轴。
- 使用 Embedding 与可选 Reranker 检索时间轴。
- 基于检索证据回答问题，并返回时间戳引用。
- 导出自包含 HTML 或 PDF 报告。
- 中文视频的 PDF 只生成中文总结；其他语言视频保留原文并提供中文翻译。
- 对昂贵阶段复用已有产物；章节总结支持逐章 checkpoint 和断点恢复。

### 2.2 不在当前范围

- Web UI、用户系统、任务队列和多租户服务。
- 实时流媒体处理。
- 模型训练、微调和模型部署编排。
- 自动评估摘要事实正确性的完整评测平台。
- 对所有视频网站登录、会员内容或 DRM 的通用支持。
- 自动翻译完整逐字稿；当前翻译范围为总结和视觉摘要。

## 3. 设计原则

1. **阶段独立**：下载、ASR、OCR、关键帧和分析命令可分别运行。
2. **产物优先**：阶段间通过可检查的 JSON、媒体文件和图片交换数据。
3. **可替换 adapter**：调用方不依赖第三方库的原始返回结构。
4. **保守校正**：ASR 校验不得改写事实，仅应用高置信度且能精确定位的替换。
5. **证据可追溯**：总结、检索和问答应保持时间范围或时间戳。
6. **语言感知**：中文来源不生成重复翻译；其他语言的翻译使用独立 `_zh` 字段且不得覆盖原文。
7. **可恢复**：已有产物必须优先复用；长任务应尽早落盘。
8. **安全归档**：目录迁移不得覆盖现有运行，且必须同步更新 JSON 中的绝对路径。

## 4. 系统结构

```text
在线视频
  │
  ▼
yt-dlp ──► source 视频 / 字幕 / metadata
  │
  ├──► FFmpeg 音频 ──► ASR adapter ──► Transcript
  │
  └──► FFmpeg 关键帧 ──► OCR adapter ──► Frame + OCR text
                                      │
                  LLM 术语/校正 ◄─────┤
                  VLM 视觉分析  ◄─────┤
                                      ▼
                               多模态 Timeline
                                │           │
                     分层语言感知总结    检索与问答
                                │
                                ▼
                         HTML / PDF 报告
```

模块职责：

| 模块 | 职责 |
|---|---|
| `archive.py` | 运行目录命名、迁移和内部路径重写 |
| `ingest.py` | yt-dlp 下载及元数据压缩 |
| `media.py` | ffprobe 探测、FFmpeg 音频提取 |
| `keyframes.py` | 基于场景变化提取关键帧 |
| `asr.py` | ASR interface、注册表和内置 adapter |
| `ocr.py` | OCR interface、注册表和内置 adapter |
| `llm.py` | Chat、Embedding、Reranker HTTP adapter |
| `terminology.py` | 从 metadata 与 OCR 发现术语 |
| `verification.py` | 选择可疑 ASR 片段并保守校正 |
| `vision.py` | 选择代表帧、VLM 分析及中文翻译 |
| `timeline.py` | 合并多模态证据为固定时间窗口 |
| `summarize.py` | 章节总结、总摘要、中文翻译和 checkpoint |
| `retrieval.py` | 向量召回、余弦排序和 Reranker |
| `qa.py` | 基于检索片段生成带引用回答 |
| `pipeline.py` | 分析阶段编排与缓存 |
| `report.py` | 自包含 HTML 和 WeasyPrint PDF |
| `cli.py` | 命令行 interface |

## 5. 运行目录规格

### 5.1 命名

自动下载模式必须使用：

```text
runs/<video_id>-<sanitized_title>/
```

- `video_id` 来自 yt-dlp metadata 的 `id`，Bilibili 分 P ID 需要保留，例如 `BV..._p4`。
- 标题中的 `\ / : * ? " < > |`、控制字符和多余空白必须清理。
- ID 最长 40 字符，标题最长 100 字符。
- 目标目录已存在时必须报错，不得合并或覆盖。
- 下载未完成时使用 `runs/.pending-*`；失败目录保留用于诊断。

### 5.2 目录布局

```text
runs/<id>-<title>/
├── source/
│   ├── source.mp4
│   ├── source.info.json
│   ├── metadata.json
│   └── source.<subtitle extension>
├── audio.wav
├── keyframes/
│   ├── keyframes.json
│   └── frame_*.jpg
├── asr-<engine>-<model>.json
├── ocr-<engine>.json
├── subtitles/
│   ├── *.srt
│   └── *.txt
├── analysis/
│   ├── terminology.json
│   ├── corrections.json
│   ├── verified-transcript.json
│   ├── visual-analysis.json
│   ├── timeline.json
│   ├── chapters.json
│   └── summary.json
└── *.html / *.pdf
```

文件名可因实验配置不同而变化；pipeline 通过 `asr-*.json` 和 `ocr-*.json` 发现输入。

## 6. 数据契约

所有 JSON 使用 UTF-8、保留 Unicode，并以缩进格式写入。写入必须先生成同目录 `.tmp` 文件再原子替换目标文件。

### 6.1 Frame

```json
{
  "path": "/absolute/path/frame_000001.jpg",
  "timestamp": 27.133,
  "scene_score": null,
  "text": ["recognized OCR text"]
}
```

### 6.2 Transcript

```json
{
  "language": "en",
  "duration": 3134.95,
  "engine": "faster-whisper",
  "model": "small",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "original speech",
      "confidence": 0.91,
      "words": [{"start": 0.0, "end": 0.5, "word": "original", "probability": 0.95}]
    }
  ]
}
```

### 6.3 VisualAnalysis

```json
{
  "timestamp": 265.1,
  "summary": "Original-language visual summary.",
  "summary_zh": "视觉总结的中文翻译。",
  "visual_type": "presentation_slide",
  "entities": ["Federal Reserve"],
  "relations": ["FOMC sets interest rates"]
}
```

`summary_zh` 不得替换 `summary`。旧产物缺少中文字段时，vision pipeline 应使用 LLM 补齐并回写。

### 6.4 TimelineSegment

```json
{
  "start": 0.0,
  "end": 60.0,
  "speech": "combined transcript text",
  "screen_text": ["deduplicated OCR text"],
  "visuals": [],
  "terms": ["Federal Reserve"]
}
```

默认窗口为 60 秒。片段按时间升序且不得重叠，最后一个片段可以短于 60 秒。

### 6.5 Chapter

```json
{
  "title": "Original title",
  "title_zh": "中文标题",
  "start": 0.0,
  "end": 300.0,
  "summary": "Original-language summary.",
  "summary_zh": "中文摘要。",
  "key_points": ["Original point"],
  "key_points_zh": ["中文要点"]
}
```

非中文来源的 `key_points` 与 `key_points_zh` 必须按索引一一对应。中文来源只填写原文字段，所有 `_zh` 字段保持空值。

### 6.6 Summary

非中文来源的 `summary.json` 包含：

- `summary` 与 `summary_zh`
- `key_concepts` 与 `key_concepts_zh`
- `takeaways` 与 `takeaways_zh`
- `chapters`（模型生成的报告级章节引用，可为空）

成对数组必须保持相同顺序和语义对应。

中文来源只包含 `summary`、`key_concepts`、`takeaways`，不得生成重复的 `_zh` 字段。所有报告还必须记录 `source_language` 和 `translation_mode`；后者为 `monolingual` 或 `bilingual_zh`。

## 7. 处理阶段

### 7.1 下载

- 必须使用 `--no-playlist`，单次只处理一个视频或一个分 P。
- 必须请求 info JSON、人工字幕和自动字幕。
- 合并输出优先为 MP4。
- `metadata.json` 只保留下游使用的字段：`id`、`webpage_url`、`title`、`description`、`tags`、`chapters`、`duration`、`uploader`、`upload_date`。

### 7.2 媒体与关键帧

- ASR 音频为 16 kHz、单声道、PCM signed 16-bit WAV。
- 关键帧使用 FFmpeg scene score；默认阈值 `0.3`，允许范围 `0..1`。
- 第一帧必须纳入候选。
- 默认最多输出 500 帧。

### 7.3 OCR

统一 interface：

```python
recognize(frames: list[Frame], options: OCROptions, engine: str) -> list[Frame]
```

内置 adapter：

- `rapidocr`
- `paddleocr`，兼容 PaddleOCR 2.x 与 3.x 返回结构

外部 adapter 使用 `register_engine()` 注册。调用方只消费 `Frame.text`。

### 7.4 ASR

统一 interface：

```python
transcribe(audio: Path, options: ASROptions, engine: str) -> Transcript
```

内置 adapter：

- `faster-whisper`
- `openai-whisper`

ASR 应请求 word timestamps。`faster-whisper` 支持独立 hotwords；`openai-whisper` 不支持时必须明确报错，并提示合并到 initial prompt。

### 7.5 术语发现

- 输入为 metadata 和 OCR 帧。
- 输出为 canonical name、aliases、category、confidence 和 evidence。
- evidence 应注明来源，并在适用时携带时间戳。
- 已有 `analysis/terminology.json` 时复用。

### 7.6 ASR 校验

- 只检查 segment confidence `< 0.75` 或任一 word probability `< 0.15` 的片段。
- 默认每批 4 个片段。
- 允许修改名称、术语、同音词、拼写和标点；禁止释义、添加或删除观点。
- correction confidence `< 0.8` 时不得应用。
- `original` 必须是目标 segment 的精确子串，否则不得应用。
- 所有建议写入 `corrections.json`，实际结果写入 `verified-transcript.json`。

### 7.7 视觉分析

- 从 OCR 信息量或语音视觉指代中选择候选帧。
- 默认最多分析 24 帧，相邻帧至少相隔 20 秒。
- 超过上限时必须在全片时间范围内均匀抽样，不能只保留前半段。
- 每帧输出简短原文总结、中文翻译、类型、实体与关系。
- 已有英文视觉分析缺少 `summary_zh` 时，使用 LLM 分批补译。

### 7.8 时间轴

- 将重叠时间窗口内的 ASR、OCR、VLM 和术语合并。
- OCR 文本在窗口内去重并保持首次出现顺序。
- 术语匹配不区分大小写。
- `timeline.json` 是检索、问答和总结的统一证据来源。

### 7.9 分层语言感知总结

- 默认每 5 个一分钟窗口形成一个章节。
- ASR `Transcript.language` 是语言模式的权威来源。
- 中文（`zh`、`cmn`、`yue`）来源只生成中文原文字段，不生成 `_zh` 翻译字段。
- 非中文来源保留视频主要语言，并在 `_zh` 字段提供中文翻译。
- 中文关键概念必须以自然中文为主；必要外文专名可在括号中保留，不得整套概念改写为英文。
- 章节不得引入输入证据之外的事实。
- 每完成一个章节必须立即 checkpoint 到 `chapters.json`。
- 重启时必须先比较 `summary-mode.json`。模式一致且章节字段符合该模式时，才可从已有章节数量对应的时间偏移继续。
- 总摘要基于章节压缩结果生成，而不是重新输入完整逐字稿。
- 结构化提取与总结使用 `enable_thinking=false`，避免隐藏推理耗尽输出预算。

### 7.10 检索与问答

- 检索文档包含时间范围、speech、OCR 和 visual summary。
- 查询使用 retrieval instruction 前缀。
- 先通过 Embedding 余弦相似度召回，默认候选 20 条、返回 5 条。
- 配置 Reranker 时，对候选进行二次排序。
- 问答只能使用传入 evidence；输出 answer、citations 和 insufficient_evidence。
- 每个事实性结论应有 start/end 时间引用。

## 8. 远程模型 interface

项目默认使用统一的 `config/config.yaml`；`config/config-full.yaml` 是全功能示例。`llm` 是总结流程的必需 endpoint；其余三个 endpoint 均可缺省：

| 名称 | HTTP interface | 用途 |
|---|---|---|
| `llm` | `/v1/chat/completions` | 术语、校正、翻译、总结、问答 |
| `vlm` | `/v1/chat/completions` | 关键帧视觉理解 |
| `embedding` | `/v1/embeddings` | 时间轴向量召回 |
| `reranker` | `/v1/rerank` | 候选重排 |

基础模式只使用 `config/config.yaml`。只有显式使用 `--vision` 时才要求 VLM；只有执行 `search` 或 `ask` 时才要求 Embedding，Reranker 始终是可选增强。配置加载器以 YAML 为默认，同时兼容已有 JSON endpoint 配置。

每个 endpoint 配置 `base_url`、`model`、`api_key` 和 `timeout`。环境变量可覆盖：

```text
BBVS_<NAME>_BASE_URL
BBVS_<NAME>_MODEL
BBVS_<NAME>_API_KEY
BBVS_<NAME>_TIMEOUT
```

`<NAME>` 为 `LLM`、`VLM`、`EMBEDDING` 或 `RERANKER`。

模型输出需要 JSON 时必须请求 `response_format={"type":"json_object"}`。客户端允许清理 Markdown code fence，但无效或截断 JSON 必须报错，不得静默猜测。

## 9. 缓存与恢复

`analyze` 的复用规则：

| 产物 | 存在时行为 |
|---|---|
| `terminology.json` | 直接加载 |
| `verified-transcript.json` | 使用校验结果，不重复调用 LLM |
| `visual-analysis.json` | 直接加载；仅补齐缺失中文翻译 |
| `chapters.json` | 语言模式一致时复用已有章节并继续 |
| `summary.json` | 不重复生成总结 |

任何缓存文件存在但 JSON 无效时应直接失败并暴露问题，不应自动忽略损坏产物。

## 10. 报告规格

### 10.1 格式

- HTML 必须自包含，图片以内嵌 data URI 表示。
- PDF 使用 WeasyPrint，A4 页面和可显示中文的 Noto CJK 字体。
- HTML 与 PDF 必须由同一个 document builder 生成。
- 缺少摘要、章节、视觉分析或校验阶段时，报告必须显示明确警告。

### 10.2 内容顺序

1. 封面与 metadata
2. 总体摘要；仅非中文来源附中文翻译
3. 语言感知核心概念
4. 语言感知核心结论
5. 带时间范围的章节；仅非中文来源显示翻译
6. 术语表
7. 图文时间轴
8. 转录校正审计
9. 可选完整逐字稿

### 10.3 图文时间轴

- 每个节点固定对应一个 VLM 已分析的关键帧。
- 节点按 timestamp 升序。
- 每个节点包含图片、可点击时间戳、visual type、原文总结和中文翻译。
- 默认均匀展示全片 12 个节点；`--max-images` 可修改数量。
- 图片与总结必须通过相同 visual timestamp 关联；最多允许与 OCR frame 相差 1 秒。
- 不得把无关章节摘要按最近时间临时绑定到图片。

### 10.4 完整转录

只有指定 `--include-transcript` 时才加入。优先使用 `verified-transcript.json`，否则使用第一个 `asr-*.json`。

## 11. CLI 规格

```text
bbvs run URL_OR_RUN_DIR [--config config/config.yaml]
bbvs download URL [--runs-dir runs | --output-dir DIR]
bbvs rename-run RUN_DIR
bbvs probe VIDEO [--output FILE]
bbvs audio VIDEO --output FILE
bbvs keyframes VIDEO --output-dir DIR [--threshold N] [--max-frames N]
bbvs ocr KEYFRAMES_JSON --output FILE [--engine NAME] [--language LANG]
bbvs asr AUDIO --output FILE [--engine NAME] [ASR options]
bbvs export-transcript TRANSCRIPT --format srt|txt --output FILE
bbvs llm-ping --model MODEL [endpoint options]
bbvs analyze RUN_DIR [--services FILE] [--verify] [--vision] [--summarize]
bbvs search TIMELINE QUERY [--services FILE] [--top-k N]
bbvs ask TIMELINE QUESTION [--services FILE] [--top-k N]
bbvs export-report RUN_DIR --output FILE [--include-transcript] [--max-images N]
```

`bbvs run` 是默认产品入口，负责下载、非 LLM 模型、LLM 分析和报告导出。传入已有运行目录时，按产物存在性跳过下载、音频、关键帧、OCR 和 ASR；分步命令用于实验与诊断。

默认 `download URL` 自动创建归档目录。`--output-dir` 是兼容旧脚本的低级模式，只直接写入指定 source 目录。

## 12. 依赖

基础包无强制 Python 第三方运行时依赖。能力通过 extras 安装：

| Extra | 主要依赖 |
|---|---|
| `download` | yt-dlp |
| `asr-faster-whisper` | faster-whisper |
| `asr-openai-whisper` | openai-whisper |
| `ocr-rapidocr` | rapidocr-onnxruntime、Pillow |
| `ocr-paddleocr` | paddleocr |
| `report` | weasyprint |
| `dev` | pytest |

系统命令依赖：FFmpeg、ffprobe。PDF 中文显示依赖可用的 Noto Sans CJK 字体。

## 13. 错误与安全行为

- 缺失外部命令或可选依赖时，错误必须包含明确安装建议。
- HTTP 非 2xx、超时、网络错误和模型返回结构异常必须转换为可读错误。
- 目录归档不得覆盖已有目标。
- `rename-run` 必须更新所有 JSON 字符串中的旧绝对路径。
- 下载失败的 pending 目录不得自动删除。
- 报告生成不得调用 LLM/VLM；它只读取现有结构化产物。
- HTML 中所有 metadata、模型文本和 OCR 文本必须转义。

## 14. 验收标准

### 14.1 自动化

- `pytest` 全部通过。
- ASR/OCR adapter 可通过 fake 或标准化数据测试，而无需加载实际模型。
- LLM/VLM 测试必须覆盖结构化 JSON 解析、截断/无效输出错误。
- 归档测试必须覆盖非法字符、路径重写和禁止覆盖。
- 报告测试必须覆盖 HTML escaping、中文单语/非中文双语字段和图片—时间戳—总结配对。

### 14.2 端到端

一个完整课程视频运行应满足：

- 视频、metadata 和可用字幕成功下载。
- 音频、关键帧、OCR 与 ASR 产物可独立检查。
- 时间轴覆盖完整视频时长且按时间连续。
- 所有选中视觉节点同时具有 `summary` 和 `summary_zh`。
- 中文章节不得包含重复翻译；非中文章节必须具有原文与中文字段，成对数组长度一致。
- PDF 可由 `pdfinfo` 识别、无加密、A4 页面。
- 抽查封面、语言感知摘要、图文时间轴和末页转录均无溢出或缺字。

当前已验证样例：Bilibili `BV1k9rAYZEzM_p4`，约 52 分钟课程视频，完整运行产物保存在 `runs/` 下对应标题目录。

## 15. 已知限制

- 目前完整低成本阶段尚未由单个 CLI 命令自动串联，媒体、ASR、OCR 仍可按实验需要手动执行。
- LLM/VLM 请求按批次顺序执行，服务排队会显著影响总耗时。
- 视觉翻译按批回写，尚未提供逐条 translation checkpoint。
- ASR 校验只依赖文本、置信度和术语，不重新听取原始音频。
- 场景阈值对讲座 PPT 有效，但不能保证适用于运动镜头或强转场视频。
- 报告的图文节点总结描述对应画面，不等同于该时间窗口的完整语义摘要。
- Embedding 当前对整个 timeline 即时计算，尚未持久化向量索引。
- `runs/` 默认被 Git 忽略，分析产物不作为源码提交。

## 16. 变更规则

- 修改 JSON 字段时必须保持向后兼容，或提供明确迁移函数。
- 新增 ASR/OCR 实现应优先注册 adapter，不修改下游 pipeline。
- 新增模型供应方应适配现有 Chat/Embedding/Reranker interface。
- 改变报告对应关系时，应先修改分析数据契约，再修改展示层。
- 本文档描述已实现的项目级行为；未实现想法应记录为 issue，不得混入本规格作为既成事实。
