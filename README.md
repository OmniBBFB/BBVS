# BBVS

设计与维护文档分为三层：

- [`INTENT.md`](INTENT.md)：产品目标、优先级与取舍。
- [`SPEC.md`](SPEC.md)：必须满足且可验收的行为。
- [`IMPL.md`](IMPL.md)：当前实现、默认值与已知限制。

这是一个面向实验的模块化视频理解项目。当前阶段先独立验证下载、媒体处理、关键帧、OCR 和 ASR；LLM/vLLM 只提供 OpenAI-compatible 客户端，不进入默认处理链。项目优先组合成熟的开源项目，不重新实现已有模型。

完整的已实现行为、数据契约和验收标准见 [SPEC.md](SPEC.md)。

## 一行运行完整流水线

默认配置使用 `config/pipeline-basic.yaml`。它会依次完成下载、音频、关键帧、OCR、ASR、LLM 分析和 PDF：

首次使用先复制 DeepSeek 模板；`config/pipeline-basic.yaml` 已被 Git 忽略，可以保存本地 API Key：

```bash
cp config/pipeline-deepseek.example.yaml config/pipeline-basic.yaml
```

```bash
uv run bbvs run 'VIDEO_URL'
```

指定其他 YAML：

```bash
uv run bbvs run 'VIDEO_URL' --config config/pipeline-basic.yaml
```

任务中断后，传入已经创建的运行目录即可按现有产物续跑：

```bash
uv run bbvs run 'runs/<视频ID>-<标题>' --config config/pipeline-basic.yaml
```

批量总结时仍使用同一个 `run` 命令。TXT 清单每行填写一个 BV 号，也支持完整的
Bilibili 视频 URL、空行和以 `#` 开头的注释：

```bash
uv run bbvs run collections/3278335-bvids.txt \
  --config config/pipeline-deepseek.yaml
```

视频会按清单顺序逐个处理；单个视频失败不会阻塞后续任务。进度持续写入
`runs/batches/3278335-bvids.json`。重新执行同一命令时，已存在的阶段产物会自动复用，
失败或未完成的视频从断点继续。批次中存在失败项时，全部条目处理完毕后命令返回非零状态。

流水线按步骤保存实验产物：`audio/`、`keyframes/<variant>/`、`ocr/<variant>/`、
`asr/<variant>/`、`analysis/<variant>/` 和 `reports/<variant>/`。同一配置再次运行会复用；
切换模型或参数会创建新的 variant，因此可以保留 ASR/OCR 等上游结果，只重跑受影响的下游步骤。

七个步骤均由独立阶段对象执行，并共享一个阶段上下文。每个阶段默认最多尝试 3 次，
失败后等待 2 秒、4 秒再试；确定性的配置错误、缺少输入和依赖缺失会立即失败。
远程分析重试会重新创建模型客户端，并从已写入的 checkpoint 继续。可在 YAML 中调整：

```yaml
retry:
  attempts: 3
  initial_delay: 2
  multiplier: 2
  max_delay: 30
```

分步命令仍然保留，用于单独评测 ASR、OCR 或关键帧参数。

## 环境

- Python 3.11+
- FFmpeg / ffprobe
- `uv`（推荐）

按需安装，避免第一天就拉取所有模型：

```bash
uv sync --extra dev
uv sync --extra download   # 只测试 yt-dlp 时
uv sync --extra asr-faster-whisper
uv sync --extra asr-openai-whisper
uv sync --extra ocr-rapidocr
uv sync --extra ocr-paddleocr
```

## 分模块验证

所有命令相互独立，输出可检查、可复用。

```bash
# 1. 下载视频、字幕和元数据
uv run bbvs download 'VIDEO_URL'
# 自动创建 runs/<视频ID>-<清理后的标题>/source

# 仍可使用旧模式直接指定 source 目录
uv run bbvs download 'VIDEO_URL' --output-dir runs/demo/source

# 本地视频可以从这里直接开始
uv run bbvs probe video.mp4 --output runs/demo/media.json

# 2. 提取 ASR 使用的音频
uv run bbvs audio video.mp4 --output runs/demo/audio.wav

# 3. 关键帧。阈值越低越敏感；PPT 可从 0.15～0.25 开始试
uv run bbvs keyframes video.mp4 \
  --output-dir runs/demo/keyframes --threshold 0.25

# 4. OCR（读取上一步的 keyframes.json）
uv run bbvs ocr runs/demo/keyframes/keyframes.json \
  --engine paddleocr --language ch --output runs/demo/ocr-paddle.json

uv run bbvs ocr runs/demo/keyframes/keyframes.json \
  --engine rapidocr --output runs/demo/ocr-rapid.json

# 5. ASR baseline
uv run bbvs asr runs/demo/audio.wav \
  --engine faster-whisper --model small --language zh \
  --output runs/demo/asr-faster-whisper.json

uv run bbvs asr runs/demo/audio.wav \
  --engine openai-whisper --model small --language zh \
  --output runs/demo/asr-openai-whisper.json

# 6. 带人工上下文的 ASR；用于先验证 context/hotwords 的收益
uv run bbvs asr runs/demo/audio.wav \
  --engine faster-whisper \
  --model small --language zh \
  --initial-prompt '本视频讨论 vLLM、FlashInfer、PagedAttention。' \
  --hotwords 'vLLM FlashInfer PagedAttention' \
  --output runs/demo/asr-context.json
```

首次运行 ASR/OCR 可能下载模型。建议用 3～10 分钟、包含 PPT 和专业术语的视频先做小样本，不要直接测试一小时视频。

## LLM / vLLM 客户端

服务部署以后，用同一客户端测试任何 OpenAI-compatible endpoint：

```bash
uv run bbvs llm-ping \
  --base-url http://127.0.0.1:8000/v1 \
  --model your-model-name
```

`OpenAICompatibleClient` 位于 `bbvs.llm`，没有第三方依赖，后续可用于术语抽取、转录校正和分层总结。

## 可替换后端

ASR 和 OCR 的调用方只依赖各自的小型 interface。内置 adapter：

- ASR：`faster-whisper`、`openai-whisper`
- OCR：`paddleocr`、`rapidocr`

第三方 adapter 可以用 `bbvs.asr.register_engine()` 或
`bbvs.ocr.register_engine()` 注册。adapter 负责把第三方库的参数和返回结构转换成统一的 `Transcript` / `Frame`，下游流水线不接触厂商特有对象。

## 建议的第一阶段验收

1. `yt-dlp`：至少各测一个 YouTube/Bilibili URL，检查视频、标题、描述、章节、字幕是否齐全。
2. 关键帧：为同一视频比较阈值 0.15/0.25/0.35，记录帧数、漏掉的页面和重复帧。
3. OCR：人工标注 30～50 张关键帧，分别统计普通文本和专业实体的识别率。
4. ASR：固定同一模型，比较 baseline、metadata prompt、metadata+OCR hotwords 三组结果。
5. 指标：除 WER/CER 外，单列 Terminology Error Rate（专业词错误数 / 专业词总数）。

自动术语发现、LLM 校正、VLM、Timeline、分层总结和检索现已作为按需阶段提供；默认命令仍只运行便宜阶段，便于分别评测。

## 远程模型与完整分析

基础流水线配置默认位于 `config/pipeline-basic.yaml`；全能力配置为 `config/pipeline-full.yaml`。也可用环境变量覆盖，例如
`BBVS_LLM_BASE_URL`、`BBVS_LLM_MODEL` 和 `BBVS_LLM_API_KEY`。

`services.llm.provider` 支持 `openai`（OpenAI-compatible/vLLM）和 `deepseek`。DeepSeek adapter 会把流水线中的思考开关转换为 DeepSeek 的 `thinking` 参数；模板见 `config/pipeline-deepseek.example.yaml`。

本地 vLLM 与 DeepSeek API 都使用同一份配置结构，不需要两套 schema：本地通常设置
`provider: openai` 和本地 `base_url`；DeepSeek 设置 `provider: deepseek`、官方 `base_url`
及 API Key。`run`/`analyze` 只用 `--config`/`--services` 选择配置，也可通过
`BBVS_LLM_PROVIDER`、`BBVS_LLM_BASE_URL`、`BBVS_LLM_MODEL`、`BBVS_LLM_API_KEY`
逐项覆盖。仅 `llm-ping` 提供直接的 `--provider --base-url --model --api-key` 参数。

已有下载、ASR、OCR 产物后，先生成术语和多模态 Timeline：

```bash
uv run bbvs analyze 'runs/BVxxxx-视频标题'
```

按需启用昂贵阶段：

```bash
uv run bbvs analyze 'runs/BVxxxx-视频标题' \
  --verify --vision --summarize
```

当一个 run 中已有多个 ASR 或 OCR variant 时，单独运行 `analyze` 需用
`--transcript .../transcript.json --ocr-input .../frames.json` 明确选择输入；一行 `run`
命令会根据配置自动选择对应 variant。

各阶段会写入 `run_dir/analysis/`，已有术语文件会复用，从而支持断点恢复。

### 基础模式（不使用 VLM、Embedding、Reranker）

基础模式只需要 `config/pipeline-basic.yaml` 中的 LLM。不要传 `--vision`，也不要调用 `search` 或 `ask`：

```bash
# 安装基础流水线需要的开源依赖
uv sync --extra download --extra asr-faster-whisper --extra ocr-rapidocr --extra report

# 下载后记下输出中的 run_dir 和 video
uv run bbvs download 'VIDEO_URL'

# 以下用实际输出路径替换 RUN_DIR 和 VIDEO
uv run bbvs audio 'VIDEO' --output 'RUN_DIR/audio.wav'
uv run bbvs keyframes 'VIDEO' --output-dir 'RUN_DIR/keyframes' --threshold 0.25
uv run bbvs ocr 'RUN_DIR/keyframes/keyframes.json' \
  --engine rapidocr --output 'RUN_DIR/ocr-rapid.json'
uv run bbvs asr 'RUN_DIR/audio.wav' \
  --engine faster-whisper --model small --device auto \
  --output 'RUN_DIR/asr-faster-whisper-small.json'

# 只有 LLM：术语、ASR 校验、时间轴、语言感知总结；不传 --vision
uv run bbvs analyze 'RUN_DIR' --services config/pipeline-basic.yaml \
  --verify --summarize

uv run bbvs export-report 'RUN_DIR' \
  --output 'RUN_DIR/final-report-basic.pdf' --include-transcript
```

此模式不会请求 VLM、Embedding 或 Reranker。Embedding/Reranker 只在 `search`、`ask` 时需要；VLM 只在显式传入 `--vision` 时需要。无 VLM 时报告保留 metadata、语言感知总结、章节、术语、校正审计和完整转录，但不生成基于视觉分析的图文时间轴。中文视频只生成中文内容；其他语言视频保留原文并附中文翻译。
Timeline 可以直接检索：

```bash
uv run bbvs search \
  'runs/BVxxxx-视频标题/analysis/timeline.json' \
  'What determines the demand for money?'
```

生成只基于检索证据、带时间戳引用的回答：

```bash
uv run bbvs ask \
  'runs/BVxxxx-视频标题/analysis/timeline.json' \
  'What determines the demand for money?'
```

## HTML / PDF 报告

HTML 无额外依赖，并会内嵌代表性关键帧：

```bash
uv run bbvs export-report 'runs/BVxxxx-视频标题' \
  --output 'runs/BVxxxx-视频标题/report.html'
```

PDF 使用 WeasyPrint 和系统 Noto CJK 字体：

```bash
uv sync --extra report
uv run bbvs export-report 'runs/BVxxxx-视频标题' \
  --output 'runs/BVxxxx-视频标题/report.pdf'
```

增加 `--include-transcript` 可附带完整转录。报告只读取已有结构化产物，不会自动调用 LLM/VLM；缺失阶段会在报告中明确标注。
