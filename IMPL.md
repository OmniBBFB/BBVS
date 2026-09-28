# BBVS Implementation

本文件描述当前实现及其已知限制；它不是规范。产品意图见 `INTENT.md`，强制契约见 `SPEC.md`。

## 当前主链路

默认 `run` interface 只接收 BV 号。`runner.resolve_run_dir` 在内部隐藏标题目录查找：唯一匹配时续跑，
没有匹配时由 DownloadStage 构造 Bilibili URL；调用者不需要知道运行目录名。

```text
yt-dlp
  ├─ 平台字幕解析（manual > automatic）
  └─ 无可用字幕时 FFmpeg audio → faster-whisper/openai-whisper

FFmpeg scene frames → OCR
Transcript + OCR → EvidenceUnit → ContentMap → Global Outline
                                    │               │
                                    └─ visual requests
                                             ↓
                                 request-local frame candidates → VLM
                                             ↓
                              chapter drafting → coverage review → synthesis
```

## Authoring module

`bbvs.authoring` 是写作主链路的深模块。它隐藏以下实现步骤：

- `build_evidence_units`：产生稳定的原子证据封装；默认约 120 秒，但明确不代表章节。
- `map_content`：抽取原子 Knowledge Item；`form/role/importance` 使用固定英文枚举，`content_label` 使用模型生成的中文领域词汇，同时保留公式/代码和视觉需求。
- `plan_outline`：跨 Evidence Unit 聚合为读者级 Report Topic；每章最多 4 个主题，每主题最多 3 个核心证据和 2 个辅助证据，并验证所有 evidence unit 恰好覆盖一次。
- `select_requested_frames`：按视觉需求时间范围选择候选；每个需求的候选与保留数量均可配置，无视频级上限。
- `draft_chapters`：只读取大纲主题显式选择的核心项与辅助项；同一主题内综合表达，不逐项复述，输出纯文本章节。
- `review_coverage`：只审查已选 Report Topic 的遗漏、推理链、失真、无证据论断和连贯性，不要求补回未选候选项。
- `synthesize`：以核心知识和跨章联系为主生成纯文本全局综合，不逐章复述。

长文本产物写入 `chapters/*.txt` 和 `report.txt`。`chapters-manifest.json` 仅由程序维护文件名、标题和时间范围，
用于断点续跑；结构化抽取、outline 和 coverage review 仍使用 JSON。报告优先读取纯文本产物，同时只读兼容旧的
Markdown、`chapters.json` 与 `summary.json`。

`authoring-version.json` 和完成状态记录 `AUTHORING_PROMPT_VERSION`。版本改变时复用原始 Evidence Unit，重建
Content Map 及其下游 authoring 产物；runner 的 analysis variant 同时包含该版本，因此默认流程不会混用旧契约。

## 字幕

下载阶段保存 `subtitle_tracks`，标记 `manual` 或 `automatic`。ASR 阶段先调用
`preferred_platform_transcript()`；找到可解析 SRT 时直接生成统一 `Transcript`，否则运行配置的 ASR adapter。

旧运行目录如果没有 `subtitle_tracks` 元数据，不会猜测字幕来源，会继续使用 ASR。重新下载可获得新字段。
Bilibili 登录字幕可通过 `download.cookies_from_browser` 或 `download.cookies_file` 显式启用；两者互斥，默认均关闭。
Linux Chrome 通常配置为 `chrome+gnomekeyring`，`download` extra 包含其 cookie 解密依赖 `secretstorage`。

## 视觉与公式

初始候选仍由 FFmpeg scene score 产生。ContentMap 根据字幕/OCR 生成带时间区间的视觉需求。
每个请求在局部范围内按 OCR 信息量和较晚时间优先选择候选，再交给 VLM 做视觉解释。
没有配置或启用 VLM 时，整个图文时间轴关闭，不生成 `semantic_candidate_unverified`，也不展示历史残留的视觉候选。

当前不转录公式，也没有数学 OCR。公式相关 speech/OCR 会生成 `formula or code slide` 请求，目标是让原始公式
PPT 进入候选与报告。后续可增加固定间隔补采样、感知哈希聚类和 progressive-reveal 稳定性评分。

## Retrieval view

`timeline.py` 的固定窗口、Embedding、Reranker 和 QA 暂时保留，作为可选 Retrieval View。
Authoring module 不使用“每 5 个 Timeline 作为章节”的旧策略。

## Results Web

`bbvs.web.RunCatalog` 是读取 `runs/` 的深模块：调用者只使用 `all/search/get/asset` interface，目录扫描、
新旧报告发现、元数据归一化和路径越界防护都隐藏在实现中。`JobManager` 隐藏 URL/BV 归一化、后台线程、
同 BV 去重和进度快照，并通过已有 runner 写入正常运行产物。`bbvs serve` 使用标准库线程 HTTP server，
提供服务端渲染的搜索、详情、报告、关键帧和任务进度页面，无前端构建步骤。

## 已知限制

- 字幕质量目前只按 manual/automatic 和语言排序，尚未计算覆盖率与异常空洞。
- scene score 仍可能漏掉低变化的板书或静态动画，需要周期采样补充。
- 文本阶段只能根据 OCR/字幕筛选候选；真正的图形关系和完整性仍需要 VLM。
- coverage review 当前只生成审计产物，尚未自动触发定向补写。
- 旧 `summarize.py` 为兼容分步调用暂时保留，不再是默认一行流水线的 authoring 路径。
