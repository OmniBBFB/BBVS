# BBVS Implementation

本文件描述当前实现及其已知限制；它不是规范。产品意图见 `INTENT.md`，强制契约见 `SPEC.md`。

## 当前主链路

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
- `map_content`：高召回抽取教学目标、论点、机制、例子、限制、公式/代码和视觉需求。
- `plan_outline`：按概念依赖规划全局章节，并验证所有 evidence unit 恰好覆盖一次。
- `select_requested_frames`：按视觉需求时间范围选择候选；每个需求的候选与保留数量均可配置，无视频级上限。
- `draft_chapters`：重新组织教学逻辑，并直接读取原始证据而非上一轮短摘要。
- `review_coverage`：独立报告遗漏、失真、无证据论断和连贯性问题。
- `synthesize`：基于完整章节做全局综合，不设置 500 汉字硬上限。

## 字幕

下载阶段保存 `subtitle_tracks`，标记 `manual` 或 `automatic`。ASR 阶段先调用
`preferred_platform_transcript()`；找到可解析 SRT 时直接生成统一 `Transcript`，否则运行配置的 ASR adapter。

旧运行目录如果没有 `subtitle_tracks` 元数据，不会猜测字幕来源，会继续使用 ASR。重新下载可获得新字段。

## 视觉与公式

初始候选仍由 FFmpeg scene score 产生。ContentMap 根据字幕/OCR 生成带时间区间的视觉需求。
每个请求在局部范围内按 OCR 信息量和较晚时间优先，保留至多 3 个不同候选。没有 VLM 时，这些候选
仍以 `semantic_candidate_unverified` 写入图文报告；有 VLM 时再做视觉解释与验证。

当前不转录公式，也没有数学 OCR。公式相关 speech/OCR 会生成 `formula or code slide` 请求，目标是让原始公式
PPT 进入候选与报告。后续可增加固定间隔补采样、感知哈希聚类和 progressive-reveal 稳定性评分。

## Retrieval view

`timeline.py` 的固定窗口、Embedding、Reranker 和 QA 暂时保留，作为可选 Retrieval View。
Authoring module 不使用“每 5 个 Timeline 作为章节”的旧策略。

## 已知限制

- 字幕质量目前只按 manual/automatic 和语言排序，尚未计算覆盖率与异常空洞。
- scene score 仍可能漏掉低变化的板书或静态动画，需要周期采样补充。
- 文本阶段只能根据 OCR/字幕筛选候选；真正的图形关系和完整性仍需要 VLM。
- coverage review 当前只生成审计产物，尚未自动触发定向补写。
- 旧 `summarize.py` 为兼容分步调用暂时保留，不再是默认一行流水线的 authoring 路径。
