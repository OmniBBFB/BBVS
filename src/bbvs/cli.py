from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

from . import asr, ingest, keyframes, media, ocr
from .archive import archive_run
from .errors import BBVSError
from .config import Services
from .io import read_json, write_json
from .llm import OpenAICompatibleClient
from .models import Frame, to_dict
from .transcript import export as export_transcript
from .transcript import from_dict as transcript_from_dict
from .pipeline import VideoPipeline, _timeline
from .retrieval import search
from .qa import answer_question
from .report import ReportOptions, export_report
from .llm import EmbeddingClient, RerankerClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bbvs", description="BBVS 模块化视频理解工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    probe_cmd = sub.add_parser("probe", help="查看本地媒体信息")
    probe_cmd.add_argument("video", type=Path)
    probe_cmd.add_argument("--output", type=Path)

    download_cmd = sub.add_parser("download", help="用 yt-dlp 下载视频、字幕和元数据")
    download_cmd.add_argument("url")
    download_target = download_cmd.add_mutually_exclusive_group()
    download_target.add_argument("--runs-dir", type=Path, default=Path("runs"), help="按 <视频ID>-<标题> 自动归档")
    download_target.add_argument("--output-dir", type=Path, help="旧模式：直接指定 source 输出目录")

    rename_cmd = sub.add_parser("rename-run", help="按 <视频ID>-<标题> 重命名已有运行目录")
    rename_cmd.add_argument("run_dir", type=Path)

    audio_cmd = sub.add_parser("audio", help="提取 16kHz 单声道 WAV")
    audio_cmd.add_argument("video", type=Path)
    audio_cmd.add_argument("--output", type=Path, required=True)

    frames_cmd = sub.add_parser("keyframes", help="按场景变化提取关键帧")
    frames_cmd.add_argument("video", type=Path)
    frames_cmd.add_argument("--output-dir", type=Path, required=True)
    frames_cmd.add_argument("--threshold", type=float, default=0.3)
    frames_cmd.add_argument("--max-frames", type=int, default=500)

    ocr_cmd = sub.add_parser("ocr", help="对 keyframes.json 中的帧运行 OCR")
    ocr_cmd.add_argument("frames", type=Path)
    ocr_cmd.add_argument("--output", type=Path, required=True)
    ocr_cmd.add_argument("--engine", choices=ocr.available_engines(), default="rapidocr")
    ocr_cmd.add_argument("--language", default="ch")

    asr_cmd = sub.add_parser("asr", help="用可替换的 ASR engine 转录音频")
    asr_cmd.add_argument("audio", type=Path)
    asr_cmd.add_argument("--output", type=Path, required=True)
    asr_cmd.add_argument("--engine", choices=asr.available_engines(), default="faster-whisper")
    asr_cmd.add_argument("--model", default="small")
    asr_cmd.add_argument("--device", default="auto")
    asr_cmd.add_argument("--compute-type", default="default")
    asr_cmd.add_argument("--language")
    asr_cmd.add_argument("--initial-prompt")
    asr_cmd.add_argument("--hotwords")

    llm_cmd = sub.add_parser("llm-ping", help="测试 OpenAI-compatible LLM/vLLM 客户端")
    llm_cmd.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    llm_cmd.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    llm_cmd.add_argument("--model", required=True)
    llm_cmd.add_argument("--prompt", default="只回复 pong")

    export_cmd = sub.add_parser("export-transcript", help="将 ASR JSON 导出为 SRT 或纯文本")
    export_cmd.add_argument("transcript", type=Path)
    export_cmd.add_argument("--format", choices=("srt", "txt"), required=True)
    export_cmd.add_argument("--output", type=Path, required=True)

    pipeline_cmd = sub.add_parser("analyze", help="生成术语、Timeline，并按需校正/视觉分析/总结")
    pipeline_cmd.add_argument("run_dir", type=Path)
    pipeline_cmd.add_argument("--services", type=Path, default=Path("config/services.json"))
    pipeline_cmd.add_argument("--verify", action="store_true")
    pipeline_cmd.add_argument("--vision", action="store_true")
    pipeline_cmd.add_argument("--summarize", action="store_true")

    search_cmd = sub.add_parser("search", help="用 Embedding + Reranker 检索 Timeline")
    search_cmd.add_argument("timeline", type=Path)
    search_cmd.add_argument("query")
    search_cmd.add_argument("--services", type=Path, default=Path("config/services.json"))
    search_cmd.add_argument("--top-k", type=int, default=5)

    ask_cmd = sub.add_parser("ask", help="检索 Timeline 并生成带时间戳证据的回答")
    ask_cmd.add_argument("timeline", type=Path)
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--services", type=Path, default=Path("config/services.json"))
    ask_cmd.add_argument("--top-k", type=int, default=5)

    report_cmd = sub.add_parser("export-report", help="将现有分析产物导出为自包含 HTML 或 PDF")
    report_cmd.add_argument("run_dir", type=Path)
    report_cmd.add_argument("--output", type=Path, required=True)
    report_cmd.add_argument("--include-transcript", action="store_true")
    report_cmd.add_argument("--max-images", type=int, default=12)
    return parser


def _emit(value: object, output: Path | None = None) -> None:
    payload = to_dict(value) if hasattr(value, "__dataclass_fields__") else value
    if output:
        write_json(output, payload)
        print(output)
    else:
        import json

        print(json.dumps(payload, ensure_ascii=False, indent=2))


def run(args: argparse.Namespace) -> None:
    if args.command == "probe":
        _emit(media.probe(args.video), args.output)
    elif args.command == "download":
        if args.output_dir:
            path, metadata = ingest.download(args.url, args.output_dir)
            _emit({"video": str(path), "metadata": metadata})
        else:
            run_dir, path, metadata = ingest.download_to_run(args.url, args.runs_dir)
            _emit({"run_dir": str(run_dir), "video": str(path), "metadata": metadata})
    elif args.command == "rename-run":
        print(archive_run(args.run_dir))
    elif args.command == "audio":
        print(media.extract_audio(args.video, args.output))
    elif args.command == "keyframes":
        frames = keyframes.extract_keyframes(args.video, args.output_dir, args.threshold, args.max_frames)
        output = args.output_dir / "keyframes.json"
        write_json(output, [asdict(frame) for frame in frames])
        print(output)
    elif args.command == "ocr":
        frames = [Frame(**item) for item in read_json(args.frames)]
        recognized = ocr.recognize(
            frames, ocr.OCROptions(language=args.language), engine=args.engine
        )
        write_json(args.output, [asdict(frame) for frame in recognized])
        print(args.output)
    elif args.command == "asr":
        transcript = asr.transcribe(
            args.audio,
            asr.ASROptions(
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
                initial_prompt=args.initial_prompt,
                hotwords=args.hotwords,
            ),
            engine=args.engine,
        )
        _emit(transcript, args.output)
    elif args.command == "llm-ping":
        client = OpenAICompatibleClient(args.base_url, args.api_key)
        print(client.chat(model=args.model, messages=[{"role": "user", "content": args.prompt}]))
    elif args.command == "export-transcript":
        transcript = transcript_from_dict(read_json(args.transcript))
        print(export_transcript(transcript, args.output, args.format))
    elif args.command == "analyze":
        pipeline = VideoPipeline(Services.load(args.services))
        print(pipeline.analyze(args.run_dir, verify=args.verify, vision=args.vision, summarize=args.summarize))
    elif args.command == "search":
        services = Services.load(args.services)
        if not services.embedding:
            raise ValueError("search 需要配置 embedding endpoint")
        hits = search(
            args.query, _timeline(read_json(args.timeline)),
            EmbeddingClient(services.embedding.base_url, services.embedding.api_key, services.embedding.timeout),
            services.embedding.model,
            RerankerClient(services.reranker.base_url, services.reranker.api_key, services.reranker.timeout) if services.reranker else None,
            services.reranker.model if services.reranker else None, top_k=args.top_k,
        )
        _emit([{"index": hit.index, "score": hit.score, "segment": asdict(hit.segment)} for hit in hits])
    elif args.command == "ask":
        services = Services.load(args.services)
        if not services.embedding:
            raise ValueError("ask 需要配置 embedding endpoint")
        embedding = EmbeddingClient(services.embedding.base_url, services.embedding.api_key, services.embedding.timeout)
        reranker = RerankerClient(services.reranker.base_url, services.reranker.api_key, services.reranker.timeout) if services.reranker else None
        hits = search(
            args.question, _timeline(read_json(args.timeline)), embedding, services.embedding.model,
            reranker, services.reranker.model if services.reranker else None, top_k=args.top_k,
        )
        llm = OpenAICompatibleClient(services.llm.base_url, services.llm.api_key, services.llm.timeout)
        _emit(answer_question(args.question, hits, llm, services.llm.model))
    elif args.command == "export-report":
        options = ReportOptions(
            include_transcript=args.include_transcript,
            max_images=args.max_images,
        )
        print(export_report(args.run_dir, args.output, options))


def main() -> None:
    try:
        run(_parser().parse_args())
    except (BBVSError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
