"""Export source and portable assets without invoking a TeX compiler."""
import shutil
from pathlib import Path

from .teaching_document import SIGNALS, validate


def escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(text))


def export(document, output):
    validate(document)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / (output.stem + "-assets")
    assets.mkdir(exist_ok=True)
    figures = {f["figure_id"]: f for f in document["figures"]}
    parts = [r"\documentclass{ctexart}", r"\usepackage{graphicx,float,tcolorbox}",
             r"\usepackage[margin=2.4cm]{geometry}", r"\begin{document}",
             r"\tableofcontents"]
    if document.get("cover_path") and Path(document["cover_path"]).is_file():
        cover_name = "cover" + Path(document["cover_path"]).suffix
        shutil.copyfile(document["cover_path"], assets / cover_name)
        parts[4:4] = [r"\begin{titlepage}\centering", r"\Huge " + escape(document.get("title", "课程笔记")),
                      r"\par\includegraphics[width=\textwidth]{" + assets.name + "/" + cover_name + "}", r"\end{titlepage}"]
    for chapter in document["chapters"]:
        parts.append(r"\section{" + escape(chapter["title"]) + "}")
        for block in chapter["blocks"]:
            if block["kind"] == "figure":
                f = figures[block["figure_id"]]
                name = f["figure_id"] + Path(f["path"]).suffix
                shutil.copyfile(f["path"], assets / name)
                parts.extend([r"\begin{figure}[H]\centering",
                    r"\includegraphics[width=\textwidth]{" + assets.name + "/" + name + "}",
                    r"\caption{" + escape(f["caption"]) + r"\protect\footnotemark}\end{figure}",
                    r"\footnotetext{视频来源：" + f"{f['start']:.1f}--{f['end']:.1f}" + " 秒。}"])
            else:
                if block.get("title"):
                    parts.append(r"\subsection*{" + escape(block["title"]) + "}")
                boxed = block["kind"] in {"callout", "dialogue"}
                if boxed:
                    color = {"important": "yellow", "knowledge": "blue", "warning": "red", "dialogue": "green"}[SIGNALS[block["signal"]]]
                    parts.append(r"\begin{tcolorbox}[colback=" + color + "!5]")
                parts.append(escape(block.get("text", "")) + "\n")
                if boxed:
                    parts.append(r"\end{tcolorbox}")
    if document.get("synthesis"):
        parts.extend([r"\section{总结与延伸}", escape(document["synthesis"])])
    for appendix in document.get("appendices", []):
        parts.extend([r"\section{" + escape(appendix["title"]) + "}",
                      escape(appendix["text"]).replace("\n", "\n\n")])
    parts.append(r"\end{document}")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output
