#!/usr/bin/env python3
"""Render Japanese study-design diagrams; these contain no measured results.

Run from the repository root:
    .venv/bin/python scripts/visualize_concepts.py

The default exports are SVG (outlined Japanese text), PNG, and PDF (embedded
font subsets). A system Japanese font is required; see reports/README.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import shlex

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
PALETTE = {
    "EDC": "#6B7280",
    "linear": "#2677B8",
    "ESN": "#D97706",
    "DBP": "#25846C",
}
INK = "#202B3A"
MUTED = "#526174"
BORDER = "#CDD6DF"
PAPER = "#FFFFFF"
SOFT = "#F5F7FA"
DISPERSION = "#2677B8"
KERR = "#7C60A3"
ASE = "#B95D40"


@dataclass(frozen=True)
class JapaneseFont:
    family: str
    regular: Path
    bold: Path

    def properties(self, size: float, bold: bool = False) -> fm.FontProperties:
        return fm.FontProperties(fname=self.bold if bold else self.regular, size=size)


def select_font(explicit: Path | None) -> JapaneseFont:
    if explicit is not None:
        regular = explicit.expanduser().resolve()
        if not regular.is_file():
            raise SystemExit(f"Japanese font not found: {explicit}")
        family = fm.FontProperties(fname=regular).get_name()
        font = JapaneseFont(family, regular, regular)
    else:
        for family in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic"):
            try:
                regular = Path(fm.findfont(fm.FontProperties(family=family), fallback_to_default=False))
                bold = Path(
                    fm.findfont(fm.FontProperties(family=family, weight="bold"), fallback_to_default=False)
                )
            except ValueError:
                continue
            font = JapaneseFont(family, regular, bold)
            break
        else:
            raise SystemExit(
                "A Japanese font is required. Install fonts-noto-cjk or fonts-ipaexfont-gothic, "
                "or pass --font /path/to/a/Japanese-font.ttf. See reports/README.md."
            )
    required = set("光通信学習評価分散雑音正解復元偏重")
    available = fm.get_font(str(font.regular)).get_charmap()
    missing = [char for char in required if ord(char) not in available]
    if missing:
        raise SystemExit(f"Font {font.family!r} lacks Japanese glyphs: {''.join(missing)}")
    return font


class Diagram:
    """A diagram canvas with text-bound checks before any export."""

    def __init__(self, font: JapaneseFont, size: tuple[float, float], dpi: int):
        self.font = font
        self.figure, self.ax = plt.subplots(figsize=size, dpi=dpi)
        self.figure.patch.set_facecolor(PAPER)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set(xlim=(0, 100), ylim=(0, 100))
        self.ax.set_axis_off()
        self.text_bounds: list[tuple[object, tuple[float, float, float, float] | None]] = []
        self.glyphs = fm.get_font(str(font.regular)).get_charmap()

    def text(
        self,
        x: float,
        y: float,
        label: str,
        size: float = 12,
        *,
        color: str = INK,
        bold: bool = False,
        ha: str = "center",
        va: str = "center",
        bounds: tuple[float, float, float, float] | None = None,
    ):
        missing = {char for char in label if not char.isspace() and ord(char) not in self.glyphs}
        if missing:
            raise ValueError(f"Font {self.font.family} lacks glyphs: {''.join(sorted(missing))}")
        artist = self.ax.text(
            x,
            y,
            label,
            fontproperties=self.font.properties(size, bold),
            color=color,
            ha=ha,
            va=va,
            linespacing=1.55,
        )
        self.text_bounds.append((artist, bounds))
        return artist

    def box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = SOFT,
        edge: str = BORDER,
        radius: float = 1.3,
        linewidth: float = 1.2,
    ):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=fill,
        )
        self.ax.add_patch(patch)
        return (x, y, width, height)

    def arrow(self, start, end, *, color: str = MUTED, width: float = 1.6):
        self.ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=width,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )

    def line(self, start, end, color: str = BORDER, width: float = 1):
        self.ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=width)

    def validate_layout(self):
        self.figure.canvas.draw()
        renderer = self.figure.canvas.get_renderer()
        canvas = self.figure.bbox
        for artist, bounds in self.text_bounds:
            extent = artist.get_window_extent(renderer)
            if bounds is None:
                container = canvas
            else:
                x, y, w, h = bounds
                points = self.ax.transData.transform(((x, y), (x + w, y + h)))
                container = Bbox.from_extents(*points.reshape(-1))
            tolerance = 0.8
            if (
                extent.x0 < container.x0 - tolerance
                or extent.y0 < container.y0 - tolerance
                or extent.x1 > container.x1 + tolerance
                or extent.y1 > container.y1 + tolerance
            ):
                raise ValueError(f"Text exceeds its diagram bounds: {artist.get_text()!r}")


def link_overview(font: JapaneseFont, dpi: int) -> Diagram:
    diagram = Diagram(font, (13.6, 7.6), dpi)
    diagram.text(4, 94, "長距離の光通信を、ソフトウェアで再現", 22, bold=True, ha="left")
    diagram.text(
        4,
        87,
        "主評価は 1600 km。単一波長・理想的な送受信機を使ったシミュレーション。",
        12.5,
        color=MUTED,
        ha="left",
    )

    message = diagram.box(3, 57, 9, 22, fill="#F5F7FA")
    transmitter = diagram.box(15, 57, 14, 22, fill="#EEF4FA", edge="#AECBE4")
    channel = diagram.box(32, 57, 23, 22, fill="#F6F3FA", edge="#CBBBDD")
    receiver = diagram.box(58, 57, 13, 22, fill="#EEF4FA", edge="#AECBE4")
    reservoir = diagram.box(74, 57, 14, 22, fill="#FFF5E8", edge="#E7BA7D")
    recovered = diagram.box(91, 57, 7, 22, fill="#EDF6F2", edge="#ADCEC1")

    diagram.text(7.5, 71.8, "元の\nメッセージ", 12.5, bold=True, bounds=message)
    diagram.text(7.5, 61.8, "ビット", 11, color=MUTED, bounds=message)
    diagram.text(22, 73, "光を送る", 13, bold=True, bounds=transmitter)
    diagram.text(22, 64, "２つの偏光\n16QAM", 12, color=MUTED, bounds=transmitter)
    diagram.text(43.5, 73.3, "光ファイバ ＋ 増幅器", 13, bold=True, bounds=channel)
    diagram.text(43.5, 64, "80 km を 20回くり返す", 12, color=MUTED, bounds=channel)
    diagram.text(64.5, 73.1, "光を受け取る", 12.5, bold=True, bounds=receiver)
    diagram.text(64.5, 63.7, "コヒーレント受信\n分散を補正", 11.1, color=MUTED, bounds=receiver)
    diagram.text(81, 73.1, "リザバー", 13, bold=True, bounds=reservoir)
    diagram.text(81, 63.7, "ソフトウェアで\nひずみを補償", 11.5, color=MUTED, bounds=reservoir)
    diagram.text(94.5, 68, "復元\nビット", 12.4, bold=True, bounds=recovered)
    for left, right in ((12, 15), (29, 32), (55, 58), (71, 74), (88, 91)):
        diagram.arrow((left, 68), (right, 68))

    diagram.line((43.5, 56.6), (43.5, 50), color="#B9A6CE", width=1.1)
    diagram.text(50, 44.7, "伝送の途中で加わる、３つの影響", 15, bold=True)
    cards = [
        (4, DISPERSION, "分散", "パルスが時間方向に広がる", "隣り合う信号が重なり、\n見分けにくくなる。"),
        (36, KERR, "Kerr 効果", "光の強さで、ひずみが変わる", "送る光の強さが変わると、\n信号の形や位相も変わる。"),
        (68, ASE, "増幅器の雑音（ASE）", "ランダムな雑音が加わる", "弱くなった光を強めるたびに、\n余分な雑音も加わる。"),
    ]
    for x, color, heading, summary, explanation in cards:
        bounds = diagram.box(x, 13, 28, 24, fill=PAPER, edge=BORDER)
        diagram.line((x + 2, 34), (x + 26, 34), color=color, width=3.5)
        diagram.text(x + 14, 29.5, heading, 13, bold=True, color=color, bounds=bounds)
        diagram.text(x + 14, 23.4, summary, 12.1, bold=True, bounds=bounds)
        diagram.text(x + 14, 17.6, explanation, 11.5, color=MUTED, bounds=bounds)

    diagram.text(
        50,
        5.8,
        "補償後も誤りは残りうる。復元ビットを正解と比べ、誤り率（BER）で評価する。",
        12.2,
        color=MUTED,
    )
    return diagram


def learning_workflow(font: JapaneseFont, dpi: int) -> Diagram:
    diagram = Diagram(font, (13.6, 8.5), dpi)
    diagram.text(4, 95, "答え合わせに使うデータは、学習から分ける", 22, bold=True, ha="left")
    diagram.text(
        4,
        88.5,
        "送信電力ごとに別のモデルを学習・選択し、最後に固定して比較する。",
        12.5,
        color=MUTED,
        ha="left",
    )

    specifications = [
        (4, "① 学習", "A", "#FFF5E8", PALETTE["ESN"]),
        (36, "② 設定選択", "B", "#EEF4FA", PALETTE["linear"]),
        (68, "③ 最終テスト", "C", "#EDF6F2", PALETTE["DBP"]),
    ]
    for x, heading, split, fill, color in specifications:
        bounds = diagram.box(x, 39, 28, 43, fill=PAPER)
        diagram.box(x + 0.6, 74.2, 26.8, 7.1, fill=fill, edge=fill, radius=0.7)
        diagram.text(x + 14, 77.7, heading, 15, bold=True, color=color, bounds=bounds)
        diagram.text(
            x + 14,
            66.9,
            f"独立したメッセージ {split}\n＋ 独立した雑音 {split}",
            12,
            bounds=bounds,
        )
        diagram.line((x + 2, 60.2), (x + 26, 60.2))

    diagram.text(18, 54.2, "学習用だけで\n受信信号を校正・正規化", 12.2, bold=True, bounds=(4, 39, 28, 43))
    diagram.text(18, 43.8, "リザバーの出力の重みを学習", 11.8, color=MUTED, bounds=(4, 39, 28, 43))
    diagram.text(50, 53.4, "候補モデルで予測\n正解と比べ、設定を選ぶ", 12.2, bold=True, bounds=(36, 39, 28, 43))
    diagram.text(50, 43.8, "ここで選択を終える", 11.8, color=MUTED, bounds=(36, 39, 28, 43))
    diagram.text(82, 53.4, "選んだモデルを固定\n同じ受信信号で４方式を比較", 12.2, bold=True, bounds=(68, 39, 28, 43))
    diagram.text(82, 43.8, "正解は採点だけに使う", 11.8, color=MUTED, bounds=(68, 39, 28, 43))
    diagram.arrow((32.3, 48.5), (35.7, 48.5))
    diagram.arrow((64.3, 48.5), (67.7, 48.5))

    diagram.text(
        50,
        33.7,
        "校正・正規化の値は学習用から引き継ぐ。状態はデータ集合・フレームの境界でリセット。",
        11.8,
        color=MUTED,
    )

    inset = diagram.box(4, 6.5, 59, 22, fill=SOFT)
    diagram.text(6, 24.4, "リザバーで学習する場所", 13.5, bold=True, ha="left", bounds=inset)
    diagram.text(9, 14.3, "受信\n信号", 11.5, bounds=inset)
    diagram.arrow((12, 14), (15, 14), width=1.3)
    fixed = diagram.box(15.3, 9.4, 17.6, 11.9, fill=PAPER, edge="#BFCBD5", radius=0.9)
    diagram.text(24.1, 19.1, "内部の接続は固定", 10.8, bold=True, bounds=fixed)
    nodes = [(18, 12.7), (22.3, 15.3), (26.9, 12.1), (30.3, 15.1)]
    for first, second in ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0)):
        diagram.line(nodes[first], nodes[second], color="#AEBCCA", width=0.9)
    for center in nodes:
        diagram.ax.add_patch(Circle(center, 0.68, facecolor="#7C91A4", edgecolor=PAPER, linewidth=0.8))
    diagram.arrow((33, 14), (36.8, 14), color=PALETTE["ESN"], width=1.3)
    readout = diagram.box(37.2, 9.4, 16.8, 11.9, fill="#FFF5E8", edge="#E7BA7D", radius=0.9)
    diagram.text(45.6, 17.3, "出力の重み", 12, bold=True, color=PALETTE["ESN"], bounds=readout)
    diagram.text(45.6, 12.8, "ここだけ学習", 11.2, color=MUTED, bounds=readout)
    diagram.arrow((54.3, 14), (57.4, 14), width=1.3)
    diagram.text(59.4, 14.3, "予測", 11.5, bounds=inset)

    legend = diagram.box(66, 6.5, 30, 22, fill=SOFT)
    diagram.text(68, 24.4, "最終テストの比較方式", 13.5, bold=True, ha="left", bounds=legend)
    comparisons = [
        (68.3, 17.8, "EDC", "分散補正のみ"),
        (82.4, 17.8, "linear", "線形補償"),
        (68.3, 11.4, "ESN", "リザバー補償"),
        (82.4, 11.4, "DBP", "逆向き伝搬"),
    ]
    for x, y, name, label in comparisons:
        diagram.line((x, y + 0.2), (x + 1.8, y + 0.2), color=PALETTE[name], width=3)
        diagram.text(x + 2.5, y, name, 11.9, bold=True, ha="left", color=PALETTE[name], bounds=legend)
        diagram.text(x, y - 3.1, label, 10.7, ha="left", color=MUTED, bounds=legend)

    diagram.text(
        50,
        2.7,
        "未使用のテストデータで誤り率（BER）を測る。テストの正解でモデルを調整しない。",
        11.8,
        color=MUTED,
    )
    return diagram


FIGURES = {
    "link_overview": {
        "renderer": link_overview,
        "title": "長距離光通信のシミュレーション",
        "description": "1600 km の模擬通信路と分散・Kerr 効果・増幅器の ASE 雑音を説明する設計図。実測波形・性能の主張を含まない。",
        "caption_ja": "図1　シミュレーションの流れ。光ファイバと増幅器で生じる３つの影響の補償を試み、復元ビットの誤り率で性能を測る。",
    },
    "learning_workflow": {
        "renderer": learning_workflow,
        "title": "学習・設定選択・最終テストの分離",
        "description": "送信電力ごとの学習、独立したメッセージと雑音、学習用だけの校正・正規化、固定リザバーと学習する出力、４方式の公平なテスト比較を説明する設計図。",
        "caption_ja": "図2　学習と評価の分け方。設定を選んだ後はモデルを固定し、独立したテスト信号で４方式を比較する。テストの正解は採点だけに使う。",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "figures")
    parser.add_argument("--format", choices=("svg", "png", "pdf"), nargs="+", default=["svg", "png", "pdf"])
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--font", type=Path, help="Explicit Japanese .ttf/.otf/.ttc font; font file is not copied")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside this repository, so manifest paths remain relative")
    output_dir.mkdir(parents=True, exist_ok=True)
    font = select_font(args.font)
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "path",
            "svg.hashsalt": "optical-comm-lab-concepts-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    command = [".venv/bin/python", "scripts/visualize_concepts.py"]
    if output_dir != ROOT / "reports" / "figures":
        command.extend(["--output-dir", str(output_dir.relative_to(ROOT))])
    if args.dpi != 160:
        command.extend(["--dpi", str(args.dpi)])
    if args.format != ["svg", "png", "pdf"]:
        command.extend(["--format", *args.format])
    if args.font is not None:
        # Do not disclose a machine-specific absolute path in saved artifacts.
        explicit_font = args.font.expanduser().resolve()
        font_argument = str(explicit_font.relative_to(ROOT)) if explicit_font.is_relative_to(ROOT) else "<matching-Japanese-font-file>"
        command.extend(["--font", font_argument])
    manifest = {
        "schema_version": 1,
        "kind": "study_design_schematics_not_experimental_results",
        "generator": {"path": "scripts/visualize_concepts.py", "sha256": sha256(Path(__file__))},
        "regeneration_command": shlex.join(command),
        "source_documents": [
            {"path": path, "sha256": sha256(ROOT / path)}
            for path in ("docs/requirements.md", "docs/experiment-plan.md")
        ],
        "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "font": {
            "family": font.family,
            "regular_file": font.regular.name,
            "bold_file": font.bold.name,
            "regular_sha256": sha256(font.regular),
            "bold_sha256": sha256(font.bold),
            "svg_text": "glyph outlines",
            "pdf_text": "embedded TrueType subsets",
        },
        "comparison_palette": PALETTE,
        "dpi": args.dpi,
        "figures": [],
    }
    for name, specification in FIGURES.items():
        diagram = specification["renderer"](font, args.dpi)
        diagram.validate_layout()
        record = {
            "name": name,
            "title_ja": specification["title"],
            "description_ja": specification["description"],
            "caption_ja": specification["caption_ja"],
            "size_inches": list(diagram.figure.get_size_inches()),
            "layout_check": "all text inside canvas and assigned boxes",
            "outputs": [],
        }
        for extension in dict.fromkeys(args.format):
            path = output_dir / f"{name}.{extension}"
            if extension == "svg":
                metadata = {"Title": specification["title"], "Description": specification["description"], "Date": None}
            elif extension == "pdf":
                metadata = {"Title": specification["title"], "Subject": specification["description"], "CreationDate": None, "ModDate": None}
            else:
                metadata = {"Title": specification["title"], "Description": specification["description"]}
            diagram.figure.savefig(path, dpi=args.dpi, facecolor=PAPER, metadata=metadata)
            record["outputs"].append(
                {"path": str(path.relative_to(ROOT)), "format": extension, "sha256": sha256(path), "bytes": path.stat().st_size}
            )
        plt.close(diagram.figure)
        manifest["figures"].append(record)
    (output_dir / "concepts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(FIGURES)} study-design diagrams ({', '.join(dict.fromkeys(args.format))}); font: {font.family}")


if __name__ == "__main__":
    main()
