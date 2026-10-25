#!/usr/bin/env python3
"""Build the Japanese visual report and an offline HTML from verified saved results.

No new model is trained and no test result is selected by this command. Review
and normal verification records are separate inputs supplied by the coordinator.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import platform
import shlex

from .visualize_results import (ROOT, LABELS, METHODS, display_path, load_run, read_json, sha256)


CSS = """
:root{color-scheme:light;--ink:#202b3a;--muted:#526174;--edge:#d8e0e8;--paper:#fff;--soft:#f4f7fa;--orange:#d97706}
*{box-sizing:border-box}body{margin:0;background:var(--soft);color:var(--ink);font:16px/1.85 system-ui,-apple-system,'Noto Sans CJK JP','Yu Gothic',sans-serif}
main{max-width:1120px;margin:0 auto;padding:48px 52px 64px;background:var(--paper)}h1{font-size:2rem;line-height:1.45;margin:0 0 20px}h2{font-size:1.4rem;margin:42px 0 14px;border-bottom:1px solid var(--edge);padding-bottom:8px}p{margin:14px 0}a{color:#216aa0;text-underline-offset:.18em}.lead{font-size:1.15rem;border-left:4px solid var(--orange);padding:15px 22px;background:#fff9f0}.scope{font-size:.9rem;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}.card{background:var(--soft);padding:16px 20px;border:1px solid var(--edge);border-radius:9px}.card strong{display:block;font-size:1.5rem;line-height:1.5}.card span{color:var(--muted);font-size:.88rem}figure{margin:24px 0 32px;break-inside:avoid}figure img{width:100%;height:auto;display:block}figcaption{margin-top:9px;font-size:.9rem;line-height:1.75;color:var(--muted)}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:.86rem;line-height:1.7}th{background:var(--soft);font-weight:600;text-align:left}th,td{padding:10px 12px;border-bottom:1px solid var(--edge);vertical-align:top}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}small{color:var(--muted);font-size:.83rem}pre{overflow-x:auto;background:#f0f4f7;border-radius:6px;padding:18px;font-size:.84rem;line-height:1.6}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}li{margin:7px 0}details{border:1px solid var(--edge);border-radius:8px;margin:14px 0;padding:0 18px}summary{cursor:pointer;font-weight:600;padding:15px 0;line-height:1.6}details[open] summary{border-bottom:1px solid var(--edge);margin-bottom:14px}summary:focus-visible{outline:2px solid #2677b8;outline-offset:4px}footer{margin-top:44px;border-top:1px solid var(--edge);padding-top:18px;font-size:.88rem;color:var(--muted)}.pilot{background:#fff2ee;padding:14px;border:1px solid #d97757}
@media(max-width:700px){main{padding:28px 18px}h1{font-size:1.55rem}.cards{grid-template-columns:1fr}.card{padding:12px 16px}table{font-size:.8rem}th,td{padding:8px}}
@media print{@page{size:A4;margin:14mm}body{background:white;font-size:10pt;line-height:1.6}main{max-width:none;padding:0}h1{font-size:19pt}h2{font-size:13pt;break-after:avoid;margin-top:24px}p{orphans:3;widows:3}.cards{gap:8px}.card strong{font-size:15pt}details{padding:0 8px;break-inside:auto}summary{padding:9px 0;font-size:10pt}figure{margin:16px 0}figure img{max-height:85mm;object-fit:contain}a{color:inherit}pre{white-space:pre-wrap;overflow:visible;font-size:8pt}table{font-size:8pt}.table-wrap{overflow:visible}.lead{padding:10px 16px}footer{font-size:8pt}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def e(value):
    return html.escape(str(value), quote=True)


def rate(row: dict) -> str:
    if row["bit_errors"] == 0:
        return f"0（95%上限 {row['ber_ci_high']:.2e}）"
    return f"{row['ber']:.2e}"


def extra_time(row: dict) -> str:
    seconds = row["inference_seconds"] / row["n_frames"]
    return f"{seconds*1000:.3g} ms" if seconds < .1 else f"{seconds:.3g} s"


def relpath(path: Path, output: Path) -> str:
    import os
    return Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()


def load_waveform_diagnostics(run):
    """Link completed diagnostics only after checking their recorded provenance."""
    manifest_path = run.path / "waveform_diagnostics/manifest.json"
    if not manifest_path.is_file():
        return None, []
    record = read_json(manifest_path)
    if record.get("schema_version") != 1 or record.get("run_id") != run.manifest["run_id"]:
        raise ValueError("Waveform diagnostics refer to a different run/schema")
    if record.get("original_source_hash") != run.manifest["compatibility"]["source_hash"]:
        raise ValueError("Waveform diagnostic source provenance differs from the run")
    if (record.get("existing_input_hashes_unchanged") is not True
            or record.get("scientific_source_hash_unchanged") is not True
            or record.get("new_independent_test_frames") != 0
            or record.get("evaluation_bits_added") != 0
            or record.get("fit_selection_prediction_or_training_executed") is not False):
        raise ValueError("Diagnostics lack existing-frame/no-added-evaluation evidence")
    inputs = [manifest_path]
    for name, expected in record["input_files_sha256"].items():
        source = (run.path / name).resolve()
        if not source.is_relative_to(run.path) or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Waveform diagnostic input changed: {name}")
        inputs.append(source)
    for name, expected in record["scientific_source_sha256"].items():
        source = ROOT / name
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Scientific source differs from the diagnostic record: {name}")
        inputs.append(source)
    for key in ("exporter_code", "saved_data_audit_code"):
        source = Path(record[key]["path"])
        if not source.is_absolute():
            source = ROOT / source
        if not source.is_file() or sha256(source) != record[key]["sha256"]:
            raise ValueError(f"Diagnostic provenance code changed: {key}")
        inputs.append(source)
    entries = record["files"]
    if len(entries) != len(run.powers) or {float(entry["launch_power_dbm"]) for entry in entries} != set(run.powers):
        raise ValueError("Waveform diagnostics do not cover the saved run powers")
    for entry in entries:
        source = (manifest_path.parent / entry["file"]).resolve()
        if not source.is_relative_to(manifest_path.parent) or not source.is_file() or sha256(source) != entry["sha256"]:
            raise ValueError("Waveform diagnostic output hash differs")
        replicas = run.frames[float(entry["launch_power_dbm"])]["calibrated_edc"]
        matched = [row for row in replicas if row["frame_index"] == entry["original_frame_index"]]
        if (entry["split"] != "test" or len(matched) != 1
                or any(entry[key] != matched[0][key] for key in ("bit_seed", "noise_seed"))):
            raise ValueError("Waveform diagnostic seeds are not an existing test frame")
        inputs.append(source)
    return record, inputs


class Report:
    def __init__(self, output: Path):
        self.output = output
        self.md = []
        self.htm = []
        self.inputs = []

    def title(self, text):
        self.md += [f"# {text}", ""]
        self.htm += [f"<h1>{e(text)}</h1>"]

    def heading(self, text):
        self.md += [f"## {text}", ""]
        self.htm += [f"<h2>{e(text)}</h2>"]

    def paragraph(self, text, css=""):
        self.md += [text, ""]
        self.htm += [f'<p class="{e(css)}">{e(text)}</p>']

    def links(self, intro, links):
        self.md += [intro + " " + " / ".join(f"[{label}]({url})" for label, url in links) + "。", ""]
        self.htm += [f"<p>{e(intro)} " + " / ".join(f'<a href="{e(url)}">{e(label)}</a>' for label, url in links) + "。</p>"]

    def bullets(self, items):
        self.md += ["\n".join(f"- {item}" for item in items), ""]
        self.htm += ["<ul>" + "".join(f"<li>{e(item)}</li>" for item in items) + "</ul>"]

    def table(self, headers, rows):
        self.md += ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        self.md += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows] + [""]
        self.htm += ['<div class="table-wrap"><table><thead><tr>' + "".join(f"<th scope=\"col\">{e(cell)}</th>" for cell in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f'<td>{e(cell)}</td>' for cell in row) + "</tr>" for row in rows) + "</tbody></table></div>"]

    def figure(self, path: Path, caption: str, alt: str):
        self.inputs.append(path)
        self.md += [f"![{alt}]({relpath(path, self.output)})", "", caption, ""]
        blob = base64.b64encode(path.read_bytes()).decode("ascii")
        self.htm += [f'<figure><img src="data:image/png;base64,{blob}" alt="{e(alt)}" loading="lazy"><figcaption>{e(caption)}</figcaption></figure>']

    def open_details(self, summary):
        self.md += [f"<details><summary>{e(summary)}</summary>", ""]
        self.htm += [f"<details><summary>{e(summary)}</summary>"]

    def close_details(self):
        self.md += ["</details>", ""]
        self.htm += ["</details>"]

    def command(self, lines):
        code = "\n".join(lines)
        self.md += ["```bash", code, "```", ""]
        self.htm += [f"<pre><code>{e(code)}</code></pre>"]


def review_text(record: dict | None):
    if record is None:
        return "独立評価は指定1回、試行0回、完了0回。この版は評価待ち。通常検証と独立評価は別の記録として扱う。", []
    requested = record.get("requested", 1)
    attempted, completed = record.get("attempted", 0), record.get("completed", 0)
    intro = f"独立評価：指定{requested}回、試行{attempted}回、完了{completed}回。"
    verdict = record.get("original_verdict", record.get("verdict"))
    if verdict:
        intro += f"原判定：{verdict}。"
    intro += str(record.get("summary_ja", ""))
    findings = []
    for finding in record.get("findings", []):
        identity = "/".join(str(finding[key]) for key in ("id", "severity") if finding.get(key))
        prefix = identity + "：" if identity else ""
        findings.append(f"{prefix}{finding.get('title_ja', finding.get('title', '指摘'))}：{finding.get('disposition_ja', finding.get('disposition', '未記録'))}。{finding.get('reason_ja', finding.get('reason', ''))}")
    if record.get("post_review_changes"):
        intro += " 評価後の修正は今回の独立評価の対象外で、独立再評価を実施していない。修正状況と通常検証は以下の記録に従う。"
    if record.get("runtime_verification_ja"):
        intro += " " + record["runtime_verification_ja"]
    return intro, findings


def build(args):
    run = load_run(args.run, args.allow_pilot)
    output = args.output_dir.resolve()
    if run.pilot and output.is_relative_to(ROOT / "reports"):
        raise ValueError("Pilot report belongs in .work, not the final reports directory")
    output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.figures_dir.resolve()
    figure_manifest = read_json(figures_dir / "results_manifest.json")
    if figure_manifest["run_id"] != run.manifest["run_id"] or figure_manifest["source_run"] != display_path(run.path):
        raise ValueError("Figures use a different source run")
    for record in figure_manifest["inputs"]:
        source = Path(record["path"])
        if not source.is_absolute():
            source = ROOT / source
        if not source.is_file() or record["sha256"] != sha256(source):
            raise ValueError(f"Figure input changed after generation: {source}")
    for figure in figure_manifest["figures"]:
        for artifact in figure["outputs"]:
            path = Path(artifact["path"])
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                raise ValueError(f"Figure hash differs: {path}")
    qualification = read_json(args.qualification / "production_recommendation.json") if args.qualification else None
    hardware_path = run.path / "execution-hardware.json"
    hardware_record = read_json(hardware_path) if hardware_path.is_file() else None
    review = read_json(args.review_record) if args.review_record else None
    verification = read_json(args.verification_record) if args.verification_record else None
    diagnostics, diagnostic_inputs = load_waveform_diagnostics(run)
    original_review_path = None
    if review and review.get("original_result"):
        original_review_path = Path(review["original_result"])
        if not original_review_path.is_absolute():
            original_review_path = ROOT / original_review_path
        if not original_review_path.is_file():
            raise ValueError("Original independent review record is missing")
    n = len(run.powers)
    improved = [p for p in run.powers if run.metrics[p]["esn"]["ber"] < run.metrics[p]["calibrated_edc"]["ber"]]
    worsened = [p for p in run.powers if run.metrics[p]["esn"]["ber"] > run.metrics[p]["calibrated_edc"]["ber"]]
    better_linear = [p for p in run.powers if run.metrics[p]["esn"]["ber"] < run.metrics[p]["linear"]["ber"]]
    link, config = run.config["link"], run.config
    bit_counts = {run.metrics[p]["esn"]["n_bits"] for p in run.powers}
    bits_text = f"{next(iter(bit_counts)):,}" if len(bit_counts) == 1 else f"{min(bit_counts):,}～{max(bit_counts):,}"
    report = Report(output)
    report.title("光ファイバ長距離伝送をリザバーで補償できるか")
    if run.pilot:
        report.paragraph("探索的 pilot の描画・動作確認版。以下は最終結果ではなく、研究の結論には使用しない。", "pilot")
    if len(worsened) == n:
        lead = f"今回の数値シミュレーションでは、リザバー補償のビット誤り率（BER）が{n}つの送信電力すべてで通常補償より高かった。モデルの学習は完了したが、通信品質の改善は確認できなかった。"
    elif not improved:
        lead = f"今回の設定では、リザバー補償が通常補償より低いBERを示した電力条件はなかった（{n}条件を評価）。モデルの学習は完了したが、通信品質の改善は確認できなかった。"
    else:
        lead = f"リザバー補償は、{n}電力条件のうち{len(improved)}条件で通常補償より低いBERを示し、{len(worsened)}条件では高かった。これは今回の未学習データで数えた結果で、別の伝送条件での改善を保証しない。"
    report.paragraph(lead, "lead")
    report.paragraph("BERは『間違って受信したビット数 ÷ 調べたビット数』。小さいほどよい。今回の結果は誤り訂正（FEC）を行う前の値である。")
    constellation_record = next(record for record in figure_manifest["figures"] if record["name"] == "result_constellations")
    example_power = constellation_record["power_dbm"]
    fixed = run.metrics[example_power]
    report.table([f"{example_power:+g} dBmでの比較", "誤り数 / 調べたビット数", "BER"], [
        [LABELS[m], f"{fixed[m]['bit_errors']:,} / {fixed[m]['n_bits']:,}", rate(fixed[m])]
        for m in ("calibrated_edc", "esn")
    ])
    report.paragraph(f"同じ受信信号・同じ採点区間を比較した。各電力では、独立に作った{config['test_frames']}テストフレーム、各方式 {bits_text} bit を調べた。実機の伝送実験ではなく、ソフトウェア上の数値シミュレーションである。", "scope")
    report.htm += ['<div class="cards">' + ''.join(f'<div class="card"><strong>{e(value)}</strong><span>{e(label)}</span></div>' for value,label in [(f"{link['n_spans'] * link['span_km']:g} km", "シミュレーションの伝送距離"),(f"{bits_text} bit", "各方式・各電力の評価数"),(f"{len(improved)} / {n} 条件", "通常補償よりESNのBERが低い条件")]) + '</div>']

    report.heading("何を測ったか")
    report.paragraph("16種類の点の位置でビットを表す光信号を、増幅器をはさんだ光ファイバに通した。伝送中の広がり、光の強さに依存するひずみ、増幅器が足す雑音を再現し、受信側の4方式を比べた。DBPは物理計算を逆向きに解く比較対象で、雑音を消せる理想的な上限とは仮定していない。")
    report.table(["項目", "今回の範囲"], [
        ["伝送", f"{link['n_spans'] * link['span_km']:g} km（{link['n_spans']} × {link['span_km']:g} km）、単一波長・2偏波、16QAM・{link['symbol_rate_hz']/1e9:g} GBd"],
        ["送信電力", "、".join(f"{p:+g}" for p in run.powers) + " dBm（X/Y偏波の合計）"],
        ["データ", f"電力ごとに独立フレーム：学習 {config['train_frames']}、候補選択 {config['validation_frames']}、テスト {config['test_frames']}"],
        ["送受信器", "理想的な同期。実機の位相ずれ、量子化、多波長間の干渉などは含まない"],
    ])
    report.figure(ROOT / "reports/figures/link_overview.png", "図1　通信路と4方式の説明図。測定値ではない。同じ伝送結果・同じ採点区間で比べた。", "送信、20区間のファイバと増幅器、4つの受信補償方式の流れ")
    units = sorted({candidate.get("units",128) for candidate in config["esn_candidates"]})
    report.paragraph(f"リザバー（ESN）は過去の入力を内部状態に残すモデルで、今回は{'、'.join(map(str,units))}状態を使った。内部の接続は固定し、受信点を送信点へ近づける出口の重みだけを学習した。線形方式とは同じ明示的な入力窓を使うが、ESNは内部状態にも過去の履歴を保持する。")
    report.figure(ROOT / "reports/figures/learning_workflow.png", "図2　学習・候補選択・最終評価の分離。校正と入力の尺度は学習データだけから決め、テストを見る前にモデルを固定した。説明図であり、性能の測定値ではない。", "独立した学習・検証・テストフレームと、テスト前に設定を固定する学習の流れ")

    report.heading("結果の意味")
    report.figure(figures_dir / "result_ber.png", "図3　全電力条件のBER。点は全フレームの合算、縦線は95%計数区間、短い横線は各独立フレーム。縦軸は対数で1目盛りが10倍。0誤りは上限からの下向き矢印で示す。", "7つの送信電力で4つの補償方式の直接計数BERと不確かさを比較するグラフ")
    report.paragraph("95%区間は各ビットの誤りが独立という仮定に基づき、連続する誤りをすべて扱うものではない。このため独立フレーム間のばらつきも示した。区間の重なりだけで方式間の有意差を断定していない。")
    zero_dbp = [p for p in run.powers if run.metrics[p]["calibrated_dbp"]["bit_errors"] == 0]
    if zero_dbp:
        bounds = sorted({run.metrics[p]["calibrated_dbp"]["ber_ci_high"] for p in zero_dbp})
        bound_text = f"{bounds[0]:.2e}" if len(bounds) == 1 else f"{min(bounds):.2e}～{max(bounds):.2e}"
        report.paragraph("DBPの " + "、".join(f"{p:+g} dBm" for p in zero_dbp) + f" では0誤りだったが、有限の評価数に対する95%上限は {bound_text}。雑音を完全に消した、将来も誤らない、という意味ではない。")
    rows = [[f"{p:+g}"] + [f"{rate(run.metrics[p][m])} [{run.metrics[p][m]['bit_errors']:,}]" for m in METHODS] for p in run.powers]
    report.table(["電力 [dBm]", "通常補償 BER [誤り数]", "線形方式 BER [誤り数]", "ESN BER [誤り数]", "DBP BER [誤り数]"], rows)
    report.paragraph(f"各方式・各電力の評価数は {bits_text} bit。7電力すべてを表示し、結果のよい条件だけを選んでいない。")
    if len(better_linear) == 1 and run.metrics[better_linear[0]]["linear"]["bit_errors"] - run.metrics[better_linear[0]]["esn"]["bit_errors"] == 1:
        p = better_linear[0]
        report.paragraph(f"同じ明示的な入力窓を使う線形方式に対しても、ESNのBERが低かったのは{p:+g} dBmだけで、誤り数 {run.metrics[p]['esn']['bit_errors']:,} 対 {run.metrics[p]['linear']['bit_errors']:,}（1個の差）だった。この結果だけでESNの優位性は判断できない。")
    else:
        report.paragraph(f"同じ明示的な入力窓を使う線形方式との比較では、{n}条件中{len(better_linear)}条件でESNのBERが低かった。この結果は今回の固定したモデル・データに限る。")
    report.figure(figures_dir / "result_constellations.png", f"図4　事前指定の{example_power:+g} dBm、X偏波の同じ{constellation_record['n_points']:,}点。十字は送信位置、色点は受信結果。最初のテストフレームから等間隔に保存した点を、共通の軸で表示した。", "同じX偏波の受信点を通常補償、線形方式、ESN、DBPの4方式で並べた配置図")
    report.paragraph(f"この電力で、送信点からの相対的なずれ（EVM）は通常補償 {fixed['calibrated_edc']['evm']*100:.2f}%、ESN {fixed['esn']['evm']*100:.2f}%。EVMは小さいほど点に近いが、通信品質は実際の誤り数で判断する。")
    if fixed["esn"]["evm"] < fixed["calibrated_edc"]["evm"] and fixed["esn"]["ber"] > fixed["calibrated_edc"]["ber"]:
        report.paragraph(f"今回は平均的なずれが少し小さくても、誤りは {fixed['calibrated_edc']['bit_errors']:,} 個から {fixed['esn']['bit_errors']:,} 個へ増えた。EVMの改善だけではBERの改善を証明できない。配置図の小標本だけでも優劣を決めない。")

    report.heading("限界と次の問い")
    report.bullets([
        "各電力で別モデルを学習した。未学習の電力・距離への適用や、実機・多波長伝送・誤り訂正後の性能は検証していない。",
        "独立フレームのばらつきは確認したが、リザバーの初期化は各電力1通り。今回の結果を、リザバー全般の頑健性や性能の結論には広げない。",
        "改善が得られなかった原因はこの比較だけでは特定できない。次は内部重みを別に初期化し、新しい独立データでも今回の結果が再現するかを調べる。",
    ])
    report.links("実験条件と方法：", [("実験計画", relpath(ROOT / "docs/experiment-plan.md", output)), ("通信路", relpath(ROOT / "docs/channel-methods.md", output)), ("学習と評価", relpath(ROOT / "docs/learning-methods.md", output))])

    report.heading("技術補足")
    report.paragraph("必要な項目を開いて読める。印刷するときは、含めたい補足を開いてから印刷する。")
    report.open_details("補足1：実装条件と、検証データでの候補選択")
    report.table(["項目", "保存した設定"], [
        ["ファイバ・増幅器", f"損失 {link['alpha_db_per_km']:g} dB/km、分散 {link['dispersion_ps_nm_km']:g} ps/(nm km)、非線形係数 {link['gamma_per_w_km']:g} /(W km)、雑音指数 {link['edfa_noise_figure_db']:g} dB"],
        ["数値計算", f"{link['sps']} samples/symbol、伝搬刻み {link['step_km']:g} km、DBP刻み {config['dbp_step_km']:g} km、倍精度"],
        ["波形・境界", f"RRC roll-off {link['rolloff']:g}、{config['n_symbols']:,} symbols/偏波、両端 {link['guard_symbols']:,} symbolsを採点から除外"],
        ["入力窓と状態", f"前{config['context']['past_symbols']}・後{config['context']['future_symbols']} symbols。ESNは内部状態にも履歴を保持し、フレームごとにリセット。立上り {config['context']['washout_symbols']} symbolsを除外"],
        ["端末", "理想同期。レーザー位相雑音、周波数ずれ、PMD、量子化、多波長干渉は含まない"],
    ])
    report.figure(figures_dir / "result_validation.png", "図5　すべての候補の検証BER。枠と星が選択結果。BER、EVM、保存配列量、候補IDの順で決め、テストで選び直していない。読出しはリッジ回帰で一括計算するためepoch曲線はない。", "電力ごとの線形方式とESNの候補別検証BERと、選択した候補のヒートマップ")
    chosen_ids = {str(p): run.selections[p]["selected"] for p in run.powers}
    report.paragraph(f"電力ごとにESN {len(config['esn_candidates']) * len(config['esn_ridges'])}候補、線形方式 {len(config['linear_ridges'])}候補を比較。校正と正規化は学習フレームだけで決めた。モデルの保存・復元後も検証フレームへの予測が完全一致した。候補と選択IDは各電力の selection.json に保存している。")
    report.close_details()

    report.open_details("補足2：学習性能とは別に行った数値条件の確認")
    if qualification:
        checks = qualification.get("checks", {})
        recommended_link = qualification.get("recommended_link", {})
        if not run.pilot and (any(link.get(key) != value for key,value in recommended_link.items()) or config["n_symbols"] != qualification.get("n_symbols", config["n_symbols"]) or config["dbp_step_km"] != qualification.get("dbp_step_km", config["dbp_step_km"])):
            raise ValueError("Reported final configuration differs from numerical recommendation")
        report.paragraph("刻み幅・サンプル数・フレーム長・境界の除外数を別々に変えた。目標は有効SNR差0.1 dB以内とBERの統計的整合性。計算条件への依存を調べる確認で、ESNの改善を示す証拠ではない。")
        if (figures_dir / "result_qualification.png").is_file():
            report.figure(figures_dir / "result_qualification.png", "図6　計算条件変更による有効SNRの最大差。破線は0.1 dBの目標。BERの整合性も含めて確認した。学習モデルのテストとは別のデータである。", "数値刻み・サンプル数の変更と、フレーム長・境界除外による有効SNR差")
        report.table(["確認の対象", "条件ごとの比較数", "有効SNR差の最大値", "目標"], [
            ["刻み幅・サンプル数", str(checks.get('refinement_pair_summaries', '記録参照')), f"{checks['maximum_abs_refinement_snr_drift_db']:.2g} dB", "0.1 dB以内 + BER整合"],
            ["フレーム長・境界除外", str(checks.get('boundary_pair_summaries', '記録参照')), f"{checks['maximum_abs_boundary_snr_drift_db']:.2g} dB", "0.1 dB以内 + BER整合"],
        ])
        initial = qualification.get("accepted_initial", {})
        initial_passed = all(initial.values()) if isinstance(initial, dict) else initial is True
        if initial_passed:
            report.paragraph("初期の4 samples/symbol・1 km刻み・256 symbols除外も基準内だった。境界に対する測定上の余裕を確保するため、最終評価は1024 symbols除外を採用した。")
        if checks.get("full_receiver_controls_accepted"):
            report.paragraph("雑音だけを加えた3条件でも、実測BERは独立に求めた16QAM理論値と事前指定の99.9%計数区間内で整合し、有効SNR差は0.1 dB以内だった。")
        report.paragraph("数値確認は0/+6 dBmの代表2電力と有限の比較に限る。伝搬波形に非ゼロの差が残り、受信指標の安定性を確認した。短い雑音波形の境界確認は、長い波形の同じ先頭部分を切り出して行った。すべての電力や連続方程式の厳密解を証明するものではない。")
        report.links("精密な値・比較条件・限界：", [("数値確認の記録", relpath(args.qualification / "production_recommendation.json", output)), ("数値確認の方法", relpath(ROOT / "docs/numerical-qualification.md", output))])
    else:
        report.paragraph("数値条件の確認結果は未添付。学習runnerだけでは数値収束を判定していない。")
    report.close_details()

    report.open_details("補足3：実装に依存する追加処理時間と保存重み")
    report.figure(figures_dir / "result_resources.png", "図7　共通の通信路生成・EDC後の追加処理時間とモデル配列量。DBPは逆伝搬を含む。線形方式/ESNはCPU、CuPy使用時のDBPはGPU。時間は1テストフレームの平均で、入出力と候補選択を含まない。", "各電力の追加処理時間と、線形方式・ESNが保存するモデル配列量")
    hardware = args.hardware_label or (f"{hardware_record['cpu']}（ESN・線形方式）、{hardware_record['gpu']}（通信路・DBP）、{hardware_record['environment']}" if hardware_record else ("ESN/線形方式：CPU、DBP：CuPy GPU" if config["backend"] == "cupy" else "NumPy/Numba CPU"))
    report.paragraph("実行装置：" + hardware + "。異なる装置・処理範囲の測定なので、実機の遅延・消費電力・スループットや理論的な計算量の優劣を示すものではない。配列量はピークRAMではない。")
    report.table([f"{example_power:+g} dBmでの方式", "BER", "誤り数 / bit数", "EVM", "追加時間 / frame", "モデル配列"], [[LABELS[m], rate(fixed[m]), f"{fixed[m]['bit_errors']:,} / {fixed[m]['n_bits']:,}", f"{fixed[m]['evm']*100:.2f}%", extra_time(fixed[m]), f"{fixed[m]['model_array_bytes']/1024:.2f} KiB"] for m in METHODS])
    report.paragraph("通常補償とDBPの配列量は校正係数だけで32 bytes。実行コードや作業メモリを含まない。線形/ESNは model_array_bytes を使い、圧縮checkpointのファイルサイズとは分けた。")
    future_ps = config["context"]["future_symbols"] / link["symbol_rate_hz"] * 1e12
    report.paragraph(f"入力窓は未来の{config['context']['future_symbols']}シンボルも参照し、入力待ちだけで {future_ps:g} ps が必要。フレームの蓄積、EDC、計算、装置の遅延を含まない。")
    if hardware_record:
        report.links("装置・ドライバ・実際のthread環境：", [("execution-hardware.json", relpath(hardware_path, output))])
    report.close_details()

    report.open_details("補足4：再現コマンド、元データ、通常検証・独立評価")
    report.paragraph(f"実行ID：{run.manifest['run_id']}。全候補、校正値、bit/ASEの種、モデル、ソフトウェア版とハッシュを保存した。大きな生波形はレシピから再生成する。pilotは最終BERに合算していない。")
    report.links("元データと図の来歴：", [("summary.csv", relpath(run.path / "summary.csv", output)), ("config.json", relpath(run.path / "config.json", output)), ("manifest.json", relpath(run.path / "manifest.json", output)), ("結果図の入力・出力ハッシュ", relpath(figures_dir / "results_manifest.json", output))])
    if diagnostics:
        report.paragraph("同じ保存済みテストフレームのbit/ASE seedから、EDC・受信フィルタ後のXY連続波形を再生成し、短い2-sps診断区間（1シンボル当たり2サンプル）と送信点・mask・フレーム位置を保存した。追加の評価ビットや再学習はない。保存時の記録に加え、この補足生成時も既存入力・科学計算コード・診断ファイルのハッシュ一致を確認した。")
        report.links("連続診断波形のseed・区間・再生成コマンド・ハッシュ：", [("waveform_diagnostics/manifest.json", relpath(run.path / "waveform_diagnostics/manifest.json", output))])
    cmd_run = f".venv/bin/python -m experiments.esn_longhaul --config {display_path(run.path / 'config.json')} --run-id {'pilot' if run.pilot else 'final'}-reproduction"
    cmd_export = f".venv/bin/python -m experiments.esn_longhaul.export_waveform_diagnostics --run artifacts/{'pilot' if run.pilot else 'final'}-reproduction"
    cmd_fig = figure_manifest["regeneration_command"]
    cmd_report = [".venv/bin/python", "-m", "experiments.esn_longhaul.build_report", "--run", display_path(run.path), "--figures-dir", display_path(figures_dir), "--output-dir", display_path(output)]
    if args.qualification:
        cmd_report += ["--qualification", display_path(args.qualification)]
    if args.review_record:
        cmd_report += ["--review-record", display_path(args.review_record)]
    if args.verification_record:
        cmd_report += ["--verification-record", display_path(args.verification_record)]
    if args.hardware_label:
        cmd_report += ["--hardware-label", args.hardware_label]
    if args.allow_pilot:
        cmd_report += ["--allow-pilot"]
    sync_command = "uv sync --python 3.12 --locked --group dev" + (" --extra gpu" if config["backend"] == "cupy" else "")
    report.command([sync_command + "  # CUDA準備はREADMEを参照", ".venv/bin/python -m pytest", cmd_run, cmd_export, cmd_fig, shlex.join(cmd_report)])
    report.paragraph(f"HTMLは全{len(report.inputs)}図を埋め込み、ネット接続なしで読める。図は保存結果から生成し、結果の置換や都合のよい条件・受信点の選別をしていない。")
    if verification:
        report.paragraph(str(verification.get("summary_ja", "通常検証は添付記録を参照。")))
        report.links("通常検証：", [("ordinary_checks.json", relpath(args.verification_record, output))])
    else:
        report.paragraph("通常検証の集約記録は添付待ち。")
    review_intro, findings = review_text(review)
    report.paragraph(review_intro)
    if findings:
        report.bullets(findings)
    if args.review_record:
        review_links = [("親の採否・対応・通常検証", relpath(args.review_record, output))]
        if original_review_path:
            review_links.insert(0, ("validatorの原判定・指摘", relpath(original_review_path, output)))
        report.links("独立評価の原記録と対応：", review_links)
    if review and review.get("post_review_changes"):
        report.bullets([str(item) for item in review["post_review_changes"]])
    report.close_details()
    report.links("研究設計・開発管理：", [("元の要件", relpath(ROOT / "docs/requirements.md", output)), ("Issue #6", "https://github.com/Motoki0705/optical-comm-lab/issues/6"), ("統合PR #7", "https://github.com/Motoki0705/optical-comm-lab/pull/7"), ("scoutによる保守点検", relpath(ROOT / "docs/repository-maintenance.md", output)), ("AGENTS.md", relpath(ROOT / "AGENTS.md", output))])
    report.links("基礎資料：", [("OptiCommPy Manakov", "https://opticommpy.readthedocs.io/en/latest/generated/optic.models.channels.manakovSSF.html"), ("EDFA", "https://opticommpy.readthedocs.io/en/latest/generated/optic.models.devices.edfa.html"), ("DBP：Ip・Kahn（2008）", "https://www-ee.stanford.edu/~jmk/pubs/digital.backpropagation.jlt.10-08.pdf"), ("学習評価の注意：Freireら", "https://arxiv.org/html/2109.14942")])
    md = output / "final_report.md"
    htm = output / "final_report.html"
    md.write_text("\n".join(report.md), encoding="utf-8")
    htm.write_text('<!doctype html>\n<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="1600 km光伝送のリザバー補償を独立テストフレームのBERで比較した日本語レポート"><title>光伝送リザバー補償・結果レポート</title><style>' + CSS + '</style></head><body><main>' + "\n".join(report.htm) + '<footer>optical-comm-lab · ' + e(run.manifest['run_id']) + ' · 保存済み計数結果から生成</footer></main></body></html>\n', encoding="utf-8")
    inputs = set(run.inputs + report.inputs + [Path(__file__).with_name("visualize_results.py"), Path(__file__).with_name("visualize_concepts.py"), ROOT / "reports/figures/concepts_manifest.json", ROOT / "docs/repository-maintenance.md", figures_dir / "results_manifest.json", ROOT / "docs/requirements.md", ROOT / "docs/experiment-plan.md", ROOT / "docs/channel-methods.md", ROOT / "docs/learning-methods.md", ROOT / "docs/numerical-qualification.md"])
    if hardware_record:
        inputs.add(hardware_path)
    if args.qualification:
        inputs.add(args.qualification / "production_recommendation.json")
    if args.review_record:
        inputs.add(args.review_record)
    if args.verification_record:
        inputs.add(args.verification_record)
    inputs.update(diagnostic_inputs)
    if original_review_path:
        inputs.add(original_review_path)
    manifest = {"schema_version":1, "source_run":display_path(run.path), "run_id":run.manifest["run_id"], "evidence_scope":run.scope, "generator":{"path":display_path(Path(__file__)), "sha256":sha256(Path(__file__))}, "runtime":{"python":platform.python_version()}, "regeneration_command":shlex.join(cmd_report),
        "inputs":[{"path":display_path(path), "sha256":sha256(path)} for path in sorted(inputs)], "outputs":[{"path":display_path(path), "sha256":sha256(path), "bytes":path.stat().st_size} for path in (md,htm)], "measured_summary":{"esn_lower_ber_than_edc_powers_dbm":improved,"esn_higher_ber_than_edc_powers_dbm":worsened,"esn_lower_ber_than_linear_powers_dbm":better_linear,"selected_candidate_ids":chosen_ids}, "review":{"requested":review.get("requested",1) if review else 1,"attempted":review.get("attempted",0) if review else 0,"completed":review.get("completed",0) if review else 0,"original_verdict":review.get("original_verdict",review.get("verdict")) if review else None,"post_review_changes_independently_evaluated":False if review and review.get("post_review_changes") else None}}
    (output / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    print(f"Saved {display_path(md)} and {display_path(htm)} from {run.manifest['run_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--review-record", type=Path, help="coordinator-supplied JSON with counts, findings/dispositions, runtime verification")
    parser.add_argument("--verification-record", type=Path, help="normal verification JSON with summary_ja")
    parser.add_argument("--hardware-label", help="verified CPU/GPU label, not an inferred hardware performance claim")
    parser.add_argument("--allow-pilot", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
