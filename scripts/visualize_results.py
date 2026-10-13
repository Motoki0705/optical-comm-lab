#!/usr/bin/env python3
"""Plot saved, directly counted receiver results without fitting or selecting test data.

Use --allow-pilot and a .work output directory for exploratory rendering checks.
Final figures require every configured power and verified artifact hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from visualize_concepts import JapaneseFont, select_font

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("calibrated_edc", "linear", "esn", "calibrated_dbp")
LABELS = {"calibrated_edc": "通常補償（EDC）", "linear": "線形学習", "esn": "リザバー（ESN）", "calibrated_dbp": "逆伝搬（DBP）"}
COLORS = dict(zip(METHODS, ("#6B7280", "#2677B8", "#D97706", "#25846C")))
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D")))
INK = "#202B3A"
MUTED = "#526174"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


@dataclass
class RunData:
    path: Path
    config: dict
    manifest: dict
    metrics: dict
    frames: dict
    selections: dict
    constellations: dict
    inputs: list[Path]

    @property
    def powers(self):
        return sorted(self.metrics)

    @property
    def pilot(self):
        return self.config["mode"] == "pilot"

    @property
    def scope(self):
        return "探索的 pilot：最終結論には使用しない" if self.pilot else "独立した未学習フレームでの評価"


def load_run(path: Path, allow_pilot: bool = False) -> RunData:
    """Check saved provenance and comparison denominators before displaying data."""
    path = path.resolve()
    config, manifest, summary = (read_json(path / name) for name in ("config.json", "manifest.json", "summary.json"))
    if config["mode"] == "pilot" and not allow_pilot:
        raise ValueError("Exploratory pilot requires --allow-pilot; it is not final evidence")
    if config["mode"] != summary["mode"] or manifest["run_id"] != summary["run_id"]:
        raise ValueError("Run identifiers/modes disagree")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest["compatibility"]["config_hash"]:
        raise ValueError("Recorded configuration hash differs from manifest")
    metrics, frames, selections, constellations = {}, {}, {}, {}
    inputs = [path / name for name in ("config.json", "manifest.json", "summary.json")]
    for folder in sorted(path.glob("power_*")):
        if not (folder / "complete.json").exists():
            continue
        complete = read_json(folder / "complete.json")
        if complete["compatibility"] != manifest["compatibility"]:
            raise ValueError(f"Provenance differs: {folder}")
        for name, expected in complete["files_sha256"].items():
            artifact = folder / name
            if artifact.parent != folder or not artifact.is_file() or sha256(artifact) != expected:
                raise ValueError(f"Missing or changed artifact: {artifact}")
        power = float(complete["launch_power_dbm"])
        if power in metrics:
            raise ValueError("Duplicate completed launch power")
        result = read_json(folder / "metrics.json")
        metrics[power] = {row["method"]: row for row in result["metrics"]}
        if set(metrics[power]) != set(METHODS) or len(result["metrics"]) != len(METHODS):
            raise ValueError("Exactly four receiver methods are required")
        frame_rows = read_json(folder / "test_frames.json")
        frames[power] = {method: sorted((row for row in frame_rows if row["method"] == method), key=lambda row: row["frame_index"]) for method in METHODS}
        expected_indices = list(range(config["test_frames"]))
        expected_bits = (config["n_symbols"] - 2*config["link"]["guard_symbols"] - config["context"]["past_symbols"] - config["context"]["future_symbols"] - config["context"]["washout_symbols"])*8
        for method in METHODS:
            row, replicas = metrics[power][method], frames[power][method]
            if [item["frame_index"] for item in replicas] != expected_indices:
                raise ValueError(f"Missing independent test frames: {folder}/{method}")
            if any(item["n_bits"] != expected_bits for item in replicas):
                raise ValueError("Scored bit counts differ from saved guard/context/washout recipe")
            for key in ("bit_errors", "n_bits", "error_energy", "signal_energy", "inference_seconds"):
                if not np.isclose(row[key], sum(item[key] for item in replicas), rtol=1e-12, atol=1e-14):
                    raise ValueError(f"Summary is not a pooled {key}: {folder}/{method}")
            if not np.isclose(row["ber"], row["bit_errors"] / row["n_bits"], rtol=1e-13, atol=0):
                raise ValueError("BER differs from directly counted errors/bits")
            if not np.isclose(row["evm"], np.sqrt(row["error_energy"] / row["signal_energy"]), rtol=1e-13, atol=0):
                raise ValueError("EVM does not pool signal/error energy")
            if not 0 <= row["ber_ci_low"] <= row["ber"] <= row["ber_ci_high"] <= 1:
                raise ValueError("Invalid counting interval")
        for index in expected_indices:
            compared = [frames[power][method][index] for method in METHODS]
            if len({(row["n_bits"], row["n_symbols_xy"], row["bit_seed"], row["noise_seed"]) for row in compared}) != 1:
                raise ValueError("Receiver methods do not share test frames/scored bit counts")
        selection = read_json(folder / "selection.json")
        if selection["selection_split"] != "validation" or selection["test_labels_used_for_selection"] is not False or selection["checkpoint_reload_prediction_equal"] is not True:
            raise ValueError("Missing validation-only selection/checkpoint evidence")
        for method in ("linear", "esn"):
            candidates = [row for row in selection["candidates"] if row["method"] == method]
            chosen = min(candidates, key=lambda row: (row["ber"], row["evm"], row["model_array_bytes"], row["candidate_id"]))
            if chosen["candidate_id"] != selection["selected"][method] or metrics[power][method]["selected_candidate"] != chosen["candidate_id"]:
                raise ValueError("Saved choice disagrees with frozen validation rule")
        selections[power] = selection
        with np.load(folder / "constellations.npz", allow_pickle=False) as archive:
            diagnostic = {name: archive[name] for name in archive.files}
        n = len(diagnostic["indices"])
        for key in ("tx_symbols", *METHODS):
            if diagnostic[key].shape != (n, 2) or not np.isfinite(diagnostic[key]).all():
                raise ValueError("Constellation arrays must have the same finite XY samples")
        if len(np.unique(diagnostic["indices"])) != n or np.any(np.diff(diagnostic["indices"]) <= 0):
            raise ValueError("Constellation indices must be unique and in time order")
        if any(int(diagnostic[key]) != frames[power]["esn"][0][key] for key in ("bit_seed", "noise_seed")):
            raise ValueError("Constellations are not the same saved first test frame")
        constellations[power] = diagnostic
        inputs.extend(folder / name for name in ("complete.json", "metrics.json", "test_frames.json", "selection.json", "constellations.npz", "seeds.json", "calibration.json"))
    if not metrics:
        raise ValueError("No completed powers")
    if config["mode"] == "final" and set(metrics) != set(config["launch_powers_dbm"]):
        raise ValueError("Final figures require all configured powers; partial results are not publishable")
    if config["mode"] == "final" and set(config["launch_powers_dbm"]) != {-6, -4, -2, 0, 2, 4, 6}:
        raise ValueError("The final study requires the accepted seven-power sweep")
    for row in summary["metrics"]:
        if row != metrics[float(row["launch_power_dbm"])][row["method"]]:
            raise ValueError("Top-level summary differs from completed power artifact")
    if len(summary["metrics"]) != len(metrics) * len(METHODS):
        raise ValueError("Summary does not contain all completed receivers/powers")
    return RunData(path, config, manifest, metrics, frames, selections, constellations, inputs)


def style(font: JapaneseFont):
    plt.rcParams.update({"font.family": font.family, "font.size": 10, "axes.titlesize": 11,
                        "axes.labelsize": 10, "axes.labelcolor": INK, "text.color": INK,
                        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#CDD6DF",
                        "axes.spines.top": False, "axes.spines.right": False,
                        "figure.facecolor": "white", "axes.facecolor": "white",
                        "svg.fonttype": "path", "svg.hashsalt": "optical-comm-results-v1", "pdf.fonttype": 42})


def frame_note(run: RunData) -> str:
    counts = sorted({run.metrics[p]["esn"]["n_bits"] for p in run.powers})
    count = f"{counts[0]:,}" if len(counts) == 1 else f"{min(counts):,}–{max(counts):,}"
    return f"各方式・各電力 {count} bit、独立 {run.config['test_frames']} フレーム。{run.scope}。"


def figure_ber(run: RunData):
    fig, ax = plt.subplots(figsize=(11.3, 5.7))
    fig.subplots_adjust(left=.10, right=.98, bottom=.25, top=.87)
    positive = []
    for method in METHODS:
        rows = [run.metrics[p][method] for p in run.powers]
        ber = np.array([row["ber"] for row in rows])
        positive.extend(value for row in rows for value in (row["ber"], row["ber_ci_low"], row["ber_ci_high"], row["frame_ber_min"]) if value > 0)
        positive.extend(replica["ber_ci_high"] for p in run.powers for replica in run.frames[p][method] if replica["bit_errors"] == 0)
        color = COLORS[method]
        plotted = np.where(ber > 0, ber, np.nan)
        ax.plot(run.powers, plotted, color=color, marker=MARKERS[method], linewidth=1.7, markersize=5, label=LABELS[method])
        for power, row in zip(run.powers, rows):
            for replica in run.frames[power][method]:
                if replica["ber"] > 0:
                    ax.plot(power, replica["ber"], marker="_", color=color, alpha=.45, markersize=11, linestyle="none", zorder=2)
                else:
                    upper = replica["ber_ci_high"]
                    ax.plot(power, upper, marker="v", markerfacecolor="white", markeredgecolor=color, alpha=.35, markersize=4)
                    ax.annotate("", (power, upper / 1.5), (power, upper), arrowprops={"arrowstyle": "->", "color": color, "lw": .7, "alpha": .3})
            if row["bit_errors"]:
                ax.vlines(power, row["ber_ci_low"], row["ber_ci_high"], color=color, linewidth=1.4, alpha=.85)
            else:
                # Zero is not representable on a log axis: mark the actual upper
                # counting bound and a downward arrow, never a fabricated BER.
                upper = row["ber_ci_high"]
                ax.plot(power, upper, marker="v", markerfacecolor="white", markeredgecolor=color, markersize=7)
                ax.annotate("", (power, upper / 2.5), (power, upper), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
    ax.set_yscale("log")
    ax.set_ylim(min(positive) / 3.2, min(.8, max(run.metrics[p][m]["ber_ci_high"] for p in run.powers for m in METHODS) * 2))
    ax.set_xticks(run.powers)
    ax.set_xlabel("送信電力［dBm、X/Y偏波の合計］")
    ax.set_ylabel("ビット誤り率 BER（小さいほどよい）")
    ax.grid(True, which="major", color="#E6EBF0", linewidth=.8)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=9)
    fig.suptitle("受信したビットを直接数えた補償性能", x=.10, ha="left", fontsize=17, fontweight="bold")
    fig.text(.10, .12, "点・線：全フレームの合算 BER　縦線：95%計数区間　短い横線：各フレームの BER", fontsize=9, color=MUTED)
    fig.text(.10, .075, "縦軸は対数：1目盛りで10倍。白い下向き印・矢印は0誤り時の上限（BER=0の点ではない）。", fontsize=9, color=MUTED)
    fig.text(.10, .025, frame_note(run) + " 誤りの独立性を仮定した区間で、相関は保証しない。", fontsize=8.5, color=MUTED)
    return fig, {"title_ja": "送信電力とビット誤り率", "caption_ja": "同じ未学習フレーム・同じ採点区間を4方式で比較。95%の計数区間は独立ビット近似で、独立フレームのばらつきも横線で示す。", "zero_error_policy": "upper confidence bound with downward arrow; no fabricated/clamped measured BER"}


def figure_constellations(run: RunData, power: float):
    diagnostic = run.constellations[power]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.985, bottom=.26, top=.72, wspace=.12)
    samples = [diagnostic[method][:, 0] for method in METHODS]
    limit = max(1.35, max(np.max(np.abs(np.r_[array.real, array.imag])) for array in samples) * 1.03)
    intended = np.unique(diagnostic["tx_symbols"][:, 0])
    for ax, method, values in zip(axes, METHODS, samples):
        ax.scatter(values.real, values.imag, s=3, color=COLORS[method], alpha=.45, rasterized=True, edgecolors="none")
        ax.scatter(intended.real, intended.imag, s=48, color=INK, marker="+", linewidths=1.1, zorder=5)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="同相 I（正規化振幅）")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.set_title(LABELS[method], color=COLORS[method], pad=10, fontweight="bold")
        ax.grid(True, color="#EDF0F4", linewidth=.6)
    axes[0].set_ylabel("直交 Q（正規化振幅）")
    fig.suptitle(f"同じ受信点で見る補償の違い：{power:+g} dBm", x=.055, ha="left", fontsize=17, fontweight="bold")
    fig.text(.055, .86, "＋ は送った16種類の位置。色の点は受信後の位置。位置が重なるほど誤判定を避けやすい。", fontsize=10, color=MUTED)
    fig.text(.055, .12, f"X偏波、最初の未学習フレームから等間隔の同じ {len(diagnostic['indices']):,} 点。4図の軸範囲・縮尺は共通。", fontsize=9, color=MUTED)
    fig.text(.055, .065, "図示電力は事前指定（既定 0 dBm）。この小標本だけでは BER の良し悪しを判定しない。", fontsize=9, color=MUTED)
    fig.text(.055, .02, run.scope, fontsize=9, color=MUTED)
    return fig, {"title_ja": "同じ受信点の配置", "caption_ja": f"{power:+g} dBmのX偏波。最初のテストフレームの保存済み等間隔インデックスを全方式で共有し、全点を表示。十字は送信した16QAM位置。軸は全方式で共通。", "power_dbm": power, "polarization": "X", "n_points": len(diagnostic["indices"]), "sample_selection": "saved first test frame, identical evenly spaced indices", "xylim": [-float(limit), float(limit)]}


def figure_validation(run: RunData):
    selections = [run.selections[p] for p in run.powers]
    orders = {method: [row["candidate_id"] for row in selections[0]["candidates"] if row["method"] == method] for method in ("linear", "esn")}
    all_rows = [row for selection in selections for row in selection["candidates"]]
    positive = [row["ber"] for row in all_rows if row["ber"] > 0]
    lower = min(positive) / 1.4 if positive else 1e-6
    upper = max(max(positive, default=1e-5), lower * 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [max(4, len(orders['linear'])), len(orders['esn'])]})
    fig.subplots_adjust(left=.075, right=.82, bottom=.30, top=.76, wspace=.17)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#EEF2F6")
    for ax, method in zip(axes, ("linear", "esn")):
        ids = orders[method]
        matrix = np.zeros((len(run.powers), len(ids)))
        zero_cells = []
        for pi, selection in enumerate(selections):
            lookup = {row["candidate_id"]: row for row in selection["candidates"] if row["method"] == method}
            if set(lookup) != set(ids):
                raise ValueError("Validation candidate grids differ between powers")
            for ci, candidate in enumerate(ids):
                matrix[pi, ci] = lookup[candidate]["ber"]
                if lookup[candidate]["bit_errors"] == 0:
                    zero_cells.append((ci, pi))
        heat = ax.imshow(np.ma.masked_equal(matrix, 0), norm=LogNorm(lower, upper), cmap=cmap, aspect="auto", interpolation="nearest")
        for x, y in zero_cells:
            ax.text(x, y, "0", ha="center", va="center", fontsize=8, color=MUTED)
        for pi, selection in enumerate(selections):
            ci = ids.index(selection["selected"][method])
            ax.add_patch(Rectangle((ci-.43, pi-.40), .86, .80, fill=False, edgecolor="#F0A45B", linewidth=2.1))
            ax.plot(ci, pi, marker="*", markersize=7, color="#FFFFFF", markeredgecolor=INK, markeredgewidth=.5)
        if method == "linear":
            ticks, labels = list(range(len(ids))), [str(i+1) for i in range(len(ids))]
        else:
            groups = [candidate.split("-ridge")[0] for candidate in ids]
            unique = list(dict.fromkeys(groups))
            ticks = [np.mean([i for i,g in enumerate(groups) if g == group]) for group in unique]
            labels = [str(i+1) for i in range(len(unique))]
            for group in unique[1:]:
                ax.axvline(groups.index(group)-.5, color="white", linewidth=1)
        ax.set_xticks(ticks, labels)
        ax.set_yticks(range(len(run.powers)), [f"{p:+g}" for p in run.powers])
        ax.set_title(LABELS[method], color=COLORS[method], fontweight="bold", pad=10)
        ax.set_xlabel("リッジ候補番号" if method == "linear" else "内部状態候補番号（各群内はリッジ強度の順）")
    axes[0].set_ylabel("送信電力［dBm、XY合計］")
    colorax = fig.add_axes([.855, .31, .018, .44])
    colorbar = fig.colorbar(heat, cax=colorax, label="検証用データの BER（対数）")
    ticks = np.geomspace(lower, upper, 3)
    colorbar.set_ticks(ticks, labels=[f"{value:.1e}" for value in ticks])
    colorbar.minorticks_off()
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("学習候補の選択に使ったのは検証用データ", x=.075, ha="left", fontsize=17, fontweight="bold")
    fig.text(.075, .87, "各マスは1候補。枠・★ が選ばれた候補。テスト用データで選び直していない。", fontsize=10, color=MUTED)
    ridges = ", ".join(f"{value:g}" for value in run.config["esn_ridges"])
    fig.text(.075, .16, f"ESNの各群内リッジ：{ridges}。0誤りのマスは『0』と表示し、正の値に置き換えない。", fontsize=9, color=MUTED)
    fig.text(.075, .105, "選択順：検証 BER → EVM → 保存する重みの量 → 候補 ID。読出し重みは一括で解くため epoch 曲線はない。", fontsize=9, color=MUTED)
    fig.text(.075, .04, f"独立した検証 {run.config['validation_frames']} フレーム／電力。{run.scope}。全候補は selection.json に保存。", fontsize=9, color=MUTED)
    return fig, {"title_ja": "検証データによる候補選択", "caption_ja": "全候補の検証BERと、事前に定めたルールで選んだ候補を表示。テストBERや架空の学習epoch曲線は使っていない。", "candidate_order": orders, "selection_split": "validation", "zero_cells": "literal zero annotation, masked color"}


def figure_resources(run: RunData):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.095, right=.98, bottom=.27, top=.75, wspace=.30)
    for method in METHODS:
        times = [run.metrics[p][method]["inference_seconds"] / run.metrics[p][method]["n_frames"] for p in run.powers]
        label = LABELS[method] + (" / GPU" if method == "calibrated_dbp" and run.config["backend"] == "cupy" else " / CPU")
        axes[0].plot(run.powers, times, color=COLORS[method], marker=MARKERS[method], label=label, linewidth=1.5, markersize=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("1テストフレーム当たりの追加処理［秒、対数］")
    axes[0].set_title("保存された処理時間の平均", fontweight="bold")
    axes[0].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(.01, .32))
    for method in ("linear", "esn"):
        axes[1].plot(run.powers, [run.metrics[p][method]["model_array_bytes"] / 1024 for p in run.powers], color=COLORS[method], marker=MARKERS[method], label=LABELS[method], linewidth=1.5)
    axes[1].set_ylabel("保存するモデル配列［KiB］")
    axes[1].set_title("重み・固定変換の量", fontweight="bold")
    axes[1].legend(fontsize=9, frameon=False)
    for ax in axes:
        ax.set_xticks(run.powers)
        ax.set_xlabel("送信電力［dBm、XY合計］")
        ax.grid(True, which="major", color="#E6EBF0", linewidth=.8)
    fig.suptitle("性能とは別に、追加処理の費用を測る", x=.095, ha="left", fontsize=17, fontweight="bold")
    fig.text(.095, .86, "共通の通信路生成・EDC後の追加処理。DBPだけは逆伝搬を含む。異なる装置の比較に注意。", fontsize=9.5, color=MUTED)
    fig.text(.095, .155, f"1フレームは {run.config['n_symbols']:,} symbols／偏波。ディスク入出力とモデル選択の時間は含まない。", fontsize=9, color=MUTED)
    fig.text(.095, .10, "この秒数は通信装置の実時間遅延・スループットではない。配列量はピークRAMや実行コード量ではない。", fontsize=9, color=MUTED)
    fig.text(.095, .045, run.scope + "。学習時間は候補群で共有するため selection.json の群単位で記録。", fontsize=9, color=MUTED)
    return fig, {"title_ja": "追加処理時間と保存重み量", "caption_ja": "同じテストフレームで記録した追加処理時間の平均と、線形/ESNの保存モデル配列量。CPUのESNとCuPy GPUのDBPを含み、装置性能や実時間処理の優劣は示さない。", "timing_scope": "extra receiver processing after shared channel/EDC, excludes disk I/O, includes DBP inverse propagation", "memory_scope": "stored model arrays, not peak RAM"}


def qualification_figure(directory: Path):
    required = [directory / name for name in ("refinement.json", "boundaries.json", "production_recommendation.json")]
    completions = [directory / f"completion_{stage}.json" for stage in ("controls", "refinement", "boundaries")]
    if not all(path.is_file() for path in completions):
        raise ValueError("Numerical qualification stages are not marked complete")
    if not all(path.is_file() for path in required):
        raise ValueError("Numerical qualification is incomplete; do not plot partial evidence")
    refinement, boundaries, recommendation = map(read_json, required)
    qualification_manifest = directory / "manifest.json"
    provenance = read_json(qualification_manifest)
    for name, expected in provenance.get("artifact_sha256", {}).items():
        artifact = directory / name
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"Numerical qualification artifact changed: {artifact}")
    required.append(qualification_manifest)
    for name, expected in recommendation.get("input_result_sha256", {}).items():
        source = directory / name
        if not source.is_file() or sha256(source) != expected:
            raise ValueError("Numerical recommendation source hash differs")
    required.extend(directory / name for name in recommendation.get("input_result_sha256", {}) if directory / name not in required)
    pairs = refinement["pair_summary"]
    boundary_rows = boundaries["summary"]
    if not isinstance(pairs, list) or not isinstance(boundary_rows, list):
        raise ValueError("Expected completed numerical pair/boundary summaries")
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.4))
    fig.subplots_adjust(left=.095, right=.98, bottom=.30, top=.76, wspace=.28)
    kinds = list(dict.fromkeys(row["kind"] for row in pairs))
    kind_labels = {"step": "伝搬刻み", "sampling": "サンプル数", "accepted_initial_reference": "初期 vs 8sps/0.5km", "initial_to_finest": "初期 vs 8sps/0.25km"}
    for i, kind in enumerate(kinds):
        rows = [row for row in pairs if row["kind"] == kind]
        powers = sorted({row["power_dbm"] for row in rows})
        maximum = [max(row["maximum_abs_snr_difference_db"] for row in rows if row["power_dbm"] == power) for power in powers]
        axes[0].scatter(powers, maximum, s=40, marker=("o", "s", "^", "D")[i%4], label=kind_labels.get(kind, kind))
    for ax in axes:
        ax.axhline(.1, color="#B95D40", linestyle="--", linewidth=1.4, label="目標 0.1 dB")
        ax.set_ylabel("基準との差の最大値［有効SNR、dB］")
        ax.grid(True, color="#E6EBF0", linewidth=.8)
        ax.set_yscale("log")
    axes[0].set_title("刻み幅・サンプル数の変更", fontweight="bold")
    axes[0].set_xlabel("送信電力［dBm、XY合計］")
    axes[0].set_xticks(sorted({row["power_dbm"] for row in pairs}))
    axes[0].legend(fontsize=8, frameon=False)
    lengths = sorted({row["n_symbols"] for row in boundary_rows})
    for i, length in enumerate(lengths):
        rows = [row for row in boundary_rows if row["n_symbols"] == length]
        guards = sorted({row["guard_symbols"] for row in rows})
        maximum = [max(row["maximum_abs_snr_difference_db"] for row in rows if row["guard_symbols"] == guard) for guard in guards]
        axes[1].plot(guards, maximum, marker=("o", "s", "^")[i%3], label=f"{length:,} symbols", linewidth=1.3)
    axes[1].set_xticks(sorted({row["guard_symbols"] for row in boundary_rows}))
    axes[1].set_xlabel("両端で採点から除外する数［symbols／端］")
    axes[1].set_title("フレーム長・端の除外範囲", fontweight="bold")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("計算の細かさを変えて、結果のずれを調べる", x=.095, ha="left", fontsize=17, fontweight="bold")
    fig.text(.095, .865, "数値精度の確認。学習モデルの性能評価とは別のフレームを使用。小さいほど計算条件への依存が小さい。", fontsize=9.5, color=MUTED)
    fig.text(.095, .18, "各点は0/+6 dBm・雑音あり/なしの対応比較から最大差を集約。縦軸は対数（1目盛り10倍）。", fontsize=9, color=MUTED)
    fig.text(.095, .12, "図にない物理条件への収束は保証しない。条件を細かくしても、学習による改善の証明にはならない。", fontsize=9, color=MUTED)
    fig.text(.095, .06, "採用設定・確認できた範囲・残る制約は production_recommendation.json とレポート本文を参照。", fontsize=9, color=MUTED)
    return fig, {"title_ja": "数値条件の確認", "caption_ja": "刻み幅・サンプル数・フレーム長・端の除外数を独立に変えた比較。0/+6 dBm・雑音あり/なしの保存済み対応比較から最大差を集約。目標は有効SNR差0.1 dB以内とBERの統計的整合性。モデル性能のテストとは別。", "recommendation": recommendation}, required + completions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--constellation-power", type=float, default=0)
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--format", nargs="+", choices=("svg", "png", "pdf"), default=["svg", "png"])
    args = parser.parse_args()
    run = load_run(args.run, args.allow_pilot)
    output = args.output_dir.resolve()
    if run.pilot and output.is_relative_to(ROOT / "reports"):
        raise ValueError("Pilot figures belong in .work, not the final reports directory")
    if args.constellation_power not in run.powers:
        raise ValueError("Constellation power must be available; never silently pick a favorable power")
    font = select_font(args.font)
    style(font)
    output.mkdir(parents=True, exist_ok=True)
    renderers = [("result_ber", lambda: figure_ber(run)), ("result_constellations", lambda: figure_constellations(run, args.constellation_power)), ("result_validation", lambda: figure_validation(run)), ("result_resources", lambda: figure_resources(run))]
    inputs = list(run.inputs)
    figures = []
    for name, renderer in renderers:
        fig, record = renderer()
        record.update(name=name, outputs=[])
        for extension in dict.fromkeys(args.format):
            path = output / f"{name}.{extension}"
            metadata = {"Title": record["title_ja"]}
            if extension == "svg":
                metadata.update(Date=None, Description=record["caption_ja"])
            elif extension == "pdf":
                metadata.update(CreationDate=None, ModDate=None, Subject=record["caption_ja"])
            fig.savefig(path, dpi=args.dpi, metadata=metadata, facecolor="white")
            record["outputs"].append({"path": display_path(path), "format": extension, "sha256": sha256(path), "bytes": path.stat().st_size})
        plt.close(fig)
        figures.append(record)
    if args.qualification:
        fig, record, more_inputs = qualification_figure(args.qualification.resolve())
        inputs.extend(more_inputs)
        record.update(name="result_qualification", outputs=[])
        for extension in dict.fromkeys(args.format):
            path = output / f"result_qualification.{extension}"
            metadata = {"Title": record["title_ja"]}
            if extension == "svg":
                metadata.update(Date=None)
            elif extension == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            fig.savefig(path, dpi=args.dpi, metadata=metadata, facecolor="white")
            record["outputs"].append({"path": display_path(path), "format": extension, "sha256": sha256(path), "bytes": path.stat().st_size})
        plt.close(fig)
        figures.append(record)
    command = [".venv/bin/python", "scripts/visualize_results.py", "--run", display_path(run.path)]
    if args.qualification:
        command += ["--qualification", display_path(args.qualification)]
    command += ["--output-dir", display_path(output), "--constellation-power", f"{args.constellation_power:g}", "--dpi", str(args.dpi), "--format", *dict.fromkeys(args.format)]
    if args.allow_pilot:
        command += ["--allow-pilot"]
    if args.font:
        command += ["--font", display_path(args.font)]
    manifest = {"schema_version": 1, "kind": "exploratory_pilot_results" if run.pilot else "held_out_receiver_results", "run_id": run.manifest["run_id"], "source_run": display_path(run.path), "scope": run.scope,
        "generator": {"path": display_path(Path(__file__)), "sha256": sha256(Path(__file__)), "font_helper_path": "scripts/visualize_concepts.py", "font_helper_sha256": sha256(ROOT / "scripts/visualize_concepts.py")}, "regeneration_command": shlex.join(command),
        "inputs": [{"path": display_path(path), "sha256": sha256(path)} for path in sorted(set(inputs))],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "font": {"family": font.family, "regular_sha256": sha256(font.regular), "bold_sha256": sha256(font.bold)}, "comparison_palette": COLORS, "figures": figures,
        "visual_inspection": "Not automated; inspect exported PNGs and record the outcome with the final review."}
    (output / "results_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(figures)} result figures from {run.manifest['run_id']} to {display_path(output)}")


if __name__ == "__main__":
    main()
