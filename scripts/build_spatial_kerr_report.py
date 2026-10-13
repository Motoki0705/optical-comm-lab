#!/usr/bin/env python3
"""Build figures and an offline Japanese report from saved spatial Kerr films.

No field propagation or Blender rendering is performed. Missing inputs, invalid
array contracts, or mismatched hashes prevent publication of a completed page.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import shlex
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = ("linear", "weak_reference", "intermediate_kerr", "educational_kerr")
LABELS = {
    "linear": "Kerr なし",
    "weak_reference": "弱い参照条件",
    "intermediate_kerr": "仮の中間条件",
    "educational_kerr": "仮の強条件",
}
COLORS = {"linear": "#536472", "weak_reference": "#2383a5",
          "intermediate_kerr": "#b2722d", "educational_kerr": "#ac3b4e"}
FILMS = (
    {"file": "01_reflection.mp4", "number": "01", "title": "境界で、進む向きが戻る。",
     "lead": "コアの縁に近づく光の流れと、その後の折り返しを見る。",
     "caption": "線は計算した場から得た、近似的なエネルギーの流れ。個々の光子の軌道ではない。輪郭は同じ強度の場所をつないだ面で、光そのものの外殻ではない。"},
    {"file": "02_kerr_comparison.mp4", "number": "02", "title": "光の強さが、その先の分布を変える。",
     "lead": "同じ入射光を、Kerr なし・ありで並べる。反射した後の広がりと位置を見比べる。",
     "caption": "明るい領域ほど屈折率が上がり、位相に差がつく。回折・境界・干渉と組み合わさり、分布と流れに差が現れる。"},
    {"file": "03_phase_change.mp4", "number": "03", "title": "波の山谷のずれが、重なり方を変える。",
     "lead": "色で読むのは、Kerr なしの場合に対する位相の差。",
     "caption": "位相は波の山谷の位置を表す。場所ごとにずれ方が違うと、波が重なるときの強め合い・弱め合いも変わる。"},
)

CSS = """
:root{color-scheme:light;--ink:#18313d;--muted:#65747a;--paper:#fbfaf6;--edge:#dce3e3;--accent:#277b8a;--warm:#b46c35;--width:1120px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.8 -apple-system,BlinkMacSystemFont,'Noto Sans CJK JP','Yu Gothic',Meiryo,sans-serif}a{color:var(--accent);text-underline-offset:.22em}a:hover{color:#a75f2f}a:focus-visible,video:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid #b46c35;outline-offset:4px}.skip{position:absolute;top:-100px;left:16px;padding:8px;background:white;z-index:10}.skip:focus{top:10px}.shell{max-width:var(--width);margin:auto;padding:0 36px}.topbar{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid var(--edge);padding-top:20px;padding-bottom:18px;font-size:.77rem}.topbar span{color:var(--muted)}.topbar nav{display:flex;gap:22px}.intro{padding-top:38px;padding-bottom:36px}.eyebrow{margin:0 0 10px;color:var(--accent);font-size:.73rem;letter-spacing:.1em}h1{font:500 clamp(1.75rem,3.8vw,2.9rem)/1.45 'Noto Serif CJK JP','Yu Mincho',serif;margin:0 0 14px;text-wrap:balance}.lead{margin:0;max-width:860px;font-size:1rem}.scope{color:var(--muted);font-size:.83rem;margin:16px 0 0;max-width:880px}.chapters{display:flex;gap:28px;flex-wrap:wrap;margin-top:22px;font-size:.83rem}.chapters a{text-decoration:none}.chapter{margin-bottom:48px}.chapter-head{display:flex;gap:16px;align-items:baseline}.number{font:italic 1.35rem Georgia,serif;color:var(--warm)}h2{font-weight:500;font-size:clamp(1.3rem,2.4vw,1.75rem);line-height:1.55;margin:0 0 12px;text-wrap:balance}.chapter p{margin:0 0 17px;font-size:.91rem;max-width:950px}figure{margin:0}.screen{background:#0d171e;line-height:0;border:1px solid #14252f}.screen video{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain}.film-meta{display:flex;gap:18px;justify-content:space-between;align-items:center;margin:9px 0 13px;font-size:.76rem;color:var(--muted)}.film-meta .links{display:flex;gap:16px;flex-wrap:wrap}.rate{display:flex;gap:8px;align-items:center;white-space:nowrap}select{font:inherit;border:1px solid #bdcdcf;color:var(--ink);background:white;padding:3px 8px;border-radius:3px}.caption{font-size:.88rem;line-height:1.8;color:var(--muted);max-width:990px}.takeaway{background:#edf3f2;border-left:3px solid var(--accent);padding:18px 22px;margin:5px 0 30px;font-size:.96rem}.results{margin:0 0 48px;padding-top:12px}.results .reading{font-size:.88rem;color:var(--muted);margin:0 0 18px}.plot{margin-bottom:22px;background:white;border:1px solid var(--edge)}.plot img{display:block;width:100%;height:auto}.plot figcaption{padding:12px 18px;font-size:.81rem;color:var(--muted)}.plot figcaption a{white-space:nowrap}.tables{display:grid;grid-template-columns:1.4fr 1fr;gap:24px;margin-top:28px}.table-box{min-width:0}.table-box h3{font-weight:500;font-size:.97rem;margin:0 0 9px}.table-scroll{overflow-x:auto;max-width:100%;border-top:1px solid var(--edge)}table{border-collapse:collapse;width:100%;font-size:.79rem;line-height:1.7;font-variant-numeric:tabular-nums}th,td{padding:10px 9px;border-bottom:1px solid var(--edge);text-align:left;vertical-align:top}th{font-weight:600;background:#f0f4f3}td.value{white-space:nowrap;text-align:right}.table-note{font-size:.76rem;color:var(--muted);margin-top:10px}.closing{border-top:1px solid var(--edge);padding:26px 0 0;margin-top:30px;font-size:.95rem;max-width:970px}.supplement{margin-bottom:35px}details{border-top:1px solid var(--edge);padding:16px 0}details:last-child{border-bottom:1px solid var(--edge)}summary{cursor:pointer;font-size:.87rem;font-weight:600;margin-left:18px}.detail-body{font-size:.83rem;color:var(--muted);padding:14px 0 0}.detail-body p{margin:0 0 13px}.downloads{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}.downloads h3{font-size:.85rem;font-weight:600;margin:0 0 7px}.downloads a{display:block;margin-bottom:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f1;border:1px solid var(--edge);padding:14px;font-size:.73rem}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.foot{padding-top:15px;padding-bottom:30px;font-size:.73rem;color:var(--muted);display:flex;justify-content:space-between;flex-wrap:wrap;gap:15px}
@media(max-width:720px){.shell{padding-inline:20px}.intro{padding-top:28px;padding-bottom:28px}.topbar{font-size:.7rem;gap:10px}.topbar nav{gap:14px}.topbar span{display:none}.lead{font-size:.92rem}.scope{font-size:.76rem}.chapters{gap:15px;font-size:.74rem}.chapter{margin-bottom:35px}.chapter p{font-size:.85rem}.chapter-head{gap:11px}.screen{margin-inline:-20px;border-left:0;border-right:0}.film-meta{font-size:.69rem;align-items:flex-start;gap:10px}.film-meta .links{gap:9px}.caption{font-size:.81rem}.takeaway{padding:14px 16px;font-size:.88rem}.plot figcaption{padding:10px 12px;font-size:.73rem}.tables{grid-template-columns:1fr;gap:24px}table{font-size:.73rem}th,td{padding:9px 7px}.closing{font-size:.87rem}.downloads{grid-template-columns:1fr;gap:16px}.detail-body{font-size:.77rem}.foot{font-size:.67rem}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:white;color:#111;font-size:10pt}.shell{max-width:none;padding:0}.topbar,.chapters,.film-meta,.skip{display:none}.intro{padding:0 0 15px}h1{font-size:24pt}h2{font-size:17pt}.scope,.caption,.reading,.table-note,.detail-body{color:#444}.chapter{break-inside:avoid;margin-bottom:22px}.screen video{max-height:70mm}.plot{break-inside:avoid}.tables{grid-template-columns:1fr}details{break-inside:avoid}.foot{padding:10px 0}}
.case-controls{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 14px}.case-controls button{font:inherit;font-size:.82rem;border:1px solid #b7cace;border-radius:4px;background:white;color:var(--ink);padding:9px 14px;cursor:pointer}.case-controls button[aria-pressed=true]{background:var(--ink);color:white;border-color:var(--ink)}.case-controls button:focus-visible{outline:3px solid var(--warm);outline-offset:3px}.case-controls button:hover{border-color:var(--accent)}.switch-label{font-size:.8rem;color:var(--muted);margin:0 0 12px}.print-comparison{display:none}[hidden]{display:none!important}
@media(max-width:720px){.case-controls{display:grid;grid-template-columns:1fr 1fr}.case-controls button{font-size:.75rem;padding:9px 7px}}
@media print{.case-controls,.switch-stack,.switch-label{display:none}.print-comparison{display:block}}
.entrance{margin:0 0 38px}.entrance img{display:block;width:100%;height:auto;aspect-ratio:16/9;background:#0d171e}.entrance figcaption{margin-top:12px;font-size:.85rem;color:var(--muted);max-width:960px}.entrance figcaption span{display:block;font-size:.76rem;margin-top:5px}
@media(max-width:720px){.entrance{margin-bottom:30px}.entrance figcaption{font-size:.79rem}.entrance figcaption span{font-size:.73rem}}
@media print{.entrance{break-inside:avoid;margin-bottom:20px}.entrance img{max-height:70mm;object-fit:contain}}
"""

JS = """
const videos = [...document.querySelectorAll('video')];
videos.forEach(video => {
  video.addEventListener('play', () => videos.forEach(other => { if (other !== video) other.pause(); }));
  video.addEventListener('keydown', event => {
    if (event.target !== video || event.defaultPrevented) return;
    if (event.key.toLowerCase() === 'k') {
      event.preventDefault(); video.paused ? video.play().catch(() => {}) : video.pause();
    }
  });
});
document.querySelectorAll('[data-video-rate]').forEach(select => {
  select.addEventListener('change', () => { document.getElementById(select.dataset.videoRate).playbackRate = Number(select.value); });
});
document.querySelectorAll('[data-case-button]').forEach(button => {
  button.addEventListener('click', () => {
    const choice = button.dataset.caseButton;
    document.querySelectorAll('[data-case-button]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
    document.querySelectorAll('[data-case-panel]').forEach(panel => { panel.hidden = panel.dataset.casePanel !== choice; });
    document.getElementById('selected-case').textContent = button.dataset.caseDescription;
  });
});
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {display(path)}")
    return value


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def relative(path: Path, output: Path) -> str:
    return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/.")


def link(label: str, path: Path, output: Path, *, download: bool = False) -> str:
    flag = " download" if download else ""
    return f'<a href="{relative(path, output)}"{flag}>{escape(label)}</a>'


def number(value: Any, *, digits: int = 3) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("A displayed measurement is nonfinite")
    if parsed != 0 and (abs(parsed) < .001 or abs(parsed) >= 10000):
        return f"{parsed:.{digits - 1}e}"
    return f"{parsed:.{digits}g}"


def local_file(name: str, base: Path) -> Path:
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"Missing local input or invalid manifest path: {name}")
    return path


def verify_hash(path: Path, expected: str, inputs: dict[Path, str]) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {display(path)}")
    inputs[path.resolve()] = expected


def load_sections(path: Path) -> dict[str, np.ndarray]:
    keys = ("x_um", "y_um", "z_um", "central_psi", "cross_z_um", "cross_psi")
    with np.load(path, allow_pickle=False) as saved:
        missing = set(keys) - set(saved.files)
        if missing:
            raise ValueError(f"Missing section keys in {path.name}: {sorted(missing)}")
        result = {key: saved[key] for key in keys}
    for key in ("x_um", "y_um", "z_um", "cross_z_um"):
        axis = result[key]
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
            raise ValueError(f"Invalid increasing axis {key} in {path.name}")
    for key, shape in (("central_psi", (len(result["z_um"]), len(result["x_um"]))),
                       ("cross_psi", (len(result["cross_z_um"]), len(result["y_um"]), len(result["x_um"])) )):
        if result[key].shape != shape or result[key].dtype != np.complex128 or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid complex128 field {key} in {path.name}")
    return result


def load_flows(path: Path) -> dict[str, np.ndarray]:
    keys = ("position_um", "valid", "seed_quantiles", "seed_position_um", "intensity", "velocity_xy_um_per_um")
    with np.load(path, allow_pickle=False) as saved:
        if not set(keys).issubset(saved.files):
            raise ValueError(f"Incomplete flow contract in {path.name}")
        result = {key: saved[key] for key in keys}
    shape = result["position_um"].shape
    if len(shape) != 3 or shape[0] != 25 or shape[-1] != 3 or result["valid"].shape != shape[:2]:
        raise ValueError(f"Invalid 25-seed XYZ flow shape in {path.name}")
    if result["valid"].dtype != np.bool_ or not np.isfinite(result["position_um"][result["valid"]]).all():
        raise ValueError(f"Invalid flow validity or positions in {path.name}")
    if result["seed_quantiles"].shape != (25, 2) or result["seed_position_um"].shape != (25, 3):
        raise ValueError(f"Invalid flow seed contract in {path.name}")
    if result["intensity"].shape != shape[:2] or result["velocity_xy_um_per_um"].shape != (*shape[:2], 2):
        raise ValueError(f"Invalid intensity/velocity flow shape in {path.name}")
    return result


def resolve_font(path: Path | None) -> tuple[font_manager.FontProperties, dict[str, str]]:
    if path is not None:
        selected = path.resolve()
        if not selected.is_file():
            raise ValueError(f"Missing font: {selected}")
    else:
        selected = None
        for name in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic"):
            try:
                selected = Path(font_manager.findfont(name, fallback_to_default=False)).resolve()
                break
            except ValueError:
                pass
        if selected is None:
            raise ValueError("Japanese font missing; provide --font /path/to/font.ttf")
    font = font_manager.FontProperties(fname=str(selected))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "svg.fonttype": "path", "pdf.fonttype": 42})
    return font, {"path": str(selected), "sha256": sha256(selected), "family": font.get_name()}


def save_figure(figure: plt.Figure, basename: str, stage: Path, formats: tuple[str, ...] = ("png", "svg")) -> list[Path]:
    destination = stage / "figures"
    destination.mkdir(exist_ok=True)
    outputs = []
    for extension in formats:
        path = destination / f"{basename}.{extension}"
        metadata = {"Software": "optical-comm-lab spatial Kerr report"} if extension == "png" else {"Date": None}
        figure.savefig(path, dpi=170, facecolor=figure.get_facecolor(), metadata=metadata)
        outputs.append(path)
    plt.close(figure)
    return outputs


def intensity_figure(sections: dict[str, dict[str, np.ndarray]], config: dict[str, Any], stage: Path,
                     intensity_max: float = 3.0) -> tuple[list[Path], dict[str, Any]]:
    radius = float(config["core_radius_um"])
    center = float(config["core_center_x_um"])
    view_half_width = 1.4 * radius
    first = sections["linear"]
    visible = abs(first["x_um"] - center) <= view_half_width
    values = {case: abs(sections[case]["central_psi"][:, visible]) ** 2 for case in CASES}
    observed_max = max(float(value.max()) for value in values.values())
    if observed_max > intensity_max:
        raise ValueError(f"Saved central intensity {observed_max:g} exceeds --intensity-max {intensity_max:g}; choose an explicit shared scale")
    vmax = intensity_max
    cmap = LinearSegmentedColormap.from_list("spatial_intensity", ["#081421", "#16506a", "#41bbc2", "#ffebba"])
    norm = Normalize(0, vmax)
    fig, axes = plt.subplots(4, 1, figsize=(10.9, 8.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.83, bottom=.08, top=.965, hspace=.24)
    mesh = None
    for ax, case in zip(axes, CASES):
        mesh = ax.pcolormesh(first["z_um"] / 1000, first["x_um"][visible], values[case].T,
                            cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        for boundary in (center - radius, center + radius):
            ax.axhline(boundary, color="#ffffff", ls=(0, (5, 4)), lw=.8, alpha=.8)
        ax.set_title(LABELS[case], loc="left", fontsize=10, pad=5)
        ax.set_ylabel("横の位置 x [µm]")
        ax.set_ylim(center - view_half_width, center + view_half_width)
        ax.set_yticks([center - radius, center, center + radius])
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("進んだ距離 z [mm]")
    color_axis = fig.add_axes([.86, .18, .022, .66])
    colorbar = fig.colorbar(mesh, cax=color_axis)
    colorbar.set_label("強度（入力ピーク = 1）", labelpad=12)
    return save_figure(fig, "four_case_intensity", stage), {
        "quantity": "abs(central_psi)**2", "section": "y=0", "normalization": "shared input peak=1; no per-case/per-distance normalization",
        "intensity_scale": [0.0, vmax], "x_view_um": [center - view_half_width, center + view_half_width],
        "observed_visible_section_max": observed_max, "clipped_intensity_values": 0,
        "field_interpolation": "saved central_psi uses complex linear interpolation to y=0; nearest-cell plotting without smoothing",
        "boundary_lines": "linear core boundary",
    }


def case_intensity_figures(sections: dict[str, dict[str, np.ndarray]], config: dict[str, Any], stage: Path,
                           encoding: dict[str, Any]) -> list[Path]:
    """Four saved cases share exactly the static figure's coordinates and scale."""
    first = sections["linear"]
    lower, upper = encoding["x_view_um"]
    visible = (first["x_um"] >= lower) & (first["x_um"] <= upper)
    center = float(config["core_center_x_um"])
    radius = float(config["core_radius_um"])
    cmap = LinearSegmentedColormap.from_list("spatial_switch_intensity", ["#081421", "#16506a", "#41bbc2", "#ffebba"])
    outputs = []
    for case in CASES:
        fig, ax = plt.subplots(figsize=(10.9, 2.9))
        fig.subplots_adjust(left=.105, right=.825, bottom=.24, top=.88)
        values = abs(sections[case]["central_psi"][:, visible]) ** 2
        artist = ax.pcolormesh(first["z_um"] / 1000, first["x_um"][visible], values.T,
                              cmap=cmap, vmin=encoding["intensity_scale"][0], vmax=encoding["intensity_scale"][1],
                              shading="nearest", rasterized=True)
        for boundary in (center - radius, center + radius):
            ax.axhline(boundary, color="#ffffff", ls=(0, (5, 4)), lw=.8, alpha=.8)
        ax.set(xlabel="進んだ距離 z [mm]", ylabel="横の位置 x [µm]", ylim=(lower, upper))
        ax.set_yticks([center - radius, center, center + radius])
        ax.set_title(LABELS[case], loc="left", fontsize=11, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        color_axis = fig.add_axes([.86, .24, .021, .64])
        bar = fig.colorbar(artist, cax=color_axis)
        bar.set_label("強度（入力ピーク = 1）", labelpad=12)
        outputs.extend(save_figure(fig, f"intensity_{case}", stage))
        mobile, ax = plt.subplots(figsize=(5.3, 3.6))
        mobile.subplots_adjust(left=.17, right=.79, bottom=.23, top=.87)
        artist = ax.pcolormesh(first["z_um"] / 1000, first["x_um"][visible], values.T, cmap=cmap,
                              vmin=encoding["intensity_scale"][0], vmax=encoding["intensity_scale"][1],
                              shading="nearest", rasterized=True)
        for boundary in (center - radius, center + radius):
            ax.axhline(boundary, color="#ffffff", ls=(0, (5, 4)), lw=.8, alpha=.8)
        ax.set_ylim(lower, upper)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([center - radius, center, center + radius])
        ax.set_xlabel("距離 z [mm]", fontsize=13)
        ax.set_ylabel("横 x [µm]", fontsize=13)
        ax.set_title(LABELS[case], loc="left", fontsize=14, pad=8)
        ax.tick_params(labelsize=12)
        color_axis = mobile.add_axes([.84, .23, .025, .64])
        bar = mobile.colorbar(artist, cax=color_axis, ticks=np.linspace(*encoding["intensity_scale"], 4))
        bar.set_label("強度（入力ピーク = 1）", fontsize=12, labelpad=9)
        bar.ax.tick_params(labelsize=12)
        outputs.extend(save_figure(mobile, f"intensity_{case}_mobile", stage, ("png",)))
    return outputs


def phase_figure(sections: dict[str, dict[str, np.ndarray]], config: dict[str, Any], stage: Path, threshold: float,
                 intensity_max: float = 1.0) -> tuple[list[Path], dict[str, Any]]:
    linear, strong = sections["linear"], sections["educational_kerr"]
    radius = float(config["core_radius_um"])
    cx, cy = float(config["core_center_x_um"]), float(config["core_center_y_um"])
    index = len(linear["cross_z_um"]) - 1
    off, on = linear["cross_psi"][index], strong["cross_psi"][index]
    xx, yy = np.meshgrid(linear["x_um"], linear["y_um"])
    valid = (abs(off) ** 2 >= threshold) & (abs(on) ** 2 >= threshold)
    valid &= (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
    difference = np.ma.array(np.angle(on * off.conj()), mask=~valid)
    observed_max = max(float(np.max(abs(field) ** 2)) for field in (off, on))
    if observed_max > intensity_max:
        raise ValueError(f"Saved cross-section intensity {observed_max:g} exceeds --output-intensity-max {intensity_max:g}")
    peak = intensity_max
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 4.8))
    fig.subplots_adjust(left=.055, right=.98, bottom=.275, top=.875, wspace=.22)
    intensity_map = LinearSegmentedColormap.from_list("spatial_cross_intensity", ["#081421", "#16506a", "#41bbc2", "#ffebba"])
    artists = []
    for ax, field, label in zip(axes[:2], (off, on), ("Kerr なし", "仮の強条件")):
        artists.append(ax.pcolormesh(linear["x_um"], linear["y_um"], abs(field) ** 2, cmap=intensity_map,
                                     vmin=0, vmax=peak, shading="nearest", rasterized=True))
        ax.set_title(label, fontsize=11)
    phase_map = plt.get_cmap("twilight_shifted").copy()
    phase_map.set_bad("#e2e5e5")
    phase_artist = axes[2].pcolormesh(linear["x_um"], linear["y_um"], difference, cmap=phase_map,
                                    vmin=-np.pi, vmax=np.pi, shading="nearest", rasterized=True)
    axes[2].set_title("位相差（強条件 − なし）", fontsize=11)
    fig.suptitle(f"終端 z = {number(linear['cross_z_um'][index] / 1000)} mm", fontsize=11, y=.98)
    for ax in axes:
        ax.add_patch(plt.Circle((cx, cy), radius, color="#8d9698", fill=False, lw=.9))
        ax.set(xlim=(cx - 1.2 * radius, cx + 1.2 * radius), ylim=(cy - 1.2 * radius, cy + 1.2 * radius),
               xlabel="横の位置 x [µm]", aspect="equal")
        ax.set_xticks([-radius + cx, cx, radius + cx])
        ax.set_yticks([-radius + cy, cy, radius + cy])
    axes[0].set_ylabel("横の位置 y [µm]")
    for i, artist in enumerate(artists):
        position = axes[i].get_position()
        color_axis = fig.add_axes([position.x0, .15, position.width, .03])
        bar = fig.colorbar(artist, cax=color_axis, orientation="horizontal")
        bar.set_label("終端強度（入力ピーク = 1）", fontsize=9)
        bar.ax.tick_params(labelsize=9)
    position = axes[2].get_position()
    color_axis = fig.add_axes([position.x0, .15, position.width, .03])
    bar = fig.colorbar(phase_artist, cax=color_axis, orientation="horizontal",
                       ticks=[-np.pi, 0, np.pi])
    bar.ax.set_xticklabels(["−π", "0", "+π"])
    bar.set_label("位相差 [rad]", fontsize=8)
    fig.text(.5, .03, f"位相差は両方の強度が {threshold:g} 以上のコア内。灰色は対象外。",
             ha="center", fontsize=9, color="#65747a")
    outputs = save_figure(fig, "output_intensity_and_phase", stage)
    mobile = plt.figure(figsize=(5.3, 14.0))
    mobile.suptitle(f"終端 z = {number(linear['cross_z_um'][index] / 1000)} mm", fontsize=15, y=.992)
    for row, (field, label) in enumerate(((abs(off) ** 2, "Kerr なし"), (abs(on) ** 2, "仮の強条件"), (difference, "位相差（強条件 − なし）"))):
        bottom = .74 - row * .31
        ax = mobile.add_axes([.18, bottom, .71, .205])
        artist = ax.pcolormesh(linear["x_um"], linear["y_um"], field,
                              cmap=phase_map if row == 2 else intensity_map,
                              vmin=-np.pi if row == 2 else 0, vmax=np.pi if row == 2 else peak,
                              shading="nearest", rasterized=True)
        ax.add_patch(plt.Circle((cx, cy), radius, color="#8d9698", fill=False, lw=.9))
        ax.set(xlim=(cx - 1.2 * radius, cx + 1.2 * radius), ylim=(cy - 1.2 * radius, cy + 1.2 * radius), aspect="equal")
        ax.set_xticks([cx - radius, cx, cx + radius])
        ax.set_yticks([cy - radius, cy, cy + radius])
        ax.set_xlabel("横 x [µm]", fontsize=13, labelpad=3)
        ax.set_ylabel("横 y [µm]", fontsize=13, labelpad=3)
        ax.set_title(label, fontsize=15, pad=6)
        ax.tick_params(labelsize=12)
        position = ax.get_position()
        color_axis = mobile.add_axes([position.x0, bottom - .05, position.width, .012])
        bar = mobile.colorbar(artist, cax=color_axis, orientation="horizontal",
                              ticks=[-np.pi, 0, np.pi] if row == 2 else np.linspace(0, peak, 3))
        if row == 2:
            bar.ax.set_xticklabels(["−π", "0", "+π"])
        bar.ax.tick_params(labelsize=12, pad=2)
        bar.set_label("位相差 [rad]" if row == 2 else "終端強度（入力ピーク = 1）", fontsize=12, labelpad=3)
    mobile.text(.5, .003, f"位相：両方の強度が {threshold:g} 以上のコア内\n灰色は対象外", ha="center", va="bottom", fontsize=11, color="#65747a")
    outputs.extend(save_figure(mobile, "output_intensity_and_phase_mobile", stage, ("png",)))
    return outputs, {
        "cross_z_um": float(linear["cross_z_um"][index]), "phase_quantity": "arg(psi_educational_kerr * conj(psi_linear))",
        "phase_range_rad": [-math.pi, math.pi], "phase_valid": "both intensities >= fixed input-based threshold and inside core",
        "phase_intensity_threshold": threshold, "intensity_scale": [0.0, peak],
        "intensity_scale_scope": "output cross section only; shared OFF/ON fixed range, absolute input-peak=1 units; no case-maximum normalization",
        "observed_output_section_max": observed_max, "clipped_intensity_values": 0,
    }


class LocalReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths: set[str] = set()
        self.anchors: set[str] = set()
        self.used_anchors: set[str] = set()

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if attrs.get("id"):
            self.anchors.add(attrs["id"])
        for name in ("href", "src", "poster"):
            value = attrs.get(name)
            if not value:
                continue
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc:
                if tag != "a":
                    raise ValueError("Runtime images, video, and scripts must be local")
                continue
            if parsed.path:
                self.paths.add(unquote(parsed.path))
            elif parsed.fragment:
                self.used_anchors.add(parsed.fragment)
        for candidate in (attrs.get("srcset") or "").split(","):
            if not candidate.strip():
                continue
            parsed = urlsplit(candidate.strip().split()[0])
            if parsed.scheme or parsed.netloc:
                raise ValueError("Responsive runtime images must be local")
            self.paths.add(unquote(parsed.path))


def check_local_links(page: str, output: Path, staged_files: set[str]) -> int:
    references = LocalReferences()
    references.feed(page)
    for name in references.paths:
        if name in staged_files:
            continue
        if not (output / name).resolve().is_file():
            raise ValueError(f"Broken local report reference: {name}")
    if not references.used_anchors.issubset(references.anchors):
        raise ValueError("A report navigation anchor is missing")
    return len(references.paths)


def probe_video(path: Path) -> dict[str, Any]:
    process = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                              "stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(path)],
                             check=True, capture_output=True, text=True)
    observed = json.loads(process.stdout)
    stream = observed["streams"][0]
    numerator, denominator = map(float, stream["r_frame_rate"].split("/"))
    return {"codec": stream["codec_name"], "width": int(stream["width"]), "height": int(stream["height"]),
            "fps": numerator / denominator, "frames": int(stream["nb_read_frames"]),
            "duration_seconds": float(observed["format"]["duration"])}


def read_physics(physics: Path) -> dict[str, Any]:
    required = [physics / "manifest.json", physics / "metrics.json"]
    missing = [display(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Required production inputs are not ready: " + ", ".join(missing))
    manifest, metrics = (read_json(path) for path in required)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported spatial physics manifest schema")
    inputs = {path.resolve(): sha256(path) for path in required}
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Physics manifest has no saved file hashes")
    for name, record in files.items():
        path = local_file(name, physics)
        verify_hash(path, record["sha256"], inputs)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Saved byte count mismatch: {display(path)}")
    for name, expected in manifest["source_hashes"].items():
        verify_hash(local_file(name, ROOT), expected, inputs)
    config_record = manifest["input_config"]
    config_path = local_file(config_record["path"], ROOT)
    verify_hash(config_path, config_record["sha256"], inputs)
    saved_config = read_json(config_path)
    config = manifest["computational_config"]
    if any(config.get(key) != value for key, value in saved_config["model"].items()):
        raise ValueError("Input and computational configuration disagree")
    if set(manifest["case_contract"]) != set(CASES) or set(metrics["cases"]) != set(CASES):
        raise ValueError("Four production cases are required")
    sections, flows = {}, {}
    for case in CASES:
        contract = manifest["case_contract"][case]
        if float(contract["kappa"]) != float(metrics["cases"][case]["kappa"]):
            raise ValueError(f"Case Kerr coefficient mismatch: {case}")
        for name in (contract["sections_file"], contract["flows_file"]):
            if name not in files:
                raise ValueError(f"Unhashed production array: {name}")
        sections[case] = load_sections(local_file(contract["sections_file"], physics))
        flows[case] = load_flows(local_file(contract["flows_file"], physics))
        for axis in ("x_um", "y_um", "z_um", "cross_z_um"):
            if not np.array_equal(sections[case][axis], sections["linear"][axis]):
                raise ValueError(f"Cases do not share the same {axis} grid")
        if not np.isclose(sections[case]["z_um"][0], 0) or not np.isclose(sections[case]["z_um"][-1], config["zmax_um"]):
            raise ValueError(f"Saved section does not cover the model distance: {case}")
        if not np.isclose(sections[case]["cross_z_um"][-1], config["zmax_um"]):
            raise ValueError(f"The saved output cross section is missing: {case}")
    return {"physics": physics, "manifest": manifest, "metrics": metrics,
            "config": config, "saved_config": saved_config, "config_path": config_path, "inputs": inputs,
            "sections": sections, "flows": flows}


def require_complete_render(render: dict[str, Any]) -> list[dict[str, Any]]:
    if render.get("schema_version") != 1:
        raise ValueError("Unsupported spatial render manifest schema")
    if render.get("stage") != "render" or render.get("production_movies_complete") is not True:
        raise ValueError(f"Production films are incomplete: stage={render.get('stage')!r}, production_movies_complete={render.get('production_movies_complete')!r}; preview/proof assets cannot build the report")
    films = render.get("films")
    if not isinstance(films, list) or len(films) != len(FILMS):
        raise ValueError("All three completed production film records are required")
    for record in films:
        if not isinstance(record, dict) or not isinstance(record.get("mp4"), str) or not record["mp4"].strip():
            film_id = record.get("id", "unknown") if isinstance(record, dict) else "unknown"
            raise ValueError(f"Production film {film_id} has no completed MP4 path")
        for kind in ("mp4", "poster", "scene"):
            if not isinstance(record.get(kind), str) or not record[kind].strip():
                raise ValueError(f"Production film {record.get('id', 'unknown')} has no {kind} path")
            hashes = record.get("sha256")
            if not isinstance(hashes, dict) or not isinstance(hashes.get(kind), str) or len(hashes[kind]) != 64:
                raise ValueError(f"Production film {record.get('id', 'unknown')} has no final {kind} SHA-256")
    return films


def read_inputs(physics: Path, renders: Path) -> dict[str, Any]:
    manifest_path = renders / "render_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Required completed render manifest is not ready: {display(manifest_path)}")
    render = read_json(manifest_path)
    film_records = require_complete_render(render)
    entrance_record = render.get("entrance_image")
    if not isinstance(entrance_record, dict) or entrance_record.get("file") != "entrance_01.png":
        raise ValueError("Production entrance_01.png with an entrance_image provenance record is required")
    if entrance_record.get("source_film") != "01_reflection" or entrance_record.get("source_frame") != 1:
        raise ValueError("Entrance image must identify film 01_reflection, source frame 1")
    if not isinstance(entrance_record.get("sha256"), str) or len(entrance_record["sha256"]) != 64:
        raise ValueError("Production entrance image has no final SHA-256")
    context = read_physics(physics)
    context["inputs"][manifest_path.resolve()] = sha256(manifest_path)
    films = []
    for definition in FILMS:
        candidates = [row for row in film_records if Path(row["mp4"]).name == definition["file"]]
        if len(candidates) != 1:
            raise ValueError(f"A required completed film is missing: {definition['file']}")
        record = candidates[0]
        media = {name: local_file(record[name], renders) for name in ("mp4", "poster", "scene")}
        films.append({**definition, "record": record, **media})
    entrance = local_file(entrance_record["file"], renders)
    context.update(renders=renders, render=render, films=films, entrance=entrance, entrance_record=entrance_record)
    return context


def measurement_view(context: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Read the published summary and bounded refinements, without refitting."""
    cases = context["metrics"]["cases"]
    qualification = context["metrics"]["qualification"]
    if qualification.get("status") == "not executed":
        raise ValueError("Numerical qualification is absent; exploratory exports are not a completed report input")
    comparisons = {
        "dz_base_to_fine": qualification["dz"]["base_to_fine"],
        "finer_dx": qualification["finer_dx"]["comparison"],
        "half_cell_shift": qualification["half_cell_shift"]["comparison"],
        "azimuth_45deg": qualification["azimuth_45deg"]["comparison"],
        "window_linear": qualification["window"]["linear"]["comparison"],
        "window_educational_kerr": qualification["window"]["educational_kerr"]["comparison"],
    }
    quantities = (("コア内の横位置", "core_centroid_x_um_max_abs"),
                  ("コア内の幅", "core_radial_rms_um_max_abs"))
    accuracy = []
    for label, key in quantities:
        observed = float(cases["educational_kerr"]["difference_vs_linear"][key])
        bounds = {name: float(record[key]) for name, record in comparisons.items()}
        guide = max(bounds.values())
        accuracy.append({"label": label, "key": key, "observed_difference_um": observed,
                         "numerical_comparison_max_um": guide, "comparison_values_um": bounds})
    values = [row["observed_difference_um"] > row["numerical_comparison_max_um"] for row in accuracy]
    if all(values):
        strong_sentence = "仮の強条件では、反射後の位置と幅の差が、今回調べた計算条件の差より大きく現れた。"
    elif any(values):
        labels = "と".join(row["label"] for row, resolved in zip(accuracy, values) if resolved)
        strong_sentence = f"仮の強条件では、{labels}の差が、今回調べた計算条件の差より大きく現れた。"
    else:
        strong_sentence = "仮の強条件にも差は生じたが、今回の数値確認から位置と幅の差を十分に切り分けたとは言えない。"
    weak_below = all(float(cases["weak_reference"]["difference_vs_linear"][row["key"]])
                     <= row["numerical_comparison_max_um"] for row in accuracy)
    weak_sentence = ("弱い参照条件は線形の場合とほぼ重なり、その差は今回の数値確認の目安より小さい。"
                     if weak_below else "弱い参照条件も含め、4条件の差を同じ尺度で表示した。")
    rows = []
    for case in CASES:
        record = cases[case]
        final = record["summary"]["final"]
        rows.append(f'<tr><th scope="row">{LABELS[case]}</th><td class="value">{number(record["kappa"])}</td>'
                    f'<td class="value">{number(final["core_radial_rms_um"])}</td><td class="value">{number(100 * final["core_power"])}</td></tr>')
    accuracy_rows = "".join(
        f'<tr><th scope="row">{row["label"]}</th><td class="value">{number(row["observed_difference_um"])}</td>'
        f'<td class="value">{number(row["numerical_comparison_max_um"])}</td></tr>' for row in accuracy
    )
    shortened_mm = float(qualification["finer_dx"]["zmax_um"]) / 1000
    full_mm = float(context["config"]["zmax_um"]) / 1000
    qualification_text = (f"解析的な2次元Gaussian、一様Kerr、円形LP01との照合と、積分強度・吸収収支を記録した。"
                          f"距離刻みと窓の比較は {number(full_mm)} mm、横格子・半セル位置・45°の比較は {number(shortened_mm)} mm まで。"
                          "全距離での横格子収束や完全な電磁場の精度は確立していない。")
    interpolation_distances = [float(row["max_distance_um"])
                               for checks in context["metrics"]["flows"]["comparisons"].values()
                               for row in checks["interpolation"] if row["max_distance_um"] is not None]
    if not interpolation_distances:
        raise ValueError("Saved flow interpolation comparisons are missing")
    return {"takeaway": strong_sentence + weak_sentence, "case_rows": "".join(rows),
            "accuracy_rows": accuracy_rows, "accuracy": accuracy, "phase_threshold": threshold,
            "weak_below_numerical_comparison": weak_below,
            "flow_interpolation_max_um": max(interpolation_distances),
            "accuracy_scope": (f"距離刻み・窓は {number(full_mm)} mm、横格子・半セル位置・45°の比較は {number(shortened_mm)} mm まで。"),
            "closing": "Kerr の作用は、反射のたびに加える演出ではなく、強度に応じて伝搬中に蓄積する位相から生まれる。光が常に細くなるとは限らず、境界と干渉を含む分布として読む。",
            "qualification_text": qualification_text}


def film_markup(film: dict[str, Any], output: Path) -> str:
    film_id = "film-" + film["number"]
    record = film["record"]
    return f"""<section class="chapter" id="chapter-{film['number']}" aria-labelledby="heading-{film['number']}">
<div class="chapter-head"><span class="number">{film['number']}</span><h2 id="heading-{film['number']}">{escape(film['title'])}</h2></div>
<p>{escape(film['lead'])}</p>
<figure><div class="screen"><video id="{film_id}" controls playsinline preload="none" poster="{relative(film['poster'], output)}" tabindex="0" aria-label="{escape(film['title'])}" aria-describedby="caption-{film['number']}">
<source src="{relative(film['mp4'], output)}" type="video/mp4">{link('動画を開く', film['mp4'], output)}</video></div>
<div class="film-meta"><div class="links"><span>{number(record['frames'] / record['fps'], digits=2)}秒 · {escape(record['width'])}×{escape(record['height'])}</span>{link('MP4 を保存', film['mp4'], output, download=True)}{link('Blender シーン', film['scene'], output, download=True)}</div>
<label class="rate" for="rate-{film['number']}">再生速度 <select id="rate-{film['number']}" data-video-rate="{film_id}" aria-label="動画{film['number']}の再生速度"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option></select></label></div>
<figcaption class="caption" id="caption-{film['number']}">{escape(film['caption'])}</figcaption></figure></section>"""


def render_page(context: dict[str, Any], view: dict[str, Any], output: Path, command: str) -> str:
    config = context["config"]
    intro_scope = (f"教育用の円形コア径 {number(2 * config['core_radius_um'])} µm・"
                   f"{number(config['zmax_um'] / 1000)} mm の計算。変化を読み取るため、仮の強い Kerr 係数を含む4条件を比べる。"
                   "元の1600 kmの単一モード通信・学習研究とは別の計算。")
    descriptions = {
        case: f"{LABELS[case]}：入射ピークでの屈折率変化 {number(context['metrics']['cases'][case]['kappa'])}"
        for case in CASES
    }
    controls = "".join(
        f'<button type="button" data-case-button="{case}" data-case-description="{escape(descriptions[case])}" '
        f'aria-pressed="{str(case == "linear").lower()}" aria-controls="panel-{case}">{LABELS[case]}</button>'
        for case in CASES
    )
    panels = "".join(
        f'<div id="panel-{case}" data-case-panel="{case}"{(" hidden" if case != "linear" else "")}>'
        f'<picture><source media="(max-width:720px)" srcset="figures/intensity_{case}_mobile.png"><img src="figures/intensity_{case}.png" width="1853" height="493" alt="{escape(descriptions[case])}。中心断面の強度を距離と横位置に対して示した図。"></picture></div>'
        for case in CASES
    )
    downloads = "".join(
        f'<div><h3>{film["number"]} {escape(film["title"])}</h3>{link("MP4", film["mp4"], output, download=True)}'
        f'{link("編集できる Blender シーン", film["scene"], output, download=True)}{link("ポスター", film["poster"], output, download=True)}</div>'
        for film in context["films"]
    )
    physical_links = " · ".join((link("方法と出典", ROOT / "docs/spatial-kerr.md", output),
                                 link("数値・検証記録", context["physics"] / "metrics.json", output),
                                 link("原データの来歴", context["physics"] / "manifest.json", output)))
    phase_threshold = number(view["phase_threshold"])
    entrance_markup = (f'<figure class="entrance"><img src="{relative(context["entrance"], output)}" width="1920" height="1080" '
                       'alt="切り欠きから見た光ファイバの入口。中央のコアと周囲のクラッド、内部の計算場を示す。">'
                       '<figcaption>中央のコアは、周りのクラッドより屈折率が高い。切り欠きから内部を見て、次の動画で光の流れを追う。'
                       '<span>動画01の最初のフレーム。連続光の定常分布を距離を変えて観察し、光の実際の飛行時間を表したものではない。</span></figcaption></figure>')
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="円形ファイバ内の反射とKerr効果を、計算した場を撮影した3本のBlender動画と4条件の比較で読む。"><title>反射する光と Kerr 効果</title><style>{CSS}</style></head>
<body><a class="skip" href="#main">本文へ移動</a><header class="topbar shell"><span>OPTICAL COMMUNICATION LAB / 空間の光を読む</span><nav aria-label="ページ内案内"><a href="#chapter-01">動画</a><a href="#comparison">4条件の比較</a><a href="#records">資料</a></nav></header>
<main id="main" class="shell"><div class="intro"><p class="eyebrow">光ファイバの内部 / 3本の計算映像</p><h1>反射する光は、強さによってどう変わるか。</h1><p class="lead">境界で向きが戻る光の流れを追い、Kerr 効果が位相と反射後の分布にどう働くかを見比べる。</p><p class="scope">{escape(intro_scope)}</p><nav class="chapters" aria-label="3本の動画"><a href="#chapter-01">01 反射を見る</a><a href="#chapter-02">02 Kerr の有無を比べる</a><a href="#chapter-03">03 位相の違いを読む</a></nav></div>
{entrance_markup}
{"".join(film_markup(film, output) for film in context['films'])}
<section class="results" id="comparison" aria-labelledby="comparison-heading"><h2 id="comparison-heading">作用の強さを、同じ尺度で比べる。</h2>
<p class="reading">右へ行くほど伝搬が進む。白い破線はコアの境界。明るい帯が上下に戻る場所を見る。ボタンは計算済みの4条件をそのまま切り替える。</p>
<div class="case-controls" role="group" aria-label="強度図の条件">{controls}</div><p class="switch-label" id="selected-case" aria-live="polite">{escape(descriptions['linear'])}</p>
<figure class="plot"><div class="switch-stack">{panels}</div><div class="print-comparison"><img src="figures/four_case_intensity.png" width="1853" height="1411" alt="4条件の中心断面強度を、共通の色と座標で並べた比較。"></div><noscript><img src="figures/four_case_intensity.png" width="1853" height="1411" alt="4条件の静的比較。"></noscript><figcaption>中心断面 y=0。縦と横は異なる尺度。強度は入力ピークを1とする共通の0〜{number(view['longitudinal_intensity_max'])}で、各図を最大値へ正規化していない。{link('4条件を並べた図', output / 'figures/four_case_intensity.png', output)} · {link('拡大用 SVG', output / 'figures/four_case_intensity.svg', output, download=True)}</figcaption></figure>
<p class="takeaway">{escape(view['takeaway'])}</p>
<figure class="plot"><picture><source media="(max-width:720px)" srcset="figures/output_intensity_and_phase_mobile.png"><img src="figures/output_intensity_and_phase.png" width="1853" height="816" alt="終端の横断面。Kerrなしと仮の強条件の強度、および両者の位相差。" loading="lazy"></picture><figcaption>{number(config['zmax_um'] / 1000)} mm の同じ断面。Kerrなし・強条件は共通の0〜{number(view['output_intensity_max'])}の強度尺度で、入力ピークは1。位相差は循環色。灰色はコア外または両方の場を比べるには弱い領域。{link('拡大用 SVG', output / 'figures/output_intensity_and_phase.svg', output, download=True)}</figcaption></figure>
<div class="tables"><div class="table-box"><h3>4条件の終端</h3><div class="table-scroll"><table><thead><tr><th scope="col">条件</th><th scope="col">入力の屈折率変化</th><th scope="col">コア内の幅<br>[µm]</th><th scope="col">コア内に残る光<br>[%]</th></tr></thead><tbody>{view['case_rows']}</tbody></table></div><p class="table-note">幅はコア内の重心から測る RMS 半径。残る光は入力の積分強度に対する割合。</p></div>
<div class="table-box"><h3>見せた差と、計算条件による差</h3><div class="table-scroll"><table><thead><tr><th scope="col">比較量 [µm]</th><th scope="col">強条件 − なし</th><th scope="col">数値確認の目安</th></tr></thead><tbody>{view['accuracy_rows']}</tbody></table></div><p class="table-note">伝搬中の差の最大値と、今回の条件変更で観測した差の最大値。{escape(view['accuracy_scope'])} 誤差の絶対的な上限とは扱わない。</p></div></div>
<p class="closing">{escape(view['closing'])}</p></section>
<div class="supplement" id="records"><details><summary>映像の線・面・色をどう読むか</summary><div class="detail-body"><p>流線は、計算した複素場から求めた近似的なエネルギーの流れ。個々の光子の軌道ではない。補間を変えた線の位置の差は最大{number(view['flow_interpolation_max_um'])} µmであり、細かな反射点を定量的に測る表示としては使わない。強度の低い場所では線を止め、位置や幅の数値は全場から求める。</p><p>映像の等強度面には、いくつかの離散的な強度の色を使う。断面の強度図は連続色、位相差は −π〜+π の循環色で表す。この静的な位相図は、両方の場の強度が入力ピークに対して {phase_threshold} 以上の場所だけを着色する。</p><p>連続光の定常分布を表示している。カメラの移動や距離の走査は見るための時間配分で、光の実際の飛行時間ではない。切り欠き・赤外光の色・形状の拡大は説明用の描画。</p><p>{link('描画の記録', context['renders'] / 'render_manifest.json', output)}</p></div></details>
<details><summary>計算条件と通常の数値確認</summary><div class="detail-body"><p>円形の屈折率段差を持つ、横2次元のスカラー近軸モデル。時間パルス・偏波・軸方向の後方反射・材料損傷は含めない。強い仮係数を実材料の電力へ換算していない。</p><p>{escape(view['qualification_text'])} 独立 validator 評価は今回0回。元の通信研究の評価は流用していない。</p><p>{physical_links}</p></div></details>
<details><summary>動画・原データ・再生成</summary><div class="detail-body"><div class="downloads">{downloads}</div><p>動画と図はこのページの隣にあるローカルファイルを読む。リポジトリのフォルダ構成を保てば、ネット接続なしで動画を再生し、資料を開ける。</p><p>{link('原データ', context['physics'] / 'README.md', output)} · {link('物理計算のソース', ROOT / 'src/optical_comm/visualization/spatial.py', output)} · {link('解説と図のソース', ROOT / 'scripts/build_spatial_kerr_report.py', output)} · {link('再生成手順', output / 'README.md', output)} · {link('この解説の来歴', output / 'explanation_manifest.json', output)}</p><pre><code>{escape(command)}</code></pre></div></details></div>
</main><footer class="foot shell"><span>計算した場と流れを Blender で可視化</span><span>{link('方法・出典', ROOT / 'docs/spatial-kerr.md', output)}</span></footer><script>{JS}</script></body></html>
"""


def render_readme(context: dict[str, Any], view: dict[str, Any], output: Path, command: str) -> str:
    films = "\n".join(
        f"| {film['number']} | {film['title']} | [{film['mp4'].name}]({relative(film['mp4'], output)}) | [{film['scene'].name}]({relative(film['scene'], output)}) |"
        for film in context["films"]
    )
    if output.is_relative_to(ROOT):
        page_url = "http://127.0.0.1:8000/" + quote((output.relative_to(ROOT) / "index.html").as_posix(), safe="/.")
        http_guide = f"""保存リンクをHTTPで使う場合は、リポジトリ直下で次のローカルサーバーを
起動し、`{page_url}` を開く。外部サービスへの接続は不要。

```bash
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```"""
    else:
        http_guide = """この出力先はリポジトリ外なので、`index.html` をローカルファイルとして開く。
HTTPで保存リンクを使う場合は `--output-dir .work/spatial-kerr-report-reproduction` などで
リポジトリ内へ出力し、その出力先のREADMEにあるHTTP案内を使う。"""
    return f"""# 反射する光と Kerr 効果

[説明ページ](index.html)で、3本の動画と4条件の比較を順に読める。
教育用の円形コア径 {number(2 * context['config']['core_radius_um'])} µm、
長さ {number(context['config']['zmax_um'] / 1000)} mm の連続光の計算であり、
仮の強い係数を含む。1600 km の単一モード通信・学習研究とは別の計算。

{view['takeaway']}

| 動画 | 見るところ | MP4 | Blender |
| --- | --- | --- | --- |
{films}

線は場のエネルギーの流れ、面は同じ強度の場所をつないだ表示。
カメラ移動や距離の走査は観察の時間配分で、光の飛行時間ではない。
冒頭の [入口画像]({relative(context['entrance'], output)}) は動画01の最初のフレームのコピー。
画像と動画のハッシュ、および元フレームの来歴をマニフェストに記録する。

4条件のボタンは、保存済みの別々の計算画像を同じ座標・色尺度で切り替える。
係数の間を補間しない。[4条件の静的比較](figures/four_case_intensity.png)も保存し、
印刷時にはこの比較図を表示する。図の SVG は文字を輪郭で保存している。

## 再生成

リポジトリ直下で実行する。保存済みの計算データ、3本の MP4、Blender シーン、
ポスター、両方のマニフェストが必要。NumPy、Matplotlib、日本語フォント、
FFprobe を使う。標準では Noto Sans CJK JP、IPAexGothic、IPAGothic の順に探す。

```bash
{command}
```

このコマンドは図の PNG/SVG、`index.html`、この README、
`explanation_manifest.json` を再生成する。物理伝搬や Blender の描画は行わない。
通常コマンドはブラウザ確認を実行・復元しない。
`--browser-record /path/to/record.json` は、同じHTML・動画・render manifestに対する
成功記録を添付するための引数で、UIテストを自動実行するものではない。
入力不足・ハッシュ不一致では終了し、完成ページを出力しない。
別の出力先は `--output-dir .work/spatial-kerr-report-reproduction`、
日本語フォントは `--font /path/to/font.ttf` で指定できる。

元の動画、図、データと相対パスを保持して `index.html` を開く。
ブラウザの標準操作で再生・シーク・全画面表示を行い、各動画の再生速度を選べる。
動画にフォーカスしたときは Space と左右キーをブラウザの標準操作として使える。
K でも再生・一時停止を切り替えられる。キーによる移動幅はブラウザの標準に従う。
再生を始めると他の動画は一時停止する。外部サービス、外部フォント、配信 CDN は使わない。

ローカルファイルを直接開いた場合、保存リンクの扱いはブラウザによって異なる。
動画の標準メニューから保存できる。
{http_guide}

[方法と出典]({relative(ROOT / 'docs/spatial-kerr.md', output)})、
[原データ]({relative(context['physics'] / 'README.md', output)})、
[数値・検証記録]({relative(context['physics'] / 'metrics.json', output)})、
[映像の記録]({relative(context['renders'] / 'render_manifest.json', output)})を参照できる。
`explanation_manifest.json` に入力・出力・フォント・コードのハッシュ、生成条件、
通常の確認を記録する。自己参照を避け、同マニフェスト自身のハッシュは内包しない。
この改訂の独立 validator は指定0・試行0・完了0。元研究の1回を今回の評価に流用していない。
"""


def verify_media(context: dict[str, Any]) -> dict[str, Any]:
    render = context["render"]
    provenance = render.get("provenance", {})
    verify_hash(context["physics"] / "manifest.json", provenance.get("numerical_manifest_sha256"), context["inputs"])
    records = {}
    for film in context["films"]:
        for kind in ("mp4", "poster", "scene"):
            path = film[kind]
            verify_hash(path, film["record"]["sha256"][kind], context["inputs"])
        observed = probe_video(film["mp4"])
        expected = film["record"]
        for key in ("width", "height", "frames", "fps"):
            if not math.isclose(observed[key], float(expected[key]), rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Movie {key} does not match its manifest: {film['mp4'].name}")
        if not math.isclose(observed["duration_seconds"], observed["frames"] / observed["fps"], abs_tol=.05):
            raise ValueError(f"Movie duration/frame count mismatch: {film['mp4'].name}")
        records[film["number"]] = observed
    entrance = context["entrance"]
    verify_hash(entrance, context["entrance_record"].get("sha256"), context["inputs"])
    with entrance.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Entrance image is not a PNG with an IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (records["01"]["width"], records["01"]["height"]):
        raise ValueError("Entrance image dimensions do not match film 01")
    context["entrance_dimensions"] = {"width": width, "height": height}
    records["entrance_image"] = {**context["entrance_record"], "width": width, "height": height}
    return records


def verify_browser_record(path: Path | None, page_path: Path, context: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    record = read_json(path)
    if record.get("scope") != "final_delivered_media":
        raise ValueError("Browser record must inspect the final delivered media; layout preview checks are not accepted")
    render_hash = context["inputs"][(context["renders"] / "render_manifest.json").resolve()]
    if record.get("render_manifest_sha256") != render_hash:
        raise ValueError("Browser record refers to a different render manifest revision")
    if record.get("all_passed") is not True or record.get("html_sha256") != sha256(page_path):
        raise ValueError("Browser record is unsuccessful or refers to a different HTML page")
    observed_media = record.get("media_sha256", {})
    expected_media = {film["mp4"].name: context["inputs"][film["mp4"].resolve()] for film in context["films"]}
    if observed_media != expected_media:
        raise ValueError("Browser record refers to different movies")
    required_checks = ("local_file_playback", "seek", "playback_speed", "video_switching", "mobile_width",
                       "no_horizontal_overflow", "case_switching", "local_references", "svg_rendering")
    if any(record.get("checks", {}).get(key) is not True for key in required_checks):
        raise ValueError("Browser record does not contain all required ordinary checks")
    return {"record_path": display(path), "record_sha256": sha256(path), "record": record}


def generate_figures(context: dict[str, Any], stage: Path, intensity_max: float,
                     phase_threshold: float, output_intensity_max: float = 1.0) -> tuple[list[Path], dict[str, Any], dict[str, Any]]:
    figures, intensity_encoding = intensity_figure(context["sections"], context["config"], stage, intensity_max)
    figures.extend(case_intensity_figures(context["sections"], context["config"], stage, intensity_encoding))
    phase_outputs, phase_encoding = phase_figure(context["sections"], context["config"], stage, phase_threshold, output_intensity_max)
    figures.extend(phase_outputs)
    return figures, intensity_encoding, phase_encoding


def assert_inputs_unchanged(inputs: dict[Path, str]) -> None:
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError(f"An input changed during generation: {display(path)}")


def copy_outputs(stage: Path, paths: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        destination = output / path.relative_to(stage)
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())


def write_figure_manifest(args: argparse.Namespace, context: dict[str, Any], font_record: dict[str, str],
                          stage: Path, figures: list[Path], intensity_encoding: dict[str, Any],
                          phase_encoding: dict[str, Any], output: Path) -> Path:
    source = Path(__file__).resolve()
    physics = context["physics"]
    argv = [".venv/bin/python", display(source), "--figures-only", "--physics-run", display(physics),
            "--output-dir", display(output), "--intensity-max", str(args.intensity_max),
            "--output-intensity-max", str(args.output_intensity_max), "--phase-threshold", str(args.phase_threshold)]
    if args.font is not None:
        argv.extend(["--font", display(args.font)])
    numerical_paths = {physics / "manifest.json", physics / "metrics.json", context["config_path"], source,
                       Path(font_record["path"])}
    numerical_paths.update(physics / name for name in context["manifest"]["files"])
    numerical_paths.update(ROOT / name for name in context["manifest"]["source_hashes"])
    record = {
        "schema_version": 1, "status": "figures_generated",
        "scope": "Saved spatial Kerr data figures only; no field propagation, movies, HTML, or completed explanation",
        "regeneration_command": shlex.join(argv),
        "inputs": {display(path): {"sha256": context["inputs"][path.resolve()], "bytes": path.stat().st_size}
                   for path in sorted(numerical_paths, key=str)},
        "outputs": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in figures},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__, "font": font_record},
        "visual_encoding": {"intensity": intensity_encoding, "phase": phase_encoding,
                            "case_order": list(CASES), "interpolation_between_cases": False,
                            "responsive_assets": "Same saved samples, coordinates, fixed color limits and phase mask; larger type and vertically stacked output panels on mobile"},
        "ordinary_checks": {"input_hashes_match": True, "shared_case_coordinates": True,
                            "shared_longitudinal_intensity_scale": [0.0, args.intensity_max],
                            "shared_output_intensity_scale": [0.0, args.output_intensity_max], "intensity_clipping": False},
        "independent_validator": {"requested": 0, "attempted": 0, "completed": 0},
    }
    path = stage / "figures/figures_manifest.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def build_figures_only(args: argparse.Namespace, physics: Path, output: Path) -> dict[str, Any]:
    """Publish only saved-data figures; no render manifest, HTML, or report state."""
    context = read_physics(physics)
    _, font_record = resolve_font(args.font)
    source = Path(__file__).resolve()
    context["inputs"][source] = sha256(source)
    context["inputs"][Path(font_record["path"])] = font_record["sha256"]
    scratch_root = ROOT / ".work"
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spatial-figures-build-", dir=scratch_root) as raw_stage:
        stage = Path(raw_stage)
        figures, intensity_encoding, phase_encoding = generate_figures(context, stage, args.intensity_max, args.phase_threshold, args.output_intensity_max)
        assert_inputs_unchanged(context["inputs"])
        manifest_path = write_figure_manifest(args, context, font_record, stage, figures, intensity_encoding, phase_encoding, output)
        copy_outputs(stage, figures + [manifest_path], output)
    return {"output": display(output / "figures"), "status": "figures_generated", "case_count": 4,
            "figure_files": len(figures), "intensity_scale": [0.0, args.intensity_max],
            "output_intensity_scale": [0.0, args.output_intensity_max],
            "phase_range_rad": [-math.pi, math.pi], "phase_intensity_threshold": args.phase_threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-run", type=Path, default=ROOT / "artifacts/spatial_kerr")
    parser.add_argument("--renders", type=Path, default=ROOT / "reports/animations/spatial_kerr")
    parser.add_argument("--output-dir", type=Path, help="Default: the renders directory")
    parser.add_argument("--figures-only", action="store_true", help="Generate only saved-data figures; completed movies are not required")
    parser.add_argument("--font", type=Path, help="Japanese font; default: a locally installed Noto/IPA font")
    parser.add_argument("--intensity-max", type=float, default=3.0, help="Fixed shared longitudinal intensity maximum, in input-peak=1 units")
    parser.add_argument("--output-intensity-max", type=float, default=1.0, help="Fixed shared endpoint OFF/ON intensity maximum, in the same absolute input-peak=1 units")
    parser.add_argument("--phase-threshold", type=float, default=1e-4, help="Fixed input-normalized threshold for both phase fields")
    parser.add_argument("--browser-record", type=Path, help="Optional completed ordinary browser checks for the same HTML and movies")
    args = parser.parse_args()
    if not math.isfinite(args.phase_threshold) or args.phase_threshold <= 0:
        parser.error("--phase-threshold must be finite and positive")
    if not math.isfinite(args.intensity_max) or args.intensity_max <= 0:
        parser.error("--intensity-max must be finite and positive")
    if not math.isfinite(args.output_intensity_max) or args.output_intensity_max <= 0:
        parser.error("--output-intensity-max must be finite and positive")
    if args.figures_only and args.browser_record is not None:
        parser.error("--browser-record applies only to the full report")
    physics, renders = args.physics_run.resolve(), args.renders.resolve()
    output = (args.output_dir or renders).resolve()
    if output == physics or output.is_relative_to(physics):
        parser.error("Report outputs must be separate from physics inputs")
    try:
        if args.figures_only:
            print(json.dumps(build_figures_only(args, physics, output), ensure_ascii=False))
            return
        context = read_inputs(physics, renders)
        media_checks = verify_media(context)
        font, font_record = resolve_font(args.font)
        context["inputs"][Path(font_record["path"])] = font_record["sha256"]
        source = Path(__file__).resolve()
        context["inputs"][source] = sha256(source)
        for document in (ROOT / "docs/spatial-kerr.md", ROOT / "docs/requirements.md"):
            context["inputs"][document] = sha256(document)
        view = measurement_view(context, args.phase_threshold)
        argv = [".venv/bin/python", display(source), "--physics-run", display(physics), "--renders", display(renders)]
        if output != renders:
            argv.extend(["--output-dir", display(output)])
        if args.font is not None:
            argv.extend(["--font", display(args.font)])
        if args.phase_threshold != 1e-4:
            argv.extend(["--phase-threshold", str(args.phase_threshold)])
        if args.intensity_max != 3.0:
            argv.extend(["--intensity-max", str(args.intensity_max)])
        if args.output_intensity_max != 1.0:
            argv.extend(["--output-intensity-max", str(args.output_intensity_max)])
        command = shlex.join(argv)
        scratch_root = ROOT / ".work"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="spatial-report-build-", dir=scratch_root) as raw_stage:
            stage = Path(raw_stage)
            figures, intensity_encoding, phase_encoding = generate_figures(context, stage, args.intensity_max, args.phase_threshold, args.output_intensity_max)
            figure_manifest_path = write_figure_manifest(args, context, font_record, stage, figures, intensity_encoding, phase_encoding, output)
            view.update(longitudinal_intensity_max=args.intensity_max, output_intensity_max=args.output_intensity_max)
            page = render_page(context, view, output, command)
            (stage / "index.html").write_text(page, encoding="utf-8")
            (stage / "README.md").write_text(render_readme(context, view, output, command), encoding="utf-8")
            planned = {path.relative_to(stage).as_posix() for path in figures + [figure_manifest_path]} | {"index.html", "README.md", "explanation_manifest.json"}
            reference_count = check_local_links(page, output, planned)
            browser = verify_browser_record(args.browser_record, stage / "index.html", context)
            assert_inputs_unchanged(context["inputs"])
            outputs = figures + [figure_manifest_path, stage / "index.html", stage / "README.md"]
            record = {
                "schema_version": 1, "status": "assembled_and_browser_checked" if browser else "assembled",
                "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Offline explanatory report; saved circular-core CW fields, no new propagation or communications evaluation",
                "regeneration_command": command,
                "regeneration_scope": {
                    "ordinary_command": "Regenerates figures, HTML, README and provenance only; does not execute or restore browser checks",
                    "browser_record_option": "Attaches a successful record for the same HTML, movies and render manifest; does not execute UI tests"
                },
                "inputs": {display(path): {"sha256": value, "bytes": path.stat().st_size} for path, value in sorted(context["inputs"].items(), key=lambda item: str(item[0]))},
                "outputs": {path.relative_to(stage).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__, "font": font_record},
                "visual_encoding": {"intensity": intensity_encoding, "phase": phase_encoding,
                                    "entrance_image": context["entrance_record"],
                                    "interactive_comparison": {"cases": list(CASES), "interpolation_between_cases": False, "shared_coordinates_and_scale": True}},
                "displayed_difference_vs_numerical_comparisons": view["accuracy"],
                "weak_reference_below_observed_numerical_comparisons": view["weak_below_numerical_comparison"],
                "ordinary_checks": {"input_hashes_match": True, "local_reference_count": reference_count, "video_probe": media_checks, "browser": browser},
                "independent_validator": {"requested": 0, "attempted": 0, "completed": 0,
                                          "original_research_review_reused": False},
            }
            manifest_path = stage / "explanation_manifest.json"
            manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            outputs.append(manifest_path)
            copy_outputs(stage, outputs, output)
        print(json.dumps({"output": display(output), "status": record["status"], "case_count": 4, "film_count": 3,
                          "figure_files": len(figures), "local_references": reference_count}, ensure_ascii=False))
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Spatial Kerr report was not generated: {error}\n")


if __name__ == "__main__":
    main()

