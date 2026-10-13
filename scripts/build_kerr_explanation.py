#!/usr/bin/env python3
"""Build the offline film essay from verified Kerr simulation/render records.

This entry point does not propagate a field, render a frame, or infer measured
results. It checks the delivered inputs before writing the reader-facing page.
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
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
CASE_ORDER = ("pure_spm", "dispersion_only", "combined")
FILMS = {
    "pure_spm": {
        "number": "01",
        "file": "01_self_phase_modulation.mp4",
        "blend": "01_self_phase_modulation.blend",
        "poster": "poster_01.png",
        "eyebrow": "SELF-PHASE MODULATION",
        "title": "強さが、位相を変える。",
        "lead": "まず、Kerr効果だけを見る。光の強い中心と、弱い裾では、蓄積する位相が違う。",
        "watch": "パルスの中心と裾を比べる。損失で弱くなりながらも時間幅を保ち、位相の分布が変化する。",
        "change": "シリカでは、光が強いほど屈折率がわずかに増す。その結果、光自身の強さに応じて位相が変わる。これが自己位相変調。",
        "meaning": "位相の変化が時間の場所ごとに違うと、周波数にも幅が生まれる。分散を切ったこの比較では、Kerr効果だけでパルスの強度幅は広がらない。",
    },
    "dispersion_only": {
        "number": "02",
        "file": "02_chromatic_dispersion.mp4",
        "blend": "02_chromatic_dispersion.blend",
        "poster": "poster_02.png",
        "eyebrow": "CHROMATIC DISPERSION",
        "title": "周波数が違うと、到着がずれる。",
        "lead": "次に、波長分散だけを見る。短いパルスにも周波数の幅があり、それぞれの成分が同じ速さでは進まない。",
        "watch": "伝搬が進むにつれて、パルスが時間方向へ広がるところを見る。",
        "change": "このファイバでは、周波数成分ごとに到着時刻がずれる。重なっていた成分がずれるため、パルスの強度幅が広がる。",
        "meaning": "光の強さに依存しない変化もある。位相が変わることと、時間方向へ広がることを分けて見ると、Kerr効果の役割が読み取りやすい。",
    },
    "combined": {
        "number": "03",
        "file": "03_kerr_dispersion_interplay.mp4",
        "blend": "03_kerr_dispersion_interplay.blend",
        "poster": "poster_03.png",
        "eyebrow": "KERR × DISPERSION",
        "title": "二つの変化が、影響し合う。",
        "lead": "最後に、Kerr効果と分散を同時に入れる。位相が変わると、分散による広がり方も変わる。",
        "watch": "同じ入射パルスから始めた「分散だけ」の映像と、輪郭の変化を見比べる。",
        "change": "Kerr効果が生む周波数の変化を、分散が時間方向の変化へつなぐ。一方、形が変われば光の強さも変わり、その後の位相の蓄積も変わる。",
        "meaning": "作用の組み合わせによって、変化の向きと大きさが変わる。その結果を、下の計算値で確かめる。",
    },
}


CSS = r"""
:root{color-scheme:dark;--bg:#111212;--panel:#191b1b;--paper:#f0eade;--muted:#a7a69f;--gold:#d7aa71;--rule:#ffffff1c;--width:1160px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:5rem}body{margin:0;background:var(--bg);color:var(--paper);font-family:-apple-system,BlinkMacSystemFont,"Noto Sans CJK JP","Yu Gothic",Meiryo,sans-serif;font-size:16px;line-height:1.9;letter-spacing:.025em}a{color:inherit;text-decoration-color:#746a5c;text-underline-offset:.28em}a:hover{color:var(--gold)}button,select{font:inherit}button,a,select,video{touch-action:manipulation}button:focus-visible,a:focus-visible,select:focus-visible,video:focus-visible,summary:focus-visible{outline:2px solid var(--gold);outline-offset:5px}::selection{background:#d7aa7145}.skip{position:absolute;left:1rem;top:-5rem;padding:.4rem 1rem;background:var(--paper);color:var(--bg);z-index:20}.skip:focus{top:1rem}
.serif,h1,h2,.statement{font-family:"Noto Serif CJK JP","Yu Mincho","Hiragino Mincho ProN",serif;font-weight:500}.shell{width:min(calc(100% - 96px),var(--width));margin-inline:auto}.topbar{display:flex;align-items:center;justify-content:space-between;gap:2rem;padding:24px 0;border-bottom:1px solid var(--rule);font-size:.72rem;letter-spacing:.15em}.brand{text-decoration:none;font-weight:600}.topbar nav{display:flex;gap:1.7rem}.topbar nav a{text-decoration:none;color:var(--muted)}.topbar nav a:hover{color:var(--paper)}
.hero{position:relative;min-height:660px;min-height:min(860px,92svh);display:grid;align-items:end;isolation:isolate;overflow:hidden;margin-top:20px}.hero img{position:absolute;inset:-12% 0 auto;width:100%;height:124%;object-fit:cover;object-position:center;z-index:-3}.hero::before{content:"";position:absolute;inset:0;z-index:-2;background:linear-gradient(90deg,#080a0adb 0%,#0b0d0d9a 34%,#080a0a12 78%),linear-gradient(0deg,#111212 1%,#11121233 42%,transparent 65%)}.hero-copy{padding:90px 0 70px;max-width:850px}.eyebrow{font-size:.7rem;letter-spacing:.25em;color:var(--gold);margin:0 0 1.4rem;text-transform:uppercase}.hero h1{font-size:clamp(2.6rem,5.7vw,5rem);line-height:1.45;letter-spacing:.1em;margin:0 0 1.6rem;text-wrap:balance}.hero-lead{max-width:530px;font-size:1rem;color:#d6d3c9;line-height:2.05;margin:0}.hero-foot{display:flex;gap:2rem;flex-wrap:wrap;margin-top:2.2rem;font-size:.72rem;letter-spacing:.12em;color:var(--muted)}
.intro{max-width:740px;margin:80px auto 100px}.intro .statement{font-size:clamp(1.45rem,2.5vw,2rem);line-height:1.9;letter-spacing:.07em;margin-bottom:1.8rem}.intro p{color:var(--muted);font-size:.95rem}.intro .observation{color:#d9d4c9;border-left:2px solid var(--gold);padding-left:18px;margin-top:1.5rem}.contents{display:grid;grid-template-columns:repeat(3,1fr);gap:2rem;margin-top:3.8rem;padding-top:1.5rem;border-top:1px solid var(--rule)}.contents a{text-decoration:none;display:flex;flex-direction:column;gap:.35rem}.contents small{font-size:.65rem;color:var(--gold);letter-spacing:.15em}.contents span{font-size:.83rem;color:var(--paper)}
.chapter{margin:0 auto 128px;scroll-margin-top:28px}.chapter-heading{display:flex;align-items:end;justify-content:space-between;gap:4rem;margin-bottom:26px}.chapter-title{flex:1}.chapter-title .eyebrow{margin:0 0 .7rem}.chapter-title h2{font-size:clamp(1.8rem,3.2vw,2.7rem);letter-spacing:.06em;line-height:1.65;margin:0;text-wrap:balance}.chapter-number{display:inline-block;font-family:Georgia,serif;font-style:italic;font-size:1.2rem;color:var(--gold);margin-right:1.1rem;vertical-align:middle}.chapter-lead{max-width:360px;margin:0 0 .2rem;color:var(--muted);font-size:.86rem;line-height:1.85}.film{margin:0}.screen{background:#080909;box-shadow:0 14px 60px #0005;line-height:0}.screen video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;max-height:82svh}.film-caption{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:15px 0 30px;font-size:.7rem;color:var(--muted);line-height:1.8}.film-caption .left{display:flex;gap:1rem;flex-wrap:wrap}.rate{display:flex;gap:.65rem;align-items:center;white-space:nowrap}.rate select{border:1px solid #ffffff2a;border-radius:2px;background:#202121;color:var(--paper);padding:.13rem .5rem;font-size:.73rem;min-width:70px}.explanation{display:grid;grid-template-columns:repeat(3,1fr);gap:2.6rem;border-top:1px solid var(--rule);padding-top:28px}.explanation h3{font-size:.7rem;font-weight:500;letter-spacing:.16em;color:var(--gold);margin:0 0 .8rem}.explanation p{font-size:.9rem;line-height:1.95;margin:0;color:#cecec6}.observed{margin-top:26px;padding:19px 0 0;border-top:1px solid var(--rule);display:flex;gap:2rem;align-items:baseline;flex-wrap:wrap}.observed-label{font-size:.7rem;color:var(--muted);letter-spacing:.12em}.measurement{font-size:.84rem}.measurement strong{font-family:Georgia,serif;font-size:1.3rem;font-weight:400;color:var(--paper);letter-spacing:.03em}.measurement .unit{font-size:.74rem;color:var(--muted);margin-left:.3rem}.measurement-note{font-size:.73rem;color:var(--muted);margin:10px 0 0}
.closing{display:grid;grid-template-columns:1fr 1fr;gap:5rem;padding:65px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);margin-bottom:65px}.closing h2{font-size:clamp(1.6rem,2.7vw,2.2rem);margin:0;line-height:1.8;letter-spacing:.06em}.closing p{font-size:.91rem;color:var(--muted);margin:0}.closing p+p{margin-top:1.3rem}.reading{max-width:870px;margin:0 auto 64px}.reading h2{font-family:inherit;font-size:.85rem;letter-spacing:.12em;margin-bottom:1.4rem}.reading dl{display:grid;grid-template-columns:160px 1fr;gap:1rem 2rem;margin:0;font-size:.84rem}.reading dt{color:var(--gold)}.reading dd{margin:0;color:var(--muted)}
.supplement{margin:0 auto 60px;max-width:870px}.supplement details{border-top:1px solid var(--rule);padding:20px 0}.supplement details:last-child{border-bottom:1px solid var(--rule)}.supplement summary{cursor:pointer;font-size:.83rem;color:var(--paper);list-style-position:outside;margin-left:1rem;padding-left:.5rem}.detail-body{padding:22px 0 2px;font-size:.81rem;color:var(--muted)}.detail-body p{margin:0 0 1rem}.detail-body ul{padding-left:1.2rem}.detail-body li{margin:.6rem 0}.detail-body code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#e4dfd2;font-size:.78rem;overflow-wrap:anywhere}.detail-body pre{background:#090b0b;padding:18px;overflow:auto;line-height:1.7;border:1px solid var(--rule)}.detail-body pre code{white-space:pre-wrap}.specs{border-collapse:collapse;width:100%}.specs th,.specs td{padding:.65rem 0;border-bottom:1px solid var(--rule);vertical-align:top;text-align:left;font-weight:400}.specs th{width:41%;color:var(--paper)}.downloads{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.downloads a{display:block}.foot{display:flex;gap:1rem;flex-wrap:wrap;justify-content:space-between;padding:25px 0 45px;color:#81857f;font-size:.66rem;letter-spacing:.08em}.foot a{color:var(--muted)}
@media(min-width:1500px){.hero{max-width:1600px;margin-inline:auto}}
@media(max-width:900px){.shell{width:calc(100% - 48px)}.hero{min-height:650px}.hero-copy{padding:60px 0}.hero::before{background:linear-gradient(90deg,#080a0ac2,#080a0a40),linear-gradient(0deg,#111212 1%,#11121233 65%,transparent)}.chapter-heading{gap:2rem}.chapter-lead{max-width:280px}.chapter-title h2{font-size:1.85rem}.explanation{gap:1.7rem}.closing{gap:3rem}.intro{margin-top:60px}}
@media(max-width:650px){body{font-size:15px}.shell{width:calc(100% - 36px)}.topbar{padding:18px 0}.topbar nav{gap:1rem;font-size:.65rem}.topbar nav a:nth-child(2){display:none}.hero{min-height:620px;min-height:82svh;margin-top:10px}.hero-copy{padding:55px 0 48px}.hero h1{font-size:2.7rem;letter-spacing:.05em}.hero-lead{font-size:.9rem;max-width:340px}.hero-foot{gap:1rem;font-size:.65rem}.intro{margin:45px auto 68px}.intro .statement{font-size:1.4rem}.intro p{font-size:.87rem}.contents{gap:.8rem;margin-top:2.4rem}.contents span{font-size:.74rem;line-height:1.7}.contents small{font-size:.58rem}.chapter{margin-bottom:80px}.chapter-heading{display:block;margin-bottom:21px}.chapter-title h2{font-size:1.7rem}.chapter-lead{max-width:100%;margin-top:1.1rem;font-size:.83rem}.chapter-number{margin-right:.5rem;font-size:1rem}.screen{width:calc(100% + 36px);margin-left:-18px}.film-caption{align-items:start;font-size:.62rem;margin-bottom:22px}.film-caption .left{gap:.15rem;display:flex;flex-direction:column}.rate{gap:.35rem}.rate select{font-size:.68rem;min-width:60px}.explanation{grid-template-columns:1fr;gap:1.35rem;padding-top:23px}.explanation h3{margin-bottom:.35rem;font-size:.66rem}.explanation p{font-size:.86rem;line-height:1.95}.observed{gap:1rem;display:block;padding-top:17px}.observed-label{display:block;margin-bottom:.7rem}.measurement{display:inline-block;margin-right:1.2rem;font-size:.78rem}.measurement strong{font-size:1.2rem}.closing{display:block;padding:35px 0;margin-bottom:40px}.closing h2{font-size:1.6rem;margin-bottom:1.3rem}.closing p{font-size:.86rem}.reading{margin-bottom:35px}.reading dl{grid-template-columns:1fr;gap:.4rem;font-size:.8rem}.reading dd{margin-bottom:1rem}.downloads{grid-template-columns:1fr}.detail-body{font-size:.77rem}.detail-body pre{padding:12px}.specs th{width:45%}.foot{font-size:.61rem}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{html,body{background:white;color:#111}body{font-size:10pt}.topbar,.film-caption,.skip,.contents{display:none}.hero{min-height:auto;margin:0;display:block}.hero img,.hero::before{display:none}.hero-copy{padding:20px 0}.hero h1{font-size:32pt}.shell{width:100%}.hero-lead,.intro p,.chapter-lead,.explanation p,.closing p,.reading dd,.detail-body{color:#333}.intro{margin:20px 0}.chapter{break-inside:avoid;margin-bottom:30px}.chapter-heading{margin-bottom:10px}.chapter-title h2{font-size:20pt}.screen video{max-height:260px}.screen{box-shadow:none}.explanation{gap:1rem;padding-top:15px}.closing{margin-bottom:20px;padding:20px 0}.closing h2{font-size:18pt}.eyebrow,.explanation h3,.reading dt{color:#78502d}.measurement strong,.specs th{color:#111}.foot{padding:10px 0}.detail-body pre{background:#f4f3ef}a{color:#111}}
"""

JAVASCRIPT = r"""
// Local files only. Native media controls remain usable if JavaScript is off.
document.querySelectorAll('video').forEach(video => {
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => {
      if (other !== video) other.pause();
    });
  });
  video.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    // Space belongs to the browser's native media controls. Handling it here
    // as well can toggle twice when a native play button is focused.
    if (event.code === 'KeyK') {
      event.preventDefault();
      if (video.paused) video.play().catch(() => {}); else video.pause();
    } else if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
      if (!Number.isFinite(video.duration)) return;
      event.preventDefault();
      video.currentTime = Math.max(0, Math.min(video.duration,
        video.currentTime + (event.code === 'ArrowLeft' ? -5 : 5)));
    } else if (event.code === 'KeyF' && video.requestFullscreen) {
      event.preventDefault();
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      else video.requestFullscreen().catch(() => {});
    }
  });
});
document.querySelectorAll('[data-video-rate]').forEach(select => {
  select.addEventListener('change', () => {
    document.getElementById(select.dataset.videoRate).playbackRate = Number(select.value);
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
    with path.open(encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object: {display(path)}")
    return record


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def relative(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, output)).as_posix()


def recorded_path(raw: str, base: Path) -> Path:
    """Manifests use repository-relative paths or local output basenames."""
    path = Path(raw)
    if path.is_absolute():
        resolved = path.resolve()
    elif path.parts and path.parts[0] in {
        "artifacts", "configs", "docs", "reports", "scripts", "src", "tests"
    }:
        resolved = (ROOT / path).resolve()
    else:
        resolved = (base / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Manifest path is outside the repository: {raw}")
    return resolved


def check_hash_records(records: Any, base: Path, inputs: set[Path]) -> int:
    """Check dict or list records without silently accepting unhashed entries."""
    if isinstance(records, dict):
        rows = []
        for name, value in records.items():
            if isinstance(value, str):
                rows.append({"path": name, "sha256": value})
            elif isinstance(value, dict):
                rows.append({"path": name, **value})
            else:
                raise ValueError(f"Invalid hash record for {name}")
    elif isinstance(records, list):
        rows = records
    else:
        raise ValueError("Missing or invalid manifest hash records")
    if not rows:
        raise ValueError("A manifest hash-record section is empty")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid manifest hash record")
        raw = row.get("path", row.get("file"))
        expected = row.get("sha256")
        if not isinstance(raw, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing path/SHA-256 in hash record: {row}")
        path = recorded_path(raw, base)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Input integrity check failed: {display(path)}")
        size = row.get("bytes")
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"Recorded file size differs: {display(path)}")
        inputs.add(path)
    return len(rows)


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite measured value: {name}")
    return float(value)


def fmt(value: float) -> str:
    return f"{value:.3g}"


def link(label: str, target: str) -> str:
    return f'<a href="{escape(target)}">{escape(label)}</a>'


def paragraph(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def load_physics(physics: Path, inputs: set[Path]) -> dict[str, Any]:
    manifest_path = physics / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("The physical run is not a completed supported artifact")
    if manifest.get("qualification_passed") is not True or manifest.get("ordinary_tests_passed") is not True:
        raise ValueError("The physical run did not pass its recorded ordinary checks")
    inputs.add(manifest_path)
    check_hash_records(manifest["source_hashes"], ROOT, inputs)
    check_hash_records([manifest["input_config"]], ROOT, inputs)
    check_hash_records(manifest["artifact_hashes"], physics, inputs)
    required = ("config.json", "derived_metrics.json", "qualification.json", "tests.json")
    for name in required:
        if name not in manifest["artifact_hashes"]:
            raise ValueError(f"Unhashed physical input: {name}")
    raw_config = read_json(physics / "config.json")
    link_config = raw_config["link"]
    config = {
        "length_km": number(link_config["span_km"], "span_km"),
        "wavelength_nm": number(link_config["wavelength_m"], "wavelength_m") * 1e9,
        "t0_ps": number(raw_config["pulse"]["intensity_1e_half_width_s"], "intensity_1e_half_width_s") * 1e12,
        "peak_power_w": number(raw_config["pulse"]["peak_xy_power_W"], "peak_xy_power_W"),
        "loss_db_per_km": number(link_config["alpha_db_per_km"], "alpha_db_per_km"),
        "dispersion_ps_nm_km": number(link_config["dispersion_ps_nm_km"], "dispersion_ps_nm_km"),
        "gamma_w_inv_km": number(link_config["gamma_per_w_km"], "gamma_per_w_km"),
    }
    if link_config["n_spans"] != 1 or link_config["ase_enabled"] is not False:
        raise ValueError("The film essay requires one unamplified ASE-free fiber")
    if config["length_km"] != 80.0 or config["dispersion_ps_nm_km"] <= 0 or config["gamma_w_inv_km"] <= 0:
        raise ValueError("The essay's physical scope differs from the recorded run")
    cases = raw_config["cases"]
    if set(cases) != set(CASE_ORDER):
        raise ValueError("The three separated physical cases are required")
    expected_cases = {
        "pure_spm": (0.0, config["gamma_w_inv_km"]),
        "dispersion_only": (config["dispersion_ps_nm_km"], 0.0),
        "combined": (config["dispersion_ps_nm_km"], config["gamma_w_inv_km"]),
    }
    for case, (dispersion, gamma) in expected_cases.items():
        if cases[case]["dispersion_ps_nm_km"] != dispersion or cases[case]["gamma_per_w_km"] != gamma:
            raise ValueError(f"A physical control does not match its explanation: {case}")
        if f"{case}.npz" not in manifest["artifact_hashes"]:
            raise ValueError(f"Missing hash for the field data: {case}")
    qualification = read_json(physics / "qualification.json")
    stages = ("analytic_limits", "energy", "segmented_vs_whole", "step_refinement", "time_grid_and_window", "display_export")
    if qualification.get("passed") is not True or any(qualification[stage].get("passed") is not True for stage in stages):
        raise ValueError("A physical or display-export qualification check failed")
    if read_json(physics / "tests.json").get("passed") is not True:
        raise ValueError("The physical run's substantive tests did not pass")
    metrics_path = physics / "derived_metrics.json"
    metrics = read_json(metrics_path)
    measurements = {}
    initial_widths = []
    for case in CASE_ORDER:
        measured = metrics["cases"][case]
        z = [number(value, f"{case}.z_km") for value in measured["z_km"]]
        if len(z) < 2 or z[0] != 0 or z[-1] != config["length_km"] or any(b <= a for a, b in zip(z, z[1:])):
            raise ValueError(f"Invalid distance samples: {case}")
        arrays = {}
        for name in ("rms_time_ps", "rms_frequency_GHz", "center_phase_rad", "energy_outside_time_view_fraction"):
            values = [number(value, f"{case}.{name}") for value in measured[name]]
            if len(values) != len(z):
                raise ValueError(f"Metric/distance length mismatch: {case}.{name}")
            arrays[name] = values
        if min(arrays["rms_time_ps"] + arrays["rms_frequency_GHz"]) <= 0:
            raise ValueError(f"Nonpositive RMS width: {case}")
        measurements[case] = {
            "time_width_ratio": arrays["rms_time_ps"][-1] / arrays["rms_time_ps"][0],
            "frequency_width_ratio": arrays["rms_frequency_GHz"][-1] / arrays["rms_frequency_GHz"][0],
            "peak_nonlinear_phase_rad": arrays["center_phase_rad"][-1] - arrays["center_phase_rad"][0],
            "rms_time_in_ps": arrays["rms_time_ps"][0],
            "rms_time_out_ps": arrays["rms_time_ps"][-1],
            "rms_frequency_in_GHz": arrays["rms_frequency_GHz"][0],
            "rms_frequency_out_GHz": arrays["rms_frequency_GHz"][-1],
            "max_fraction_outside_physics_time_view": max(arrays["energy_outside_time_view_fraction"]),
        }
        initial_widths.append((arrays["rms_time_ps"][0], arrays["rms_frequency_GHz"][0]))
    if any(not math.isclose(value, expected, rel_tol=1e-10) for pair in initial_widths[1:] for value, expected in zip(pair, initial_widths[0])):
        raise ValueError("The three cases do not share the same input widths")
    source_uses = {
        "10.1103/PhysRevA.17.1448": "シリカファイバの自己位相変調と周波数幅の変化。今回のパルス条件の実測ではない。",
        "10.1109/50.622902": "二偏波の非線形結合を平均化したManakovモデルの基礎。",
        "10.1063/1.1654836": "非線形性と分散の相互作用の基礎。今回の損失を伴うガウスパルスをソリトンとは呼ばない。",
        "10.1109/JLT.2003.808628": "SSFMの距離刻みと精度の検討。今回の数値検証は保存した記録を参照。",
    }
    sources = [{**source, "use": source_uses.get(source.get("doi"), source["use"])} for source in manifest["sources"]]
    doc = ROOT / "docs/kerr-physics.md"
    if doc not in inputs:
        raise ValueError("The physical-methods document is not covered by its source manifest")
    return {
        "config": config,
        "physics_manifest": manifest_path,
        "physics_doc": doc,
        "metrics_path": metrics_path,
        "measurements": measurements,
        "sources": sources,
        "physics_record": manifest,
        "verification_text": "解析極限との照合、エネルギー保存と損失の確認、距離刻み・時間刻み・計算窓を変えた比較、保存用データの確認を行い、記録した通常検証の基準を満たした。独立validator評価は今回実施していない。",
    }


def load_context(physics: Path, renders: Path, output: Path) -> tuple[dict[str, Any], set[Path], dict[str, Any]]:
    inputs: set[Path] = set()
    context = load_physics(physics, inputs)
    candidates = [renders / name for name in ("render_manifest.json", "manifest.json") if (renders / name).is_file()]
    if len(candidates) != 1:
        raise ValueError("Exactly one completed render manifest is required")
    manifest_path = candidates[0]
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "optical-comm-kerr-physics-films/v1":
        raise ValueError("Unsupported physical-film manifest schema")
    inputs.add(manifest_path)
    render_inputs: set[Path] = set()
    check_hash_records(manifest["inputs"], ROOT, render_inputs)
    inputs.update(render_inputs)
    render_outputs: set[Path] = set()
    check_hash_records(manifest["outputs"], renders, render_outputs)
    inputs.update(render_outputs)
    for required in (context["physics_manifest"], physics / "config.json", physics / "derived_metrics.json", *(physics / f"{case}.npz" for case in CASE_ORDER)):
        if required not in render_inputs:
            raise ValueError(f"The renderer used a different or unhashed physical input: {display(required)}")
    films = {}
    max_omission = 0.0
    if len(manifest["films"]) != 3:
        raise ValueError("The explanation requires all three completed films")
    encoding = manifest["visual_encoding"]
    distance = encoding["distance_mapping"]
    if distance["start_km"] != 0 or distance["end_km"] != context["config"]["length_km"]:
        raise ValueError("The render distance range does not match the physical run")
    for film in manifest["films"]:
        case = film["case"]
        if case not in CASE_ORDER or case in films:
            raise ValueError("Missing, duplicated or unknown film case")
        names = FILMS[case]
        for key in ("file", "blend", "poster"):
            if film[key] != names[key] or (renders / film[key]).resolve() not in render_outputs:
                raise ValueError(f"Missing or unhashed film deliverable: {case}.{key}")
        scene = film["scene"]
        source = scene["source"]
        if source["file"] != f"{case}.npz" or source["sha256"] != sha256(physics / source["file"]):
            raise ValueError(f"The film scene does not identify the supplied physical data: {case}")
        if scene["visual_encoding"] != encoding:
            raise ValueError(f"The saved scene's visual mapping differs from the common mapping: {case}")
        duration = number(film["duration_seconds"], f"{case}.duration_seconds")
        fps = number(film["fps"], f"{case}.fps")
        if duration <= 0 or not math.isclose(duration * fps, film["frame_count"], abs_tol=1e-6):
            raise ValueError(f"Inconsistent frame count and duration: {case}")
        resolution = film["resolution"]
        if len(resolution) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolution):
            raise ValueError(f"Invalid film resolution: {case}")
        if scene["frame_count"] != film["frame_count"] or scene["resolution"] != resolution:
            raise ValueError(f"Saved scene and film settings differ: {case}")
        if encoding["distance_mapping"]["end_frame"] > film["frame_count"]:
            raise ValueError(f"The film does not reach the end of the distance sweep: {case}")
        crop = scene["display_crop"]
        if crop["requested_window_ps"] != encoding["retarded_time_window_ps"]:
            raise ValueError(f"Scene time crop differs from the declared common view: {case}")
        omission = number(crop["max_fractional_energy_omission_over_distance"], f"{case}.cropped_energy_fraction")
        if not 0 <= omission <= 1:
            raise ValueError(f"Invalid omitted-energy fraction: {case}")
        max_omission = max(max_omission, omission)
        films[case] = {
            **film,
            "resolution_label": f"{resolution[0]} × {resolution[1]}",
            "distance_caption": f'{fmt(distance["start_km"])} → {fmt(distance["end_km"])} kmの伝搬を表示',
        }
    font_record = manifest["assets"]["font"]
    font = Path(font_record["file"])
    font = (ROOT / font).resolve() if not font.is_absolute() else font.resolve()
    if not font.is_file() or sha256(font) != font_record["sha256"]:
        raise ValueError("The rendered text font differs from its recorded input")
    inputs.add(font)
    render_doc = ROOT / "docs/kerr-rendering.md"
    if not render_doc.is_file():
        raise ValueError("The renderer's reproduction document is missing")
    inputs.add(render_doc)
    core = number(encoding["core_diameter_um"], "core_diameter_um")
    cladding = number(encoding["cladding_diameter_um"], "cladding_diameter_um")
    magnification = number(encoding["magnified_view_scale_relative_to_intro"], "magnified_view_scale_relative_to_intro")
    carrier_ps = number(encoding["illustrated_carrier_period_ps"], "illustrated_carrier_period_ps")
    amplitude_scale = number(encoding["field_geometry_scale_scene_units_per_sqrt_W"], "field_geometry_scale")
    envelope_scale = number(encoding["amplitude_envelope_scale_scene_units_per_sqrt_W"], "envelope_geometry_scale")
    window = [number(value, "retarded_time_window_ps") for value in encoding["retarded_time_window_ps"]]
    if len(window) != 2 or window[0] >= window[1] or min(core, cladding, carrier_ps, amplitude_scale, envelope_scale, magnification) <= 0:
        raise ValueError("Invalid geometric/carrier mapping in the renderer")
    actual_period_ps = context["config"]["wavelength_nm"] * 1e-9 / 299792458.0 * 1e12
    phase = encoding["phase_color"]
    # This essay explains the renderer's declared palette. Refuse a changed
    # palette rather than displaying a stale description beside a new film.
    if phase["input"] != "arg(Ax), circular phase modulo 2pi" or phase["mapping"] != "cyclic teal at 0, warm orange toward +pi/2, blue toward -pi/2, magenta at +/-pi":
        raise ValueError("The phase-color mapping needs an updated Japanese explanation")
    loss_db = context["config"]["loss_db_per_km"] * context["config"]["length_km"]
    time_upsampling = encoding["display_time_upsampling"]
    upsampling_factor = number(time_upsampling["factor"], "display_time_upsampling.factor")
    if upsampling_factor != 4:
        raise ValueError("The display interpolation needs an updated explanation")
    mappings = [
        ("横方向", f'パルスと一緒に進む時間の座標。生成する電場形状は{fmt(window[0])}〜{fmt(window[1])} psの共通範囲で、カメラがその内部を接写する。低い裾は画面の外に出ることがある。映像中のガラスの長さは説明用の縮尺。'),
        ("線の揺れ", f'計算したX・Y偏波の電場から作る。見えるように振動の周期を{fmt(carrier_ps)} psへ拡大した。実際の{context["config"]["wavelength_nm"]:.0f} nmの光に対して約{carrier_ps / actual_period_ps:,.0f}倍。この入力の偏光方向は一定。'),
        ("透明な輪郭", f'電力の平方根に比例する振幅を、大きさに置き換えた模式表現。線と輪郭には、3本・全距離で共通の尺度を使う。{fmt(loss_db)} dBの伝送損失も残している。'),
        ("色", "位相を見分けるための色分け。青緑は0、橙は正側、青は負側、折り返しの境界は紫。光の見える色を撮影したものとしては読まない。ごく弱い裾では色を隠す。"),
        ("映像の進行", f'距離0〜{fmt(context["config"]["length_km"])} kmに伴う変化を、ゆっくり見る。1 km刻みの81波形の間は描画する形状と色を補間し、各動画フレームをBlenderでレンダリングした。上映時間は実際の飛行時間を表す値ではない。'),
    ]
    geometry = f'冒頭の横断面は、クラッド直径{fmt(cladding)} µmとコア直径{fmt(core)} µmの比を保持した。細部ではコアの表示を冒頭に対して約{fmt(magnification)}倍に拡大する。ガラスの切り欠きと器具は説明用のモデルで、電場と透明な輪郭はその中に重ねた計算量の可視化。'
    context.update({
        "render_dir": renders,
        "render_manifest": manifest_path,
        "render_doc": render_doc,
        "render_record": manifest,
        "films": films,
        "mapping": mappings,
        "geometry": geometry,
        "max_render_crop_omission": max_omission,
        "geometry_scale_details": {"field_scene_units_per_sqrt_W": amplitude_scale, "envelope_scene_units_per_sqrt_W": envelope_scale},
        "display_time_upsampling_factor": upsampling_factor,
    })
    context["sources"].append({"title": "Corning SMF-28 ULL データシート", "url": encoding["geometry_reference"], "use": "コアとクラッドの直径比の参考。描画したガラスと器具は説明用のモデル。"})
    checks = {
        "checked_file_count": len(inputs),
        "physical_run_complete_and_ordinary_checks_passed": True,
        "three_movies_scenes_and_posters_hash_verified": True,
        "renderer_physical_input_identity_verified": True,
        "shared_visual_encoding_and_scene_settings_verified": True,
        "metrics_derived_from_full_solver_records": True,
        "max_render_crop_energy_omission_fraction": max_omission,
        "html_external_runtime_assets": 0,
        "scope": "Input integrity and structural generation checks. Browser interaction and visual inspection are recorded separately by the coordinating agent.",
    }
    return context, inputs, checks


def build_page(context: dict[str, Any], output: Path) -> str:
    """Use only values extracted from checked simulation/render manifests."""
    renders = context["render_dir"]
    config = context["config"]
    geometry = context["geometry"]
    mapping = context["mapping"]
    measurements = context["measurements"]
    length = fmt(config["length_km"])
    contents = "".join(
        f'<a href="#film-{case}"><small>{FILMS[case]["number"]} / {FILMS[case]["eyebrow"]}</small>'
        f'<span>{escape(FILMS[case]["title"])}</span></a>' for case in CASE_ORDER
    )
    chapters = []
    download_rows = []
    for case in CASE_ORDER:
        film = FILMS[case]
        record = context["films"][case]
        video = escape(relative(renders / film["file"], output))
        poster = escape(relative(renders / film["poster"], output))
        measured = measurements[case]
        metrics_html = (
            '<span class="measurement">時間幅 <strong>' + fmt(measured["time_width_ratio"]) + '</strong><span class="unit">倍</span></span>'
            '<span class="measurement">周波数幅 <strong>' + fmt(measured["frequency_width_ratio"]) + '</strong><span class="unit">倍</span></span>'
        )
        note = "入射時に対する出口の比。どちらも強度で重み付けしたRMS幅。"
        if case == "pure_spm":
            metrics_html += '<span class="measurement">中心の非線形位相 <strong>' + fmt(measured["peak_nonlinear_phase_rad"]) + '</strong><span class="unit">rad</span></span>'
        meaning = film["meaning"]
        if case == "combined":
            time_comparison = measured["time_width_ratio"] / measurements["dispersion_only"]["time_width_ratio"]
            time_direction = "小さく" if time_comparison < 1 else "大きく"
            frequency_direction = "狭く" if measured["frequency_width_ratio"] < 1 else "広く"
            meaning = f"この条件の出口では、時間幅は分散だけの場合より{time_direction}なり、周波数幅は入射時より{frequency_direction}なった。作用の組み合わせによって、変化の向きと大きさが変わる。"
        chapters.append(f'''<section class="chapter shell" id="film-{case}" aria-labelledby="title-{case}">
  <header class="chapter-heading"><div class="chapter-title"><p class="eyebrow">{film["eyebrow"]}</p>
  <h2 id="title-{case}"><span class="chapter-number">{film["number"]}</span>{escape(film["title"])}</h2></div>
  <p class="chapter-lead">{escape(film["lead"])}</p></header>
  <figure class="film"><div class="screen"><video id="video-{case}" controls playsinline preload="none" poster="{poster}" tabindex="0" aria-label="{escape(film["title"])}" aria-describedby="caption-{case}">
    <source src="{video}" type="video/mp4">このブラウザでは動画を再生できません。{link("MP4を開く", relative(renders / film["file"], output))}
  </video></div><figcaption class="film-caption" id="caption-{case}"><div class="left"><span>{escape(record["resolution_label"])} · {fmt(record["duration_seconds"])}秒 · 無音</span><span>{escape(record["distance_caption"])}</span></div>
  <label class="rate">再生速度<select data-video-rate="video-{case}" aria-label="映像{film["number"]}の再生速度"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option></select></label></figcaption></figure>
  <div class="explanation"><div><h3>見るところ</h3>{paragraph(film["watch"])}</div><div><h3>起きていること</h3>{paragraph(film["change"])}</div><div><h3>何が分かるか</h3>{paragraph(meaning)}</div></div>
  <div class="observed"><span class="observed-label">{length} kmでの計算結果</span>{metrics_html}</div><p class="measurement-note">{note}</p>
</section>''')
        download_rows.append(
            f'<div><strong>{film["number"]} {escape(film["title"])}</strong><br>'
            + link("MP4", relative(renders / film["file"], output)) + " · "
            + link("編集できるBlenderシーン", relative(renders / film["blend"], output)) + "</div>"
        )
    mappings = "".join(f"<dt>{escape(title)}</dt><dd>{escape(value)}</dd>" for title, value in mapping)
    rows = [
        ("伝搬モデル", "二偏波のManakov方程式 / 対称SSFM"),
        ("距離", f"{length} km / 増幅器なし"),
        ("波長", f'{config["wavelength_nm"]:.0f} nm'),
        ("入射パルス", f'ガウス型 / 強度の1/e半幅 {fmt(config["t0_ps"])} ps'),
        ("入射ピーク電力", f'XY合計 {fmt(config["peak_power_w"] * 1000)} mW / 二偏波に等分'),
        ("損失", f'{fmt(config["loss_db_per_km"])} dB/km'),
        ("波長分散", f'{fmt(config["dispersion_ps_nm_km"])} ps/(nm·km) / 映像01では0'),
        ("非線形係数", f'{fmt(config["gamma_w_inv_km"])} /(W·km) / 映像02では0'),
        ("雑音・学習", "ASEなし / 学習なし / BER評価なし"),
    ]
    table = "".join(f"<tr><th scope=\"row\">{escape(key)}</th><td>{escape(value)}</td></tr>" for key, value in rows)
    sources = "".join(
        '<li>' + link(source["title"], source["url"]) + '<br>' + escape(source["use"]) + '</li>'
        for source in context["sources"]
    )
    technical = link("物理モデルと数値検証", relative(context["physics_doc"], output))
    render_technical = link("映像の作り方と再描画", relative(context["render_doc"], output))
    physical_manifest = link("計算データのマニフェスト", relative(context["physics_manifest"], output))
    render_manifest = link("映像のマニフェスト", relative(context["render_manifest"], output))
    explanation_manifest = link("この解説のマニフェスト", "explanation_manifest.json")
    qualified = context["verification_text"]
    crop_note = f'生成する電場形状の共通時間窓の外に残るエネルギーは、3条件・全距離を通して最大{fmt(context["max_render_crop_omission"] * 100)}%。元の物理データの標本から計算した値で、接写した画面の外に出るエネルギーを測った値ではない。RMS幅の計算には、2 nsの全計算窓を用いる。'
    interpolation_note = f'滑らかに描くため、複素電場の実部・虚部をパルス内の時間方向へ{fmt(context["display_time_upsampling_factor"])}倍の点密度で線形補間し、その点で模式的な振動を描く。元データの標本点は保持した。これは描画の補間で、物理計算の精度を追加する処理ではない。'
    total_duration = fmt(sum(context["films"][case]["duration_seconds"] for case in CASE_ORDER))
    combined = measurements["combined"]
    dispersion = measurements["dispersion_only"]
    difference = combined["time_width_ratio"] / dispersion["time_width_ratio"]
    if abs(difference - 1) < 0.005:
        comparison = "今回の出口の時間幅は、分散だけの場合に近い。位相や周波数の変化まで同じという意味ではない。"
    else:
        comparison = f"今回の出口では、Kerr効果と分散を同時に入れた時間幅が、分散だけの場合の{fmt(difference)}倍になった。"
    if combined["frequency_width_ratio"] < 1:
        comparison += f' 周波数幅は入射時の{fmt(combined["frequency_width_ratio"])}倍。二つを同時に入れたときの結果は、この条件での計算値だ。'
    reports_readme = escape(relative(ROOT / "reports/README.md", output))
    final_report = relative(ROOT / "reports/final_report.html", output)
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="数値計算した光パルスをBlenderで映像化し、自己位相変調、波長分散、その相互作用を3本の動画で見る日本語解説。"><meta name="color-scheme" content="dark light"><title>光が、光を変えていく。— 光ファイバの中のKerr効果</title><style>{CSS}</style></head>
<body><a class="skip" href="#films">動画と解説へ進む</a><header class="topbar shell"><a class="brand" href="{reports_readme}">OPTICAL COMM LAB</a><nav aria-label="ページ内案内"><a href="#films">3つの映像</a><a href="#reading">映像の読み方</a><a href="#sources">出典・記録</a></nav></header>
<main><section class="hero" aria-labelledby="page-title"><img src="{escape(relative(renders / FILMS["combined"]["poster"], output))}" alt="光ファイバと、計算された光パルスの立体的な可視化"><div class="hero-copy shell"><p class="eyebrow">INSIDE AN OPTICAL FIBER</p><h1 id="page-title">光が、光を<br>変えていく。</h1><p class="hero-lead">光ファイバの中で、パルスはどう変わるのか。<br>同じパルスを、三つの条件で追いかける。</p><div class="hero-foot"><span>3 FILMS / {total_duration} SECONDS</span><span>PHYSICS SIMULATION + BLENDER</span><span>{length} km</span></div></div></section>
<div class="intro shell" id="films"><p class="statement">光の「強さ」と「位相」。<br>二つを分けて見ると、変化の理由が見えてくる。</p><p>強さは、パルスのどこに光のエネルギーが集まっているか。位相は、光の振動が周期のどこにあるか。この二つは別の性質だ。Kerr効果は強さに応じて位相を変え、波長分散は周波数成分の到着をずらす。</p><p>以下の映像は、ガウス型の短い光パルスを{length} km伝搬させた計算から作った。まず一つずつ、最後に同時に。横方向は、一つのパルスの中での相対時刻。パルスと一緒に進む視点で見ている。</p><p class="observation">波が低くなるのは、3条件に共通するファイバの損失です。Kerr効果だけの条件では、時間方向の幅は変わりません。</p><nav class="contents" aria-label="各映像へ">{contents}</nav></div>
{"".join(chapters)}
<section class="closing shell" aria-labelledby="conclusion"><h2 id="conclusion">位相の変化が、<br>形の変化へつながる。</h2><div><p>{escape(comparison)}</p><p>光通信では、多数のパルスが続いて情報を運ぶ。位相や形が変わると、受信した情報を読み取りにくくなることがある。この三つの映像は、その前段にある物理を、単独のパルスで見ている。</p><p>本リポジトリの1600 km・DP-16QAM・増幅雑音を含む学習評価とは、別の説明用実験。ここから補償性能やビット誤り率の改善を結論づけることはできない。</p></div></section>
<section class="reading shell" id="reading" aria-labelledby="reading-title"><h2 id="reading-title">映像の読み方</h2><dl>{mappings}<dt>操作</dt><dd>自動再生はしない。動画にフォーカスした状態で Space / K は再生・停止、← / → は5秒移動、F は全画面。各動画の速度は0.5倍にもできる。</dd></dl></section>
<div class="supplement shell" id="sources"><details><summary>計算条件と、ここから言えること</summary><div class="detail-body"><table class="specs"><caption class="eyebrow">3本に共通する入射条件</caption><tbody>{table}</tbody></table><p>{escape(qualified)}</p><p>強度幅と周波数幅は、分布全体を重み付けして求めたRMS幅。見かけの輪郭の幅、半値全幅、中心の明るさとは区別する。入射に対する比を示し、各フレームを個別の最大値で描き直して結果を強調しない。</p><p>{technical} · {physical_manifest} · {render_manifest}</p></div></details>
<details><summary>映像の表現と、実際の光との対応</summary><div class="detail-body">{paragraph(geometry)}{paragraph(crop_note)}{paragraph(interpolation_note)}<p>ごく弱い裾では、位相の色を隠す。基準は各保存距離のピーク電力に対する10⁻⁶。振幅を毎フレームの最大値で拡大せず、損失を含む絶対値を共通の表示尺度へ写している。</p><p>ガラスの見え方、背景、照明、カメラの動きは説明のための表現。使用した表現と倍率は映像マニフェストに保存した。</p><p>{render_technical} · {render_manifest}</p></div></details>
<details><summary>動画・Blenderシーン・再生成記録</summary><div class="detail-body"><div class="downloads">{"".join(download_rows)}</div><p style="margin-top:1.5rem">動画はこのHTMLの隣に保存したMP4を読む。フォルダ全体を保持すれば、ネット接続を使わずに閲覧できる。外部フォントや画像を取得しない。出典リンクを開くときだけ接続が必要。</p><p>{explanation_manifest} · {link("再生成手順", "README.md")}</p><pre><code>{escape(context["regeneration_command"])}</code></pre><p>ファイルのSHA-256は、今回届けた版の識別に用いる。Blender、GPU、エンコード環境が変わると、再描画した画素やMP4のバイト列は一致しないことがある。</p></div></details>
<details><summary>出典</summary><div class="detail-body"><ul>{sources}</ul></div></details></div></main>
<footer class="foot shell"><span>OPTICAL COMM LAB · KERR PHYSICS FILMS</span><span>{link("研究の結果レポート", final_report)} · {link("記録と再生成", "README.md")}</span></footer><script>{JAVASCRIPT}</script></body></html>
'''


def build_readme(context: dict[str, Any], output: Path) -> str:
    movies = "\n".join(
        f'| {FILMS[case]["number"]} | [{FILMS[case]["title"]}]({relative(context["render_dir"] / FILMS[case]["file"], output)}) | '
        f'[{FILMS[case]["blend"]}]({relative(context["render_dir"] / FILMS[case]["blend"], output)}) | '
        f'{fmt(context["films"][case]["duration_seconds"])}秒 / {context["films"][case]["resolution_label"]} |'
        for case in CASE_ORDER
    )
    return f'''# 光ファイバの中のKerr効果 — 3本の物理映像

[日本語の映像解説を開く](index.html)。3本の動画を順に見て、自己位相変調、波長分散、両者が同時に起きる場合を比べる。動画の下に「見るところ」「起きていること」「何が分かるか」と、計算したRMS幅の比を示す。

| 映像 | MP4 | 編集できるBlenderシーン | 保存仕様 |
| --- | --- | --- | --- |
{movies}

HTMLと動画・ポスターを同じフォルダ構成で保持すれば、ブラウザでオフライン閲覧できる。外部のJavaScript、画像、フォントは読み込まない。無音・手動再生で、速度は0.5 / 1 / 1.5倍を選べる。動画にフォーカスすると Space / K で再生・停止、← / → で5秒移動、F で全画面にできる。JavaScriptを無効にしても標準の動画操作は利用できる。

## 物理計算と表現

{fmt(context["config"]["length_km"])} kmの損失・二偏波Manakov伝搬を、既存のSSFM通信路で計算した説明用実験。ガウスパルスの瞬間ピーク電力はXY合計{fmt(context["config"]["peak_power_w"] * 1000)} mW、強度の1/e半幅は{fmt(context["config"]["t0_ps"])} ps。自己位相変調だけ、分散だけ、両方の三つの条件を分けた。増幅器・ASE・通信シンボル・学習・BER評価は含まない。本研究の1600 km評価とは別のデータであり、補償性能の根拠には用いない。

{context["geometry"]}

''' + "\n\n".join(f"**{title}**：{value}" for title, value in context["mapping"]) + f'''

計算条件・数値確認・出典は [物理モデルの文書]({relative(context["physics_doc"], output)})、原データは [計算マニフェスト]({relative(context["physics_manifest"], output)}) と [導出量]({relative(context["metrics_path"], output)}) を参照する。映像の設定と再描画コマンドは [{context["render_manifest"].name}]({relative(context["render_manifest"], output)}) と [描画手順]({relative(context["render_doc"], output)}) に保存した。{context["verification_text"]}

## 解説の再生成

リポジトリ直下で、保存済みの計算データと3本の映像から再生成する。Python標準ライブラリだけで動作し、伝搬・学習・レンダリングは行わない。

```bash
{context["regeneration_command"]}
```

計算マニフェストに記録されたソース・設定・全artifact、映像マニフェストの入力・出力をSHA-256で照合する。未完成の計算、数値確認に失敗した計算、欠落・変更された動画や入力からは生成しない。`--output-dir .work/kerr-explanation-check` を追加すれば、公開済みのHTMLとREADMEを変更せずに確認できる。

計算と動画をやり直すときは、[描画手順]({relative(context["render_doc"], output)})にある `.venv/bin/python` のコマンドを使い、保存済み成果物と分けた出力先で実行する。描画マニフェストの `python3` は、Pillowを含むプロジェクト環境を有効にしたPythonを想定している。GPU・Blender・FFmpegの差で画素やエンコード済みのバイト列が変わる可能性があり、SHA-256は今回保存した版を識別するための記録。

描画ではパルス内の時刻方向に4倍の密度で複素電場を線形補間する。元の物理標本点と、数値結果の導出量は保持する。詳細は映像マニフェストの `display_time_upsampling` を参照する。接写で画面の外に出る低い裾と、数値データ窓の切り取りによる省略は区別する。

[explanation_manifest.json](explanation_manifest.json) は入力と出力のハッシュ、生成CLIの版、環境、再生成コマンドと通常確認を記録する。自己参照を避けるため、そのマニフェスト自身のハッシュは内包しない。独立validator評価は今回のタスクでは指定0・試行0・完了0。

旧版の30秒の模式アニメーションは [当時のREADME]({relative(ROOT / "reports/animations/kerr/README.md", output)}) とマニフェストを保持している。こちらは別の生成物で、今回の物理計算動画の入力ではない。
'''


def inspect_page(page: str, output: Path) -> dict[str, Any]:
    """Check local media and links before publishing any generated files."""
    class Elements(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.ids: set[str] = set()
            self.references: list[tuple[str, str]] = []
            self.videos = 0
            self.sources = 0

        def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
            attrs = dict(attributes)
            if attrs.get("id"):
                identity = attrs["id"]
                if identity in self.ids:
                    raise ValueError(f"Duplicate HTML ID: {identity}")
                self.ids.add(identity)
            if tag == "video":
                self.videos += 1
                if "controls" not in attrs or "autoplay" in attrs or attrs.get("tabindex") != "0":
                    raise ValueError("Video controls/manual playback/keyboard focus are required")
            if tag == "source":
                self.sources += 1
                if attrs.get("type") != "video/mp4":
                    raise ValueError("Unexpected video media type")
            for name in ("src", "poster", "href"):
                if attrs.get(name):
                    self.references.append((name, attrs[name]))

    elements = Elements()
    elements.feed(page)
    if elements.videos != 3 or elements.sources != 3:
        raise ValueError("The page must contain three complete MP4 players")
    planned = {output / name for name in ("index.html", "README.md", "explanation_manifest.json")}
    local_links = 0
    for kind, raw in elements.references:
        url = urlsplit(raw)
        if url.scheme or url.netloc:
            if kind != "href" or url.scheme != "https":
                raise ValueError(f"External runtime asset is forbidden: {raw}")
            continue
        if url.path:
            path = (output / unquote(url.path)).resolve()
            if not path.is_file() and path not in planned:
                raise ValueError(f"Broken local page reference: {raw}")
            local_links += 1
        elif url.fragment and url.fragment not in elements.ids:
            raise ValueError(f"Broken HTML anchor: {raw}")
    return {"local_video_players": elements.videos, "local_link_and_media_references_checked": local_links,
            "duplicate_ids": 0, "broken_local_references": 0, "external_runtime_assets": 0,
            "autoplay": False, "keyboard_focus_and_native_controls": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-run", type=Path, default=Path("artifacts/kerr-physics"))
    parser.add_argument("--renders", type=Path, default=Path("reports/animations/kerr_physics"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    physics = (ROOT / args.physics_run).resolve()
    renders = (ROOT / args.renders).resolve()
    output = (ROOT / args.output_dir).resolve() if args.output_dir else renders
    if not output.is_relative_to(ROOT) or (output != renders and not output.is_relative_to(ROOT / ".work")):
        raise ValueError("Output must be the render directory or an isolated .work directory")
    context, inputs, checks = load_context(physics, renders, output)
    command = [".venv/bin/python", "scripts/build_kerr_explanation.py", "--physics-run", display(physics), "--renders", display(renders)]
    if output != renders:
        command.extend(["--output-dir", display(output)])
    context["regeneration_command"] = shlex.join(command)
    inputs.add(Path(__file__).resolve())
    checks["checked_file_count"] = len(inputs)
    frozen = {path: sha256(path) for path in inputs}
    page = build_page(context, output)
    readme = build_readme(context, output)
    checks["html_structure"] = inspect_page(page, output)
    for path, expected in frozen.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"An input changed during page generation: {display(path)}")
    # Construct both deliverables before touching the output directory.
    output.mkdir(parents=True, exist_ok=True)
    page_path = output / "index.html"
    readme_path = output / "README.md"
    page_path.write_text(page, encoding="utf-8")
    readme_path.write_text(readme, encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    record = {
        "schema": "optical-comm-kerr-explanation/v1",
        "issue": 17,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "scope": "Offline Japanese explanation of three films generated from a separate 80 km deterministic dual-polarization Manakov pulse experiment. No learning or communications-performance evaluation.",
        "generator": {"path": display(Path(__file__)), "sha256": sha256(Path(__file__))},
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "regeneration_command": context["regeneration_command"],
        "input_manifests": {"physics": display(context["physics_manifest"]), "render": display(context["render_manifest"])},
        "inputs": [{"path": display(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(inputs)],
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in (page_path, readme_path)},
        "ordinary_checks": checks,
        "independent_validator": {"requested": 0, "attempted": 0, "completed": 0},
        "browser": {"external_runtime_assets": [], "autoplay": False, "local_video_count": 3, "font_source": "browser/OS native fonts; no network requests", "keyboard": "With video focused: Space/K play or pause; left/right seek 5 seconds; F fullscreen; native video controls retained."},
    }
    (output / "explanation_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {display(page_path)} after verifying {checks['checked_file_count']} input files.")


if __name__ == "__main__":
    main()
