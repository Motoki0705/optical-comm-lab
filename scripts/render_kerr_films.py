#!/usr/bin/env python3
"""Render three data-driven optical-fiber films in native Blender/Cycles.

The propagation is supplied by ``artifacts/kerr-physics``; this script does
not solve the channel.  Its enlarged field geometry and illustrated carrier
make the complex envelope visible inside a cutaway fiber.  Start with
``--mode stills`` or ``--mode proof`` before an expensive full render.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import time
import zlib


FILMS = {
    "pure_spm": {"stem": "01_self_phase_modulation", "poster": "poster_01.png", "title": "Kerr効果：光の強さが位相を変える"},
    "dispersion_only": {"stem": "02_chromatic_dispersion", "poster": "poster_02.png", "title": "分散：パルスが時間方向に広がる"},
    "combined": {"stem": "03_kerr_dispersion_interplay", "poster": "poster_03.png", "title": "Kerr効果と分散が同時に働く"},
}
WIDTH, HEIGHT, FPS = 1920, 1080, 24
DURATION = 12.0
TIME_WINDOW_PS = 240.0
CARRIER_PERIOD_PS = 12.0
DISPLAY_TIME_UPSAMPLING = 4
PEAK_TOTAL_W = .050
COMPONENT_REFERENCE_SQRT_W = math.sqrt(PEAK_TOTAL_W / 2)
AMPLITUDE_SCALE = .68 / COMPONENT_REFERENCE_SQRT_W
ENVELOPE_SCALE = 1.02 / math.sqrt(PEAK_TOTAL_W)
AXIAL_UNITS_PER_PS = 5.4 / TIME_WINDOW_PS
CORE_DIAMETER_UM, CLAD_DIAMETER_UM = 8.2, 125.
CORE_RADIUS = .8 * CORE_DIAMETER_UM / CLAD_DIAMETER_UM
INTRO_MAGNIFICATION = CORE_RADIUS / 1.1
ILLUSTRATION_MAGNIFICATION = 1. / INTRO_MAGNIFICATION


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: dict | list) -> None:
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def input_record_key(path: Path, root: Path) -> str:
    """Keep repository inputs portable; retain an explicit external location."""
    resolved=path.resolve()
    return str(resolved.relative_to(root.resolve())) if resolved.is_relative_to(root.resolve()) else str(resolved)


def selected_proof_case(cases: list[str]) -> str:
    return "combined" if "combined" in cases else cases[-1]


def gpu_report() -> dict:
    try:
        value=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version","--format=csv,noheader"],text=True,timeout=10)
        rows=sorted(row.strip() for row in value.splitlines() if row.strip())
        return {"known":bool(rows),"query":"uuid,name,driver_version","rows":rows}
    except (FileNotFoundError,subprocess.SubprocessError):
        return {"known":False,"query":"uuid,name,driver_version","rows":[],"reason":"nvidia-smi unavailable; cross-run frame reuse disabled"}


def render_signature(parameters: dict, environment: dict) -> str:
    return hashlib.sha256(json.dumps({"parameters":parameters,"environment":environment},sort_keys=True).encode()).hexdigest()


def require_resume_signature(previous: dict, signature: str, environment: dict) -> None:
    if not environment.get("gpu_driver_report",{}).get("known"):
        raise RuntimeError("Resume refused: GPU driver identity is unavailable")
    if previous.get("signature")!=signature:
        raise RuntimeError("Resume refused: source, data, render settings, Blender build, device, or driver changed")


def inspect_png(path: Path, expected_size: tuple[int,int]=(WIDTH,HEIGHT)) -> dict:
    """Decode the lossless native RGB/RGBA PNG stream and verify its chunks."""
    blob=path.read_bytes()
    if blob[:8]!=b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    offset=8
    header=None
    payload=[]
    ended=False
    while offset<len(blob):
        if offset+12>len(blob):
            raise ValueError("Truncated PNG chunk")
        size=struct.unpack_from(">I",blob,offset)[0]
        kind=blob[offset+4:offset+8]
        end=offset+12+size
        if end>len(blob):
            raise ValueError("Truncated PNG payload")
        chunk=blob[offset+8:offset+8+size]
        crc=struct.unpack_from(">I",blob,offset+8+size)[0]
        if zlib.crc32(kind+chunk)&0xFFFFFFFF!=crc:
            raise ValueError("PNG chunk CRC mismatch")
        if kind==b"IHDR":
            if header is not None or len(chunk)!=13:
                raise ValueError("Invalid PNG header")
            header=struct.unpack(">IIBBBBB",chunk)
        elif kind==b"IDAT":
            payload.append(chunk)
        elif kind==b"IEND":
            if size!=0 or end!=len(blob):
                raise ValueError("Invalid PNG ending")
            ended=True
            break
        offset=end
    if not ended or header is None or not payload:
        raise ValueError("Incomplete PNG")
    width,height,depth,color,compression,filter_method,interlace=header
    if (width,height)!=expected_size or depth!=8 or color not in [2,6] or compression or filter_method or interlace:
        raise ValueError("Unexpected PNG dimensions or native RGB format")
    row_size=1+width*(3 if color==2 else 4)
    decoder=zlib.decompressobj()
    try:
        decoded=decoder.decompress(b"".join(payload),row_size*height+1)
    except zlib.error as error:
        raise ValueError("PNG raster decompression failed") from error
    if not decoder.eof or decoder.unused_data or len(decoded)!=row_size*height or any(decoded[index]>4 for index in range(0,len(decoded),row_size)):
        raise ValueError("PNG raster is truncated or has invalid filters")
    return {"bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),"width":width,"height":height}


def cached_frame_valid(path: Path, record: dict | None, expected_size: tuple[int,int]=(WIDTH,HEIGHT)) -> bool:
    if not record:
        return False
    try:
        return inspect_png(path,expected_size)==record
    except (OSError,ValueError):
        return False


def arguments(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blender", help="Blender executable, including a Windows .exe under WSL")
    p.add_argument("--data-dir", type=Path, default=Path("artifacts/kerr-physics"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/animations/kerr_physics"))
    p.add_argument("--scratch-dir", type=Path, default=Path(".work/kerr-cinema"))
    p.add_argument("--mode", choices=["stills", "proof", "build", "render"], default="stills")
    p.add_argument("--case", choices=list(FILMS), nargs="+", default=list(FILMS))
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--adaptive-threshold", type=float, default=.014)
    p.add_argument("--duration", type=float, default=DURATION)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"))
    p.add_argument("--font-license",type=Path,help="License text for a custom caption font; default IPAexGothic uses its installed package license")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    p.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    p.add_argument("--resume", action="store_true", help="Reuse frames only when their source/config/data signature matches")
    p.add_argument("--inside-blender", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gpu-report",default="",help=argparse.SUPPRESS)
    p.add_argument("--proof-motion-case",choices=list(FILMS),help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.samples < 16 or not 0 < a.adaptive_threshold <= .1:
        p.error("Use at least 16 samples and an adaptive threshold in (0, .1]")
    if not 10 <= a.duration <= 20 or a.fps != 24:
        p.error("These films use 24fps and a duration between 10 and 20 seconds")
    return a


def frame_count(args: argparse.Namespace) -> int:
    return round(args.duration * args.fps)


def distance_at_frame(frame: int, count: int) -> float:
    """An explicitly illustrative distance sweep, not the optical flight time."""
    start, end = 2.5 * FPS + 1, count - 1.5 * FPS
    return 80. * max(0., min(1., (frame - start) / (end - start)))


def native_path(path: Path, windows_blender: bool) -> str:
    if windows_blender and os.name != "nt":
        return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()
    return str(path.resolve())


def visual_encoding(count: int) -> dict:
    return {
        "scientific_scope": "Deterministic dual-polarization Manakov pulse propagation: one 80 km passive span, loss included, no EDFA/ASE or receiver/training result.",
        "geometry": "A realistic macro fiber and precision fixtures give context. A cutaway exposes the computed field. The detailed field is an enlarged schematic of electric-field values, not a photon path, ray bundle, or physical motion of the fiber.",
        "core_diameter_um": CORE_DIAMETER_UM,
        "cladding_diameter_um": CLAD_DIAMETER_UM,
        "geometry_reference": "https://www.corning.com/media/worldwide/coc/documents/Fiber/SMF-28%20ULL.pdf",
        "real_ratio_intro_frames": [1, 37],
        "expanded_core_transition_frames": [37, 73],
        "magnified_view_scale_relative_to_intro": ILLUSTRATION_MAGNIFICATION,
        "detailed_cladding_radius_relative_to_intro": 1.5,
        "retarded_time_window_ps": [-TIME_WINDOW_PS, TIME_WINDOW_PS],
        "axial_units_per_ps": AXIAL_UNITS_PER_PS,
        "axial_meaning": "Position along the exposed segment encodes retarded time within a pulse at the displayed propagation distance; the centimeter-like specimen length is illustrative and is not 80 km mapped literally to a single spatial instant.",
        "camera_framing": "The camera approaches the exposed core for a macro closeup. +/-240ps is the generated field geometry/data window, not an axis drawn across the image; the low-power outer tails may be outside the closeup framing. Camera paths are identical for the three cases.",
        "illustrated_carrier_period_ps": CARRIER_PERIOD_PS,
        "display_time_upsampling": {"factor":DISPLAY_TIME_UPSAMPLING,"method":"Linear interpolation of real/imaginary complex-envelope components in retarded time, followed by analytic evaluation of the illustrated carrier on that finer grid. Original exported sample knots and physical data are unchanged. This is display smoothing, not propagation refinement.","phase_mask":"Conservative interpolation: phase color is present only where the bracketing original phase_valid samples are valid."},
        "carrier": "E_pol(t)=Re[A_pol(z,t) exp(-i 2 pi t / 12ps)]. The 1550 nm infrared carrier is invisible and far faster. Its period is enlarged for visibility; the envelope and envelope phase come from the numerical fields.",
        "field_components": "The field curve coordinates are (displayed retarded time, E_y, E_x). The same phase in Ax and Ay preserves linear polarization at 45 degrees; envelope phase is never used as a polarization rotation.",
        "field_amplitude_units": "sqrt(W), proportional to electric field for the supplied normalization",
        "field_geometry_scale_scene_units_per_sqrt_W": AMPLITUDE_SCALE,
        "amplitude_envelope_scale_scene_units_per_sqrt_W": ENVELOPE_SCALE,
        "envelope_geometry": "The translucent envelope radius encodes absolute sqrt(Px+Py) on one fixed scale. It is an amplitude glyph, not a changing physical mode-field radius or a transverse electromagnetic simulation.",
        "amplitude_scaling": "A single fixed mapping is shared by all cases and all distances. The 16 dB passive loss is retained; no per-frame or per-case normalization, gain compensation, or synthetic enhancement of the numerical field is applied.",
        "phase_color": {"input": "arg(Ax), circular phase modulo 2pi", "range_rad": [-math.pi, math.pi], "mapping": "cyclic teal at 0, warm orange toward +pi/2, blue toward -pi/2, magenta at +/-pi", "mask": "The supplied phase_valid mask hides phase colors below 1e-6 of each snapshot's peak. Absolute field and power arrays remain unchanged."},
        "distance_mapping": {"start_km": 0., "end_km": 80., "start_frame": 61, "end_frame": count-36, "start_timestamp_seconds":2.5,"end_timestamp_seconds":(count-37)/FPS,"sweep_seconds":(count-97)/FPS,"formula":"z_km=80*clip((frame-61)/(frame_count-97),0,1)","hold_start_seconds": 2.5, "hold_end_seconds_between_rendered_endpoints": 1.5, "meaning": "An illustrative sweep through propagation distance, not a real-time transit video."},
        "display_interpolation": "Native Blender linear shape-key interpolation between the 81 measured solver export distances (0,1,...,80 km); atlas phase colors interpolate between the same rows. Every MP4 frame is actually rendered by Cycles; no optical-flow interpolation or duplication to accelerate rendering.",
        "renderer_scope": "Cycles traces visible studio illumination to render materials. It does not solve the nonlinear optical propagation; all data-driven field geometry is generated from the separate numerical artifacts.",
    }


def make_overlay(path: Path, case: str, frame: int, args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont
    count = frame_count(args)
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0,0,0,0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(args.font), 30)
    small = ImageFont.truetype(str(args.font), 25)

    def text(pos, value, f=font, fill=(237,247,255,255)):
        draw.text((pos[0]+1,pos[1]+2), value, font=f, fill=(0,0,0,170), stroke_width=2, stroke_fill=(0,0,0,140))
        draw.text(pos, value, font=f, fill=fill)

    text((42,35), FILMS[case]["title"])
    text((1450,37), f"伝搬距離  {distance_at_frame(frame,count):5.1f} km", f=small)
    if frame <= 37:
        text((42,HEIGHT-91), "ファイバ断面の実寸比：コア 8.2 µm / クラッド 125 µm", f=small)
    else:
        text((42,HEIGHT-91), "光の電場を拡大表示 ｜ 振幅＝共通の固定倍率、色＝包絡線の位相", f=small)
    text((42,HEIGHT-52), "横＝パルス内の相対時刻（データ窓 ±240 ps）｜ 搬送波周期 12 ps は模式表示（実光は1550 nmの赤外線）", f=small, fill=(174,200,218,255))
    image.save(path)


def encode_video(args: argparse.Namespace, case: str, frames: list[int], scratch: Path, target: Path) -> dict:
    overlay = scratch / case / "overlays"
    overlay.mkdir(parents=True,exist_ok=True)
    for frame in frames:
        make_overlay(overlay / f"frame_{frame:04d}.png", case, frame, args)
    cmd = [args.ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
           "-framerate", str(args.fps), "-start_number", str(frames[0]), "-i", str(scratch/case/"frames"/"frame_%04d.png"),
           "-framerate", str(args.fps), "-start_number", str(frames[0]), "-i", str(overlay/"frame_%04d.png"),
           "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto[v]", "-map", "[v]", "-frames:v", str(len(frames)),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(target)]
    subprocess.run(cmd,check=True)
    probe = json.loads(subprocess.check_output([args.ffprobe,"-v","error","-count_frames","-show_streams","-show_format","-of","json",str(target)],text=True))
    stream = probe["streams"][0]
    if len(probe["streams"]) != 1 or (stream["width"],stream["height"]) != (WIDTH,HEIGHT) or stream["r_frame_rate"] != f"{args.fps}/1" or int(stream["nb_read_frames"]) != len(frames):
        raise RuntimeError("Encoded film has an unexpected frame count or format")
    if abs(float(probe["format"]["duration"])-len(frames)/args.fps) > .002:
        raise RuntimeError("Encoded film duration differs from the rendered frame count")
    return {"command":cmd,"ffprobe":probe}


def run_host(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    root = script.parents[1]
    data = args.data_dir.resolve()
    scratch = args.scratch_dir.resolve()
    output = args.output_dir.resolve() if args.mode in ["render","build"] else scratch/"data-proofs"
    for case in args.case:
        if not (data/f"{case}.npz").is_file():
            raise FileNotFoundError(f"Actual propagation field is required: {data/case}.npz")
    if not (data/"manifest.json").is_file():
        raise FileNotFoundError("The numerical artifact manifest is required")
    physical_manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    if physical_manifest.get("status")!="complete":
        raise ValueError("Only completed numerical artifacts may be rendered")
    for case in args.case:
        if sha256(data/(case+".npz"))!=physical_manifest["artifact_hashes"][case+".npz"]:
            raise ValueError(f"Numerical artifact hash differs from its manifest: {case}")
    blender = args.blender or shutil.which("blender")
    default = Path("/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe")
    if not blender and default.is_file():
        blender = str(default)
    if not blender:
        raise FileNotFoundError("Pass --blender /path/to/blender")
    if args.mode in ["render","proof"]:
        if not args.ffmpeg or not args.ffprobe or not args.font.is_file():
            raise FileNotFoundError("FFmpeg, FFprobe, and the Japanese overlay font are required")
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Pillow is required for captions; run with .venv/bin/python or install Pillow before rendering") from error
        default_font=Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
        if args.font_license is None:
            if not default_font.is_file() or sha256(args.font)!=sha256(default_font):
                raise ValueError("A custom font requires --font-license with its license text")
            args.font_license=Path("/usr/share/doc/fonts-ipaexfont-gothic/copyright")
        if not args.font_license.is_file():
            raise FileNotFoundError(f"Caption font license is required before rendering: {args.font_license}")
    driver_report=gpu_report()
    if args.resume and not driver_report["known"]:
        raise RuntimeError("Resume refused: GPU driver identity is unavailable")
    scratch.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    windows = str(blender).lower().endswith(".exe")
    invocations=[]
    metadata=[]
    for case in args.case:
        invocation=[str(blender),"--background","--factory-startup","--python-exit-code","1","--python",native_path(script,windows),"--","--inside-blender","--mode",args.mode,"--case",case,"--proof-motion-case",selected_proof_case(args.case),"--gpu-report",json.dumps(driver_report,sort_keys=True),"--data-dir",native_path(data,windows),"--output-dir",native_path(output,windows),"--scratch-dir",native_path(scratch,windows),"--samples",str(args.samples),"--adaptive-threshold",str(args.adaptive_threshold),"--duration",str(args.duration),"--fps",str(args.fps)]
        if args.resume:
            invocation.append("--resume")
        log = scratch / case / f"{args.mode}.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        print(f"Building/rendering {case}; log: {log}",flush=True)
        with log.open("w",encoding="utf-8") as handle:
            subprocess.run(invocation,stdout=handle,stderr=subprocess.STDOUT,check=True)
        invocations.append(invocation)
        metadata.append(json.loads((scratch/case/"scene_metadata.json").read_text(encoding="utf-8")))
        print(f"Completed {case}: {metadata[-1]['render_summary']}",flush=True)
    if args.mode in ["stills","build"]:
        return
    if args.mode == "proof":
        case = selected_proof_case(args.case)
        frames = metadata[args.case.index(case)]["motion_proof_frames"]
        if frames:
            result=encode_video(args,case,frames,scratch,output/"computed_field_motion_proof.mp4")
            (output/"motion_proof_probe.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        return
    films=[]
    encodings=[]
    for case,scene_meta in zip(args.case,metadata):
        info=FILMS[case]
        frames=list(range(1,frame_count(args)+1))
        target=output/(info["stem"]+".mp4")
        encodings.append(encode_video(args,case,frames,scratch,target))
        poster_frame=round(frame_count(args)*.55)
        from PIL import Image
        poster=Image.open(scratch/case/"frames"/f"frame_{poster_frame:04d}.png").convert("RGBA")
        overlay=Image.open(scratch/case/"overlays"/f"frame_{poster_frame:04d}.png").convert("RGBA")
        Image.alpha_composite(poster,overlay).convert("RGB").save(output/info["poster"])
        films.append({"case":case,"title":info["title"],"file":info["stem"]+".mp4","blend":info["stem"]+".blend","poster":info["poster"],"duration_seconds":args.duration,"fps":args.fps,"frame_count":frame_count(args),"resolution":[WIDTH,HEIGHT],"scene":scene_meta})
    license_source=args.font_license
    license_text=f"Caption font: {args.font.name}, used without modification to rasterize post-production captions.\nNo font binary is embedded in the native Blender scene.\nLicense input: {license_source}\n\n"+license_source.read_text(encoding="utf-8")
    (output/"LICENSE-fonts.txt").write_text(license_text,encoding="utf-8")
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        commit=None
    inputs={input_record_key(path,root):sha256(path) for path in [script,*sorted(data.glob("*.json")),*sorted(data.glob("*.npz"))]}
    delivered=[output/name for film in films for name in [film["file"],film["blend"],film["poster"]]]+[output/"LICENSE-fonts.txt"]
    outputs={path.name:{"bytes":path.stat().st_size,"sha256":sha256(path)} for path in delivered}
    manifest={"schema":"optical-comm-kerr-physics-films/v1","issue":16,"created_utc":datetime.now(timezone.utc).isoformat(),"source_commit":commit,"inputs":inputs,"outputs":outputs,"films":films,"visual_encoding":visual_encoding(frame_count(args)),"rendering":{"engine":"Cycles","backend":"OptiX","samples":args.samples,"adaptive_threshold":args.adaptive_threshold,"denoiser":"OptiX","resolution":[WIDTH,HEIGHT],"persistent_data":True,"caustics_disabled_for_readability":True},"assets":{"external_artwork":[],"procedural_scene":"All scene geometry/materials authored by the renderer. No third-party HDRI, textures, model, music, or footage.","font":{"file":str(args.font),"sha256":sha256(args.font),"use":"Small transparent post-production overlay, not the scene protagonist.","license":"LICENSE-fonts.txt"}},"regeneration":{"working_directory":"repository root","command":shlex.join(["python3","scripts/render_kerr_films.py","--mode","render","--blender",str(blender),"--samples",str(args.samples),"--adaptive-threshold",str(args.adaptive_threshold),"--data-dir",str(args.data_dir),"--output-dir",str(args.output_dir),"--scratch-dir",str(args.scratch_dir)]),"blender_invocations":invocations,"encodings":encodings,"native_scene":"Mesh shape keys, packed phase atlases, material animation, camera animation and field coordinates are baked native Blender data. No Python handler, external scientific array, or texture is required to reload/render a saved .blend."},"host":{"platform":platform.platform(),"python":sys.version},"verification":{"independent_validator":{"requested":0,"attempted":0,"completed":0}}}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Rendered and checked {len(films)} data-driven films: {output}",flush=True)


def run_blender(args: argparse.Namespace) -> None:
    import bpy
    import numpy as np
    from mathutils import Vector
    from bpy_extras.anim_utils import animdata_get_channelbag_for_assigned_slot

    case=args.case[0]
    out=args.output_dir
    scratch=args.scratch_dir/case
    scratch.mkdir(parents=True,exist_ok=True)
    frames_dir=scratch/"frames"
    frames_dir.mkdir(exist_ok=True)
    data_path=args.data_dir/(case+".npz")
    with np.load(data_path,allow_pickle=False) as archive:
        zs=np.asarray(archive["z_km"],dtype=float)
        ts=np.asarray(archive["t_ps"],dtype=float)
        field=np.asarray(archive["A_xy_sqrtW"],dtype=np.complex128)
        valid=np.asarray(archive["phase_valid"],dtype=bool)
    if zs.shape!=(81,) or np.max(abs(zs-np.arange(81)))>1e-10 or field.shape!=(81,len(ts),2) or not np.isfinite(field).all():
        raise ValueError("Unexpected or nonfinite actual field arrays")
    time_indices=np.flatnonzero(abs(ts)<=TIME_WINDOW_PS)
    source_display_ts=ts[time_indices]
    source_display_field=field[:,time_indices,:]
    source_display_valid=valid[:,time_indices,0]
    display_ts=np.linspace(source_display_ts[0],source_display_ts[-1],(len(source_display_ts)-1)*DISPLAY_TIME_UPSAMPLING+1)
    display_field=np.empty((len(zs),len(display_ts),2),dtype=np.complex128)
    display_valid=np.empty((len(zs),len(display_ts)),dtype=bool)
    for row in range(len(zs)):
        for pol in range(2):
            display_field[row,:,pol]=np.interp(display_ts,source_display_ts,source_display_field[row,:,pol].real)+1j*np.interp(display_ts,source_display_ts,source_display_field[row,:,pol].imag)
        display_valid[row]=np.interp(display_ts,source_display_ts,source_display_valid[row].astype(float))>=1.-1e-12
    count=frame_count(args)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for unused in list(bpy.data.materials):
        bpy.data.materials.remove(unused)
    scene=bpy.context.scene
    scene.name=f"Computed field in cutaway fiber — {case}"
    scene["scientific_scope"]=visual_encoding(count)["scientific_scope"]
    scene["visual_encoding"]=json.dumps(visual_encoding(count),ensure_ascii=False)
    scene["field_source_sha256"]=sha256(data_path)
    scene.render.engine="CYCLES"
    prefs=bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type="OPTIX"
    prefs.refresh_devices()
    devices=[]
    for device in prefs.devices:
        device.use=device.type=="OPTIX"
        devices.append({"name":device.name,"type":device.type,"id":device.id,"enabled":device.use})
    if not any(device["enabled"] for device in devices):
        raise RuntimeError("No OptiX GPU detected; CPU fallback is not silently allowed")
    driver_report=json.loads(args.gpu_report) if args.gpu_report else gpu_report()
    environment={"blender_version":bpy.app.version_string,"blender_build_hash":bpy.app.build_hash.decode(),"selected_devices":[device for device in devices if device["enabled"]],"gpu_driver_report":driver_report}
    parameters={"script":sha256(Path(__file__)),"data":sha256(data_path),"samples":args.samples,"threshold":args.adaptive_threshold,"duration":args.duration,"fps":args.fps,"resolution":[WIDTH,HEIGHT]}
    signature=render_signature(parameters,environment)
    signature_path=scratch/"frame_signature.json"
    ledger_path=scratch/"frame_ledger.json"
    if args.resume:
        if not signature_path.is_file():
            raise RuntimeError("Resume refused: no recorded frame-generation signature")
        require_resume_signature(json.loads(signature_path.read_text()),signature,environment)
        try:
            ledger=json.loads(ledger_path.read_text())
        except (OSError,ValueError):
            ledger={"signature":signature,"frames":{}}
        if ledger.get("signature")!=signature:
            ledger={"signature":signature,"frames":{}}
    else:
        ledger={"signature":signature,"frames":{}}
        write_json_atomic(signature_path,{"signature":signature,"case":case,"parameters":parameters,"environment":environment})
        write_json_atomic(ledger_path,ledger)
    scene["renderer_source_sha256"]=parameters["script"]
    scene["render_environment"]=json.dumps(environment,sort_keys=True)
    scene.cycles.device="GPU"
    scene.cycles.samples=args.samples
    scene.cycles.use_adaptive_sampling=True
    scene.cycles.adaptive_threshold=args.adaptive_threshold
    scene.cycles.adaptive_min_samples=24
    scene.cycles.use_denoising=True
    scene.cycles.denoiser="OPTIX"
    scene.cycles.max_bounces=12
    scene.cycles.transmission_bounces=8
    scene.cycles.transparent_max_bounces=12
    scene.cycles.glossy_bounces=4
    scene.cycles.diffuse_bounces=3
    scene.cycles.volume_bounces=0
    scene.cycles.caustics_reflective=False
    scene.cycles.caustics_refractive=False
    scene.cycles.sample_clamp_indirect=5.
    scene.render.use_persistent_data=True
    scene.render.resolution_x,scene.render.resolution_y=WIDTH,HEIGHT
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGB"
    scene.render.image_settings.color_depth="8"
    scene.render.film_transparent=False
    scene.render.fps=args.fps
    scene.frame_start,scene.frame_end=1,count
    scene.view_settings.view_transform="AgX"
    scene.view_settings.look="AgX - Medium High Contrast"
    scene.view_settings.exposure=.05
    scene.world.use_nodes=True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value=(.03,.05,.085,1.)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value=.22

    def material(name,color,metal=0.,rough=.25,transmission=0.,ior=1.45):
        mat=bpy.data.materials.new(name)
        mat.use_nodes=True
        p=mat.node_tree.nodes.get("Principled BSDF")
        p.inputs["Base Color"].default_value=(*color,1.)
        p.inputs["Metallic"].default_value=metal
        p.inputs["Roughness"].default_value=rough
        p.inputs["Transmission Weight"].default_value=transmission
        p.inputs["IOR"].default_value=ior
        return mat

    glass=material("Polished silica cladding | visible rendering only",(.96,.98,1.),rough=.026,transmission=1.,ior=1.444)
    glass["scope"]="Visible studio material. Cycles does not solve the 1550nm Kerr propagation."
    core=material("Clear silica core",(.98,.99,1.),rough=.025,transmission=1.,ior=1.45)
    metal=material("Satin titanium fixture",(.20,.24,.29),metal=.92,rough=.24)
    blackmetal=material("Graphite anodized aluminum",(.019,.027,.038),metal=.8,rough=.26)
    brass=material("Champagne brass trim",(.38,.24,.10),metal=.95,rough=.23)
    polymer=material("Acrylate protective jacket",(.015,.018,.022),rough=.30)
    slate=material("Charcoal laboratory plinth",(.023,.035,.048),rough=.31)
    ground=material("Graphite studio floor",(.008,.014,.025),rough=.35)
    for mat,scale,strength in [(metal,180,.025),(blackmetal,240,.018),(polymer,55,.045),(slate,35,.035),(ground,25,.02)]:
        nodes=mat.node_tree.nodes
        noise=nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value=scale
        noise.inputs["Detail"].default_value=2.
        bump=nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value=strength
        bump.inputs["Distance"].default_value=.02
        mat.node_tree.links.new(noise.outputs["Fac"],bump.inputs["Height"])
        mat.node_tree.links.new(bump.outputs["Normal"],nodes["Principled BSDF"].inputs["Normal"])

    def finish(obj,mat,bevel=0.):
        obj.data.materials.append(mat)
        if bevel:
            modifier=obj.modifiers.new("Manufactured edge radius","BEVEL")
            modifier.width=bevel
            modifier.segments=3
        return obj

    def cube(name,loc,size,mat,bevel=.05):
        bpy.ops.mesh.primitive_cube_add(size=1.,location=loc)
        obj=bpy.context.object
        obj.name=name
        obj.dimensions=size
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        return finish(obj,mat,bevel)

    def cylinder(name,x0,x1,radius,mat,z=2.):
        bpy.ops.mesh.primitive_cylinder_add(vertices=128,radius=radius,depth=x1-x0,location=((x0+x1)/2,0,z),rotation=(0,math.pi/2,0))
        obj=bpy.context.object
        obj.name=name
        for face in obj.data.polygons:
            face.use_smooth=len(face.vertices)==4
        return finish(obj,mat,.004)

    def annular(name,x0,x1,ri,ro,theta0,theta1,mat,segments=160,expand=False):
        full=abs(theta1-theta0-2*math.pi)<1e-8
        number=segments if full else segments+1
        vertices=[]
        for x in [x0,x1]:
            for r in [ro,ri]:
                for j in range(number):
                    theta=theta0+(theta1-theta0)*j/segments
                    vertices.append((x,r*math.cos(theta),2.+r*math.sin(theta)))
        faces=[]
        smooth=[]
        a,b,c,d=0,number,2*number,3*number
        for j in range(segments):
            k=(j+1)%number
            faces.extend([(a+j,c+j,c+k,a+k),(b+k,d+k,d+j,b+j),(a+k,b+k,b+j,a+j),(c+j,d+j,d+k,c+k)])
            smooth.extend([True,True,False,False])
        if not full:
            faces.extend([(a,b,d,c),(a+segments,c+segments,d+segments,b+segments)])
            smooth.extend([False,False])
        mesh=bpy.data.meshes.new(name)
        mesh.from_pydata(vertices,[],[tuple(reversed(face)) for face in faces])
        mesh.update()
        obj=bpy.data.objects.new(name,mesh)
        scene.collection.objects.link(obj)
        for face,sm in zip(mesh.polygons,smooth):
            face.use_smooth=sm
        finish(obj,mat,.002)
        if expand:
            obj.shape_key_add(name="Real transverse ratio")
            key=obj.shape_key_add(name="Enlarged internal field view | not real cross-section")
            enlarged=np.array(vertices)
            for offset in [b,d]:
                enlarged[offset:offset+number,1:]=(enlarged[offset:offset+number,1:]-[0,2.])*1.1/ri+[0,2.]
            for offset in [a,c]:
                enlarged[offset:offset+number,1:]=(enlarged[offset:offset+number,1:]-[0,2.])*1.2/ro+[0,2.]
            key.data.foreach_set("co",enlarged.ravel())
            for frame,value in [(1,0.),(37,0.),(73,1.),(count,1.)]:
                key.value=value
                key.keyframe_insert("value",frame=frame)
        return obj

    core_obj=cylinder("Silica core | real 8.2:125 diameter ratio",-7.8,6.25,CORE_RADIUS,core)
    alpha=core.node_tree.nodes["Principled BSDF"].inputs["Alpha"]
    for frame,value in [(1,1.),(37,1.),(73,0.),(count,0.)]:
        alpha.default_value=value
        alpha.keyframe_insert("default_value",frame=frame)
    annular("Intact cladding, jacket side",-7.6,-5.6,CORE_RADIUS,.8,0,2*math.pi,glass)
    annular("Cutaway cladding | enlarged internal view later",-5.6,5.6,CORE_RADIUS,.8,math.radians(235),math.radians(400),glass,expand=True)
    annular("Polished end",5.6,6.25,CORE_RADIUS,.8,0,2*math.pi,glass)
    cylinder("Acrylate jacket",-10.,-6.7,1.05,polymer)
    annular("Titanium ferrule",-7.1,-5.65,.801,1.23,0,2*math.pi,metal)
    annular("Ferrule brass nose",-5.65,-5.33,.801,1.13,0,2*math.pi,brass)
    for x in [-6.9,-6.74,-6.58,-6.42,-6.26]:
        annular("Fine ferrule machining groove",x,x+.035,1.19,1.241,0,2*math.pi,blackmetal,segments=64)
    cube("Monolithic charcoal plinth",(-.6,.6,.12),(18.,7.,.38),slate,.16)
    cube("Rear holder foot",(-6.1,.55,.43),(2.5,3.2,.42),blackmetal,.12)
    cube("Rear holder upright",(-6.1,.7,1.08),(1.25,1.35,1.15),metal,.12)
    cube("Far clamp foot",(5.8,1.45,.43),(2.9,2.8,.45),blackmetal,.12)
    cube("Far clamp riser",(5.8,1.8,1.45),(.7,.58,1.75),metal,.08)
    cube("Far clamp jaw",(5.8,1.05,2.18),(.85,1.4,.28),metal,.06)
    for x,y in [(-6.9,-.53),(-5.3,-.53),(-6.9,1.62),(-5.3,1.62),(5.05,1.1),(6.55,1.1)]:
        bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=.13,depth=.075,location=(x,y,.70))
        bpy.context.object.name="Countersunk hardware detail"
        finish(bpy.context.object,metal,.018)
        cube("Machined socket",(x,y,.741),(.045,.15,.012),blackmetal,.003)
    cube("Infinite graphite floor",(0,0,-.2),(200,200,.1),ground,0.)

    def area(name,loc,target,power,color,size,size_y):
        data=bpy.data.lights.new(name,"AREA")
        data.energy=power
        data.color=color
        data.shape="RECTANGLE"
        data.size,data.size_y=size,size_y
        obj=bpy.data.objects.new(name,data)
        scene.collection.objects.link(obj)
        obj.location=loc
        obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()

    area("Long overhead studio softbox",(-2,-3,10),(0,0,1.8),1050,(.90,.95,1.),11.,2.)
    area("Cool edge rim",(0,6,6),(0,0,2),1600,(.33,.65,1.),10.,1.8)
    area("Warm polished end reveal",(8,-4,7),(4,0,2),600,(1.,.70,.40),5.,4.)
    area("Low cool front fill",(-2,-7,3),(-1,0,2),130,(.5,.7,1.),8.,3.)
    area("Narrow white glass reflection",(3,2,9),(2,0,2),240,(1.,1.,1.),8.,.45)

    # A packed atlas carries actual complex-envelope phase through the shader.
    phase=np.angle(display_field[:,:,0])
    pixels=np.ones((*phase.shape,4),dtype=np.float32)
    anchors=np.array([-np.pi,-np.pi/2,0.,np.pi/2,np.pi])
    palette=np.array([[.60,.010,.42],[.015,.025,.75],[.015,.48,.62],[1.,.12,.006],[.60,.010,.42]])
    for channel in range(3):
        pixels[:,:,channel]=np.interp(phase,anchors,palette[:,channel])
    pixels[:,:,3]=display_valid.astype(np.float32)
    atlas=bpy.data.images.new(f"Actual phase atlas | {case} | packed",width=len(display_ts),height=len(zs),alpha=True,float_buffer=True)
    atlas.colorspace_settings.name="Non-Color"
    atlas.pixels.foreach_set(pixels.ravel())
    atlas.pack()

    def field_material(name,envelope=False):
        mat=material(name,(.07,.72,.68),rough=.28 if envelope else .21,metal=.0,transmission=.22 if envelope else .0,ior=1.10)
        nodes=mat.node_tree.nodes
        links=mat.node_tree.links
        p=nodes["Principled BSDF"]
        uv=nodes.new("ShaderNodeTexCoord")
        separate=nodes.new("ShaderNodeSeparateXYZ")
        combine=nodes.new("ShaderNodeCombineXYZ")
        progress=nodes.new("ShaderNodeValue")
        progress.name="Actual propagation distance, packed phase row"
        texture=nodes.new("ShaderNodeTexImage")
        texture.image=atlas
        texture.interpolation="Linear"
        texture.extension="EXTEND"
        links.new(uv.outputs["UV"],separate.inputs["Vector"])
        links.new(separate.outputs["X"],combine.inputs["X"])
        links.new(progress.outputs[0],combine.inputs["Y"])
        links.new(combine.outputs[0],texture.inputs["Vector"])
        links.new(texture.outputs["Color"],p.inputs["Base Color"])
        links.new(texture.outputs["Color"],p.inputs["Emission Color"])
        p.inputs["Emission Strength"].default_value=.045 if envelope else .35
        mask=nodes.new("ShaderNodeMath")
        mask.operation="MULTIPLY"
        mask.inputs[1].default_value=.115 if envelope else 1.
        links.new(texture.outputs["Alpha"],mask.inputs[0])
        links.new(mask.outputs[0],p.inputs["Alpha"])
        for frame,z in [(1,0.),(61,0.),(count-36,80.),(count,80.)]:
            progress.outputs[0].default_value=(z+.5)/81
            progress.outputs[0].keyframe_insert("default_value",frame=frame)
        return mat

    wave_mat=field_material("False-color computed electric field | absolute scale")
    envelope_mat=field_material("Translucent absolute amplitude envelope glyph",envelope=True)
    carrier=np.exp(-2j*np.pi*display_ts/CARRIER_PERIOD_PS)
    real_field=np.real(display_field*carrier[None,:,None])
    x=display_ts*AXIAL_UNITS_PER_PS
    rings=12
    angles=np.linspace(0,2*np.pi,rings,endpoint=False)

    def coordinates(row,envelope=False):
        if envelope:
            radius=np.sqrt(np.sum(abs(display_field[row])**2,axis=1))*ENVELOPE_SCALE
            coords=np.empty((len(x),rings,3))
            coords[:,:,0]=x[:,None]
            coords[:,:,1]=radius[:,None]*np.cos(angles)
            coords[:,:,2]=radius[:,None]*np.sin(angles)
        else:
            center=np.column_stack([x,real_field[row,:,1]*AMPLITUDE_SCALE,real_field[row,:,0]*AMPLITUDE_SCALE])
            tangent=np.gradient(center,axis=0)
            tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
            binormal=np.cross(tangent,np.array([0,0,1.]))
            binormal/=np.linalg.norm(binormal,axis=1,keepdims=True)
            normal=np.cross(binormal,tangent)
            coords=center[:,None,:]+.027*(normal[:,None,:]*np.cos(angles)[None,:,None]+binormal[:,None,:]*np.sin(angles)[None,:,None])
        return coords.reshape(-1,3).astype(np.float32)

    def animate_field(name,mat,envelope=False):
        faces=[]
        for j in range(len(x)-1):
            for k in range(rings):
                next_k=(k+1)%rings
                faces.append((j*rings+k,(j+1)*rings+k,(j+1)*rings+next_k,j*rings+next_k))
        mesh=bpy.data.meshes.new(name)
        mesh.from_pydata(coordinates(0,envelope),[],[tuple(reversed(face)) for face in faces])
        mesh.update()
        obj=bpy.data.objects.new(name,mesh)
        scene.collection.objects.link(obj)
        obj.location.z=2.
        obj.data.materials.append(mat)
        for face in mesh.polygons:
            face.use_smooth=True
        uv=mesh.uv_layers.new(name="Retarded time along the pulse")
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                vertex=mesh.loops[loop_index].vertex_index
                uv.data[loop_index].uv=((vertex//rings+.5)/len(x),0.)
        obj.shape_key_add(name="Measured z=0 km")
        start,end=61,count-36
        for row,z in enumerate(zs[1:],start=1):
            key=obj.shape_key_add(name=f"Measured z={z:g} km")
            key.data.foreach_set("co",coordinates(row,envelope).ravel())
            f=start+(end-start)*z/80
            prev=start+(end-start)*(z-1)/80
            nxt=start+(end-start)*(z+1)/80
            for frame,value in [(1,0.),(prev,0.),(f,1.),(nxt,0.)] if row<80 else [(1,0.),(prev,0.),(f,1.),(count,1.)]:
                key.value=value
                key.keyframe_insert("value",frame=frame)
        for frame,scale in [(1,INTRO_MAGNIFICATION),(37,INTRO_MAGNIFICATION),(73,1.),(count,1.)]:
            obj.scale=(1.,scale,scale)
            obj.keyframe_insert("scale",frame=frame)
        obj["field_source"]=str(data_path.name)
        obj["field_source_sha256"]=sha256(data_path)
        obj["encoding"]="Absolute sqrt(W), never per-frame normalized. Enlarged vector/amplitude glyph, not photon trajectories or a physical mode radius."
        return obj

    animate_field("Computed XY electric-field curve | same phase preserves polarization",wave_mat)
    animate_field("Computed absolute amplitude envelope | not physical mode width",envelope_mat,envelope=True)
    focus=bpy.data.objects.new("Macro focus on computed pulse",None)
    scene.collection.objects.link(focus)
    focus.location=(0,0,2.)
    cam_data=bpy.data.cameras.new("Cinematic optical-fiber macro lens")
    cam=bpy.data.objects.new("Animated macro camera",cam_data)
    scene.collection.objects.link(cam)
    scene.camera=cam
    cam_data.sensor_width=36
    cam_data.dof.use_dof=True
    cam_data.dof.focus_object=focus
    cam_data.dof.aperture_fstop=10.
    cam_data.dof.aperture_blades=9
    cam_data.clip_start,cam_data.clip_end=.05,250.
    for frame,loc,target,lens in [(1,(9.4,-18.2,10.3),(-.8,0,1.95),55.),(37,(9.05,-18.1,10.4),(-.65,0,1.95),55.),(85,(3.9,-12.2,10.1),(0,0,2.),61.),(count,(3.0,-12.0,9.7),(.1,0,2.),61.)]:
        cam.location=loc
        cam.rotation_euler=(Vector(target)-cam.location).to_track_quat("-Z","Y").to_euler()
        cam_data.lens=lens
        focus.location=target
        cam.keyframe_insert("location",frame=frame)
        cam.keyframe_insert("rotation_euler",frame=frame)
        cam_data.keyframe_insert("lens",frame=frame)
        focus.keyframe_insert("location",frame=frame)

    # Linear field/color interpolation is stated in the manifest. Camera and
    # magnification transitions retain Blender's smooth visual interpolation.
    for datablock in [obj.data.shape_keys for obj in scene.objects if obj.type=="MESH" and obj.data.shape_keys]:
        if datablock.animation_data and datablock.animation_data.action:
            channelbag=animdata_get_channelbag_for_assigned_slot(datablock.animation_data)
            for fcurve in channelbag.fcurves:
                for point in fcurve.keyframe_points:
                    point.interpolation="LINEAR"
    for mat in [wave_mat,envelope_mat]:
        if mat.node_tree.animation_data and mat.node_tree.animation_data.action:
            channelbag=animdata_get_channelbag_for_assigned_slot(mat.node_tree.animation_data)
            for fcurve in channelbag.fcurves:
                for point in fcurve.keyframe_points:
                    point.interpolation="LINEAR"
    for unused in list(bpy.data.materials):
        if unused.users==0:
            bpy.data.materials.remove(unused)
    scene.frame_set(1)
    scene.render.filepath=f"//{case}/frames/frame_"
    bpy.ops.file.pack_all()
    out.mkdir(parents=True,exist_ok=True)
    blend=out/(FILMS[case]["stem"]+".blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend),compress=True)
    stills=[1,round(count*.55),count]
    motion=list(range(97,145)) if args.mode=="proof" and case==(args.proof_motion_case or selected_proof_case(args.case)) else []
    render_frames=list(range(1,count+1)) if args.mode=="render" else sorted(set(stills+motion)) if args.mode in ["stills","proof"] else []
    timings=[]
    for frame in render_frames:
        target=frames_dir/f"frame_{frame:04d}.png"
        if args.resume and cached_frame_valid(target,ledger["frames"].get(str(frame))):
            timings.append({"frame":frame,"reused":True,"PNG":ledger["frames"][str(frame)]})
            continue
        scene.frame_set(frame)
        scene.render.filepath=str(target)
        begin=time.perf_counter()
        bpy.ops.render.render(write_still=True)
        elapsed=time.perf_counter()-begin
        record=inspect_png(target)
        ledger["frames"][str(frame)]=record
        write_json_atomic(ledger_path,ledger)
        timings.append({"frame":frame,"render_seconds":elapsed,"distance_km":distance_at_frame(frame,count),"PNG":record})
        write_json_atomic(scratch/"render_progress.json",timings)
        print("RENDER_FRAME="+json.dumps(timings[-1]),flush=True)
        if frame in stills and args.mode in ["stills","proof"]:
            shutil.copy2(target,out/f"{case}_proof_{frame:04d}.png")
    energy=np.sum(abs(field)**2,axis=(1,2))
    cropped_energy=np.sum(abs(source_display_field)**2,axis=(1,2))
    omission=np.maximum(0,1-cropped_energy/energy)
    actual_times=[item["render_seconds"] for item in timings if "render_seconds" in item]
    metadata={"schema":"optical-comm-kerr-film-scene/v1","case":case,"blender":bpy.app.version_string,"environment":environment,"renderer_source_sha256":parameters["script"],"engine":"CYCLES","compute_device_type":"OPTIX","device":"GPU","devices":devices,"signature":signature,"source":{"file":data_path.name,"sha256":sha256(data_path)},"frame_count":count,"resolution":[WIDTH,HEIGHT],"visual_encoding":visual_encoding(count),"display_crop":{"requested_window_ps":[-TIME_WINDOW_PS,TIME_WINDOW_PS],"actual_sample_bounds_ps":[float(display_ts[0]),float(display_ts[-1])],"samples":len(display_ts),"max_fractional_energy_omission_over_distance":float(np.max(omission)),"fractional_energy_omission_by_z":omission.tolist()},"native_scene":{"file":blend.name,"packed_phase_atlas":atlas.name,"shape_key_distances_km":zs.tolist(),"external_render_assets":[],"python_handlers":False},"stills":stills,"motion_proof_frames":motion,"render_summary":{"new_frames":len(actual_times),"reused_frames":sum(item.get("reused",False) for item in timings),"mean_seconds":float(np.mean(actual_times)) if actual_times else None,"seconds":sum(actual_times)},"render_timings":timings}
    (scratch/"scene_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    parsed=arguments(argv)
    run_blender(parsed) if parsed.inside_blender else run_host(parsed)
