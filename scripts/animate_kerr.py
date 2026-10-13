#!/usr/bin/env python3
"""Build and render a self-contained Blender explanation of scalar Kerr/SPM.

Run with ordinary Python; this launcher invokes Blender, then FFmpeg.  All
geometry, text and animation live in the saved .blend, without Python handlers
or external image assets.  Blender can also execute this file directly with
``-- --inside-blender``.  This is an analytic teaching example, not link data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


WIDTH, HEIGHT, FPS, FRAMES = 1280, 720, 24, 720
PEAK_PHASE = 3.0
POSTER_FRAME = 94
CHAPTERS = [
    (1, 144, "屈折率", "強い光ほど、屈折率が変わる"),
    (145, 288, "位相", "強度の山が、位相の山になる"),
    (289, 432, "波形", "包絡線は保たれ、波の位置が変わる"),
    (433, 576, "周波数", "位相の傾きが、周波数をずらす"),
    (577, 720, "まとめ", "Kerr効果は、強度がつくる位相の変化"),
]
COLORS = {
    "bg": "091322", "panel": "112035", "panel2": "152941",
    "grid": "29415A", "white": "F3F6FB", "muted": "A9BED2",
    "dim": "63809E", "cyan": "51DCE0", "gold": "FFC66E",
    "red": "FF8E97", "blue": "7DB7FF", "purple": "BBA7FF",
}
SOURCES = [
    {
        "title": "Stolen & Lin (1978), Self-phase-modulation in silica optical fibers",
        "url": "https://doi.org/10.1103/PhysRevA.17.1448",
        "use": "Experimental silica SPM and spectral broadening; positive silica Kerr response.",
    },
    {
        "title": "Arroyo Meza et al. (2013), Physical Review E 88, 053202",
        "url": "https://doi.org/10.1103/PhysRevE.88.053202",
        "use": "Standard intensity-dependent refractive-index notation n = n0 + n2 I.",
    },
    {
        "title": "Hamaide & Emplit (1989), Spectrotemporal measurement of picosecond pulses",
        "url": "https://doi.org/10.1364/OL.14.000689",
        "use": "Experimental time-resolved frequency modulation. Not a pure-SPM calibration: their propagation also includes dispersion/Raman.",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", help="Blender executable (Linux or a Windows .exe under WSL)")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--font", help="Japanese TrueType font; packed into the saved scene")
    parser.add_argument("--latin-font", help="Math/Latin TrueType font with Greek and subscript glyphs; also packed")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/animations/kerr"))
    parser.add_argument("--scratch-dir", type=Path, default=Path(".work/kerr-render"))
    parser.add_argument("--preview", action="store_true", help="Six chapter stills and a 2-second motion preview; output only to scratch")
    parser.add_argument("--build-only", action="store_true", help="Save scene without rendering/encoding")
    parser.add_argument("--inside-blender", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def find_font(specified: str | None) -> Path:
    candidates = [Path(specified)] if specified else [
        Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"),
        Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("A Japanese font is required; pass --font /path/to/font.ttf")


def native_path(path: Path, windows_blender: bool) -> str:
    if windows_blender and os.name != "nt":
        return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()
    return str(path.resolve())


def find_latin_font(specified: str | None) -> Path:
    candidates = [Path(specified)] if specified else [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("A Latin/math font is required; pass --latin-font /path/to/font.ttf")


def run_host(args: argparse.Namespace) -> None:
    blender = args.blender or shutil.which("blender")
    windows_default = Path("/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe")
    if not blender and windows_default.is_file():
        blender = str(windows_default)
    if not blender:
        raise FileNotFoundError("Blender is required; pass --blender /path/to/blender")
    if not args.build_only and (not args.ffmpeg or not args.ffprobe):
        raise FileNotFoundError("FFmpeg and FFprobe are required to encode and verify the video")
    font = find_font(args.font)
    latin_font = find_latin_font(args.latin_font)
    script = Path(__file__).resolve()
    root = script.parents[1]
    scratch = args.scratch_dir.resolve()
    output = (scratch / "preview-assets") if args.preview else args.output_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    support = root / "reports/animations/kerr"
    for name in ["README.md", "LICENSE-fonts.txt"]:
        source, target = support / name, output / name
        if not source.is_file():
            raise FileNotFoundError(f"Required animation documentation: {source}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    windows_blender = str(blender).lower().endswith(".exe")
    invocation = [
        str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", native_path(script, windows_blender), "--", "--inside-blender",
        "--font", native_path(font, windows_blender),
        "--latin-font", native_path(latin_font, windows_blender),
        "--output-dir", native_path(output, windows_blender),
        "--scratch-dir", native_path(scratch, windows_blender),
    ]
    if args.preview:
        invocation.append("--preview")
    if args.build_only:
        invocation.append("--build-only")
    print("Building Blender scene", flush=True)
    subprocess.run(invocation, check=True)
    if args.build_only:
        return
    first = 193 if args.preview else 1
    count = 48 if args.preview else FRAMES
    video = scratch / "preview.mp4" if args.preview else output / "kerr_effect.mp4"
    encode = [
        args.ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(FPS), "-start_number", str(first),
        "-i", str(scratch / "frames" / "frame_%04d.png"), "-frames:v", str(count),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", "-metadata", "title=光ファイバのKerr効果",
        str(video),
    ]
    subprocess.run(encode, check=True)
    probe = json.loads(subprocess.check_output([
        args.ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(video),
    ], text=True))
    stream = probe["streams"][0]
    assert len(probe["streams"]) == 1 and stream["codec_type"] == "video"
    assert (stream["width"], stream["height"]) == (WIDTH, HEIGHT)
    assert stream["r_frame_rate"] == "24/1" and int(stream["nb_read_frames"]) == count
    assert stream["pix_fmt"] == "yuv420p" and stream["codec_name"] == "h264"
    assert abs(float(probe["format"]["duration"]) - count / FPS) < 0.001
    if args.preview:
        (scratch / "preview_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        print(f"Preview: {video}", flush=True)
        return
    shutil.copy2(scratch / "frames" / f"frame_{POSTER_FRAME:04d}.png", output / "poster.png")
    metadata = json.loads((scratch / "scene_metadata.json").read_text(encoding="utf-8"))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    reproduction = [
        "python3", "scripts/animate_kerr.py", "--blender", str(blender),
        "--font", str(font), "--output-dir", str(args.output_dir),
        "--latin-font", str(latin_font),
        "--scratch-dir", str(args.scratch_dir), "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe,
    ]
    import shlex
    manifest = {
        "schema": "optical-comm-kerr-animation/v1", "issue": 12,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": commit,
        "scope": "Analytic scalar, lossless, dispersion-free teaching illustration. No training, measurement, link propagation or BER result is used.",
        "scene": metadata,
        "sources": SOURCES,
        "inputs": {
            str(path.relative_to(root)): sha256(path) for path in [
                script, root / "docs/requirements.md", root / "docs/experiment-plan.md", root / "docs/channel-methods.md",
            ]
        },
        "fonts": [
            {"filename": f.name, "sha256": sha256(f), "packed_in_blend": True}
            for f in [font, latin_font]
        ],
        "font_license": {"file": "LICENSE-fonts.txt", "sha256": sha256(output / "LICENSE-fonts.txt")},
        "host": {"platform": platform.platform(), "python": sys.version, "ffmpeg": subprocess.check_output([args.ffmpeg, "-version"], text=True).splitlines()[0]},
        "regeneration": {
            "working_directory": "repository root", "command": shlex.join(reproduction),
            "blender_invocation": invocation, "ffmpeg_invocation": encode,
            "preview": "Append --preview; renders six chapter stills and a 2-second MP4 in .work only.",
            "editable_scene": "All text, geometry, drivers and keyframes are native Blender data; packed font; no render handlers or external assets.",
            "visual_reproducibility": "Exact parameters and commands are recorded. GPU/Blender/FFmpeg differences can change pixels/encoded bytes; hashes identify these delivered outputs, not a cross-platform bitwise guarantee.",
        },
        "verification": {"ffprobe": probe, "ordinary_checks": metadata["formula_checks"], "independent_validator": {"requested": 0, "attempted": 0, "completed": 0}},
        "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in ["kerr_effect.mp4", "kerr_effect.blend", "poster.png", "README.md", "LICENSE-fonts.txt"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {FRAMES} frames / 30 seconds: {video}", flush=True)


def run_blender(args: argparse.Namespace) -> None:
    import bpy
    import numpy as np
    from mathutils import Vector

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.name = "Kerr — scalar teaching illustration"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, FRAMES
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 20
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    if hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 32
    bpy.context.preferences.filepaths.save_version = 0
    font = bpy.data.fonts.load(str(args.font))
    font.name = "IPAexGothic"
    font.pack()
    latin_font = bpy.data.fonts.load(str(args.latin_font))
    latin_font.name = "DejaVuSans"
    latin_font.pack()
    materials = {}
    chapter_objects: list[list] = [[] for _ in CHAPTERS]
    current_chapter: int | None = None

    def rgb(name: str) -> tuple:
        color = COLORS.get(name, name)
        values = [int(color[j:j + 2], 16) / 255 for j in (0, 2, 4)]
        return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values)

    def material(name: str, lit: bool = False):
        key = (name, lit)
        if key not in materials:
            mat = bpy.data.materials.new(f"{name} {'surface' if lit else 'graphic'}")
            mat.diffuse_color = (*rgb(name), 1)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            output_node = nodes.new("ShaderNodeOutputMaterial")
            shader = nodes.new("ShaderNodeBsdfPrincipled" if lit else "ShaderNodeEmission")
            if lit:
                shader.inputs["Base Color"].default_value = (*rgb(name), 1)
                shader.inputs["Metallic"].default_value = 0.35
                shader.inputs["Roughness"].default_value = 0.32
            else:
                shader.inputs["Color"].default_value = (*rgb(name), 1)
                shader.inputs["Strength"].default_value = 1
            mat.node_tree.links.new(shader.outputs[0], output_node.inputs["Surface"])
            materials[key] = mat
        return materials[key]

    def register(obj, name: str, color: str | None = None, lit: bool = False):
        obj.name = name
        if color is not None:
            obj.data.materials.append(material(color, lit))
        if current_chapter is not None:
            chapter_objects[current_chapter].append(obj)
        return obj

    def xy(x: float, y: float, z: float = 10) -> tuple:
        return ((x - WIDTH / 2) / 100, (HEIGHT / 2 - y) / 100, z)

    def text(body: str, x: float, y: float, size: float = 24, color: str = "white", align: str = "LEFT", name: str | None = None):
        curve = bpy.data.curves.new(name or body, "FONT")
        curve.body, curve.font, curve.size = body, font, size / 100
        # IPAex has Japanese glyphs but no Unicode subscripts. Blender permits
        # a second native font on selected characters through font_bold. The
        # linked face is regular DejaVu Sans: the flag chooses that font only.
        curve.font_bold = latin_font
        for i, char in enumerate(body):
            if ord(char) < 0x3000:
                curve.body_format[i].use_bold = True
        curve.align_x = align
        curve.space_character = 1.02
        obj = bpy.data.objects.new(name or body, curve)
        scene.collection.objects.link(obj)
        obj.location = xy(x, y, 13)
        return register(obj, name or body, color)

    def mesh(name: str, vertices: list, faces: list, color: str, lit: bool = False):
        data = bpy.data.meshes.new(name)
        data.from_pydata(vertices, [], faces)
        data.update()
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        return register(obj, name, color, lit)

    def line(name: str, points: list, color: str, width: float = 2, z: float = 11):
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth, curve.bevel_resolution = width / 200, 2
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for p, (x, y) in zip(spline.points, points):
            p.co = (*xy(x, y, z), 1)
        obj = bpy.data.objects.new(name, curve)
        scene.collection.objects.link(obj)
        return register(obj, name, color)

    def rectangle(name: str, x: float, y: float, w: float, h: float, color: str = "panel", radius: float = 14, z: float = 8):
        points = []
        for cx, cy, start in [(x + w - radius, y + radius, -90), (x + w - radius, y + h - radius, 0), (x + radius, y + h - radius, 90), (x + radius, y + radius, 180)]:
            for a in np.linspace(start, start + 90, 9):
                points.append(xy(cx + radius * math.cos(math.radians(a)), cy + radius * math.sin(math.radians(a)), z))
        return mesh(name, points, [tuple(range(len(points)))], color)

    def circle(name: str, x: float, y: float, r: float, color: str, width: float = 2):
        return line(name, [(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 97)], color, width)

    def dot(name: str, x: float, y: float, r: float, color: str, z: float = 12):
        return mesh(name, [xy(x, y, z)] + [xy(x + r * math.cos(a), y + r * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 33)], [(0, j, j + 1) for j in range(1, 33)], color)

    def arrow(name: str, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 3, head: float = 9):
        line(name + " shaft", [(x1, y1), (x2, y2)], color, width)
        angle = math.atan2(y2 - y1, x2 - x1)
        p = [(x2, y2), (x2 - head * math.cos(angle - 0.5), y2 - head * math.sin(angle - 0.5)), (x2 - head * math.cos(angle + 0.5), y2 - head * math.sin(angle + 0.5))]
        return mesh(name + " tip", [xy(*v, 11) for v in p], [(0, 1, 2)], color)

    def empty(name: str):
        obj = bpy.data.objects.new(name, None)
        scene.collection.objects.link(obj)
        return register(obj, name)

    def linear_keys(obj, path: str, pairs: list, index: int = -1):
        for frame, value in pairs:
            if index >= 0:
                getattr(obj, path)[index] = value
            else:
                setattr(obj, path, value)
            obj.keyframe_insert(data_path=path, index=index, frame=frame)
        # Blender 4.4+ uses layered/slotted actions; fcurve access is by channel bag.
        action = obj.animation_data.action
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        for key in fc.keyframe_points:
                            key.interpolation = "LINEAR"

    def driver(data, path: str, index: int, controller, prop: str, expression: str):
        drv = data.driver_add(path, index).driver
        drv.type = "SCRIPTED"
        variable = drv.variables.new()
        variable.name, variable.type = "p", "SINGLE_PROP"
        variable.targets[0].id = controller
        variable.targets[0].data_path = f'["{prop}"]'
        drv.expression = expression

    def phase_controller(name: str, start: int, reach: int):
        obj = empty(name)
        obj["phase_rad"] = 0.0
        for frame, value in [(start, 0.0), (reach, PEAK_PHASE)]:
            obj["phase_rad"] = value
            obj.keyframe_insert(data_path='["phase_rad"]', frame=frame)
        action = obj.animation_data.action
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        for key in fc.keyframe_points:
                            key.interpolation = "LINEAR"
        return obj

    def axes(x: float, top: float, w: float, h: float, xmin: float, xmax: float, ymin: float, ymax: float, xticks: list, yticks: list, xlabel: str, color: str = "grid"):
        for value in xticks:
            px = x + (value - xmin) / (xmax - xmin) * w
            line("vertical grid", [(px, top), (px, top + h)], color, 0.7)
            text(f"{value:g}", px, top + h + 22, 19, "muted", "CENTER")
        for value in yticks:
            py = top + h - (value - ymin) / (ymax - ymin) * h
            line("horizontal grid", [(x, py), (x + w, py)], color, 0.7)
            text(f"{value:g}", x - 12, py + 5, 19, "muted", "RIGHT")
        line("plot axes", [(x, top), (x, top + h), (x + w, top + h)], "dim", 1.2)
        if xlabel:
            text(xlabel, x + w / 2, top + h + 54, 21, "muted", "CENTER")

    def plot_points(x: float, top: float, w: float, h: float, values, xmin=-3, xmax=3, ymin=0, ymax=1):
        return [(x + (t - xmin) / (xmax - xmin) * w, top + h - (v - ymin) / (ymax - ymin) * h) for t, v in values]

    def chapter_title(index: int):
        text(f"{index + 1:02d} / {CHAPTERS[index][2]}", 64, 91, 16, "cyan")
        text(CHAPTERS[index][3], 64, 139, 40)
        text(f"{index + 1:02d}", 1216, 131, 40, "dim", "RIGHT")
        text(CHAPTERS[index][2], 84 + index * 238, 693, 17, "white")
        dot("active chapter", 70 + index * 238, 687, 4, "cyan")

    def caption(first: str, second: str):
        rectangle("subtitle card", 64, 577, 1152, 83, "panel2", 12)
        text(first, 87, 611, 26)
        text(second, 87, 643, 21, "muted")

    def dashed_curve(name: str, points: list, color: str, width: float = 1.5):
        for i in range(0, len(points) - 2, 8):
            line(name, points[i:i + 4], color, width)

    def fiber(center_x: float, center_y: float):
        assembly = empty("3D cutaway fiber — exaggerated dimensions")
        assembly.location = xy(center_x, center_y, 9)
        assembly.rotation_euler = (math.radians(18), math.radians(-18), math.radians(-6))
        members = []
        # The open cladding exposes the core; it is a schematic cutaway, not a
        # ray-tracing claim about modal shape or group velocity.
        angles = np.linspace(math.radians(155), math.radians(385), 81)
        vertices = [(x, 0.61 * math.cos(a), 0.61 * math.sin(a)) for x in (-2.55, 2.55) for a in angles]
        shell = mesh("cutaway cladding", vertices, [(i, i + 1, i + 82, i + 81) for i in range(80)], "436D97", True)
        members.append(shell)
        for x in (-2.55, 2.55):
            bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=10, location=(0, 0, 0), major_radius=0.61, minor_radius=0.035)
            rim = register(bpy.context.object, "cladding end rim", "A9BED2", True)
            rim.location.x = x
            rim.rotation_euler.y = math.pi / 2
            members.append(rim)
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.14, depth=5.6)
        core = register(bpy.context.object, "core", "267A94", True)
        core.rotation_euler.y = math.pi / 2
        members.append(core)
        for a in (angles[0], angles[-1]):
            curve = bpy.data.curves.new("cladding edge", "CURVE")
            curve.dimensions, curve.bevel_depth = "3D", 0.017
            spline = curve.splines.new("POLY")
            spline.points.add(1)
            for p, x in zip(spline.points, (-2.55, 2.55)):
                p.co = (x, 0.61 * math.cos(a), 0.61 * math.sin(a), 1)
            obj = bpy.data.objects.new("cladding edge", curve)
            scene.collection.objects.link(obj)
            members.append(register(obj, "cladding edge", "cyan"))
        for obj in members:
            obj.parent = assembly
            if obj.type == "MESH":
                for polygon in obj.data.polygons:
                    polygon.use_smooth = True
        pulse_vertices = []
        axial = np.linspace(-0.72, 0.72, 49)
        for x in axial:
            radius = 0.03 + 0.23 * math.exp(-((x / 0.34) ** 2) / 2)
            pulse_vertices.extend((x, radius * math.cos(a), radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 33)[:-1])
        pulse_faces = [(32 * j + k, 32 * j + (k + 1) % 32, 32 * (j + 1) + (k + 1) % 32, 32 * (j + 1) + k) for j in range(48) for k in range(32)]
        pulse = mesh("moving pulse — intensity shown schematically", pulse_vertices, pulse_faces, "gold", True)
        pulse.parent = assembly
        for polygon in pulse.data.polygons:
            polygon.use_smooth = True
        linear_keys(pulse, "location", [(1, -2.45), (144, 2.45)], 0)
        return assembly

    # Fixed camera: the axes and type are exact camera-plane graphics, with
    # native shaded 3D fiber geometry behind them.
    world = bpy.data.worlds.new("dark studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*rgb("bg"), 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3
    scene.world = world
    bpy.ops.object.camera_add(location=(0, 0, 40))
    camera = bpy.context.object
    camera.name = "fixed orthographic teaching camera"
    camera.data.type, camera.data.ortho_scale = "ORTHO", 12.8
    scene.camera = camera
    for location, energy, size in [((-4, 3, 18), 650, 8), ((5, 1, 14), 350, 6)]:
        bpy.ops.object.light_add(type="AREA", location=location)
        lamp = bpy.context.object
        lamp.name = "fiber studio light"
        lamp.data.energy, lamp.data.shape, lamp.data.size = energy, "DISK", size
        lamp.rotation_euler = (Vector((0, 0, 7)) - lamp.location).to_track_quat("-Z", "Y").to_euler()
    rectangle("background", 0, 0, WIDTH, HEIGHT, "bg", 0, -3)
    dot("brand dot", 69, 36, 5, "cyan")
    text("OPTICAL COMM LAB", 84, 42, 15, "muted")
    text("光ファイバのKerr効果  /  30秒で見るしくみ", 1216, 42, 17, "muted", "RIGHT")
    line("top divider", [(64, 58), (1216, 58)], "grid", 1)
    for i, (_, _, label, _) in enumerate(CHAPTERS):
        text(label, 84 + i * 238, 693, 17, "dim")
        dot("chapter dot", 70 + i * 238, 687, 3, "dim")
    line("timeline base", [(64, 711), (1216, 711)], "grid", 2)
    progress = line("timeline progress", [(64, 711), (1216, 711)], "cyan", 2.5)
    # Move origin to the left edge, preserving the geometry, so scale animates
    # a left-to-right progress bar rather than a center expansion.
    for point in progress.data.splines[0].points:
        point.co.x -= (64 - WIDTH / 2) / 100
    progress.location.x = (64 - WIDTH / 2) / 100
    linear_keys(progress, "scale", [(1, 0.001), (FRAMES, 1)], 0)

    tau = np.linspace(-3, 3, 241)
    power = np.exp(-tau**2)
    envelope = np.sqrt(power)

    current_chapter = 0
    chapter_title(0)
    rectangle("fiber card", 64, 168, 610, 386)
    text("光パルスが通る光ファイバ", 88, 208, 23)
    fiber(356, 348)
    text("コア", 137, 438, 19, "cyan")
    line("core label leader", [(150, 416), (154, 327)], "cyan", 1)
    text("強度の大きい中心を明るく表示", 338, 470, 18, "gold", "CENTER")
    text("断面・寸法・進行は誇張した模式表現", 88, 529, 16, "muted")
    rectangle("index formula card", 706, 168, 510, 386)
    text("光強度 → 屈折率", 736, 218, 24, "gold")
    text("n = n₀ + n₂ I", 736, 294, 62)
    text("I : 光強度 [W/m²]", 738, 352, 23, "muted")
    text("n₂ : 非線形屈折率係数 [m²/W]", 738, 394, 21, "muted")
    rectangle("positive n2 badge", 736, 421, 449, 59, "panel2", 8, 10)
    text("石英では n₂ > 0", 758, 461, 30, "gold")
    text("同じモード形状では、強度と電力は比例", 736, 523, 19, "muted")
    caption("強い光は、ファイバ自身の屈折率をわずかに変えます。", "この強度依存性が「Kerr効果」です。石英では n₂ が正です。")

    current_chapter = 1
    chapter_title(1)
    controller = phase_controller("phase accumulation — phi_peak", 145, 268)
    rectangle("intensity and phase card", 64, 168, 645, 386)
    text("同じパルスの強度と追加位相", 88, 208, 23)
    text("P/P₀", 89, 229, 24, "cyan")
    axes(151, 235, 519, 97, -3, 3, 0, 1, [-2, 0, 2], [0, 1], "")
    line("constant intensity", plot_points(151, 235, 519, 97, zip(tau, power)), "cyan", 3.5)
    text("φ_NL [rad]", 88, 367, 24, "gold")
    axes(151, 375, 519, 121, -3, 3, 0, 3, [-2, 0, 2], [0, 3], "")
    pts = plot_points(151, 375, 519, 121, zip(tau, PEAK_PHASE * power), ymax=3)
    dashed_curve("full-length phase reference", pts, "dim", 1)
    phase_curve = line("accumulating nonlinear phase", pts, "gold", 3.5)
    baseline = xy(0, 496)[1]
    for point in phase_curve.data.splines[0].points:
        point.co.y -= baseline
    phase_curve.location.y = baseline
    driver(phase_curve, "scale", 1, controller, "phase_rad", "p/3")
    text("時間 τ=t/T₀", 412, 538, 18, "muted", "CENTER")
    text("最大 3 rad（模式例）", 666, 367, 17, "gold", "RIGHT")
    rectangle("phase definition card", 742, 168, 474, 386)
    text("φ_NL(t) = γ L P(t)", 770, 234, 42)
    text("γ : 非線形係数 [1/(W·km)]", 770, 279, 20, "muted")
    text("L : 距離 [km]   P : 電力 [W]", 770, 312, 20, "muted")
    text("位相＝波の周期のどこにいるか", 979, 352, 20, "muted", "CENTER")
    for center, t, label in [(821, -1.5, "裾"), (979, 0, "頂上"), (1137, 1.5, "裾")]:
        circle("phase comparison ring", center, 425, 47, "grid", 1.8)
        line("zero phase reference", [(center, 425), (center + 38, 425)], "cyan", 2)
        pivot = empty(f"phasor tau={t:g}")
        pivot.location = xy(center, 425, 11)
        start_idx = len(chapter_objects[1])
        arrow("nonlinear phase phasor", center, 425, center + 38, 425, "gold", 4, 10)
        for part in chapter_objects[1][start_idx:]:
            part.parent = pivot
            # Mesh/curve coordinates are already in world space.
            if part.type == "MESH":
                for vertex in part.data.vertices:
                    vertex.co -= pivot.location
            else:
                for point in part.data.splines[0].points:
                    point.co.xyz -= pivot.location
        driver(pivot, "rotation_euler", 2, controller, "phase_rad", f"p*{math.exp(-t*t):.12g}")
        text(label, center, 496, 19, "gold" if t == 0 else "muted", "CENTER")
        text(f"τ={t:+g}" if t else "τ=0", center, 525, 16, "muted", "CENTER")
    caption("屈折率の変化が、強度に応じた「位相の回転」を積み重ねます。", "山の頂上ほど追加位相が大きい。これが自己位相変調（SPM）です。")

    current_chapter = 2
    chapter_title(2)
    controller = phase_controller("SPM real-field phase — phi_peak", 289, 397)
    for x, title, color, dynamic in [(64, "入力", "cyan", False), (647, "Kerrだけの出力", "gold", True)]:
        rectangle(title + " card", x, 168, 569 if dynamic else 555, 386)
        text(title, x + 28, 214, 24, color)
        text("実電場 / E₀（搬送波周期は拡大）", x + 28, 249, 19, "muted")
        left, top, w, h = x + 66, 286, 451, 174
        axes(left, top, w, h, -3, 3, -1, 1, [-2, 0, 2], [-1, 0, 1], "時間 τ=t/T₀")
        for sign in (-1, 1):
            dashed_curve("unchanged envelope", plot_points(left, top, w, h, zip(tau, sign * envelope), ymin=-1, ymax=1), "muted", 1.8)
        values = envelope * np.cos(8 * tau)
        curve = line(title + " real carrier field", plot_points(left, top, w, h, zip(tau, values), ymin=-1, ymax=1), color, 3)
        if dynamic:
            for p, t, env, intensity in zip(curve.data.splines[0].points, tau, envelope, power):
                base = xy(0, top + h / 2)[1]
                expression = f"{base:.10g}+{h/200*env:.10g}*cos({8*t:.10g}-p*{intensity:.10g})"
                driver(p, "co", 1, controller, "phase_rad", expression)
    text("灰色の破線：包絡線 ±|A|/A₀（入力と出力で同じ）", 640, 551, 19, "muted", "CENTER")
    caption("Kerr効果だけなら、振幅の包絡線は変わりません。", "変わるのは位相。時間幅の変化には、分散など別の作用が関わります。")

    current_chapter = 3
    chapter_title(3)
    controller = phase_controller("chirp and spectrum — phi_peak", 433, 544)
    rectangle("chirp card", 64, 168, 645, 386)
    text("時間ごとの周波数差（チャープ）", 89, 210, 23)
    text("前半：低い周波数", 153, 249, 19, "red")
    text("後半：高い周波数", 425, 249, 19, "blue")
    axes(151, 279, 519, 169, -3, 3, -3, 3, [-2, 0, 2], [-3, 0, 3], "時間 τ=t/T₀")
    text("Δω T₀", 89, 269, 23, "muted")
    chirp = 2 * PEAK_PHASE * tau * power
    for mask, color in [(tau <= 0, "red"), (tau >= 0, "blue")]:
        pts = plot_points(151, 279, 519, 169, zip(tau[mask], chirp[mask]), ymin=-3, ymax=3)
        curve = line("instantaneous frequency shift", pts, color, 3.5)
        baseline = xy(0, 279 + 169 / 2)[1]
        for point in curve.data.splines[0].points:
            point.co.y -= baseline
        curve.location.y = baseline
        driver(curve, "scale", 1, controller, "phase_rad", "p/3")
    text("Δω(t) = −dφ_NL(t)/dt", 412, 541, 36, "white", "CENTER")
    rectangle("spectrum card", 742, 168, 474, 386)
    text("スペクトル（同じエネルギー）", 768, 210, 23)
    text("入力", 802, 247, 18, "cyan")
    line("input legend", [(768, 240), (791, 240)], "cyan", 3)
    text("出力", 935, 247, 18, "gold")
    line("output legend", [(901, 240), (924, 240)], "gold", 3)
    axes(802, 279, 374, 169, -7, 7, 0, 1, [-6, 0, 6], [0, 1], "周波数差 Ω=(ω−ω₀)T₀")
    text("S / S_in(0)", 1175, 271, 19, "muted", "RIGHT")
    time = np.linspace(-12, 12, 16384, endpoint=False)
    dt = float(time[1] - time[0])
    gaussian = np.exp(-time**2 / 2)
    # Zero padding samples the same continuous spectrum more densely; it does
    # not change the analytic field or normalize away any spectral energy.
    fft_samples = 131072
    angular = np.fft.fftshift(np.fft.fftfreq(fft_samples, dt)) * 2 * math.pi
    input_spectrum = np.abs(np.fft.fftshift(np.fft.fft(gaussian, n=fft_samples)))**2 * dt**2
    peak_input = float(input_spectrum.max())
    omega = np.linspace(-7, 7, 321)

    def spectrum(phase: float):
        field = gaussian * np.exp(1j * phase * np.exp(-time**2))
        spectral = np.abs(np.fft.fftshift(np.fft.fft(field, n=fft_samples)))**2 * dt**2
        return np.interp(omega, angular, spectral / peak_input), spectral, field

    input_values, _, _ = spectrum(0)
    line("input spectrum reference", plot_points(802, 279, 374, 169, zip(omega, input_values), xmin=-7, xmax=7), "cyan", 2.5)

    def spectrum_ribbon(values):
        points = plot_points(802, 279, 374, 169, zip(omega, values), xmin=-7, xmax=7)
        return [xy(x, y + offset, 11.1) for x, y in points for offset in (-1.6, 1.6)]

    ribbon = mesh("SPM spectrum — FFT shape keys", spectrum_ribbon(input_values), [(2*j, 2*j+1, 2*j+3, 2*j+2) for j in range(len(omega)-1)], "gold")
    ribbon.shape_key_add(name="No nonlinear phase")
    # Linear interpolation between deterministic FFT snapshots, with equal
    # energy normalization; values are never normalized to their own maxima.
    for i, phase in enumerate(np.linspace(0, PEAK_PHASE, 17)[1:], 1):
        key = ribbon.shape_key_add(name=f"phi_peak={phase:.4f} rad")
        for vertex, co in zip(key.data, spectrum_ribbon(spectrum(float(phase))[0])):
            vertex.co = co
        drv = key.driver_add("value").driver
        drv.type = "SCRIPTED"
        variable = drv.variables.new()
        variable.name, variable.type = "p", "SINGLE_PROP"
        variable.targets[0].id = controller
        variable.targets[0].data_path = '["phase_rad"]'
        drv.expression = f"max(0,1-abs(p*16/3-{i}))"
    text("周波数成分が広がり得る", 979, 541, 21, "gold", "CENTER")
    caption("前半は低い周波数へ、後半は高い周波数へずれます。", "時間で変わる位相が「チャープ」を生み、スペクトルが広がり得ます。")

    current_chapter = 4
    chapter_title(4)
    for x, w, label, formula, color, number in [
        (64, 334, "光の強度", "n = n₀ + n₂ I", "cyan", "01"),
        (449, 334, "非線形位相", "φ_NL = γ L P", "gold", "02"),
        (834, 382, "周波数の変化", "Δω = −dφ_NL/dt", "purple", "03"),
    ]:
        rectangle(label + " summary card", x, 184, w, 207)
        text(label, x + 25, 236, 27, color)
        text(formula, x + 25, 306, 42 if w < 380 else 38)
        text(number, x + w - 24, 370, 21, "dim", "RIGHT")
    arrow("intensity to phase", 409, 289, 438, 289, "gold", 3)
    arrow("phase to frequency", 794, 289, 823, 289, "gold", 3)
    token = dot("causal chain highlight", 96, 393, 4, "gold")
    linear_keys(token, "location", [(577, 0), (654, 10.8), (720, 10.8)], 0)
    rectangle("scope card", 64, 417, 1152, 137)
    text("この動画の範囲", 88, 452, 22, "cyan")
    text("単一偏波・損失なし・分散なし。測定データではない模式例です。", 88, 491, 23)
    text("本repoの1600 km評価は、2偏波・損失・分散・増幅器雑音を含む別の計算です。", 88, 530, 21, "muted")
    caption("振幅ノイズを消す作用ではなく、光自身が位相を変える現象です。", "この位相変化と分散・雑音が、通信信号のひずみに関わります。")

    # Individual object visibility is keyed, so opening the scene and scrubbing
    # or rendering it works without registering an external Python callback.
    current_chapter = None
    for index, objects in enumerate(chapter_objects):
        first, last, label, _ = CHAPTERS[index]
        collection = bpy.data.collections.new(f"{index + 1:02d} — {label} — frames {first}-{last}")
        scene.collection.children.link(collection)
        for obj in objects:
            for col in list(obj.users_collection):
                col.objects.unlink(obj)
            collection.objects.link(obj)
            for frame, hidden in [(0, True), (first, False), (last + 1, True)]:
                obj.hide_render = hidden
                obj.hide_viewport = hidden
                obj.keyframe_insert(data_path="hide_render", frame=frame)
                obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            for layer in obj.animation_data.action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for fc in bag.fcurves:
                            if fc.data_path in {"hide_render", "hide_viewport"}:
                                for key in fc.keyframe_points:
                                    key.interpolation = "CONSTANT"
    for first, _, label, _ in CHAPTERS:
        scene.timeline_markers.new(label, frame=first)
    scene["scientific_scope"] = "Analytic scalar, lossless, dispersion-free SPM. Not the DP/Manakov 1600 km measured study."
    scene["field_convention"] = "E = Re[A(t) exp(i(kz - omega0 t))]; delta_omega = -d(phi_NL)/dt."
    scene["illustration_parameters"] = "P/P0=exp(-tau^2); tau=t/T0; phi_NL=3*u*exp(-tau^2); Omega=(omega-omega0)*T0; u=L/L0."
    scene["editing"] = "Five chapter collections; controller custom property phase_rad; native keyframes/drivers; packed Japanese font."
    scene.frame_set(POSTER_FRAME)
    # Useful default when the .blend is opened interactively.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.shading.type = "MATERIAL"
    scratch, output = args.scratch_dir, args.output_dir
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = scratch / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_")
    blend_path = scratch / "preview.blend" if args.preview else output / "kerr_effect.blend"
    license_path = output / "LICENSE-fonts.txt"
    if license_path.is_file():
        bpy.data.texts.load(str(license_path)).use_fake_user = True
    notes = bpy.data.texts.new("README — scientific scope and editing")
    notes.use_fake_user = True
    notes.write("\n".join(f"{key}: {scene[key]}" for key in ["scientific_scope", "field_convention", "illustration_parameters", "editing"]))
    # Remove factory-startup unused brush/material assets; the deliverable has
    # no dependency on the artist's local Blender installation files.
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
    final_values, final_spectrum, final_field = spectrum(PEAK_PHASE)
    envelope_error = float(np.max(np.abs(np.abs(final_field)**2 - gaussian**2)))
    energy_in, energy_out = float(np.sum(gaussian**2) * dt), float(np.sum(np.abs(final_field)**2) * dt)
    spectral_energy = float(np.sum(final_spectrum) * (angular[1] - angular[0]) / (2 * math.pi))
    phase = PEAK_PHASE * np.exp(-time**2)
    analytic_chirp = 2 * PEAK_PHASE * time * np.exp(-time**2)
    numeric_chirp = -np.gradient(phase, dt)
    derivative_error = float(np.max(np.abs(analytic_chirp[1:-1] - numeric_chirp[1:-1])))
    rms_in = float(np.sqrt(np.sum(angular**2 * input_spectrum) / np.sum(input_spectrum)))
    rms_out = float(np.sqrt(np.sum(angular**2 * final_spectrum) / np.sum(final_spectrum)))
    expected_ratio = math.sqrt(1 + 4 * PEAK_PHASE**2 / (3 * math.sqrt(3)))
    assert envelope_error < 1e-12
    assert abs(energy_out / energy_in - 1) < 1e-12
    assert abs(spectral_energy / energy_in - 1) < 1e-10
    assert derivative_error < 2e-5
    assert abs(rms_out / rms_in - expected_ratio) < 1e-10
    assert np.all(analytic_chirp[time < 0] <= 0) and np.all(analytic_chirp[time > 0] >= 0)
    checks = {
        "scope": "Ordinary analytic/numerical verification of the teaching equations, not a communications evaluation.",
        "max_power_envelope_error": envelope_error,
        "relative_time_energy_error": abs(energy_out / energy_in - 1),
        "relative_parseval_energy_error": abs(spectral_energy / energy_in - 1),
        "max_chirp_finite_difference_error": derivative_error,
        "rms_spectrum_width_ratio": rms_out / rms_in,
        "analytic_rms_width_ratio": expected_ratio,
        "leading_edge_red_trailing_edge_blue": True,
        "passed": True,
    }
    metadata = {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode(),
        "python": sys.version, "numpy": np.__version__,
        "render_engine": scene.render.engine, "eevee_render_samples": getattr(scene.eevee, "taa_render_samples", None),
        "resolution": [WIDTH, HEIGHT], "fps": FPS, "frame_range": [1, FRAMES], "duration_seconds": FRAMES / FPS,
        "poster_frame": POSTER_FRAME,
        "chapters": [{"first_frame": a, "last_frame": b, "time_seconds": [(a - 1) / FPS, b / FPS], "label": label, "title": title} for a, b, label, title in CHAPTERS],
        "model": {
            "index": "n(t)=n0+n2 I(t); n2>0 for silica; fixed scalar effective mode relates gamma*P to k0*n2*I",
            "power": "P(t)/P0=exp(-tau^2), tau=t/T0; P0,T0 are normalization units, not measured link parameters",
            "field": "A_out(t)=A_in(t)*exp(i*phi_NL(t)), A_in/A0=exp(-tau^2/2)",
            "nonlinear_phase": "phi_NL(t)=gamma*L*P(t)=3*u*exp(-tau^2), u=L/L0",
            "final_peak_phase_rad": PEAK_PHASE,
            "frequency_shift": "delta_omega*T0=-d(phi_NL)/d(tau)=6*u*tau*exp(-tau^2)",
            "physical_phase_convention": "E=Re{A(t)*exp[i(kz-omega0*t)]}",
            "illustrated_real_field": "E/E0=exp(-tau^2/2)*cos(8*tau-phi_NL); optical carrier period intentionally enlarged, not an actual wavelength or group velocity",
            "spectrum": "FFT of A with 16384 samples on tau in [-12,12), zero-padded to 131072 for display sampling; omega axis normalized by T0; density divided by input peak, never each curve's own maximum",
            "spectrum_animation": "17 exact FFT phase snapshots, interpolated for illustration; no interpolation of research results",
            "repo_distinction": "Actual repo propagation is lossy, dispersive DP Manakov with (8/9)*gamma*(Px+Py) and EDFA ASE. This animation is not an output of that simulation.",
        },
        "formula_checks": checks,
        "scene_objects": len(bpy.data.objects),
    }
    (scratch / "scene_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.build_only:
        return
    if args.preview:
        for frame in [1, POSTER_FRAME, 216, 360, 510, 666]:
            scene.frame_set(frame)
            scene.render.filepath = str(scratch / f"still_{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
        scene.frame_start, scene.frame_end = 193, 240
    scene.render.filepath = str(frames_dir / "frame_")
    bpy.ops.render.render(animation=True)


def main() -> None:
    inside = "--inside-blender" in sys.argv
    argv = sys.argv[sys.argv.index("--") + 1:] if inside and "--" in sys.argv else sys.argv[1:]
    args = arguments(argv)
    if inside:
        run_blender(args)
    else:
        run_host(args)


if __name__ == "__main__":
    main()
