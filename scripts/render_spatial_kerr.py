#!/usr/bin/env python3
"""Render saved circular-core CW meshes, power flows, and masked phase slices.

This CLI never propagates light or invents a field. Start with ``--mode
inspect``, ``--mode posters`` or ``--mode proof``. Full six-second films are
an explicit separate ``--mode render`` operation. Cache reuse is deliberately
unsupported: every run has a fresh isolated scratch directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILMS = {
    "01_reflection": {"poster": "poster_01.png", "title": "境界で戻る、光の分布", "components": ["reflection"]},
    "02_kerr_comparison": {"poster": "poster_02.png", "title": "同じ条件で、Kerrを比べる", "components": ["comparison_off", "comparison_on"]},
    "03_phase_change": {"poster": "poster_03.png", "title": "波のずれと、明暗の変化", "components": ["phase", "intensity_off", "intensity_on"]},
}
CASES = ("linear", "weak_reference", "intermediate_kerr", "educational_kerr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def physical_to_world(xyz_um: np.ndarray, um_per_unit: float = 100.) -> np.ndarray:
    """XYZ=(transverse x, transverse y, propagation z) -> Blender (z,x,y)."""
    values = np.asarray(xyz_um, dtype=np.float64)
    if values.shape[-1:] != (3,) or um_per_unit <= 0:
        raise ValueError("Expected physical XYZ and one positive common XYZ scale")
    return values[..., [2, 0, 1]] / um_per_unit


def normals_to_world(normals_xyz: np.ndarray) -> np.ndarray:
    values = np.asarray(normals_xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Invalid physical normals")
    length = np.linalg.norm(values, axis=1)
    if (length <= 0).any():
        raise ValueError("Isosurface has zero-length supplied normals")
    return values[:, [2, 0, 1]] / length[:, None]


def flow_prefix_indices(valid: np.ndarray, stride: int = 1) -> list[np.ndarray]:
    """Retain every seed, stop once, and never reconnect across an invalid point."""
    valid = np.asarray(valid)
    if valid.dtype != np.bool_ or valid.ndim != 2 or stride < 1:
        raise ValueError("Expected a boolean (seed,z) validity mask and positive stride")
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("Power-flow seed restarts after termination")
    result = []
    for row in valid:
        count = int(row.sum())
        ids = np.arange(0, count, stride, dtype=np.int64)
        if count and (not len(ids) or ids[-1] != count - 1):
            ids = np.append(ids, count - 1)
        result.append(ids)
    return result


def phase_difference(linear: np.ndarray, kerr: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    if linear.shape != kerr.shape or floor <= 0:
        raise ValueError("Phase comparison needs registered fields and a positive joint floor")
    valid = (np.abs(linear) ** 2 >= floor) & (np.abs(kerr) ** 2 >= floor)
    return np.angle(kerr * np.conj(linear)), valid


def palette(values: np.ndarray, limits: list[float], colors: list[list[float]]) -> np.ndarray:
    """A fixed mapping: never divide by a frame, distance, or case maximum."""
    low, high = limits
    if not high > low:
        raise ValueError("Fixed color limits must increase")
    anchors = np.asarray(colors, dtype=np.float64)
    coordinate = np.clip((np.asarray(values) - low) / (high - low), 0, 1)
    knots = np.linspace(0, 1, len(anchors))
    return np.stack([np.interp(coordinate, knots, anchors[:, channel]) for channel in range(3)], axis=-1)


def intensity_rgba(intensity: np.ndarray, config: dict) -> np.ndarray:
    intensity = np.asarray(intensity)
    if not np.isfinite(intensity).all() or (intensity < 0).any():
        raise ValueError("Intensity must be finite and nonnegative")
    rgba = np.ones((*intensity.shape, 4), dtype=np.float32)
    rgba[..., :3] = palette(intensity, config["intensity_range_relative_to_input_peak"], config["intensity_palette_linear_rgb"])
    return rgba


def phase_rgba(phase: np.ndarray, valid: np.ndarray, config: dict) -> np.ndarray:
    if phase.shape != valid.shape:
        raise ValueError("Phase and joint validity masks must match")
    rgba = np.ones((*phase.shape, 4), dtype=np.float32)
    rgba[..., :3] = palette(phase, config["phase_range_rad"], config["phase_palette_linear_rgb"])
    rgba[..., :3][~valid] = config["masked_phase_color_linear_rgb"]
    return rgba


def distance_at_frame(frame: int, frame_count: int, config: dict, film_id: str | None = None) -> float:
    start, end = config["camera"]["target_z_um"]
    if frame_count < 2 or not 1 <= frame <= frame_count:
        raise ValueError("Frame is outside the recorded observation sweep")
    intro = config["camera"].get("intro_frames",0) if film_id=="01_reflection" else 0
    if frame<=intro:
        return start
    return start + (end - start) * (frame - intro - 1) / (frame_count - intro - 1)


def nearest_flow_sample(z_um: np.ndarray, observation_z_um: float) -> int:
    return int(np.argmin(np.abs(np.asarray(z_um) - observation_z_um)))


def core_boundary_rails_um(config: dict) -> np.ndarray:
    """Annotation positions in the actual y=0 circular-core reflection plane."""
    radius = config["physical_display"]["core_diameter_um"] / 2
    length = config["physical_display"]["length_um"]
    return np.asarray([[[sign*radius,0,0],[sign*radius,0,length]] for sign in (1,-1)],dtype=float)


def camera_target_at_frame(frame: int, frame_count: int, config: dict, film_id: str) -> np.ndarray:
    """Camera focus is separate from the saved-flow observation cursor in the intro."""
    camera = config["camera"]
    intro,hold = camera["intro_frames"],camera["intro_hold_frames"]
    if film_id=="01_reflection" and frame<=intro:
        t = max(0.,(frame-hold)/(intro+1-hold))
        z = (1-t)*camera["intro_target_z_um"]+t*camera["target_z_um"][0]
    else:
        z = distance_at_frame(frame,frame_count,config,film_id)
    return np.asarray([z/config["physical_display"]["um_per_world_unit"],0.,0.])


def intro_camera_basis(frame: int, config: dict) -> np.ndarray:
    """Keep each integer-frame view on the target while smoothly changing roll."""
    camera = config["camera"]
    t = (frame-camera["intro_hold_frames"])/(camera["intro_frames"]+1-camera["intro_hold_frames"])
    if not 0<=t<=1:
        raise ValueError("Intro camera basis is only defined between the held and closeup poses")
    start = np.asarray(camera["intro_offset_world"],dtype=float)
    end = np.asarray(camera["offset_world"],dtype=float)
    start_back = start/np.linalg.norm(start)
    start_right = np.cross([0.,0.,1.],start_back)
    start_right /= np.linalg.norm(start_right)
    start_up = np.cross(start_back,start_right)
    end_back = end/np.linalg.norm(end)
    end_right = np.asarray([1.,0.,0.])-end_back*end_back[0]
    end_right /= np.linalg.norm(end_right)
    end_up = np.cross(end_back,end_right)
    back = (1-t)*start+t*end
    back /= np.linalg.norm(back)
    hint = (1-t)*start_up+t*end_up
    right = np.cross(hint,back)
    right /= np.linalg.norm(right)
    up = np.cross(back,right)
    return np.column_stack((right,up,back))


def _axis(values: np.ndarray, name: str, minimum_count: int = 2) -> None:
    if values.ndim != 1 or len(values) < minimum_count or not np.isfinite(values).all() or not (np.diff(values) > 0).all():
        raise ValueError(f"{name} must be finite and strictly increasing")


def load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        missing = required - set(stored.files)
        if missing:
            raise ValueError(f"{path.name}: missing keys {sorted(missing)}")
        return {key: stored[key].copy() for key in required}


def load_case(data_dir: Path, name: str, contract: dict, config: dict) -> dict:
    sections = load_npz(data_dir / contract["sections_file"], {"x_um", "y_um", "z_um", "central_psi", "cross_z_um", "cross_psi"})
    for key in ("x_um", "y_um", "z_um", "cross_z_um"):
        _axis(sections[key], key, minimum_count=1 if key=="cross_z_um" else 2)
    central, cross = sections["central_psi"], sections["cross_psi"]
    if central.shape != (len(sections["z_um"]), len(sections["x_um"])) or cross.shape != (len(sections["cross_z_um"]), len(sections["y_um"]), len(sections["x_um"])):
        raise ValueError(f"{name}: section axis order/shape mismatch")
    if central.dtype != np.complex128 or cross.dtype != np.complex128 or not np.isfinite(central).all() or not np.isfinite(cross).all():
        raise ValueError(f"{name}: expected finite complex128 fields")
    flows = load_npz(data_dir / contract["flows_file"], {"z_um", "position_um", "valid", "seed_quantiles", "seed_position_um", "intensity", "velocity_xy_um_per_um"})
    _axis(flows["z_um"], "flow z_um")
    valid = flows["valid"]
    seed_count = config["flow"]["expected_seed_count"]
    if valid.shape != (seed_count, len(flows["z_um"])):
        raise ValueError(f"{name}: all {seed_count} saved seeds are required")
    prefixes = flow_prefix_indices(valid, config["flow"]["display_stride"])
    if flows["position_um"].shape != (*valid.shape, 3) or flows["velocity_xy_um_per_um"].shape != (*valid.shape, 2) or flows["intensity"].shape != valid.shape:
        raise ValueError(f"{name}: flow data axis mismatch")
    if flows["seed_quantiles"].shape != (seed_count, 2) or flows["seed_position_um"].shape != (seed_count, 3):
        raise ValueError(f"{name}: seed identity data mismatch")
    if not np.isfinite(flows["position_um"][valid]).all() or not np.isfinite(flows["velocity_xy_um_per_um"][valid]).all() or not np.isfinite(flows["intensity"][valid]).all():
        raise ValueError(f"{name}: a valid flow point is nonfinite")
    if not np.allclose(flows["position_um"][..., 2][valid], np.broadcast_to(flows["z_um"], valid.shape)[valid], atol=1e-6, rtol=0):
        raise ValueError(f"{name}: final physical XYZ component is not propagation z")
    if (flows["intensity"][valid] < config["phase_joint_intensity_floor"]).any():
        raise ValueError(f"{name}: accepted flow point below the fixed intensity floor")
    if not np.isfinite(flows["seed_quantiles"]).all() or not np.isfinite(flows["seed_position_um"]).all():
        raise ValueError(f"{name}: nonfinite saved seed identity")
    meshes = {}
    for filename in contract.get("mesh_files", []):
        mesh = load_npz(data_dir / filename, {"vertices_um", "faces", "normals_xyz", "level"})
        verts, faces = mesh["vertices_um"], mesh["faces"]
        level = float(mesh["level"])
        if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != np.int32:
            raise ValueError(f"{filename}: physical mesh contract mismatch")
        if not len(verts) or not len(faces) or not np.isfinite(verts).all() or faces.min() < 0 or faces.max() >= len(verts):
            raise ValueError(f"{filename}: empty/nonfinite/out-of-range mesh")
        if mesh["normals_xyz"].shape != verts.shape:
            raise ValueError(f"{filename}: supplied normal shape mismatch")
        normals_to_world(mesh["normals_xyz"])
        if level in meshes:
            raise ValueError(f"{name}: duplicate intensity threshold {level}")
        meshes[level] = {**mesh, "filename": filename}
    if name in ("linear", "educational_kerr"):
        wanted = [entry["level"] for entry in config["isosurfaces"]]
        if sorted(meshes) != wanted:
            raise ValueError(f"{name}: required fixed thresholds {wanted}, found {sorted(meshes)}")
    return {"name": name, "kappa": contract["kappa"], "sections_file":contract["sections_file"], "flows_file":contract["flows_file"], "sections": sections, "flows": flows, "prefixes": prefixes, "meshes": meshes}


def verify_inputs(data_dir: Path, config: dict, *, retain_cases: bool = True) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Numerical input is not ready: {manifest_path}. No field, scene, or poster is invented.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("coordinate_contract", {}).get("length_unit") != "um":
        raise ValueError("Expected spatial artifact schema_version=1 and physical micrometres")
    expected_orders = {"central_psi": "(z,x)", "cross_psi": "(snapshot,y,x)", "position_um": "(seed,z,XYZ)"}
    orders = manifest["coordinate_contract"].get("axis_orders", {})
    if any(orders.get(key) != value for key, value in expected_orders.items()):
        raise ValueError("Numerical axis-order contract does not match the renderer")
    records = {"manifest.json": {"sha256": sha256(manifest_path), "bytes": manifest_path.stat().st_size}}
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Artifact file keys must be safe basenames")
        path = data_dir / name
        if not path.exists() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Numerical artifact hash/size mismatch: " + name)
        records[name] = expected
    model = manifest.get("computational_config", {})
    display = config["physical_display"]
    for source, target in [("core_radius_um", display["core_diameter_um"] / 2), ("zmax_um", display["length_um"]), ("ncore", display["ncore"]), ("nclad", display["nclad"])]:
        if source in model and not math.isclose(model[source], target, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Display geometry conflicts with numerical input: " + source)
    cases, summary = {}, {}
    for name in CASES:
        if name not in manifest["case_contract"]:
            raise ValueError("Missing retained numerical case: " + name)
        data = load_case(data_dir, name, manifest["case_contract"][name], config)
        field = np.abs(data["sections"]["central_psi"]) ** 2
        summary[name] = {"kappa": data["kappa"], "central_shape": list(field.shape), "complex_dtype": "complex128", "seed_count": len(data["prefixes"]), "valid_prefix_lengths": [int(row.sum()) for row in data["flows"]["valid"]], "mesh_levels": sorted(data["meshes"]), "central_intensity_max_relative_to_input": float(field.max()), "above_fixed_display_limit_fraction": float(np.mean(field > config["intensity_range_relative_to_input_peak"][1]))}
        if retain_cases:
            cases[name] = data
    linear = cases.get("linear")
    strong = cases.get("educational_kerr")
    if retain_cases:
        for key in ("x_um", "y_um", "z_um", "cross_z_um"):
            if not np.array_equal(linear["sections"][key], strong["sections"][key]):
                raise ValueError("OFF/ON field grids must be identically registered: " + key)
        if linear["kappa"] != 0 or not math.isclose(strong["kappa"], 3e-4, rel_tol=0, abs_tol=1e-15):
            raise ValueError("These films require the registered kappa=0 / 3e-4 comparison")
        phase, valid = phase_difference(linear["sections"]["central_psi"], strong["sections"]["central_psi"], config["phase_joint_intensity_floor"])
        summary["phase"] = {"principal_range_rad": [-math.pi, math.pi], "joint_floor": config["phase_joint_intensity_floor"], "valid_fraction": float(valid.mean()), "valid_phase_min_max_rad": [float(phase[valid].min()), float(phase[valid].max())] if valid.any() else None}
    accuracy = {}
    if "metrics.json" in records:
        metrics = json.loads((data_dir/"metrics.json").read_text(encoding="utf-8"))
        for name,tests in metrics.get("flows",{}).get("comparisons",{}).items():
            accuracy[name] = {}
            for kind in ("integration_step","interpolation"):
                values = [float(row["max_distance_um"]) for row in tests.get(kind,[]) if row.get("max_distance_um") is not None]
                accuracy[name][kind+"_max_position_difference_um"] = max(values) if values else None
    return {"manifest": manifest, "records": records, "cases": cases, "summary": summary, "flow_accuracy":accuracy}


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config["fps"] != 24 or config["duration_seconds"] != 6:
        raise ValueError("The declared films are six seconds at 24 fps")
    if (config["width"], config["height"]) != (1920, 1080):
        raise ValueError("Final film layout must be 1920x1080")
    if [item["level"] for item in config["isosurfaces"]] != [.1, .3, .6]:
        raise ValueError("Use the fixed input-peak isosurfaces 0.1, 0.3, 0.6")
    if config["phase_range_rad"] != [-math.pi, math.pi] or not 0 < config["phase_joint_intensity_floor"] < .1:
        raise ValueError("Use principal wrapped phase with a fixed positive joint floor")
    if not 0 < config["proof"]["seconds"] <= 3 or config["proof"]["film"] != "01_reflection":
        raise ValueError("The bounded phenomenon proof is at most three seconds of film 01")
    if config["flow"]["expected_seed_count"] != 25 or config["flow"]["display_stride"] < 1:
        raise ValueError("All 25 input seed identities must be preserved")
    if (config["proof"]["width"],config["proof"]["height"])!=(960,540):
        raise ValueError("This proof uses an exact half-resolution 960x540 native render")
    for entry in config["isosurfaces"]:
        if not 0<=entry["opacity"]<=1 or not np.isfinite(entry["color_linear_rgb"]).all():
            raise ValueError("Fixed isosurface opacity/color is invalid")


def arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inspect", "build", "posters", "proof", "render", "verify-scene", "repair-cameras"], default="posters")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/spatial_kerr_render.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/spatial_kerr")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/animations/spatial_kerr")
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / ".work/spatial-kerr-render")
    parser.add_argument("--film", choices=list(FILMS), nargs="+", default=list(FILMS))
    parser.add_argument("--poster-frame", type=int, help="Render a selected camera frame for still review; does not alter the stored field or film schedule")
    parser.add_argument("--repair-manifest", type=Path, help="Completed original render manifest; required for repair-cameras")
    parser.add_argument("--repair-cache", type=Path, help="Original full-render cache, if moved from its recorded path")
    parser.add_argument("--preflight-only", action="store_true", help="Prepare and render the fixed camera-repair review samples, without replacing delivered films")
    parser.add_argument("--repair-state", type=Path, help="Verified preflight state from this exact repair source/config/environment")
    parser.add_argument("--blender", help="Blender executable; Windows .exe under WSL is supported")
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"))
    parser.add_argument("--font-license", type=Path, default=Path("/usr/share/doc/fonts-ipaexfont-gothic/copyright"))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--inside-blender", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


# Blender-only code is in this entry point; importing numerical/data helpers
# above does not import Blender and is safe for ordinary renderer tests.
def _collection(scene, name):
    coll = bpy.data.collections.new(name)
    scene.collection.children.link(coll)
    return coll


def _object(coll, name, data):
    obj = bpy.data.objects.new(name, data)
    coll.objects.link(obj)
    return obj


def _material(name, color, roughness=.5, metallic=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


def _camera_emission(name, color, strength=1., opacity=1., image=None, interpolation="Linear", display_shading=None):
    """Direct display encoding, not artificial scattering or reflected light."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1)
    emission.inputs["Strength"].default_value = strength
    weighted = nodes.new("ShaderNodeMixShader")
    weighted.inputs[0].default_value = opacity
    ray = nodes.new("ShaderNodeLightPath")
    direct = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputMaterial")
    links = mat.node_tree.links
    links.new(transparent.outputs[0], weighted.inputs[1])
    links.new(emission.outputs[0], weighted.inputs[2])
    links.new(ray.outputs["Is Camera Ray"], direct.inputs[0])
    links.new(transparent.outputs[0], direct.inputs[1])
    links.new(weighted.outputs[0], direct.inputs[2])
    links.new(direct.outputs[0], output.inputs["Surface"])
    if image is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = interpolation
        tex.extension = "CLIP"
        links.new(tex.outputs["Color"], emission.inputs["Color"])
    if display_shading is not None:
        geometry = nodes.new("ShaderNodeNewGeometry")
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        direction = np.asarray(display_shading["direction_world"],dtype=float)
        dot.inputs[1].default_value = direction/np.linalg.norm(direction)
        positive = nodes.new("ShaderNodeMath")
        positive.operation = "MAXIMUM"
        positive.inputs[1].default_value = 0
        weight = nodes.new("ShaderNodeMath")
        weight.operation = "MULTIPLY_ADD"
        weight.inputs[1].default_value = display_shading["diffuse_weight"]
        weight.inputs[2].default_value = display_shading["minimum"]
        tint = nodes.new("ShaderNodeMixRGB")
        tint.blend_type = "MULTIPLY"
        tint.inputs[0].default_value = 1
        tint.inputs[1].default_value = (*color,1)
        links.new(geometry.outputs["Normal"],dot.inputs[0])
        links.new(dot.outputs["Value"],positive.inputs[0])
        links.new(positive.outputs[0],weight.inputs[0])
        links.new(weight.outputs[0],tint.inputs[2])
        links.new(tint.outputs[0],emission.inputs["Color"])
        mat["display_shading"] = "Common directional normal shading for 3D shape only; never quantitative scattering/intensity brightness."
    mat["scope"] = "Computed display geometry visible on camera/straight transparent rays only. No simulated scattering, field-induced light, or mirror photon paths."
    return mat


def _presentation_glass(config, *, entrance_context=False):
    """A weak curved reflection over straight transmission preserves registration."""
    mat = bpy.data.materials.new("Registered straight-through presentation glass")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    glossy = nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (.95, .97, 1, 1)
    glossy.inputs["Roughness"].default_value = config["studio"]["glass_roughness"]
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = config["physical_display"]["nclad"]
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = config["studio"]["glass_reflection_weight"]
    mix = nodes.new("ShaderNodeMixShader")
    # A small neutral surface fill makes the filled circular entrance and
    # window faces readable during the photographic context shot. It is
    # faded away for the data closeup, without refracting any field vertices.
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (.82,.87,.92,1)
    fill = nodes.new("ShaderNodeMixShader")
    fill.inputs[0].default_value = 0
    output = nodes.new("ShaderNodeOutputMaterial")
    links = mat.node_tree.links
    links.new(fresnel.outputs[0], scale.inputs[0])
    links.new(scale.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(glossy.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    links.new(mix.outputs[0],fill.inputs[1])
    links.new(diffuse.outputs[0],fill.inputs[2])
    links.new(fill.outputs[0],output.inputs["Surface"])
    if entrance_context:
        intro = config["camera"]["intro_frames"]
        hold = config["camera"]["intro_hold_frames"]
        for frame,weight,fill_weight,roughness in [
            (1,config["studio"]["intro_glass_reflection_weight"],config["studio"]["intro_glass_surface_fill"],config["studio"]["intro_glass_roughness"]),
            (hold,config["studio"]["intro_glass_reflection_weight"],config["studio"]["intro_glass_surface_fill"],config["studio"]["intro_glass_roughness"]),
            (intro+1,config["studio"]["glass_reflection_weight"],0,config["studio"]["glass_roughness"]),
        ]:
            for socket,value in [(scale.inputs[1],weight),(fill.inputs[0],fill_weight),(glossy.inputs["Roughness"],roughness)]:
                socket.default_value = value
                socket.keyframe_insert("default_value",frame=frame)
        _set_linear_action(mat.node_tree)
    mat["scope"] = "Display glass: straight transmission and Fresnel sheen. Entrance-context sheen and neutral surface fill fade into the data-closeup material; computed field colors stay fixed. Refraction is omitted to keep numerical field geometry registered. Glass-air reflection is not part of the propagation solver."
    return mat


def _mesh(coll, name, verts, faces, mat, normals=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.asarray(verts).tolist(), [], np.asarray(faces).tolist())
    mesh.update()
    if mat:
        mesh.materials.append(mat)
    if normals is not None:
        for poly in mesh.polygons:
            poly.use_smooth = True
        mesh.normals_split_custom_set(np.asarray(normals)[np.asarray(faces).ravel()].tolist())
    return _object(coll, name, mesh)


def _cube(coll, name, location, dimensions, mat, bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("Soft manufactured edges", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    return obj


def _polycurve(coll, name, points_world, radius_world, mat):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 1
    data.bevel_depth = radius_world
    data.bevel_resolution = 0
    data.use_fill_caps = True
    if len(points_world) >= 2:
        spline = data.splines.new("POLY")
        spline.points.add(len(points_world)-1)
        values = np.ones((len(points_world), 4), dtype=np.float64)
        values[:, :3] = points_world
        spline.points.foreach_set("co", values.ravel())
    obj = _object(coll, name, data)
    obj.data.materials.append(mat)
    return obj


def _solid_fiber(coll, config, mat, *, central_cut=False):
    """One welded solid: no coincident core/inner-cladding glass interfaces."""
    from mathutils import Vector
    import bmesh
    scale = config["physical_display"]["um_per_world_unit"]
    core = config["physical_display"]["core_diameter_um"] / (2*scale)
    clad = config["physical_display"]["cladding_diameter_um"] / (2*scale)
    length = config["physical_display"]["length_um"] / scale
    a0, a1 = np.asarray(config["physical_display"]["window_z_um"]) / scale
    # The upper annular half is removed, exposing both +/-x sides. For the
    # central y=0 slice the filled core is also cut only for presentation.
    cut_angles = np.linspace(0, math.pi, 257)
    remaining = np.linspace(math.pi, 2*math.pi, 257)
    full = np.concatenate((cut_angles, remaining[1:]))
    inner = 0.0 if central_cut else core
    vertices, lookup, faces, normals = [], {}, [], []

    def p(x, r, t):
        return (x, r*math.cos(t), r*math.sin(t))

    def add(points, ns):
        direction = (Vector(points[1])-Vector(points[0])).cross(Vector(points[2])-Vector(points[0]))
        if direction.dot(sum((Vector(n) for n in ns), Vector((0,0,0)))) < 0:
            points, ns = list(reversed(points)), list(reversed(ns))
        ids = []
        for point in points:
            key = tuple(round(v, 10) for v in point)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(point)
            ids.append(lookup[key])
        faces.append(ids)
        normals.extend(ns)

    def curved(x0, x1, r, angles):
        for a, b in zip(angles[:-1], angles[1:]):
            na, nb = (0,math.cos(a),math.sin(a)), (0,math.cos(b),math.sin(b))
            add([p(x0,r,a),p(x1,r,a),p(x1,r,b),p(x0,r,b)], [na,na,nb,nb])

    curved(0, a0, clad, full)
    curved(a1, length, clad, full)
    curved(a0, a1, clad, remaining)
    if inner:
        curved(a0, a1, inner, cut_angles)
    else:
        # The numerical y=0 display slice supplies the cut surface. A
        # coincident glass rectangle here would hide it through ray offsets.
        pass
    if inner:
        for angle, sign in [(0,1),(math.pi,-1)]:
            n = (0,-sign*math.sin(angle),sign*math.cos(angle))
            add([p(a0,inner,angle),p(a0,clad,angle),p(a1,clad,angle),p(a1,inner,angle)], [n]*4)
    for x, sign in [(a0,1),(a1,-1)]:
        n = (sign,0,0)
        for a,b in zip(cut_angles[:-1],cut_angles[1:]):
            if inner:
                add([p(x,inner,a),p(x,clad,a),p(x,clad,b),p(x,inner,b)], [n]*4)
            else:
                add([(x,0,0),p(x,clad,a),p(x,clad,b)], [n]*3)
    for x, sign in [(0,-1),(length,1)]:
        for a,b in zip(full[:-1],full[1:]):
            add([(x,0,0),p(x,clad,a),p(x,clad,b)], [(sign,0,0)]*3)
    mesh = bpy.data.meshes.new("Closed circular fiber, 512 radial intervals")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.normals_split_custom_set(normals)
    obj = _object(coll, "DISPLAY | circular 50 um core / 125 um cladding", mesh)
    obj["scope"] = "Presentation cutaway only. The complete circular core is the numerical domain; displayed air and surface materials are not in propagation."
    bm = bmesh.new()
    bm.from_mesh(mesh)
    nonmanifold = sum(not edge.is_manifold for edge in bm.edges)
    bm.free()
    # A central cut has cap triangles meeting the diametral cut. The solid
    # core mode must be a closed manifold; slice mode's topology is recorded.
    if not central_cut and nonmanifold:
        raise RuntimeError("Circular display glass is not closed")
    return {"glass_objects":1,"circumferential_intervals":512,"custom_normals":True,"nonmanifold_edges":nonmanifold,"central_core_cut_for_slice":central_cut}


def _area(coll, name, location, target, energy, size, size_y):
    from mathutils import Vector
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.color = (1,1,1)
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = size_y
    obj = _object(coll, name, data)
    obj.location = location
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()


def _entrance_coating(coll, scene, config, mat):
    """An annular jacket shows the stripped observation region, not a tube core."""
    scale = config["physical_display"]["um_per_world_unit"]
    z0,z1 = (config["studio"][key]/scale for key in ("intro_coating_start_um","intro_coating_end_um"))
    inner = config["physical_display"]["cladding_diameter_um"]/(2*scale)+.008
    outer = .95
    angles = np.linspace(0,2*math.pi,257)[:-1]
    vertices = [(z,r*math.cos(t),r*math.sin(t)) for z in (z0,z1) for r in (inner,outer) for t in angles]
    faces = []
    n = len(angles)
    for i in range(n):
        j = (i+1)%n
        faces.extend([(n+i,n+j,3*n+j,3*n+i),(i,2*n+i,2*n+j,j),(i,j,n+j,n+i),(2*n+i,3*n+i,3*n+j,2*n+j)])
    obj = _mesh(coll,"DISPLAY | entrance-context retained protective jacket",vertices,faces,mat)
    for i,polygon in enumerate(obj.data.polygons):
        polygon.use_smooth = i%4 in (0,1)
    obj["scope"] = "Illustrative protective coating starts at 1.08 mm to frame the stripped window. Shown only in the entrance context, then hidden to expose the saved full field. No propagation parameter."
    for frame,hidden in [(1,False),(config["camera"]["intro_frames"],False),(config["camera"]["intro_frames"]+1,True),(scene.frame_end,True)]:
        obj.hide_render = hidden
        obj.keyframe_insert("hide_render",frame=frame)
    _set_linear_action(obj)
    return {"axial_interval_um":[z0*scale,z1*scale],"outer_diameter_um":outer*2*scale,"scope":obj["scope"]}


def _studio(scene, config, *, central_cut=False):
    coll = _collection(scene, "Presentation glass and studio")
    entrance_context = scene.get("film_id")=="01_reflection"
    stats = _solid_fiber(coll, config, _presentation_glass(config,entrance_context=entrance_context), central_cut=central_cut)
    matte = _material("Matte grey protective coating | display only", (.09,.105,.12), .58)
    floor = _material("Neutral blue grey studio floor", config["studio"]["floor_color_linear_rgb"], .65)
    _cube(coll,"Studio ground",(15,0,-2.04),(200,200,.08),floor)
    # Coating and supports stay at the far end, leaving the full observation
    # region accessible. Their dimensions are explicitly nonnumerical.
    bpy.ops.mesh.primitive_cylinder_add(vertices=256, radius=.95, depth=3.5, location=(31.25,0,0), rotation=(0,math.pi/2,0))
    coat = bpy.context.object
    coat.name = "DISPLAY | protective coating beyond observation window"
    for existing in list(coat.users_collection):
        existing.objects.unlink(coat)
    coll.objects.link(coat)
    coat.data.materials.append(matte)
    for poly in coat.data.polygons:
        poly.use_smooth = len(poly.vertices)==4
    _cube(coll,"Small matte rear support",(31.2,.3,-1.4),(1.2,1.4,.7),matte,.07)
    if entrance_context:
        stats["entrance_coating"] = _entrance_coating(coll,scene,config,matte)
    lights = _collection(scene,"Broad photographic lights")
    _area(lights,"Broad soft key",(9,-4,8),(9,0,0),850,18,10)
    _area(lights,"Broad neutral wrap",(15,5,7),(15,0,0),600,20,12)
    _area(lights,"Soft end fill",(-8,4,5),(0,0,0),400,9,9)
    return stats


def _packed_image(name, rgba, scratch: Path, interpolation: str) -> tuple[object,float]:
    height, width = rgba.shape[:2]
    image = bpy.data.images.new(name, width, height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set(np.asarray(rgba,dtype=np.float32).ravel())
    image.file_format = "OPEN_EXR"
    image.filepath_raw = str(scratch / (name.replace("|","_").replace("/","_")+".exr"))
    image.save()
    image.pack()
    image["display_interpolation"] = interpolation
    actual = np.empty(rgba.size,dtype=np.float32)
    image.pixels.foreach_get(actual)
    error = float(np.max(np.abs(actual-np.asarray(rgba,dtype=np.float32).ravel())))
    if error > 1e-6:
        raise RuntimeError("Packed scalar display texture changed its values")
    return image,error


def _text(coll, name, body, font, mat, size, location, rotation=None, parent=None):
    data = bpy.data.curves.new(name,"FONT")
    data.body = body
    data.font = font
    data.size = size
    data.offset = size*.012
    data.align_y = "BOTTOM_BASELINE"
    obj = _object(coll,name,data)
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
    obj.location = location
    if rotation is not None:
        obj.rotation_euler = rotation
    return obj


def _plane(coll, name, verts, uv, mat, parent=None):
    obj = _mesh(coll,name,verts,[(0,1,2,3)],mat)
    layer = obj.data.uv_layers.new(name="Physical sample registration")
    for loop in obj.data.loops:
        layer.data[loop.index].uv = uv[loop.vertex_index]
    if parent:
        obj.parent = parent
    return obj


def _hud(scene, config, camera, font, component, scratch):
    coll = _collection(scene,"Short Japanese interpretation labels")
    white = _camera_emission("HUD white",(.95,.98,1),2.2)
    muted = _camera_emission("HUD muted",(.55,.72,.86),1.6)
    w, h = scene.render.resolution_x, scene.render.resolution_y
    depth = .12
    view_width = camera.data.ortho_scale if camera.data.type=="ORTHO" else depth * camera.data.sensor_width/camera.data.lens
    view_height = view_width * h/w

    def label(x,y,body,pixels=24,mat=white):
        return _text(coll,body,body,font,mat,pixels*view_width/w*1.35,(-view_width/2+x*view_width/w,view_height/2-y*view_height/h,-depth),parent=camera)

    titles = {"reflection":"境界で戻る、光の分布", "comparison_off":"Kerr OFF / κ = 0", "comparison_on":"Kerr ON / 仮想 κ = 3e-4", "phase":"波のずれ（Kerrあり − なし）", "intensity_off":"明暗の分布 / OFF", "intensity_on":"明暗の分布 / ON"}
    label(36,50,titles[component],30)
    label(36,80,"定常場を、距離を進めて観察",18,muted)
    is_volume = component in ("reflection","comparison_off","comparison_on")
    if is_volume:
        label(36,h-46,"青 0.1  /  緑 0.3  /  橙 0.6：等強度面（I / 入射ピーク）",20)
        if component=="reflection":
            label(36,h-76,"白線：計算されたパワーの流れ  /  紫：観察断面",19)
    else:
        phase = component=="phase"
        limits = config["phase_range_rad"] if phase else config["intensity_range_relative_to_input_peak"]
        colors = config["phase_palette_linear_rgb"] if phase else config["intensity_palette_linear_rgb"]
        rgba = np.ones((12,512,4),dtype=np.float32)
        rgba[:,:,:3] = palette(np.linspace(*limits,512)[None,:],limits,colors)
        image,_ = _packed_image(component+"_common_colorbar",rgba,scratch,"Linear")
        mat = _camera_emission(component+" common colorbar",(1,1,1),image=image)
        x,y,bw,bh = 36,h-69,min(350,w*.37),12
        def xy(px,py):
            return (-view_width/2+px*view_width/w,view_height/2-py*view_height/h,-depth)
        _plane(coll,component+" common colorbar",[xy(x,y+bh),xy(x+bw,y+bh),xy(x+bw,y),xy(x,y)],[(0,0),(1,0),(1,1),(0,1)],mat,parent=camera)
        label(x,h-32,"−π                 0                 ＋π" if phase else "0                 1.5                 3",17)
        label(x+bw+25,h-48,"位相差（rad）・弱い場所は灰色" if phase else "相対強度 I / 入射ピーク",17)
    label(w-550,h-18,"50 µmコア・3 mm / 教育用の仮係数",14,muted)
    if camera.data.type=="ORTHO":
        # Native drivers keep captions at the same screen size and margin
        # during the film-01 context-to-closeup camera zoom.
        for obj in coll.objects:
            if obj.parent!=camera:
                continue
            for axis in (0,1):
                initial = float(obj.location[axis])
                for path,value in [("location",initial),("scale",1.)]:
                    driver = obj.driver_add(path,axis).driver
                    driver.expression = f"{value:.17g}*width/{view_width:.17g}"
                    var = driver.variables.new()
                    var.name = "width"
                    var.type = "SINGLE_PROP"
                    var.targets[0].id_type = "CAMERA"
                    var.targets[0].id = camera.data
                    var.targets[0].data_path = "ortho_scale"


def _distance_and_boundary_labels(scene, config, camera, font):
    coll = _collection(scene,"Distance and true circular-core boundary annotations")
    mat = _camera_emission("Annotation blue grey",(.48,.72,.88),1.8)
    radius = config["physical_display"]["core_diameter_um"]/(2*config["physical_display"]["um_per_world_unit"])
    angles = np.linspace(0,2*math.pi,257)
    _polycurve(coll,"ANNOTATION | 50 um core boundary at input",np.column_stack((np.zeros(len(angles)),radius*np.cos(angles),radius*np.sin(angles))),.0013,mat)
    guide = config["core_boundary_guide"]
    if guide["enabled"] and scene.get("component") in ("reflection","comparison_off","comparison_on"):
        rail_material = _camera_emission("Subtle core-boundary rail annotation",guide["color_linear_rgb"],guide["emission"])
        for index,points in enumerate(core_boundary_rails_um(config)):
            obj = _polycurve(coll,f"ANNOTATION | true core boundary rail {index}",physical_to_world(points,config["physical_display"]["um_per_world_unit"]),guide["radius_um"]/config["physical_display"]["um_per_world_unit"],rail_material)
            obj["core_boundary_sign"] = 1 if index==0 else -1
            obj["scope"] = "Display annotation at physical x=+/-25 um, y=0. Not a field contour, power-flow trajectory, or measured reflection path."
    offset_x = config["camera"]["offset_world"][0]
    view_width = camera.data.ortho_scale if camera.data.type=="ORTHO" else 2*math.tan(camera.data.angle_x/2)*np.linalg.norm(config["camera"]["offset_world"])
    label_size = 22*view_width/scene.render.resolution_x*1.35
    # Labels and leaders translate with the observed section, always attached
    # to +/-x, y=0 of the actual 25 um circular core, not to a invented turn.
    for sign,text in [(1,"＋x側のコア境界"),(-1,"−x側のコア境界")]:
        obj = _text(coll,text,text,font,mat,label_size,(0,sign*.265,.24),rotation=camera.rotation_euler)
        curve = _polycurve(coll,text+" leader",np.array([[0,sign*radius,0],[0,sign*.265,.21]]),.0014,mat)
        for item in [obj,curve]:
            driver = item.driver_add("location",0).driver
            driver.expression = "cx-({})".format(offset_x)
            var = driver.variables.new()
            var.name = "cx"
            var.type = "TRANSFORMS"
            var.targets[0].id = camera
            var.targets[0].transform_type = "LOC_X"
            var.targets[0].transform_space = "WORLD_SPACE"
            if scene.get("film_id")=="01_reflection":
                for frame,hidden in [(1,True),(config["camera"].get("intro_frames",0),True),(config["camera"].get("intro_frames",0)+1,False),(scene.frame_end,False)]:
                    item.hide_render = hidden
                    item.keyframe_insert("hide_render",frame=frame)
                _set_linear_action(item)
    # Film-03 lower panels share the top panel's observation distance. Their
    # world-distance ticks would project over the fixed screen color bars.
    # Keep the distance ruler only on phase, leaving both intensity bars clear.
    distance_ticks = [] if scene.get("component") in ("intensity_off","intensity_on") else np.arange(0,config["physical_display"]["length_um"]+1,500)
    for z in distance_ticks:
        x = z/config["physical_display"]["um_per_world_unit"]
        _polycurve(coll,f"Distance tick {z:g} um",np.array([[x,-.30,0],[x,-.36,0]]),.0015,mat)
        _text(coll,f"Distance {z:g} um",f"{z/1000:g} mm",font,mat,label_size*.77,(x,-.37,0),rotation=camera.rotation_euler)


def _set_linear_action(obj):
    if not obj.animation_data or not obj.animation_data.action:
        return
    action = obj.animation_data.action
    if hasattr(action,"fcurves"):
        curves = action.fcurves
    else:
        curves = []
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(obj.animation_data.action_slot)
                if bag:
                    curves.extend(bag.fcurves)
    for curve in curves:
        for point in curve.keyframe_points:
            point.interpolation = "CONSTANT" if curve.data_path=="hide_render" else "LINEAR"


def _cursor(scene, config, case, frame_count):
    """Baked observation annotations at actual saved same-z flow samples."""
    if not config["observation_cursor"]["enabled"]:
        return {"enabled":False}
    coll = _collection(scene,"Observation section, NOT photons or a pulse")
    c = config["observation_cursor"]
    scale = config["physical_display"]["um_per_world_unit"]
    mat = _camera_emission("Purple observation annotations",c["color_linear_rgb"],c["emission"])
    angles = np.linspace(0,2*math.pi,129)
    r = c["core_ring_radius_um"]/scale
    ring = _polycurve(coll,"OBSERVATION | same-z cross-section ring",np.column_stack((np.zeros(len(angles)),r*np.cos(angles),r*np.sin(angles))),c["ring_line_radius_um"]/scale,mat)
    markers = []
    half = c["cross_half_size_um"]/scale
    for seed in range(config["flow"]["expected_seed_count"]):
        data = bpy.data.curves.new(f"OBSERVATION cross seed {seed:02d}","CURVE")
        data.dimensions = "3D"
        data.bevel_depth = c["cross_radius_um"]/scale
        data.bevel_resolution = 0
        for points in [[(0,-half,0),(0,half,0)],[(0,0,-half),(0,0,half)]]:
            spline = data.splines.new("POLY")
            spline.points.add(1)
            for point,value in zip(spline.points,points):
                point.co = (*value,1)
        obj = _object(coll,f"OBSERVATION | saved flow intersection {seed:02d}",data)
        obj.data.materials.append(mat)
        obj["scope"] = "Observation cross at saved flow sample; not a photon, moving pulse, or new field."
        markers.append(obj)
    frame_map = []
    flows = case["flows"]
    for frame in range(1,frame_count+1):
        target = distance_at_frame(frame,frame_count,config,"01_reflection")
        index = nearest_flow_sample(flows["z_um"],target)
        observed = float(flows["z_um"][index])
        ring.location.x = observed/scale
        ring.keyframe_insert("location",frame=frame)
        active = []
        for seed,obj in enumerate(markers):
            valid = bool(flows["valid"][seed,index])
            obj.hide_render = not valid
            obj.keyframe_insert("hide_render",frame=frame)
            if valid:
                obj.location = physical_to_world(flows["position_um"][seed,index],scale)
                obj.keyframe_insert("location",frame=frame)
                active.append(seed)
        frame_map.append({"frame":frame,"camera_observation_z_um":target,"saved_flow_sample":index,"observed_z_um":observed,"valid_seed_ids":active})
    for obj in [ring,*markers]:
        _set_linear_action(obj)
    return {"enabled":True,"encoding":"Purple cross-section ruler and 25 saved-flow intersection crosses, not photons or pulse fronts","sample_selection":"Nearest stored common flow z; hide after valid-prefix termination","frames":frame_map}


def _add_volume(scene, config, case, input_records):
    coll = _collection(scene,"Computed 3D iso-intensity surfaces")
    errors, records = [], []
    scale = config["physical_display"]["um_per_world_unit"]
    for style in config["isosurfaces"]:
        source = case["meshes"][style["level"]]
        world = physical_to_world(source["vertices_um"],scale)
        mat = _camera_emission(f"Iso intensity {style['level']:g}",style["color_linear_rgb"],config["isosurface_emission"],style["opacity"],display_shading=config.get("isosurface_display_shading"))
        # Emission/straight transparency does not consume a lighting normal.
        # Validate the provided physical normals, but do not duplicate them
        # at every triangle corner in the portable native file.
        normals_to_world(source["normals_xyz"])
        obj = _mesh(coll,f"ISO | {case['name']} | I={style['level']:g}",world,source["faces"],mat)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj["source_file"] = source["filename"]
        obj["source_case"] = case["name"]
        obj["source_sha256"] = input_records[source["filename"]]["sha256"]
        obj["level_relative_to_input_peak"] = style["level"]
        actual = np.empty(world.size,dtype=np.float32)
        obj.data.vertices.foreach_get("co",actual)
        error = float(np.max(np.abs(actual.reshape(world.shape)-world))*scale)
        errors.append(error)
        records.append({"level":style["level"],"vertices":len(world),"faces":len(source["faces"]),"input_file":source["filename"],"max_coordinate_error_um":error,"provided_normals_validated":True,"native_custom_loop_normals":"Omitted for unlit emissive display; original input normals and all vertices/triangles retained"})
    if max(errors)>1e-3:
        raise RuntimeError("Numerical surface coordinates changed in Blender")
    return records


def _add_flows(scene, config, case, input_records):
    coll = _collection(scene,"All 25 saved power-flow seeds")
    c = config["flow"]
    scale = config["physical_display"]["um_per_world_unit"]
    mat = _camera_emission("Saved power-flow lines, not photons",c["color_linear_rgb"],c["emission"])
    records = []
    for seed,ids in enumerate(case["prefixes"]):
        world = physical_to_world(case["flows"]["position_um"][seed,ids],scale)
        obj = _polycurve(coll,f"FLOW | {case['name']} | seed {seed:02d}",world,c["radius_um"]/scale,mat)
        obj["seed"] = seed
        obj["quantiles"] = case["flows"]["seed_quantiles"][seed].tolist()
        obj["scope"] = "Source J/I power-flow polyline; never a photon path; terminates at first invalid sample."
        obj["source_sha256"] = input_records[case["flows_file"]]["sha256"]
        error = 0.
        if len(world)>=2:
            coords = np.empty(len(world)*4,dtype=np.float32)
            obj.data.splines[0].points.foreach_get("co",coords)
            error = float(np.max(np.abs(coords.reshape(-1,4)[:,:3]-world))*scale)
            if error>1e-3:
                raise RuntimeError("Saved power-flow position changed in Blender")
        records.append({"seed":seed,"valid_prefix_points":int(case["flows"]["valid"][seed].sum()),"display_points":len(ids),"last_source_index":int(ids[-1]) if len(ids) else None,"max_coordinate_error_um":error})
    return records


def _add_section(scene, config, case, all_cases, component, scratch, input_records):
    coll = _collection(scene,"Actual central y=0 numerical section")
    source = case["sections"]
    is_phase = component=="phase"
    if is_phase:
        phase, valid = phase_difference(all_cases["linear"]["sections"]["central_psi"],all_cases["educational_kerr"]["sections"]["central_psi"],config["phase_joint_intensity_floor"])
        rgba = phase_rgba(phase,valid,config).transpose(1,0,2)
        interpolation = "Closest"
    else:
        rgba = intensity_rgba(np.abs(source["central_psi"])**2,config).transpose(1,0,2)
        interpolation = "Linear"
    image,error = _packed_image(component+"_actual_central_y0",rgba,scratch,interpolation)
    image["source_sha256"] = input_records[case["sections_file"]]["sha256"]
    image["scope"] = "Registered central y=0 complex section; fixed input-peak intensity scale or joint-masked principal phase difference."
    mat = _camera_emission(component+" numerical section",(1,1,1),image=image,interpolation=interpolation)
    x,z = source["x_um"],source["z_um"]
    dx = float(x[1]-x[0])
    scale = config["physical_display"]["um_per_world_unit"]
    # z endpoints map to first/last sample centres; x edges map to cell edges.
    verts = np.array([[z[0],x[0]-dx/2,0],[z[-1],x[0]-dx/2,0],[z[-1],x[-1]+dx/2,0],[z[0],x[-1]+dx/2,0]])/scale
    u0,u1 = .5/len(z),(len(z)-.5)/len(z)
    obj = _plane(coll,component+" | central y=0",verts,[(u0,0),(u1,0),(u1,1),(u0,1)],mat)
    obj["scope"] = "Actual longitudinal x-z section, physical y=0. Not a texture substituted for 3D light distribution."
    obj["source_file"] = case["sections_file"]
    return {"physical_y_um":0,"texture_shape_x_z":list(rgba.shape[:2]),"max_packed_rgba_error":error,"interpolation":interpolation,"fixed_limits":config["phase_range_rad"] if is_phase else config["intensity_range_relative_to_input_peak"],"joint_phase_mask":is_phase}


def _component_layout(film_id, component, config):
    w,h = config["width"],config["height"]
    if film_id=="02_kerr_comparison":
        return w,h//2,(0,0 if component=="comparison_off" else h//2)
    if film_id=="03_phase_change":
        return (w,600,(0,0)) if component=="phase" else (w//2,h-600,(0 if component=="intensity_off" else w//2,600))
    return w,h,(0,0)


def _build_component(film_id, component, config, cases, bundle, font_path, scratch):
    from mathutils import Matrix,Vector
    scene = bpy.data.scenes.new(film_id+" | "+component)
    bpy.context.window.scene = scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = config["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = config["adaptive_threshold"]
    scene.cycles.adaptive_min_samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPTIX"
    scene.cycles.max_bounces = 10
    scene.cycles.transparent_max_bounces = 24
    scene.cycles.transmission_bounces = 0
    scene.cycles.glossy_bounces = 3
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.resolution_x,scene.render.resolution_y,_ = _component_layout(film_id,component,config)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.fps = config["fps"]
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.frame_start = 1
    scene.frame_end = round(config["duration_seconds"]*config["fps"])
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = config["physical_display"]["um_per_world_unit"]*1e-6
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium Low Contrast"
    scene.view_settings.exposure = config["studio"]["exposure"]
    scene.world = bpy.data.worlds.new(component+" blue grey world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (*config["studio"]["world_color_linear_rgb"],1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = config["studio"]["world_strength"]
    scene["scope"] = "Computed stationary CW field; camera and observation ruler scan distance, not physical pulse time. Educational hypothetical kappa, not SMF watts."
    scene["source_manifest_sha256"] = bundle["records"]["manifest.json"]["sha256"]
    scene["physical_mapping"] = "world=(z,x,y)/100, all XYZ same physical scale"
    scene["component"] = component
    scene["film_id"] = film_id
    is_volume = component in ("reflection","comparison_off","comparison_on")
    geometry = _studio(scene,config,central_cut=not is_volume)
    case_name = "educational_kerr" if component in ("comparison_on","phase","intensity_on") else "linear"
    case = cases[case_name]
    record = {"component":component,"scene_name":scene.name,"case":case_name,"geometry":geometry}
    if is_volume:
        record["meshes"] = _add_volume(scene,config,case,bundle["records"])
        if component=="reflection":
            record["flows"] = _add_flows(scene,config,case,bundle["records"])
    else:
        record["section"] = _add_section(scene,config,case,cases,component,scratch,bundle["records"])
    coll = _collection(scene,"Camera observation sweep")
    cd = bpy.data.cameras.new(component+" macro lens")
    cd.type = config["camera"].get("projection","PERSP")
    if cd.type=="ORTHO":
        cd.ortho_scale = config["camera"]["ortho_height_world"]*scene.render.resolution_x/scene.render.resolution_y
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36
    cd.lens = config["camera"]["lens_mm"]
    cd.clip_start,cd.clip_end = .02,250
    cd.dof.use_dof = False  # Data surfaces and slices remain readable throughout.
    camera = _object(coll,component+" camera",cd)
    scale = config["physical_display"]["um_per_world_unit"]
    target = Vector((config["camera"]["target_z_um"][0]/scale,0,0))
    offset = Vector(config["camera"]["offset_world"])
    camera.location = target+offset
    # The dominant fold direction is physical x = world Y. Project the
    # longitudinal axis horizontally and preserve that fold vertically.
    back = offset.normalized()
    longitudinal = Vector((1,0,0))
    right = (longitudinal-back*longitudinal.dot(back)).normalized()
    up = back.cross(right).normalized()
    camera.rotation_euler = Matrix((right,up,back)).transposed().to_euler()
    close_rotation = camera.rotation_euler.copy()
    close_width = cd.ortho_scale
    intro = config["camera"].get("intro_frames",0) if film_id=="01_reflection" else 0
    if intro:
        intro_offset = Vector(config["camera"]["intro_offset_world"])
        intro_target = Vector((config["camera"]["intro_target_z_um"]/scale,0,0))
        intro_rotation = (-intro_offset).to_track_quat("-Z","Y").to_euler()
        for frame in [1,config["camera"]["intro_hold_frames"]]:
            camera.location = intro_target+intro_offset
            camera.rotation_euler = intro_rotation
            camera.keyframe_insert("location",frame=frame)
            camera.keyframe_insert("rotation_euler",frame=frame)
            cd.ortho_scale = config["camera"]["intro_width_world"]
            cd.keyframe_insert("ortho_scale",frame=frame)
        camera.location = target+offset
        camera.rotation_euler = close_rotation
        camera.keyframe_insert("rotation_euler",frame=intro+1)
        cd.ortho_scale = close_width
        cd.keyframe_insert("ortho_scale",frame=intro+1)
    for frame in ([intro+1,scene.frame_end] if intro else [1,scene.frame_end]):
        camera.location = Vector((distance_at_frame(frame,scene.frame_end,config,film_id)/scale,0,0))+offset
        camera.keyframe_insert("location",frame=frame)
    if intro:
        # Independent Euler/position interpolation misses the target as the
        # view narrows. Bake only the affected integer-frame rotations from
        # an explicit target-facing basis; approved endpoint poses remain.
        previous = intro_rotation
        for frame in range(config["camera"]["intro_hold_frames"]+1,intro+1):
            camera.rotation_euler = Matrix(intro_camera_basis(frame,config).tolist()).to_euler("XYZ",previous)
            camera.keyframe_insert("rotation_euler",frame=frame)
            previous = camera.rotation_euler.copy()
    _set_linear_action(camera)
    _set_linear_action(cd)
    camera.rotation_euler = close_rotation
    cd.ortho_scale = close_width
    scene.camera = camera
    font = bpy.data.fonts.load(str(font_path))
    _hud(scene,config,camera,font,component,scratch)
    _distance_and_boundary_labels(scene,config,camera,font)
    if component=="reflection":
        record["observation_cursor"] = _cursor(scene,config,case,scene.frame_end)
    record["camera"] = {"projection":cd.type,"ortho_width_world":close_width if cd.type=="ORTHO" else None,"offset_world":config["camera"]["offset_world"],"intro_frames":intro,"intro_target_z_um":config["camera"].get("intro_target_z_um") if intro else None,"intro_offset_world":config["camera"].get("intro_offset_world") if intro else None,"intro_width_world":config["camera"].get("intro_width_world") if intro else None,"intro_scope":"Entrance face and first approximately 1 mm of stripped circular fiber, then camera-only move into computed reflection. Presentation surface sheen/coating changes do not animate the field." if intro else None,"intro_rotation_interpolation":"Target-facing rotation keys at every integer frame 13-24; endpoint poses and camera positions/scales unchanged" if intro else None,"lens_mm_virtual":cd.lens,"sensor_fit":"HORIZONTAL","screen_right_world":list(right),"screen_up_world":list(up),"longitudinal_foreshortening":float(math.sqrt(1-back.x**2)),"depth_of_field":False,"target_z_um":config["camera"]["target_z_um"],"frame_mapping":"Hold initial distance through film-01 context frames, then linear distance observation; other films linear throughout","time_scope":"Observation scheduling only; no physical time units"}
    return scene,record


def _devices() -> dict:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.refresh_devices()
    devices = []
    for device in preferences.devices:
        device.use = device.type=="OPTIX"
        devices.append({"name":device.name,"type":device.type,"id":device.id,"enabled":device.use})
    if not any(device["enabled"] for device in devices):
        raise RuntimeError("OptiX GPU unavailable; this renderer does not silently fall back")
    return {"blender":bpy.app.version_string,"build_hash":bpy.app.build_hash.decode(),"device":"GPU/OPTIX","devices":devices}


def _poster_frame(config: dict, film_id: str | None = None) -> int:
    count = round(config["duration_seconds"]*config["fps"])
    positions = [distance_at_frame(frame,count,config,film_id) for frame in range(1,count+1)]
    desired = config["camera"].get("poster_z_um_per_film",{}).get(film_id,config["camera"]["poster_z_um"])
    return int(np.argmin(abs(np.asarray(positions)-desired)))+1


def _render_frame(scene, frame: int, path: Path, samples: int, percentage: int) -> dict:
    import time
    bpy.context.window.scene = scene
    scene.frame_set(frame)
    _verify_runtime_device(scene)
    _camera_row(scene,frame,config=json.loads(bpy.data.texts['Fixed render configuration'].as_string()))
    scene.cycles.samples = samples
    scene.render.resolution_percentage = percentage
    scene.render.filepath = str(path)
    started = time.perf_counter()
    bpy.ops.render.render(scene=scene.name,write_still=True)
    _verify_runtime_device(scene)
    camera = _camera_row(scene,frame,config=json.loads(bpy.data.texts['Fixed render configuration'].as_string()))
    return {"frame":frame,"file":path.name,"sha256":sha256(path),"render_seconds":time.perf_counter()-started,"width":round(scene.render.resolution_x*percentage/100),"height":round(scene.render.resolution_y*percentage/100),"samples":samples,"camera_target_world":camera["target_world"],"camera_observation_z_um":distance_at_frame(frame,scene.frame_end,json.loads(bpy.data.texts['Fixed render configuration'].as_string()),scene['film_id']),"active_scene_and_camera_verified":True,"runtime_device":"GPU/OPTIX"}


def _verify_runtime_device(scene):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if scene.cycles.device!="GPU" or prefs.compute_device_type!="OPTIX" or not any(d.type=="OPTIX" and d.use for d in prefs.devices) or any(d.type!="OPTIX" and d.use for d in prefs.devices):
        raise RuntimeError("Cycles runtime device preferences changed; CPU/device fallback is forbidden")


def _camera_row(scene, frame, config, *, allow_old_intro_failure=False):
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    camera = scene.camera
    target = camera_target_at_frame(frame,scene.frame_end,config,scene["film_id"])
    projected = world_to_camera_view(scene,camera,Vector(target.tolist()))
    error = max(abs(projected.x-.5),abs(projected.y-.5))
    known = allow_old_intro_failure and scene["film_id"]=="01_reflection" and 13<=frame<=24
    intro = scene["film_id"]=="01_reflection" and frame<=config["camera"]["intro_frames"]
    if intro:
        t = max(0.,(frame-config["camera"]["intro_hold_frames"])/(config["camera"]["intro_frames"]+1-config["camera"]["intro_hold_frames"]))
        offset = (1-t)*np.asarray(config["camera"]["intro_offset_world"])+t*np.asarray(config["camera"]["offset_world"])
        basis = intro_camera_basis(max(frame,config["camera"]["intro_hold_frames"]),config)
    else:
        offset = np.asarray(config["camera"]["offset_world"])
        back = offset/np.linalg.norm(offset)
        right = np.array([1.,0.,0.])-back*back[0]
        right /= np.linalg.norm(right)
        basis = np.column_stack((right,np.cross(back,right),back))
    close_width = config["camera"]["ortho_height_world"]*scene.render.resolution_x/scene.render.resolution_y
    width = (1-t)*config["camera"]["intro_width_world"]+t*close_width if intro else close_width
    actual_basis = np.asarray(camera.matrix_world.to_3x3())
    if scene.frame_current!=frame or bpy.context.scene!=scene or np.max(abs(np.asarray(camera.location)-(target+offset)))>1e-4 or abs(camera.data.ortho_scale-width)>1e-5:
        raise RuntimeError("Active component camera position/frame/scale differs from the declared observation")
    if not known and (error>1e-3 or projected.z<=0 or np.max(abs(actual_basis-basis))>1e-5):
        raise RuntimeError(f"Camera target or pose leaves the registered frame: {scene.name}, frame {frame}, projection {list(projected)}")
    return {"frame":frame,"matrix_world":[value for row in camera.matrix_world for value in row],"location_world":list(camera.location),"ortho_scale":camera.data.ortho_scale,"target_world":target.tolist(),"target_projection":list(projected),"center_error":error}


def _camera_trace(scene, config, *, allow_old_intro_failure=False):
    original = scene.frame_current
    bpy.context.window.scene = scene
    rows = []
    try:
        for frame in range(1,scene.frame_end+1):
            scene.frame_set(frame)
            rows.append(_camera_row(scene,frame,config,allow_old_intro_failure=allow_old_intro_failure))
    finally:
        scene.frame_set(original)
    return rows


def _native_verify(cases, bundle, config, *, allow_old_intro_failure=False) -> list[dict]:
    """Repeatable saved-scene fidelity checks; not a physical-model validator."""
    scale = config["physical_display"]["um_per_world_unit"]
    records = []
    stored_config = bpy.data.texts.get("Fixed render configuration")
    if stored_config is None or json.loads(stored_config.as_string())!=config:
        raise RuntimeError("Saved native scene render configuration differs")
    for scene in bpy.data.scenes:
        component = scene.get("component")
        if not component:
            continue
        if scene.get("source_manifest_sha256") != bundle["records"]["manifest.json"]["sha256"]:
            raise RuntimeError("Saved scene input-manifest hash differs")
        for obj in scene.objects:
            if obj.name.startswith("ISO |"):
                filename = obj["source_file"]
                case_name = obj["source_case"]
                source = cases[case_name]["meshes"][obj["level_relative_to_input_peak"]]
                expected = physical_to_world(source["vertices_um"],scale)
                actual = np.empty(expected.size,dtype=np.float32)
                obj.data.vertices.foreach_get("co",actual)
                error = float(np.max(abs(actual.reshape(expected.shape)-expected))*scale)
                if error>1e-3 or len(obj.data.polygons)!=len(source["faces"]):
                    raise RuntimeError("Saved mesh geometry no longer matches numerical artifact")
                records.append({"object":obj.name,"source":filename,"max_coordinate_error_um":error})
            elif obj.name.startswith("FLOW |"):
                seed = obj["seed"]
                source = cases["linear"]
                ids = source["prefixes"][seed]
                expected = physical_to_world(source["flows"]["position_um"][seed,ids],scale)
                if len(expected)>=2:
                    actual = np.empty(len(expected)*4,dtype=np.float32)
                    obj.data.splines[0].points.foreach_get("co",actual)
                    error = float(np.max(abs(actual.reshape(-1,4)[:,:3]-expected))*scale)
                    if error>1e-3:
                        raise RuntimeError("Saved power flow no longer matches its valid prefix")
                else:
                    error = 0.
                records.append({"object":obj.name,"seed":seed,"valid_prefix_points":len(expected),"max_coordinate_error_um":error})
            elif obj.name.startswith("ANNOTATION | true core boundary rail"):
                sign = obj["core_boundary_sign"]
                expected = physical_to_world(core_boundary_rails_um(config)[0 if sign==1 else 1],scale)
                actual = np.asarray([point.co[:3] for point in obj.data.splines[0].points])
                error = float(np.max(abs(actual-expected))*scale)
                if error>1e-3:
                    raise RuntimeError("Boundary annotation left the true +/-x circular-core boundary")
                records.append({"annotation":obj.name,"physical_x_um":sign*config["physical_display"]["core_diameter_um"]/2,"physical_y_um":0,"max_coordinate_error_um":error})
        if component in ("phase","intensity_off","intensity_on"):
            if component in ("intensity_off","intensity_on") and any(obj.name.startswith("Distance") for obj in scene.objects):
                raise RuntimeError("Lower intensity panel distance ticks overlap the fixed-screen colorbar")
            image = bpy.data.images[component+"_actual_central_y0"]
            if component=="phase":
                phase,valid = phase_difference(cases["linear"]["sections"]["central_psi"],cases["educational_kerr"]["sections"]["central_psi"],config["phase_joint_intensity_floor"])
                expected = phase_rgba(phase,valid,config).transpose(1,0,2)
            else:
                name = "linear" if component=="intensity_off" else "educational_kerr"
                expected = intensity_rgba(abs(cases[name]["sections"]["central_psi"])**2,config).transpose(1,0,2)
            actual = np.empty(expected.size,dtype=np.float32)
            image.pixels.foreach_get(actual)
            error = float(np.max(abs(actual-expected.ravel())))
            if error>1e-6 or image.packed_file is None:
                raise RuntimeError("Saved numerical display texture changed or is unpacked")
            records.append({"image":image.name,"max_rgba_error":error,"packed":True})
        if not any(font.packed_file is not None for font in bpy.data.fonts if font.name!="Bfont"):
            raise RuntimeError("Japanese font is not packed in the native scene")
        trace = _camera_trace(scene,config,allow_old_intro_failure=allow_old_intro_failure)
        checked = [row for row in trace if not (allow_old_intro_failure and scene["film_id"]=="01_reflection" and 13<=row["frame"]<=24)]
        records.append({"camera_scene":scene.name,"projection_checked_frames":len(checked),"allowed_old_failure_frames":[13,24] if len(checked)!=len(trace) else None,"max_screen_center_error":max(row["center_error"] for row in checked),"targets_in_view":True})
        if component=="reflection" and config["observation_cursor"]["enabled"]:
            original_frame = scene.frame_current
            flow = cases["linear"]["flows"]
            for frame in [1,_poster_frame(config,"01_reflection"),scene.frame_end]:
                scene.frame_set(frame)
                index = nearest_flow_sample(flow["z_um"],distance_at_frame(frame,scene.frame_end,config,"01_reflection"))
                for seed in range(config["flow"]["expected_seed_count"]):
                    obj = scene.objects[f"OBSERVATION | saved flow intersection {seed:02d}"]
                    valid = bool(flow["valid"][seed,index])
                    if bool(obj.hide_render)==valid:
                        raise RuntimeError("Observation cross does not stop with the saved seed mask")
                    if valid:
                        expected = physical_to_world(flow["position_um"][seed,index],scale)
                        error = float(np.max(abs(np.asarray(obj.location)-expected))*scale)
                        if error>1e-3:
                            raise RuntimeError("Observation cross moved away from the saved same-z flow point")
                records.append({"observation_frame":frame,"source_flow_z_um":float(flow["z_um"][index]),"all_25_masked_observation_positions_match":True})
            scene.frame_set(original_frame)
    return records


def _scene_field_fingerprint(scene):
    """Camera-independent numerical/display geometry and common materials."""
    scene.frame_set(1)
    rows = {}
    for obj in scene.objects:
        if not obj.name.startswith(("ISO |","FLOW |","DISPLAY |","SECTION |","ANNOTATION |","OBSERVATION |")):
            continue
        digest = hashlib.sha256()
        if obj.type=="MESH":
            co = np.empty(len(obj.data.vertices)*3,dtype=np.float32)
            indices = np.empty(len(obj.data.loops),dtype=np.int32)
            obj.data.vertices.foreach_get("co",co)
            obj.data.loops.foreach_get("vertex_index",indices)
            digest.update(co.tobytes());digest.update(indices.tobytes())
            smooth = np.empty(len(obj.data.polygons),dtype=np.bool_)
            obj.data.polygons.foreach_get("use_smooth",smooth)
            digest.update(smooth.tobytes())
        elif obj.type=="CURVE":
            for spline in obj.data.splines:
                co = np.empty(len(spline.points)*4,dtype=np.float32)
                spline.points.foreach_get("co",co)
                digest.update(co.tobytes())
            digest.update(repr((obj.data.bevel_depth,obj.data.bevel_resolution,obj.data.use_fill_caps)).encode())
        materials = []
        for material in obj.data.materials if hasattr(obj.data,"materials") else []:
            nodes = []
            for node in material.node_tree.nodes:
                inputs = []
                for socket in node.inputs:
                    if hasattr(socket,"default_value"):
                        value = socket.default_value
                        try: value = list(value)
                        except TypeError: pass
                        inputs.append((socket.name,value))
                nodes.append((node.bl_idname,node.name,inputs))
            materials.append(nodes)
        rows[obj.name] = {"geometry_sha256":digest.hexdigest(),"materials":materials,"local_transform":list(obj.location)+list(obj.rotation_euler)+list(obj.scale),"hidden_at_frame1":obj.hide_render}
    return hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()


def _inside_repair(request, bundle, result):
    config = request["config"]
    scratch = Path(request["paths"]["scratch"])
    repair = request["repair"]
    result["environment"] = _devices()
    if result["environment"]!=repair["original_environment"]:
        raise RuntimeError("Blender build/device environment differs from the original native run")
    if request["repair"]["phase"]=="finish":
        for film_id,components in CAMERA_REPAIR_COMPONENTS.items():
            bpy.ops.wm.open_mainfile(filepath=str(Path(request["paths"]["output"])/(film_id+".blend")))
            result["native_verification"][film_id] = _native_verify(bundle["cases"],bundle,config)
            result["repair_frames"] = result.get("repair_frames",{})
            for component,frames in components.items():
                scene = bpy.data.scenes[film_id+" | "+component]
                directory = scratch/film_id/component
                result["repair_frames"].setdefault(film_id,{})[component] = {}
                for frame in frames:
                    path = directory/f"frame_{frame:04d}.png"
                    if not path.exists():
                        record = _render_frame(scene,frame,path,config["samples"],100)
                        print("REPAIR_COMPONENT_FRAME="+json.dumps({"film":film_id,"component":component,**record}),flush=True)
                    else:
                        record = repair["preflight_records"][film_id][component][str(frame)]
                        if sha256(path)!=record["sha256"]:
                            raise RuntimeError("Preflight frame hash differs; no cross-run frame reuse")
                    result["repair_frames"][film_id][component][str(frame)] = record
            dump_json(scratch/"native_repair_result.json",result)
        return
    old_traces,old_fields = {},{}
    for film_id in FILMS:
        bpy.ops.wm.open_mainfile(filepath=repair["original_scenes"][film_id])
        _native_verify(bundle["cases"],bundle,config,allow_old_intro_failure=True)
        old_traces[film_id],old_fields[film_id] = {},{}
        for scene in bpy.data.scenes:
            if scene.get("film_id")==film_id:
                component = scene["component"]
                bpy.context.window.scene = scene
                old_traces[film_id][component] = _camera_trace(scene,config,allow_old_intro_failure=True)
                old_fields[film_id][component] = _scene_field_fingerprint(scene)
    output = Path(request["paths"]["output"])
    for film_id in FILMS:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if _devices()!=repair["original_environment"]:
            raise RuntimeError("Factory reset changed the required OptiX rendering environment")
        start = bpy.context.scene
        assets = scratch/film_id/"assets"
        assets.mkdir(parents=True,exist_ok=True)
        components,scenes = [],{}
        for component in FILMS[film_id]["components"]:
            scene,info = _build_component(film_id,component,config,bundle["cases"],bundle,Path(request["paths"]["font"]),assets)
            scenes[component] = scene;components.append(info)
        bpy.data.scenes.remove(start)
        for name,value in [("Numerical input manifest",bundle["manifest"]),("Fixed render configuration",config),("Source provenance",request["provenance"])]:
            bpy.data.texts.new(name).write(json.dumps(value,ensure_ascii=False,indent=2))
        poster_frame = _poster_frame(config,film_id)
        for scene in scenes.values():
            bpy.context.window.scene = scene
            scene.frame_set(poster_frame)
            scene.cycles.samples = config["samples"]
            scene.render.filepath = "//"+FILMS[film_id]["poster"]
        bpy.context.window.scene = scenes[FILMS[film_id]["components"][0]]
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.file.pack_all()
        path = output/(film_id+".blend")
        bpy.ops.wm.save_as_mainfile(filepath=str(path),compress=True)
        if path.stat().st_size>=100000000:
            raise RuntimeError("Native delivery exceeds the 100 MB limit")
        bpy.ops.wm.open_mainfile(filepath=str(path))
        result["native_verification"][film_id] = _native_verify(bundle["cases"],bundle,config)
        record = {"id":film_id,"scene":path.name,"scene_sha256":sha256(path),"components":components,"poster_components":{},"camera_checks":{},"preflight_frames":{}}
        new_traces = {}
        for component in FILMS[film_id]["components"]:
            scene = bpy.data.scenes[film_id+" | "+component]
            bpy.context.window.scene = scene
            trace = _camera_trace(scene,config)
            new_traces[component] = trace
            if _scene_field_fingerprint(scene)!=old_fields[film_id][component]:
                raise RuntimeError("A non-camera field/display geometry or material changed during camera repair")
            old = old_traces[film_id][component]
            changed = INTRO_REPAIR_FRAMES if film_id=="01_reflection" else ()
            for previous,current in zip(old,trace):
                if previous["location_world"]!=current["location_world"] or previous["ortho_scale"]!=current["ortho_scale"]:
                    raise RuntimeError("Camera repair changed a position or scale")
                if current["frame"] not in changed and previous["matrix_world"]!=current["matrix_world"]:
                    raise RuntimeError("Camera outside the permitted intro-rotation interval changed")
            record["camera_checks"][component] = {"frames":144,"max_screen_center_error":max(row["center_error"] for row in trace),"positions_scales_and_unaffected_rotations_identical":True,"field_and_material_fingerprint_sha256":old_fields[film_id][component]}
            path = scratch/film_id/("poster_"+component+".png")
            record["poster_components"][component] = _render_frame(scene,poster_frame,path,config["poster_samples"],100)
            if component in CAMERA_REPAIR_COMPONENTS[film_id]:
                directory = scratch/film_id/component
                directory.mkdir(exist_ok=True)
                record["preflight_frames"][component] = {}
                for frame in CAMERA_REPAIR_SAMPLES[component]:
                    path = directory/f"frame_{frame:04d}.png"
                    record["preflight_frames"][component][str(frame)] = _render_frame(scene,frame,path,config["samples"],100)
        dump_json(scratch/film_id/"camera_traces.json",new_traces)
        first = new_traces[FILMS[film_id]["components"][0]]
        for trace in new_traces.values():
            if any(a["matrix_world"]!=b["matrix_world"] or a["target_world"]!=b["target_world"] for a,b in zip(first,trace)):
                raise RuntimeError("Comparison components do not observe the same distance and pose")
        record["all_panel_distances_and_poses_match"] = True
        result["films"].append(record)
        dump_json(scratch/"native_repair_result.json",result)


def inside_blender(request_path: Path) -> None:
    global bpy
    import bpy
    request = json.loads(request_path.read_text(encoding="utf-8"))
    config = request["config"]
    data_dir = Path(request["paths"]["data_dir"])
    output = Path(request["paths"]["output"])
    scratch = Path(request["paths"]["scratch"])
    bundle = verify_inputs(data_dir,config)
    cases = bundle["cases"]
    result = {"environment":{},"films":[],"mode":request["mode"],"native_verification":{}}
    if request["mode"]=="repair-cameras":
        _inside_repair(request,bundle,result)
        dump_json(scratch/"native_repair_result.json",result)
        return
    if request["mode"]=="verify-scene":
        for film_id in request["films"]:
            bpy.ops.wm.open_mainfile(filepath=str(output/(film_id+".blend")))
            result["native_verification"][film_id] = _native_verify(cases,bundle,config)
        dump_json(scratch/"native_result.json",result)
        return
    for film_id in request["films"]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        result["environment"] = _devices()
        start_scene = bpy.context.scene
        assets = scratch/film_id/"assets"
        assets.mkdir(parents=True,exist_ok=True)
        scenes,components = {},[]
        for component in FILMS[film_id]["components"]:
            scene,record = _build_component(film_id,component,config,cases,bundle,Path(request["paths"]["font"]),assets)
            scenes[component] = scene
            components.append(record)
        bpy.data.scenes.remove(start_scene)
        poster_frame = request.get("poster_frame") or _poster_frame(config,film_id)
        for scene in scenes.values():
            bpy.context.window.scene = scene
            scene.frame_set(poster_frame)
            scene.cycles.samples = config["samples"]
            scene.render.resolution_percentage = 100
            scene.render.filepath = "//"+FILMS[film_id]["poster"]
        bpy.context.window.scene = scenes[FILMS[film_id]["components"][0]]
        for name,text in [("Numerical input manifest",json.dumps(bundle["manifest"],ensure_ascii=False,indent=2)),("Fixed render configuration",json.dumps(config,ensure_ascii=False,indent=2)),("Source provenance",json.dumps(request["provenance"],ensure_ascii=False,indent=2))]:
            block = bpy.data.texts.new(name)
            block.write(text)
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.file.pack_all()
        scene_path = output/(film_id+".blend")
        bpy.ops.wm.save_as_mainfile(filepath=str(scene_path),compress=True)
        record = {"id":film_id,"poster_frame":poster_frame,"poster_observation_z_um":distance_at_frame(poster_frame,round(config["duration_seconds"]*config["fps"]),config,film_id),"components":components,"scene":scene_path.name,"scene_sha256":sha256(scene_path),"poster_components":{},"frames":{},"proof_frames":{}}
        # Reopen the delivered file; check packed geometry and textures rather
        # than treating only the pre-save arrays as verification evidence.
        bpy.ops.wm.open_mainfile(filepath=str(scene_path))
        result["native_verification"][film_id] = _native_verify(cases,bundle,config)
        scenes = {component:bpy.data.scenes[film_id+" | "+component] for component in FILMS[film_id]["components"]}
        if request["mode"] in ("posters","proof","render"):
            for component,scene in scenes.items():
                path = scratch/film_id/("poster_"+component+".png")
                record["poster_components"][component] = _render_frame(scene,poster_frame,path,config["poster_samples"],100)
                print("POSTER_COMPONENT="+json.dumps({"film":film_id,"component":component,**record["poster_components"][component]}),flush=True)
            if film_id=="01_reflection":
                # This component already has final poster dimensions. Make
                # it reviewable while the other representative cuts render.
                shutil.copyfile(scratch/film_id/record["poster_components"]["reflection"]["file"],output/FILMS[film_id]["poster"])
        if request["mode"]=="render":
            count = round(config["duration_seconds"]*config["fps"])
            for component,scene in scenes.items():
                frame_dir = scratch/film_id/component
                frame_dir.mkdir(exist_ok=True)
                record["frames"][component] = {}
                for frame in range(1,count+1):
                    path = frame_dir/f"frame_{frame:04d}.png"
                    record["frames"][component][str(frame)] = _render_frame(scene,frame,path,config["samples"],100)
                    print("FINAL_COMPONENT_FRAME="+json.dumps({"film":film_id,"component":component,**record["frames"][component][str(frame)]}),flush=True)
        result["films"].append(record)
        dump_json(scratch/"native_result.json",result)
    if request["mode"]=="proof" and config["proof"]["film"] in request["films"]:
        film_id = config["proof"]["film"]
        record = next(item for item in result["films"] if item["id"]==film_id)
        bpy.ops.wm.open_mainfile(filepath=str(output/(film_id+".blend")))
        scene = bpy.data.scenes[film_id+" | reflection"]
        frame_dir = scratch/film_id/"proof_frames"
        frame_dir.mkdir(exist_ok=True)
        percentage = round(100*config["proof"]["width"]/config["width"])
        count = round(config["proof"]["seconds"]*config["fps"])
        for frame in range(1,count+1):
            path = frame_dir/f"frame_{frame:04d}.png"
            record["proof_frames"][str(frame)] = _render_frame(scene,frame,path,config["proof"]["samples"],percentage)
            print("PROOF_FRAME="+json.dumps(record["proof_frames"][str(frame)]),flush=True)
        dump_json(scratch/"native_result.json",result)


def _windows_path(path: Path, blender: str) -> str:
    path = path.resolve()
    if blender.lower().endswith(".exe") and platform.system()=="Linux":
        return subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()
    return str(path)


def _blender_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidate = Path("/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("blender")
    if not found:
        raise FileNotFoundError("Install Blender or pass --blender; no render backend is substituted")
    return found


def _gpu_report() -> dict:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"known":False,"reason":"nvidia-smi unavailable; no cross-run frame reuse is supported"}
    try:
        rows = subprocess.check_output([command,"--query-gpu=uuid,name,driver_version","--format=csv,noheader"],text=True,timeout=10)
        return {"known":True,"rows":sorted(line.strip() for line in rows.splitlines() if line.strip())}
    except subprocess.SubprocessError:
        return {"known":False,"reason":"GPU driver query failed; no frame reuse is supported"}


def _media_tools(args, purpose: str) -> dict:
    tools = {}
    for name in ("ffmpeg","ffprobe"):
        value = getattr(args,name)
        if not value:
            raise FileNotFoundError(f"{name} is required for {purpose}")
        path = Path(shutil.which(value) or value).resolve()
        version = subprocess.check_output([str(path),"-version"],text=True,stderr=subprocess.STDOUT)
        tools[name] = {"invoked_path":value,"resolved_path":str(path),"sha256":sha256(path),"version_output":version.strip()}
    return {"queried_utc":datetime.now(timezone.utc).isoformat(),"scope":purpose,"tools":tools}


INTRO_BUG_SOURCE_SHA256 = "2f34c91242dc0de30a722a6ffea2bba24c8588858fac05e117b400f5bc2c6332"
CAMERA_REPAIR_CONFIG_SHA256 = "f37d311621a7065caca8ca4229cbe3aecf9f75db391ff6a7ff6979252e487607"
INTRO_BUG_SOURCE_REVISION = "a6b623c"
INTRO_REPAIR_FRAMES = tuple(range(13,25))
CAMERA_REPAIR_COMPONENTS = {
    "01_reflection":{"reflection":INTRO_REPAIR_FRAMES},
    "02_kerr_comparison":{"comparison_on":tuple(range(1,145))},
    "03_phase_change":{"intensity_off":tuple(range(1,145)),"intensity_on":tuple(range(1,145))},
}
CAMERA_REPAIR_SAMPLES = {"reflection":(13,18,24,25),"comparison_on":(1,90,144),"intensity_off":(1,90,144),"intensity_on":(1,90,144)}


def _repair_guard(manifest: dict, original_cache: Path, output: Path, config_path: Path, config: dict, bundle: dict, host: dict) -> dict:
    """Only the known Issue-31 completed render may supply unchanged frames."""
    if manifest.get("stage")!="render" or not manifest.get("production_movies_complete") or manifest.get("repairs"):
        raise ValueError("repair-cameras requires an unmodified completed original full run")
    provenance = manifest["provenance"]
    if provenance["renderer_source_sha256"]!=INTRO_BUG_SOURCE_SHA256:
        raise ValueError("Only the recorded Issue-31 source revision can be repaired")
    if provenance["renderer_config_sha256"]!=CAMERA_REPAIR_CONFIG_SHA256 or sha256(config_path)!=CAMERA_REPAIR_CONFIG_SHA256:
        raise ValueError("Only the recorded camera-repair configuration hash can be repaired")
    if manifest["effective_config"]!=config or provenance["renderer_config_sha256"]!=sha256(config_path):
        raise ValueError("Repair configuration differs from the original run")
    if provenance["input_files"]!=bundle["records"] or provenance["numerical_manifest_sha256"]!=bundle["records"]["manifest.json"]["sha256"]:
        raise ValueError("Repair numerical input differs from the original run")
    if not provenance["host"]["gpu_driver"].get("known") or provenance["host"]!=host:
        raise ValueError("Original/current Python, NumPy or known GPU-driver identity differs")
    if sha256(original_cache/"render_spatial_kerr_snapshot.py")!=INTRO_BUG_SOURCE_SHA256:
        raise ValueError("Original renderer snapshot hash differs")
    original_request = json.loads((original_cache/"request.json").read_text())
    if original_request["mode"]!="render" or original_request["config"]!=config or original_request["provenance"]!=provenance:
        raise ValueError("Original full-render request differs from its manifest")
    for name,record in bundle["records"].items():
        if sha256(original_cache/"input_snapshot"/name)!=record["sha256"]:
            raise ValueError("Original input snapshot hash differs: "+name)
    films = {row["id"]:row for row in manifest["films"]}
    if set(films)!=set(FILMS):
        raise ValueError("All three original films must be complete before repair")
    checked = {}
    count = round(config["duration_seconds"]*config["fps"])
    native = json.loads((original_cache/"native_result.json").read_text())
    if native["environment"]!=manifest["render_environment"]:
        raise ValueError("Original native rendering environment differs")
    for film_id,film in films.items():
        if film["status"]!="rendered" or film["frames"]!=count or set(film["frame_sha256"])!={str(i) for i in range(1,count+1)}:
            raise ValueError("Original film frame inventory is incomplete")
        for kind in ("scene","poster","mp4"):
            path = output/film[kind]
            if not path.is_file() or sha256(path)!=film["sha256"][kind]:
                raise ValueError(f"Original {film_id} {kind} differs from its manifest")
        for frame,value in film["frame_sha256"].items():
            path = original_cache/film_id/"composed_frames"/f"frame_{int(frame):04d}.png"
            if sha256(path)!=value:
                raise ValueError(f"Original full-frame hash differs: {film_id} {frame}")
        original_film = next(row for row in native["films"] if row["id"]==film_id)
        for component,frames in original_film["frames"].items():
            if set(frames)!={str(i) for i in range(1,count+1)}:
                raise ValueError("Original native component inventory is incomplete")
            for frame,record in frames.items():
                if sha256(original_cache/film_id/component/f"frame_{int(frame):04d}.png")!=record["sha256"]:
                    raise ValueError(f"Original native component hash differs: {film_id} {component} {frame}")
        checked[film_id] = {"composed_frames":count,"native_component_frames":sum(len(v) for v in original_film["frames"].values()),"scene_poster_movie_hashes_match":True}
    return {"checked_original_frame_hashes":checked,"original_source_sha256":INTRO_BUG_SOURCE_SHA256,"original_source_revision":INTRO_BUG_SOURCE_REVISION,"configuration_input_host_and_original_snapshot_match":True}


def _compose(film_id: str, component_paths: dict[str,Path], output: Path, config: dict) -> None:
    from PIL import Image
    canvas = Image.new("RGB",(config["width"],config["height"]))
    for component,path in component_paths.items():
        width,height,position = _component_layout(film_id,component,config)
        with Image.open(path) as image:
            image.load()
            if image.size!=(width,height) or image.mode!="RGB":
                raise ValueError("Native component dimensions/pixel mode changed")
            canvas.paste(image,position)
    canvas.save(output)


def _encode(frame_dir: Path, path: Path, count: int, dimensions: tuple[int,int], config: dict, args) -> dict:
    if not args.ffmpeg or not args.ffprobe:
        raise FileNotFoundError("FFmpeg and FFprobe are required for video output")
    # Each input is a native frame. The fixed camera-repair mode may retain
    # hash-verified corresponding original frames outside its declared scope.
    subprocess.run([args.ffmpeg,"-y","-v","error","-framerate",str(config["fps"]),"-i",str(frame_dir/"frame_%04d.png"),"-frames:v",str(count),"-c:v","libx264","-preset",config["encoding"]["preset"],"-crf",str(config["encoding"]["crf"]),"-pix_fmt","yuv420p","-movflags","+faststart",str(path)],check=True)
    raw = subprocess.check_output([args.ffprobe,"-v","error","-select_streams","v:0","-show_entries","stream=width,height,r_frame_rate,nb_frames,duration","-of","json",str(path)],text=True)
    probe = json.loads(raw)["streams"][0]
    if (probe["width"],probe["height"])!=dimensions or probe["r_frame_rate"]!=f"{config['fps']}/1" or int(probe["nb_frames"])!=count or abs(float(probe["duration"])-count/config["fps"])>1e-6:
        raise ValueError("Encoded video does not match its declared frames, duration, or dimensions")
    subprocess.run([args.ffmpeg,"-v","error","-i",str(path),"-f","null","-"],check=True)
    return {"file":path.name,"sha256":sha256(path),"bytes":path.stat().st_size,"probe":probe,"all_frames_decode":True}


def _run_repair_blender(request, scratch: Path, blender: str, script: Path, name: str):
    path = scratch/(name+"_request.json")
    dump_json(path,request)
    with (scratch/(name+".log")).open("w",encoding="utf-8") as log:
        subprocess.run([blender,"--background","--factory-startup","--python-exit-code","1","--python",_windows_path(script,blender),"--","--inside-blender","--request",_windows_path(path,blender)],stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((scratch/"native_repair_result.json").read_text())


def _repair_host(args, config_path, config, bundle):
    output = args.output_dir.resolve()
    source = Path(__file__).resolve()
    host = {"python":platform.python_version(),"numpy":np.__version__,"gpu_driver":_gpu_report()}
    media = _media_tools(args,"Queried for camera-repair verification/encoding only; original full-run encoder versions were not recorded")
    blender = _blender_path(args.blender)
    if args.repair_state:
        state_path = args.repair_state.resolve()
        state = json.loads(state_path.read_text())
        scratch = state_path.parent
        if state["repair_source_sha256"]!=sha256(source) or state["host"]!=host or state["media_tools"]["tools"]!=media["tools"]:
            raise ValueError("Preflight source/environment changed; prepare a new camera repair")
        original = json.loads((scratch/"original_completed_manifest.json").read_text())
        original_cache = Path(state["original_cache"])
    else:
        if not args.preflight_only or not args.repair_manifest:
            raise ValueError("First run repair-cameras --preflight-only --repair-manifest; then finish with --repair-state")
        original = json.loads(args.repair_manifest.resolve().read_text())
        original_cache = (args.repair_cache or Path(original["scratch_run"])).resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        scratch = args.scratch_dir.resolve()/(stamp+"-camera-repair-"+uuid.uuid4().hex[:8])
        scratch.mkdir(parents=True,exist_ok=False)
        dump_json(scratch/"original_completed_manifest.json",original)
        state = None
    guard = _repair_guard(original,original_cache,output,config_path,config,bundle,host)
    if sha256(args.font)!=original["provenance"]["font"]["sha256"] or sha256(args.font_license)!=original["provenance"]["font"]["license_sha256"]:
        raise ValueError("Repair Japanese font/license differs from the original")
    if state is None:
        frozen = scratch/"render_spatial_kerr_snapshot.py"
        shutil.copyfile(source,frozen)
        snapshot = scratch/"input_snapshot"
        snapshot.mkdir()
        for name,record in bundle["records"].items():
            shutil.copyfile(args.data_dir.resolve()/name,snapshot/name)
            if sha256(snapshot/name)!=record["sha256"]:
                raise RuntimeError("Input changed while freezing repair snapshot")
        original_files = scratch/"original_deliverables"
        original_files.mkdir()
        for film in original["films"]:
            for kind in ("scene","poster","mp4"):
                shutil.copyfile(output/film[kind],original_files/film[kind])
        staging = scratch/"delivery"
        staging.mkdir()
        provenance = {"renderer_source_sha256":sha256(source),"renderer_config_sha256":sha256(config_path),"numerical_manifest_sha256":bundle["records"]["manifest.json"]["sha256"],"input_files":bundle["records"],"font":original["provenance"]["font"],"host":host,"media_tools":media,"scope":"Formal fixed camera repair for Issues 31/34; numerical data and display materials unchanged","original_renderer_source_sha256":INTRO_BUG_SOURCE_SHA256,"original_source_revision":INTRO_BUG_SOURCE_REVISION}
        request = {"mode":"repair-cameras","config":config,"provenance":provenance,"paths":{name:_windows_path(path,blender) for name,path in {"scratch":scratch,"output":staging,"data_dir":snapshot,"font":args.font}.items()},"repair":{"phase":"prepare","original_environment":original["render_environment"],"original_scenes":{film["id"]:_windows_path(original_files/film["scene"],blender) for film in original["films"]}}}
        print("CAMERA_REPAIR_RUN="+str(scratch),flush=True)
        native = _run_repair_blender(request,scratch,blender,frozen,"preflight")
        hashes = {}
        for film in native["films"]:
            paths = {component:scratch/film["id"]/value["file"] for component,value in film["poster_components"].items()}
            path = staging/FILMS[film["id"]]["poster"]
            _compose(film["id"],paths,path,config)
            hashes[path.name] = sha256(path)
            hashes[film["scene"]] = sha256(staging/film["scene"])
        state = {"schema_version":1,"repair_source_sha256":sha256(source),"config_sha256":sha256(config_path),"original_cache":str(original_cache),"host":host,"media_tools":media,"guard":guard,"provenance":provenance,"request":request,"native":native,"staged_file_hashes":hashes}
        dump_json(scratch/"repair_state.json",state)
        print(json.dumps({"repair_state":str(scratch/"repair_state.json"),"stage":"preflight_ready","samples":{film["id"]:{component:[str(scratch/film["id"]/component/f"frame_{frame:04d}.png") for frame in CAMERA_REPAIR_SAMPLES[component]] for component in film["preflight_frames"]} for film in native["films"]}},ensure_ascii=False),flush=True)
        return
    if state["config_sha256"]!=sha256(config_path):
        raise ValueError("Preflight configuration hash changed")
    staging = scratch/"delivery"
    for name,value in state["staged_file_hashes"].items():
        if sha256(staging/name)!=value:
            raise ValueError("A prepared native/poster changed after preflight")
    preflight = {film["id"]:film["preflight_frames"] for film in state["native"]["films"]}
    request = json.loads(json.dumps(state["request"]))
    request["repair"].update(phase="finish",preflight_records=preflight)
    for film in state["native"]["films"]:
        for component,frames in film["preflight_frames"].items():
            for frame,record in frames.items():
                if sha256(scratch/film["id"]/component/f"frame_{int(frame):04d}.png")!=record["sha256"]:
                    raise ValueError("A prepared sample frame changed after preflight")
    native = _run_repair_blender(request,scratch,blender,scratch/"render_spatial_kerr_snapshot.py","finish")
    finished = json.loads(json.dumps(original))
    replacements = {}
    old_native = json.loads((original_cache/"native_result.json").read_text())
    for film in finished["films"]:
        film_id = film["id"]
        directory = scratch/film_id/"composed_frames"
        directory.mkdir(exist_ok=False)
        hashes = {};replacements[film_id] = {}
        for frame in range(1,145):
            path = directory/f"frame_{frame:04d}.png"
            modified = {component for component,frames in CAMERA_REPAIR_COMPONENTS[film_id].items() if frame in frames}
            if not modified:
                shutil.copyfile(original_cache/film_id/"composed_frames"/path.name,path)
                if sha256(path)!=film["frame_sha256"][str(frame)]:
                    raise RuntimeError("An unchanged corresponding original frame was not preserved")
            else:
                paths = {component:(scratch if component in modified else original_cache)/film_id/component/path.name for component in FILMS[film_id]["components"]}
                _compose(film_id,paths,path,config)
            hashes[str(frame)] = sha256(path)
        for component,frames in CAMERA_REPAIR_COMPONENTS[film_id].items():
            old = next(row for row in old_native["films"] if row["id"]==film_id)["frames"][component]
            new = native["repair_frames"][film_id][component]
            replacements[film_id][component] = {str(frame):{"old_native_frame_sha256":old[str(frame)]["sha256"],"new_native_frame_sha256":new[str(frame)]["sha256"]} for frame in frames}
        video = _encode(directory,staging/film["mp4"],144,(1920,1080),config,args)
        film.update(frame_sha256=hashes,video_verification=video)
        film["sha256"] = {kind:sha256(staging/film[kind]) for kind in ("scene","poster","mp4")}
        fresh = next(row for row in state["native"]["films"] if row["id"]==film_id)
        film["components"] = fresh["components"]
        film["poster_component_records"] = fresh["poster_components"]
        film["camera_repair_checks"] = fresh["camera_checks"]
        film["scene_and_poster_generation_source_sha256"] = state["repair_source_sha256"]
        film["component_frame_generation"] = {component:{"original_renderer_source_sha256":INTRO_BUG_SOURCE_SHA256,"repair_renderer_source_sha256":state["repair_source_sha256"],"replaced_frames":list(CAMERA_REPAIR_COMPONENTS[film_id].get(component,()))} for component in FILMS[film_id]["components"]}
    for name,record in bundle["records"].items():
        if sha256(args.data_dir.resolve()/name)!=record["sha256"]:
            raise RuntimeError("Original input changed during camera repair")
    record = {"id":"camera-registration-issues-31-34","created_utc":datetime.now(timezone.utc).isoformat(),"original_source_revision":INTRO_BUG_SOURCE_REVISION,"original_source_sha256":INTRO_BUG_SOURCE_SHA256,"repair_source_sha256":state["repair_source_sha256"],"configuration_input_environment_unchanged":True,"original_manifest_sha256":sha256(scratch/"original_completed_manifest.json"),"original_deliverables":{film["id"]:film["sha256"] for film in original["films"]},"replacement_native_frame_hashes":replacements,"guard":guard,"native_verification":native["native_verification"],"media_tools":media,"original_encoder_identity":"Not recorded by the original full-run source; current identities apply only to repair verification/encoding","scratch_run":str(scratch),"preflight_state":str(scratch/"repair_state.json"),"reproduction_command":".venv/bin/python scripts/render_spatial_kerr.py --mode render","scope":"Only camera rotations on 01 frames13-24 and corrected active-scene evaluation for 02 ON / 03 lower panels. Each affected native frame was rerendered. Matching original frame IDs elsewhere are preserved by verified hashes; all three videos are reencoded.","original_pixel_generation_provenance_retained":True}
    finished["repairs"] = [record]
    finished["canonical_reproduction_source_sha256"] = state["repair_source_sha256"]
    finished["ordinary_native_verification"] = native["native_verification"]
    for film in finished["films"]:
        for kind in ("scene","poster","mp4"):
            shutil.copyfile(staging/film[kind],output/film[kind])
    dump_json(output/"render_manifest.json",finished)
    print(json.dumps({"output":str(output),"stage":"camera_repair_complete","manifest":str(output/"render_manifest.json"),"production_movies_complete":finished["production_movies_complete"]},ensure_ascii=False))


def host_main(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.poster_frame is not None and not 1<=args.poster_frame<=round(config["duration_seconds"]*config["fps"]):
        raise ValueError("Requested still-review frame is outside the recorded film schedule")
    data_dir = args.data_dir.resolve()
    bundle = verify_inputs(data_dir,config)
    if args.mode=="repair-cameras":
        _repair_host(args,config_path,config,bundle)
        return
    if args.mode=="inspect":
        print(json.dumps({"source_manifest_sha256":bundle["records"]["manifest.json"]["sha256"],"all_input_hashes_match":True,"cases":bundle["summary"],"flow_approximation_limits":bundle["flow_accuracy"],"numerical_limits":bundle["manifest"].get("limitations",[])},ensure_ascii=False,indent=2))
        return
    if not args.font.exists() or not args.font_license.exists():
        raise FileNotFoundError("A Japanese font and its license are required; pass --font and --font-license")
    blender = _blender_path(args.blender)
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scratch = args.scratch_dir.resolve()/(stamp+"-"+uuid.uuid4().hex[:8])
    scratch.mkdir(parents=True,exist_ok=False)
    snapshot = scratch/"input_snapshot"
    snapshot.mkdir()
    # Freeze external numerical input without changing it. Hash every copy;
    # this also prevents a producer's later writes from entering one movie.
    for name,record in bundle["records"].items():
        target = snapshot/name
        shutil.copyfile(data_dir/name,target)
        if target.stat().st_size!=record["bytes"] or sha256(target)!=record["sha256"]:
            raise RuntimeError("Numerical input changed during snapshot copy: "+name)
    script = Path(__file__).resolve()
    frozen_script = scratch/"render_spatial_kerr_snapshot.py"
    shutil.copyfile(script,frozen_script)
    shutil.copyfile(args.font_license,output/"font-license.txt")
    source_location = str(data_dir.relative_to(ROOT)) if data_dir.is_relative_to(ROOT) else str(data_dir)
    provenance = {"renderer_source_sha256":sha256(script),"renderer_config_sha256":sha256(config_path),"numerical_manifest_sha256":bundle["records"]["manifest.json"]["sha256"],"numerical_source_hashes":bundle["manifest"].get("source_hashes",{}),"numerical_input_config":bundle["manifest"].get("input_config",{}),"input_location":source_location,"input_files":bundle["records"],"font":{"path":str(args.font.resolve()),"sha256":sha256(args.font),"license_sha256":sha256(args.font_license)},"host":{"python":platform.python_version(),"numpy":np.__version__,"gpu_driver":_gpu_report()},"cache_policy":"Fresh isolated cache per run; no resume or cross-run frame reuse"}
    provenance["render_request"] = {"mode":args.mode,"films":args.film,"poster_frame_override":args.poster_frame}
    if args.mode in ("render","proof"):
        provenance["media_tools"] = _media_tools(args,"Identities queried for this render/proof encoding and verification")
    request = {"mode":args.mode,"films":args.film,"poster_frame":args.poster_frame,"config":config,"provenance":provenance,"paths":{name:_windows_path(path,blender) for name,path in {"data_dir":snapshot,"output":output,"scratch":scratch,"font":args.font}.items()}}
    request_path = scratch/"request.json"
    dump_json(request_path,request)
    log_path = scratch/"blender.log"
    print(f"SPATIAL_RENDER_RUN={scratch}",flush=True)
    with log_path.open("w",encoding="utf-8") as log:
        subprocess.run([blender,"--background","--factory-startup","--python-exit-code","1","--python",_windows_path(frozen_script,blender),"--","--inside-blender","--request",_windows_path(request_path,blender)],stdout=log,stderr=subprocess.STDOUT,check=True)
    native = json.loads((scratch/"native_result.json").read_text(encoding="utf-8"))
    if args.mode=="verify-scene":
        print(json.dumps(native["native_verification"],ensure_ascii=False,indent=2))
        return
    # Input hashes are sealed again. Changes in the external original are
    # reported rather than silently attaching current metadata to old images.
    for name,record in bundle["records"].items():
        if sha256(data_dir/name)!=record["sha256"]:
            raise RuntimeError("Original numerical artifact changed during rendering: "+name)
    films = []
    for record in native["films"]:
        film_id = record["id"]
        info = {"id":film_id,"mp4":None,"planned_mp4":film_id+".mp4","poster":None,"scene":record["scene"],"width":config["width"],"height":config["height"],"fps":config["fps"],"frames":round(config["duration_seconds"]*config["fps"]),"sha256":{"scene":record["scene_sha256"],"poster":None,"mp4":None},"components":record["components"],"poster_frame":record["poster_frame"],"poster_observation_z_um":record["poster_observation_z_um"],"status":"scene_only"}
        if record["poster_components"]:
            paths = {component:scratch/film_id/value["file"] for component,value in record["poster_components"].items()}
            _compose(film_id,paths,output/FILMS[film_id]["poster"],config)
            info["poster"] = FILMS[film_id]["poster"]
            info["sha256"]["poster"] = sha256(output/info["poster"])
            info["status"] = "poster_and_scene_only"
            info["poster_component_records"] = record["poster_components"]
        if args.mode=="render":
            frame_dir = scratch/film_id/"composed_frames"
            frame_dir.mkdir()
            frame_hashes = {}
            for frame in range(1,info["frames"]+1):
                paths = {component:scratch/film_id/component/f"frame_{frame:04d}.png" for component in FILMS[film_id]["components"]}
                path = frame_dir/f"frame_{frame:04d}.png"
                _compose(film_id,paths,path,config)
                frame_hashes[str(frame)] = sha256(path)
            video = _encode(frame_dir,output/(film_id+".mp4"),info["frames"],(config["width"],config["height"]),config,args)
            info.update(mp4=video["file"],status="rendered",video_verification=video,frame_sha256=frame_hashes)
            info["sha256"]["mp4"] = video["sha256"]
        films.append(info)
    manifest = {"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),"stage":args.mode,"production_movies_complete":args.mode=="render" and len(films)==3,"films":films,"provenance":provenance,"effective_config":config,"render_environment":native["environment"],"samples":{"film":config["samples"],"poster":config["poster_samples"],"proof":config["proof"]["samples"]},"physical_coordinate_map":"world=(z,x,y)/100, physical micrometres, equal XYZ scales","camera_frame_mapping":{"frame_count":round(config["duration_seconds"]*config["fps"]),"formula":"z_start+(z_end-z_start)*(frame-1)/(frames-1)","z_start_um":config["camera"]["target_z_um"][0],"z_end_um":config["camera"]["target_z_um"][1],"time_scope":"Camera/section observation schedule; not physical pulse or photon flight time"},"legends":{"isosurfaces":config["isosurfaces"],"intensity":{"limits":config["intensity_range_relative_to_input_peak"],"normalization":"Fixed analytic input peak=1; no frame/case maximum normalization"},"phase":{"limits_rad":config["phase_range_rad"],"definition":"arg(psi_Kerr*conj(psi_linear))","joint_intensity_floor":config["phase_joint_intensity_floor"],"mask_color":config["masked_phase_color_linear_rgb"]},"lines":"All 25 source J/I power streamlines, stopped at first invalid point; not photons","cursor":"Purple observation section and source-flow intersections, not a moving pulse"},"interpolation":{"mesh":"Provided physical vertices and normals; no reconstructed field or hand path","flow":"Source-valid prefix POLY segments; source display stride="+str(config["flow"]["display_stride"]),"intensity":"Bilinear display RGBA from a fixed map of saved central complex intensity","phase":"Nearest saved masked principal-phase RGBA; no unwrap or branch-crossing interpolation","camera":"Linear source-distance observation sweep"},"input_checks":bundle["summary"],"ordinary_native_verification":native["native_verification"],"limitations":config["scope"]+bundle["manifest"].get("limitations",[]),"native_layout":"A film blend contains each named native component scene. CLI joins components without rescaling: film 02 OFF above ON; film 03 phase above paired OFF/ON intensity sections.","scratch_run":str(scratch),"independent_validator":{"requested":0,"attempted":0,"completed":0}}
    if args.mode=="proof":
        record = next((item for item in native["films"] if item["id"]==config["proof"]["film"]),None)
        if record and record["proof_frames"]:
            count = round(config["proof"]["seconds"]*config["fps"])
            video = _encode(scratch/record["id"]/"proof_frames",output/"phenomenon_proof.mp4",count,(config["proof"]["width"],config["proof"]["height"]),config,args)
            manifest["phenomenon_proof"] = {**video,"film":record["id"],"source_frames":[1,count],"frame_sha256":{key:value["sha256"] for key,value in record["proof_frames"].items()},"width":config["proof"]["width"],"height":config["proof"]["height"],"fps":config["fps"],"frames":count,"last_camera_observation_z_um":distance_at_frame(count,round(config["duration_seconds"]*config["fps"]),config,"01_reflection"),"status":"Bounded phenomenon proof; not the final six-second movie"}
    manifest["flow_approximation_limits"] = {"measured_checks":bundle["flow_accuracy"],"interpretation":"Approximate J/I power-flow display. Do not measure tiny individual-ray displacement or reflection-point shifts from these lines. Rendering coordinate fidelity is not physical streamline accuracy."}
    count = round(config["duration_seconds"]*config["fps"])
    manifest["camera_frame_mapping"] = {film_id:{"frames":count,"intro_frames":config["camera"].get("intro_frames",0) if film_id=="01_reflection" else 0,"camera_observation_z_um":[distance_at_frame(frame,count,config,film_id) for frame in range(1,count+1)],"time_scope":"Distance observation schedule only, no physical pulse or photon time"} for film_id in args.film}
    manifest["isosurface_display_shading"] = {"common_material":config.get("isosurface_display_shading"),"interpretation":"Shared normal-based display shading makes the saved 3D shape visible; it is not quantitative intensity or measured scattered brightness."}
    manifest["core_boundary_guide"] = {**config["core_boundary_guide"],"positions_physical_xyz_um":core_boundary_rails_um(config).tolist(),"scope":"Display annotation on y=0, x=+/-25 um throughout the actual circular core. Not simulated light."}
    dump_json(output/"render_manifest.json",manifest)
    print(json.dumps({"output":str(output),"manifest":str(output/"render_manifest.json"),"production_movies_complete":manifest["production_movies_complete"],"films":[{"id":f["id"],"status":f["status"]} for f in films],"proof":manifest.get("phenomenon_proof",{}).get("file")},ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args = arguments(sys.argv[1:] if argv is None else argv)
    if args.inside_blender:
        if args.request is None:
            raise ValueError("Internal Blender invocation needs its frozen request")
        inside_blender(args.request)
    else:
        host_main(args)


if __name__=="__main__":
    # Blender passes its own flags before --. Only its explicit request tail
    # belongs to this CLI; ordinary Python gets the full user argument list.
    tail = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    main(tail)
