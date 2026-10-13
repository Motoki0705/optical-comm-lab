"""Renderer persistence and public CLI contracts, without Blender or a GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import zlib

import pytest


spec=importlib.util.spec_from_file_location("kerr_renderer",Path(__file__).resolve().parents[1]/"scripts/render_kerr_films.py")
renderer=importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(renderer)


def png(width: int=3,height: int=2,pixel: bytes=b"\xff\x00\x00") -> bytes:
    def chunk(kind: bytes,data: bytes) -> bytes:
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xFFFFFFFF)
    raster=b"".join(b"\x00"+pixel*width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raster))+chunk(b"IEND",b"")


def test_resume_accepts_original_and_rejects_valid_replacement(tmp_path: Path) -> None:
    path=tmp_path/"frame.png"
    path.write_bytes(png())
    record=renderer.inspect_png(path,(3,2))
    assert renderer.cached_frame_valid(path,record,(3,2))
    path.write_bytes(png(pixel=b"\x00\xff\x00"))
    assert not renderer.cached_frame_valid(path,record,(3,2))


@pytest.mark.parametrize("damage",["truncated","CRC","dimensions","raster"])
def test_resume_rejects_incomplete_or_nondecodable_png(tmp_path: Path,damage: str) -> None:
    path=tmp_path/"frame.png"
    original=png()
    path.write_bytes(original)
    record=renderer.inspect_png(path,(3,2))
    if damage=="truncated":
        damaged=original[:-9]
    elif damage=="CRC":
        damaged=bytearray(original)
        damaged[40]^=1
    elif damage=="dimensions":
        damaged=png(width=4)
    else:
        # A correct PNG chunk CRC cannot make a truncated deflate stream valid.
        payload=zlib.compress(b"\x00\xff\x00\x00")[:-1]
        kind=b"IDAT"
        bad_chunk=struct.pack(">I",len(payload))+kind+payload+struct.pack(">I",zlib.crc32(kind+payload)&0xFFFFFFFF)
        damaged=original[:33]+bad_chunk+original[-12:]
    path.write_bytes(damaged)
    assert not renderer.cached_frame_valid(path,record,(3,2))


@pytest.mark.parametrize("field",["blender_build_hash","selected_devices","gpu_driver_report"])
def test_resume_rejects_changed_render_environment(field: str) -> None:
    original={"blender_build_hash":"build-a","selected_devices":["GPU-a"],"gpu_driver_report":{"known":True,"rows":["driver-a"]}}
    changed={**original,field:{"known":True,"rows":["driver-b"]} if field=="gpu_driver_report" else "changed"}
    signature=renderer.render_signature({"source":"same","data":"same"},original)
    new_signature=renderer.render_signature({"source":"same","data":"same"},changed)
    with pytest.raises(RuntimeError,match="changed"):
        renderer.require_resume_signature({"signature":signature},new_signature,changed)


def test_unknown_driver_prevents_cross_run_reuse() -> None:
    environment={"gpu_driver_report":{"known":False}}
    signature=renderer.render_signature({},environment)
    with pytest.raises(RuntimeError,match="unavailable"):
        renderer.require_resume_signature({"signature":signature},signature,environment)


def test_single_case_proof_selects_that_case() -> None:
    for case in renderer.FILMS:
        args=renderer.arguments(["--mode","proof","--case",case])
        assert renderer.selected_proof_case(args.case)==case
    assert renderer.selected_proof_case(list(renderer.FILMS))=="combined"


def test_external_data_manifest_path_is_explicit(tmp_path: Path) -> None:
    root=tmp_path/"repository"
    external=tmp_path/"input"/"field.npz"
    assert renderer.input_record_key(external,root)==str(external.resolve())
    assert renderer.input_record_key(root/"artifacts"/"field.npz",root)=="artifacts/field.npz"
