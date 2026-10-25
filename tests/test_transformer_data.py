"""Data isolation, exact budgets/seeds, atomic restart and boundary checks."""
from dataclasses import replace
import json
import numpy as np
import pytest

from experiments.transformer_nlc.data_config import DataConfig, main16, main64, pilot_config
from experiments.transformer_nlc.data import (frame_layout, seed_sequence, simulate_physical_frame,
    generate, load_frame, load_receiver, iter_blocks, array_hash, generation_plan, file_hash)
from experiments.transformer_nlc.qualification import qualification_cases, qualification_plan, CommonPhysicalNoise, analytic_controls


def tiny_config(condition=16):
    config = pilot_config(condition,scored_symbols=256,guard_symbols=128,pilot_symbols=64)
    return replace(config,link=replace(config.link,n_spans=1,span_km=.5),bps_windows=(33,65,129))


@pytest.mark.parametrize("factory",[main16,main64])
def test_frozen_production_budgets_and_explicit_overhead(factory):
    config = factory()
    assert config.frame_counts == {"train":8,"validation":2,"test":4}
    assert generation_plan(config)["scored_DP_times"] == {"train":2**19,"validation":2**17,"test":2**18}
    layout = frame_layout(config)
    assert layout["scoring_mask"].sum() == 65536
    assert layout["pilot_mask"].sum() == 4096
    assert not np.any(layout["pilot_mask"] & layout["scoring_mask"])
    assert layout["context_valid_mask"].sum() == 65536+192
    assert DataConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError,match="Production scored"):
        replace(config,scored_symbols=1024)


def test_exact_pcg64_keys_regions_splits_and_replication():
    config = main16()
    key = seed_sequence(config,"test",2,"bits")
    assert key.entropy == 2026102543 and key.spawn_key == (16,0,2,2,0)
    derived = [seed_sequence(config,s,0,p).generate_state(4).tobytes() for s in ("train","validation","test") for p in ("bits","ase","model","shuffle")]
    assert len(set(derived)) == len(derived)
    assert not np.array_equal(seed_sequence(config,"train",0,"bits").generate_state(4),seed_sequence(replace(config,replicate=1),"train",0,"bits").generate_state(4))
    assert seed_sequence(tiny_config(),"train",0,"bits").spawn_key[2] == 3
    assert seed_sequence(tiny_config(),"validation",0,"bits").spawn_key[3] == 1_000_000
    assert seed_sequence(tiny_config(),"pilot",0,"bits").spawn_key[3] == 3_000_000


def test_regions_preserve_scored_and_pilot_bits_when_guard_changes():
    config = tiny_config()
    first = simulate_physical_frame(config,"train",0)
    changed = replace(config,guard_symbols=256,link=replace(config.link,guard_symbols=256))
    second = simulate_physical_frame(changed,"train",0)
    for name in ("scoring_mask","pilot_mask","context_valid_mask"):
        np.testing.assert_array_equal(first.arrays["tx_bits"][first.arrays[name]],second.arrays["tx_bits"][second.arrays[name]])
    # Pilot and production random streams are disjoint, as are pilot subframes.
    third = simulate_physical_frame(config,"validation",0)
    assert not np.array_equal(first.arrays["tx_bits"],third.arrays["tx_bits"])
    assert first.arrays["rx_matched"].dtype == np.complex128
    assert first.metadata["raw_arrays"]["rx_field"]["storage"] == "regeneration_recipe"


def test_generate_reload_resume_hashes_and_no_test_metrics(tmp_path):
    config = tiny_config()
    output = tmp_path/"run"
    manifest = generate(config,output)
    assert manifest["status"] == "complete" and manifest["test_metrics"] == "not_computed"
    assert len(manifest["frames"]) == 3
    first = manifest["frames"][0]
    frame = load_frame(output/first["path"],expected_sha256=first["file_sha256"])
    fit = load_receiver(output)
    np.testing.assert_array_equal(fit.apply(frame.pilots)[0],frame.arrays["rx_nlc"])
    blocks = list(iter_blocks(frame,block_symbols=128,context=96))
    assert len(blocks)==2 and blocks[0]["rx"].shape==(320,2)
    np.testing.assert_array_equal(np.concatenate([b["symbol_index"] for b in blocks]),np.flatnonzero(frame.arrays["scoring_mask"]))
    for block in blocks:
        np.testing.assert_array_equal(block["residual"],frame.arrays["rx_nlc"][block["symbol_index"]]-block["tx"])
    hashes = [entry["file_sha256"] for entry in manifest["frames"]]
    resumed = generate(config,output,resume=True)
    assert [entry["file_sha256"] for entry in resumed["frames"]] == hashes
    with pytest.raises(FileExistsError):
        generate(config,output)
    with pytest.raises(ValueError,match="Resume"):
        generate(replace(config,phase_max_deg=.025),output,resume=True)
    path = output/first["path"]
    path.write_bytes(path.read_bytes()+b"corrupt")
    with pytest.raises(ValueError,match="hash"):
        generate(config,output,resume=True)


def test_resume_after_physical_stage_interruption(tmp_path,monkeypatch):
    import experiments.transformer_nlc.data as module
    config = tiny_config()
    original = module.fit_receiver
    def fail(*args,**kwargs):
        raise RuntimeError("intentional interruption after physical cache")
    monkeypatch.setattr(module,"fit_receiver",fail)
    with pytest.raises(RuntimeError,match="interruption"):
        generate(config,tmp_path/"run")
    failed = json.loads((tmp_path/"run/manifest.json").read_text())
    assert failed["status"] == "failed" and failed["failed_attempts"][0]["exception"] == "RuntimeError"
    monkeypatch.setattr(module,"fit_receiver",original)
    assert generate(config,tmp_path/"run",resume=True)["status"] == "complete"


def test_boundaries_invalid_context_and_holes_are_rejected(tmp_path):
    output = tmp_path/"run"
    manifest = generate(tiny_config(),output)
    frame = load_frame(output/manifest["frames"][0]["path"])
    with pytest.raises(ValueError,match="invalid region"):
        list(iter_blocks(frame,context=97))
    frame.arrays["frame_id"][np.flatnonzero(frame.arrays["scoring_mask"])[0]+10] = 999
    with pytest.raises(ValueError,match="frame/index"):
        list(iter_blocks(frame))
    frame.arrays["scoring_mask"][np.flatnonzero(frame.arrays["scoring_mask"])[0]+10] = False
    with pytest.raises(ValueError,match="contiguous"):
        list(iter_blocks(frame))


def test_production_gate_and_duplicate_job_lock(tmp_path):
    with pytest.raises(ValueError,match="qualification"):
        generate(main16(),tmp_path/"production")
    output = tmp_path/"locked"; output.mkdir()
    (output/".generation.lock").write_text("123")
    with pytest.raises(FileExistsError):
        generate(tiny_config(),output,resume=True)


def test_qualification_is_separate_one_factor_plan_and_common_noise():
    plan = qualification_plan()
    assert plan["full_link_propagations"] == 36
    cases = qualification_cases(16)
    base = cases["baseline"]
    assert base.scored_symbols == 65536 and cases["frame"].scored_symbols == 131072
    assert cases["phase"].phase_max_deg == .025
    # Small analogous physical geometries test the same reference sampling API.
    tiny = tiny_config()
    geometry = {"base":tiny,"sps":replace(tiny,link=replace(tiny.link,sps=8)),
                "guard":replace(tiny,guard_symbols=256,link=replace(tiny.link,guard_symbols=256))}
    noises = {key:CommonPhysicalNoise(geometry,cfg,0)(0,cfg.n_symbols*cfg.link.sps,cfg.link) for key,cfg in geometry.items()}
    np.testing.assert_allclose(noises["sps"][::2],noises["base"],atol=1e-16,rtol=1e-12)
    shift = 128*tiny.link.sps
    np.testing.assert_allclose(noises["guard"][shift:-shift],noises["base"],atol=1e-16,rtol=1e-12)


@pytest.mark.parametrize("condition",[16,64])
def test_reproducible_analytic_qualification_controls(condition):
    result = analytic_controls(condition)
    assert result["status"] == "passed"
    assert result["CPU_GPU"]["status"] == "not_run"
