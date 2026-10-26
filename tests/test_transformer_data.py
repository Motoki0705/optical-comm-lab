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
    from experiments.transformer_nlc.data import run_lock
    with run_lock(output):
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



def test_numerical_fingerprint_ignores_unrelated_model_reports_and_locks(tmp_path):
    from pathlib import Path
    from experiments.transformer_nlc.data import numerical_source_identity, NUMERICAL_SOURCE_PATHS
    root = Path(__file__).resolve().parents[1]
    for relative in NUMERICAL_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((root/relative).read_bytes())
    before = numerical_source_identity(tmp_path)
    for relative in ("experiments/transformer_nlc/model.py", "experiments/transformer_nlc/training.py",
                     "experiments/transformer_nlc/docs/report.md", "pyproject.toml", "uv.lock"):
        path = tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("unrelated addition or optional Torch dependency change\n")
    assert numerical_source_identity(tmp_path) == before
    # Every declared contributor affects the fingerprint, including config.
    for relative in NUMERICAL_SOURCE_PATHS:
        path = tmp_path/relative
        original = path.read_bytes()
        path.write_bytes(original+b"\n# actual contributing source change\n")
        assert numerical_source_identity(tmp_path)["content_sha256"] != before["content_sha256"]
        path.write_bytes(original)
    assert numerical_source_identity(tmp_path) == before


def passing_gate_fixture(config):
    from experiments.transformer_nlc.data import (source_identity, canonical_hash,
        QUALIFICATION_SCHEMA_VERSION, SOURCE_SCOPE_VERSION)
    from experiments.transformer_nlc.data_config import scientific_contract
    contract = scientific_contract(config)
    return {"schema":QUALIFICATION_SCHEMA_VERSION,"status":"passed","condition":config.condition,
        "source_scope_version":SOURCE_SCOPE_VERSION,"source_content_sha256":source_identity()["content_sha256"],
        "scientific_contract":contract,"scientific_contract_sha256":canonical_hash(contract)}


def test_complete_qualification_contract_and_prewrite_production_gate(tmp_path):
    from experiments.transformer_nlc.data import validate_qualification, source_identity
    from experiments.transformer_nlc.data_config import scientific_contract
    config = main16()
    gate = passing_gate_fixture(config)  # Synthetic policy fixture; never saved as measured qualification.
    validate_qualification(config,gate,source_identity())
    qualification = replace(config,regime="qualification",replicate=2,master_entropy=42,
        train_frames=1,validation_frames=1,test_frames=1,save_rx_field=True)
    assert scientific_contract(qualification) == scientific_contract(config)
    # Receiver changes are scientifically meaningful even though LinkConfig is identical.
    changed = replace(config,ridge_relative=1e-4)
    gate_path = tmp_path/"synthetic-gate.json";gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match="scientific contract"):
        generate(changed,tmp_path/"must-not-be-created",qualification=gate_path)
    assert not (tmp_path/"must-not-be-created").exists()
    # Geometry changes are constrained in production and also bound by the gate.
    pilot = tiny_config()
    pilot_gate = passing_gate_fixture(pilot)
    variants = [replace(pilot,scored_symbols=512),replace(pilot,pilot_symbols=128),
        replace(pilot,context_symbols=64),replace(pilot,launch_power_dbm=2.5),
        replace(pilot,phase_max_deg=.025),replace(pilot,min_step_km=2e-6),
        replace(pilot,equalizer_taps=9),replace(pilot,bps_phases=32),
        replace(pilot,bps_windows=(33,65)),
        replace(pilot,guard_symbols=256,link=replace(pilot.link,guard_symbols=256))]
    for variant in variants:
        with pytest.raises(ValueError,match="scientific contract"):
            validate_qualification(variant,pilot_gate,source_identity())


def test_resume_checks_numeric_versions_but_not_optional_torch(tmp_path,monkeypatch):
    import experiments.transformer_nlc.data as module
    config=tiny_config();output=tmp_path/"run"
    generate(config,output)
    original=module.environment_identity
    def with_torch(backend):
        return {**original(backend),"torch":"unrelated-optional-version"}
    monkeypatch.setattr(module,"environment_identity",with_torch)
    assert generate(config,output,resume=True)["status"]=="complete"
    for dependency in ("python","numpy","scipy","physical_dtype"):
        def changed(backend,dependency=dependency):
            return {**original(backend),dependency:"changed-numeric-runtime"}
        monkeypatch.setattr(module,"environment_identity",changed)
        with pytest.raises(ValueError,match="numeric runtime"):
            generate(config,output,resume=True)
    # GPU runtime/version fields participate without requiring CUDA for this unit control.
    environment={**original("numpy"),"cupy":"14","cuda_runtime":13020,"cuda_driver":13030,"gpu_name":"example"}
    before=module.numeric_runtime(environment,"cupy")
    for key in ("cupy","cuda_runtime","cuda_driver","gpu_name"):
        assert module.numeric_runtime({**environment,key:"changed"},"cupy") != before


def test_qualification_resume_rejects_runtime_and_receiver_changes(tmp_path,monkeypatch):
    import experiments.transformer_nlc.qualification as module
    monkeypatch.setattr(module,"analytic_controls",lambda *a,**k:{"status":"passed","CPU_GPU":{"status":"not_run"}})
    def stop(*args,**kwargs):
        raise RuntimeError("bounded fixture stops before full-link propagation")
    monkeypatch.setattr(module,"simulate_physical_frame",stop)
    output=tmp_path/"qualification"
    with pytest.raises(RuntimeError,match="bounded fixture"):
        module.run_qualification(16,output,cases=("phase",))
    original=module.environment_identity
    monkeypatch.setattr(module,"environment_identity",lambda backend:{**original(backend),"numpy":"changed"})
    with pytest.raises(ValueError,match="numeric runtime"):
        module.run_qualification(16,output,cases=("phase",),resume=True)
    monkeypatch.setattr(module,"environment_identity",original)
    original_cases=module.qualification_cases
    monkeypatch.setattr(module,"qualification_cases",lambda condition:{k:replace(v,bps_phases=32) for k,v in original_cases(condition).items()})
    with pytest.raises(ValueError,match="scientific contract"):
        module.run_qualification(16,output,cases=("phase",),resume=True)


@pytest.mark.parametrize("kind",["generation","qualification"])
def test_sigkill_releases_run_flock_and_resume_works(tmp_path,monkeypatch,kind):
    import os
    import select
    import subprocess
    import sys
    from pathlib import Path
    import experiments.transformer_nlc.qualification as qualifier
    script = r"""
import signal,sys
from pathlib import Path
from dataclasses import replace
from experiments.transformer_nlc.data_config import pilot_config
import experiments.transformer_nlc.data as data
import experiments.transformer_nlc.qualification as qualifier
output=Path(sys.argv[1]);kind=sys.argv[2]
config=pilot_config(16,scored_symbols=256,guard_symbols=128,pilot_symbols=64)
config=replace(config,link=replace(config.link,n_spans=1,span_km=.5))
def pause(*args,**kwargs):
    print('LOCKED',flush=True)
    signal.pause()
if kind=='generation':
    data.fit_receiver=pause
    data.generate(config,output)
else:
    qualifier.analytic_controls=lambda *a,**k:{'status':'passed','CPU_GPU':{'status':'not_run'}}
    qualifier.simulate_physical_frame=pause
    qualifier.run_qualification(16,output,cases=('phase',))
"""
    output=tmp_path/kind
    process=subprocess.Popen([sys.executable,"-c",script,str(output),kind],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        ready,_,_=select.select([process.stdout],[],[],15)
        assert ready and process.stdout.readline().strip()=="LOCKED"
        inode=(output/".generation.lock").stat().st_ino
        # Both entrypoints must reject a concurrent writer on the same inode.
        with pytest.raises(FileExistsError,match="writer"):
            generate(tiny_config(),output,resume=True)
        with pytest.raises(FileExistsError,match="writer"):
            qualifier.run_qualification(16,output,cases=("phase",),resume=True)
        process.kill();process.wait(timeout=5)
        if kind=="generation":
            assert generate(tiny_config(),output,resume=True)["status"]=="complete"
        else:
            # Tiny transport fixtures exercise resumed orchestration only;
            # these three partial-case outputs can never qualify production.
            actual=qualifier.simulate_physical_frame
            monkeypatch.setattr(qualifier,"simulate_physical_frame",lambda *a,**k:actual(tiny_config(),"pilot",0))
            result=qualifier.run_qualification(16,output,cases=("phase",),resume=True)
            assert result["status"]=="incomplete" and len(result["frames"])==3
        assert (output/".generation.lock").stat().st_ino==inode
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=5)
        process.stdout.close();process.stderr.close()


def test_mixed_shared_study_checkouts_fail_before_output_writes(tmp_path):
    from pathlib import Path
    import os
    import shutil
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    other=tmp_path/"other-checkout"
    shutil.copytree(root/"src/optical_comm",other/"src/optical_comm",ignore=shutil.ignore_patterns("__pycache__"))
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(other/"src"),str(root)))}
    script=r'''
import sys
from pathlib import Path
from experiments.transformer_nlc.data import generate
from experiments.transformer_nlc.data_config import pilot_config
from experiments.transformer_nlc.qualification import run_qualification
output=Path(sys.argv[1])
for action in (lambda:generate(pilot_config(),output/'generate'),
               lambda:run_qualification(16,output/'qualify',cases=('baseline',))):
    try:
        action()
    except ValueError as exc:
        assert 'Mixed numerical imports' in str(exc),str(exc)
    else:
        raise AssertionError('Mixed execution was not rejected')
assert not output.exists()
'''
    subprocess.run([sys.executable,"-c",script,str(tmp_path/"must-not-exist")],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)


def test_installed_wheel_data_and_qualification_outside_checkout(tmp_path):
    """Build actual study wheel, isolate its modules, and execute tiny CPU paths."""
    from pathlib import Path
    import os
    import shutil
    import subprocess
    import sys
    import sysconfig
    root=Path(__file__).resolve().parents[1]
    uv=shutil.which("uv")
    if uv is None:
        pytest.skip("uv build is required for the installed-wheel integration control")
    wheel_dir=tmp_path/"wheels"
    subprocess.run([uv,"build","--wheel","--out-dir",str(wheel_dir)],cwd=root,check=True,capture_output=True,text=True)
    wheel=next(wheel_dir.glob("*.whl"))
    environment=tmp_path/"isolated-env"
    subprocess.run([sys.executable,"-m","venv","--system-site-packages",str(environment)],check=True,capture_output=True,text=True)
    python=environment/"bin/python"
    isolated_site=Path(subprocess.check_output([str(python),"-c","import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
    # Add only the current dependency directory after this environment's own
    # wheel. Its .pth files are not executed, preventing editable-source leaks.
    (isolated_site/"research-control-dependencies.pth").write_text(sysconfig.get_path("purelib")+"\n")
    subprocess.run([str(python),"-m","pip","install","--no-deps",str(wheel)],check=True,capture_output=True,text=True)
    env={key:value for key,value in os.environ.items() if key!="PYTHONPATH"}
    env["PYTHONNOUSERSITE"]="1"
    env["PATH"]=str(tmp_path/"no-external-commands")  # Installed paths must work without Git.
    script=r'''
from dataclasses import replace
from pathlib import Path
import json,sys
from experiments.transformer_nlc.data_config import pilot_config
from experiments.transformer_nlc.data import generate,load_frame,source_identity,ensure_bulk_ignored
from experiments.transformer_nlc.qualification import qualification_plan,analytic_controls
output=Path(sys.argv[1]);own_site=Path(sys.argv[2]).resolve()
source=source_identity()
assert source['source_layout']['kind']=='installed_package',source
assert source['commit'] is None and source['dirty_status'] is None
assert all(Path(p).is_relative_to(own_site) for p in source['source_layout']['module_paths'].values()),source
config=pilot_config(16,scored_symbols=256,guard_symbols=128,pilot_symbols=64)
config=replace(config,link=replace(config.link,n_spans=1,span_km=.5))
manifest=generate(config,output)
assert manifest['status']=='complete'
entry=manifest['frames'][0]
assert load_frame(output/entry['path'],expected_sha256=entry['file_sha256']).arrays['rx_nlc'].shape==(config.n_symbols,2)
assert generate(config,output,resume=True)['status']=='complete'
assert qualification_plan((16,))['full_link_propagations']==18
assert analytic_controls(16)['status']=='passed'
try:
    ensure_bulk_ignored(own_site/'must-not-write')
except ValueError as exc:
    assert 'installed package sources' in str(exc)
else:
    raise AssertionError('Installed source protection failed')
print(json.dumps({'fingerprint':source['content_sha256'],'layout':source['source_layout']['kind'],'modules':source['source_layout']['module_paths']}))
'''
    result=subprocess.run([str(python),"-c",script,str(tmp_path/"wheel-run"),str(isolated_site)],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
    from experiments.transformer_nlc.data import source_identity
    receipt=json.loads(result.stdout)
    assert receipt["fingerprint"]==source_identity()["content_sha256"]
    # The actual installed CLI also plans qualification from outside a checkout.
    result=subprocess.run([str(python),"-m","experiments.transformer_nlc.data_cli","qualify","--condition","64","--dry-run"],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)["full_link_propagations"]==18
