"""Scientific display contracts, without Blender, a GPU, or invented films."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spatial_renderer",ROOT/"scripts/render_spatial_kerr.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
CONFIG = json.loads((ROOT/"configs/spatial_kerr_render.json").read_text())


def test_three_physical_axes_keep_one_scale_and_right_handed_orientation():
    physical = np.eye(3)*100
    world = renderer.physical_to_world(physical)
    np.testing.assert_array_equal(world,np.array([[0,1,0],[0,0,1],[1,0,0]]))
    np.testing.assert_allclose(np.linalg.norm(world[:,None]-world[None,:],axis=-1),np.linalg.norm(physical[:,None]-physical[None,:],axis=-1)/100)
    assert np.linalg.det(world)>0
    np.testing.assert_array_equal(renderer.normals_to_world(np.eye(3)),world)


def test_float32_input_is_not_rounded_again_before_world_coordinate_check():
    value = np.array([[.3,-.125,2999.9375]],dtype=np.float32)
    world = renderer.physical_to_world(value)
    assert world.dtype==np.float64
    np.testing.assert_array_equal(world,value.astype(np.float64)[:,[2,0,1]]/100)


def test_boundary_guides_use_the_real_circle_in_the_reflection_plane():
    rails = renderer.core_boundary_rails_um(CONFIG)
    np.testing.assert_array_equal(rails[:,:,0],[[25,25],[-25,-25]])
    np.testing.assert_array_equal(rails[:,:,1],0)
    np.testing.assert_array_equal(rails[:,:,2],[[0,3000],[0,3000]])
    # Every guide point is on x²+y²=r², including both propagation endpoints.
    np.testing.assert_array_equal(np.sum(rails[:,:,:2]**2,axis=2),25**2)
    np.testing.assert_array_equal(renderer.physical_to_world(rails)[:,:,1],[[.25,.25],[-.25,-.25]])


def test_intro_camera_forward_ray_keeps_the_registered_core_target():
    for frame in range(12,26):
        t = (frame-12)/13
        offset = (1-t)*np.asarray(CONFIG["camera"]["intro_offset_world"])+t*np.asarray(CONFIG["camera"]["offset_world"])
        basis = renderer.intro_camera_basis(frame,CONFIG)
        np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-14)
        np.testing.assert_allclose((basis.T@(-offset))[:2],0,atol=1e-14)
        assert (basis.T@(-offset))[2]<0 and np.linalg.det(basis)>0
    np.testing.assert_allclose(renderer.camera_target_at_frame(1,144,CONFIG,"01_reflection"),[4.8,0,0])
    np.testing.assert_allclose(renderer.camera_target_at_frame(25,144,CONFIG,"01_reflection"),[2.5,0,0])
    # Both comparison components must observe the same physical distance.
    np.testing.assert_allclose(renderer.camera_target_at_frame(90,144,CONFIG,"02_kerr_comparison"),[17.125874125874,0,0])


def test_camera_repair_rejects_unrelated_or_changed_original_provenance(tmp_path):
    original = {"stage":"render","production_movies_complete":True,"provenance":{"renderer_source_sha256":"wrong"}}
    with pytest.raises(ValueError,match="source revision"):
        renderer._repair_guard(original,tmp_path,tmp_path,tmp_path/"missing-config",CONFIG,{}, {})
    original["provenance"].update(renderer_source_sha256=renderer.INTRO_BUG_SOURCE_SHA256,renderer_config_sha256=renderer.CAMERA_REPAIR_CONFIG_SHA256)
    config_path = ROOT/"configs/spatial_kerr_render.json"
    original["effective_config"] = {**CONFIG,"samples":32}
    with pytest.raises(ValueError,match="configuration differs"):
        renderer._repair_guard(original,tmp_path,tmp_path,config_path,CONFIG,{}, {})
    assert not list(tmp_path.iterdir())


def test_camera_repair_rejects_an_otherwise_matching_run_with_another_config(tmp_path):
    alternate = {**CONFIG,"studio":{**CONFIG["studio"],"world_strength":.8}}
    path = tmp_path/"alternate.json"
    path.write_text(json.dumps(alternate))
    original = {"stage":"render","production_movies_complete":True,"effective_config":alternate,"provenance":{"renderer_source_sha256":renderer.INTRO_BUG_SOURCE_SHA256,"renderer_config_sha256":renderer.sha256(path)}}
    with pytest.raises(ValueError,match="configuration hash"):
        renderer._repair_guard(original,tmp_path,tmp_path,path,alternate,{}, {})


@pytest.mark.parametrize("options",[
    ["--mode","repair-cameras","--preflight-only","--repair-state","state.json","--repair-manifest","original.json"],
    ["--mode","repair-cameras","--repair-state","state.json","--repair-manifest","original.json"],
    ["--mode","repair-cameras","--repair-state","state.json","--repair-cache","cache"],
    ["--mode","repair-cameras","--preflight-only"],
    ["--mode","repair-cameras"],
    ["--mode","repair-cameras","--repair-state","state.json","--film","01_reflection"],
    ["--mode","repair-cameras","--repair-state","state.json","--film","01_reflection","02_kerr_comparison","03_phase_change"],
    ["--mode","repair-cameras","--repair-state","state.json","--poster-frame","18"],
    ["--mode","inspect","--preflight-only"],
    ["--mode","render","--repair-manifest","original.json"],
    ["--mode","posters","--repair-cache","cache"],
    ["--mode","verify-scene","--repair-state","state.json"],
])
def test_unsupported_or_ambiguous_repair_requests_fail_before_output_or_input_io(tmp_path,options):
    output = tmp_path/"must-not-be-created"
    with pytest.raises(ValueError):
        renderer.main(options+["--data-dir",str(tmp_path/"nonexistent-input"),"--output-dir",str(output)])
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_supported_prepare_finish_and_normal_requests_preserve_their_contract():
    prepare = renderer.arguments(["--mode","repair-cameras","--preflight-only","--repair-manifest","original.json","--repair-cache","cache"])
    assert prepare.preflight_only and prepare.repair_state is None
    finish = renderer.arguments(["--mode","repair-cameras","--repair-state","state.json"])
    assert not finish.preflight_only and finish.repair_state==Path("state.json")
    normal = renderer.arguments(["--mode","posters","--film","01_reflection","--poster-frame","18"])
    assert normal.film==["01_reflection"] and normal.poster_frame==18
    assert renderer.arguments([]).film==list(renderer.FILMS)


def test_stopped_streamline_never_reconnects_and_final_valid_point_is_kept():
    mask = np.array([[True,True,True,False,False],[True,True,False,False,False],[False,False,False,False,False]])
    prefixes = renderer.flow_prefix_indices(mask,stride=2)
    assert len(prefixes)==3
    np.testing.assert_array_equal(prefixes[0],[0,2])
    np.testing.assert_array_equal(prefixes[1],[0,1])
    assert len(prefixes[2])==0
    invalid = mask.copy()
    invalid[0,4] = True
    with pytest.raises(ValueError,match="restarts"):
        renderer.flow_prefix_indices(invalid)


def test_phase_is_joint_masked_and_wrapped_without_fitted_phase():
    linear = np.array([1,1,1e-6,1],dtype=complex)
    strong = np.array([np.exp(1j*.2),np.exp(1j*(np.pi+.15)),1,1e-6],dtype=complex)
    phase,valid = renderer.phase_difference(linear,strong,1e-4)
    np.testing.assert_array_equal(valid,[True,True,False,False])
    np.testing.assert_allclose(phase[:2],[.2,-np.pi+.15],atol=1e-14)
    assert ((phase>=-np.pi)&(phase<=np.pi)).all()
    rgba = renderer.phase_rgba(phase,valid,CONFIG)
    np.testing.assert_allclose(rgba[2:,:3],np.broadcast_to(CONFIG["masked_phase_color_linear_rgb"],(2,3)),rtol=1e-6)


def test_brightness_mapping_does_not_follow_a_case_maximum():
    a = renderer.intensity_rgba(np.array([.1,.3,.6,1.0]),CONFIG)
    b = renderer.intensity_rgba(np.array([.1,.3,.6,8.0]),CONFIG)
    np.testing.assert_array_equal(a[:3],b[:3])
    assert not np.array_equal(a[3],b[3])
    np.testing.assert_allclose(renderer.palette(np.array([-np.pi,np.pi]),CONFIG["phase_range_rad"],CONFIG["phase_palette_linear_rgb"])[0],renderer.palette(np.array([-np.pi,np.pi]),CONFIG["phase_range_rad"],CONFIG["phase_palette_linear_rgb"])[1])


def test_observation_cursor_uses_actual_nearest_saved_z_not_physical_time():
    z = np.array([0.,.5,1.,1.5,2.])
    assert renderer.nearest_flow_sample(z,1.26)==3
    count = 144
    assert renderer.distance_at_frame(1,count,CONFIG)==250
    assert renderer.distance_at_frame(count,count,CONFIG)==2600
    assert 1400<renderer.distance_at_frame(72,count,CONFIG)<1430
    assert renderer.distance_at_frame(1,count,CONFIG,"01_reflection")==250
    assert renderer.distance_at_frame(24,count,CONFIG,"01_reflection")==250
    assert renderer.distance_at_frame(25,count,CONFIG,"01_reflection")==250
    assert renderer.distance_at_frame(144,count,CONFIG,"01_reflection")==2600
    assert 1170<renderer.distance_at_frame(72,count,CONFIG,"01_reflection")<1190


def test_final_and_proof_are_separate_bounded_cli_modes():
    assert renderer.arguments([]).mode=="posters"
    assert renderer.arguments(["--mode","proof"]).mode=="proof"
    renderer.validate_config(CONFIG)
    invalid = {**CONFIG,"proof":{**CONFIG["proof"],"seconds":4}}
    with pytest.raises(ValueError,match="three seconds"):
        renderer.validate_config(invalid)


def test_missing_input_does_not_create_a_field_or_deliverable(tmp_path):
    with pytest.raises(FileNotFoundError,match="No field, scene, or poster"):
        renderer.verify_inputs(tmp_path,CONFIG)
    assert list(tmp_path.iterdir())==[]


def test_incorrect_numerical_hash_is_rejected_before_loading_fields(tmp_path):
    content = b"not-a-real-field"
    (tmp_path/"linear_sections.npz").write_bytes(content)
    manifest = {"schema_version":1,"coordinate_contract":{"length_unit":"um","axis_orders":{"central_psi":"(z,x)","cross_psi":"(snapshot,y,x)","position_um":"(seed,z,XYZ)"}},"files":{"linear_sections.npz":{"bytes":len(content),"sha256":"0"*64}},"case_contract":{}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="hash/size mismatch"):
        renderer.verify_inputs(tmp_path,CONFIG)
    assert sorted(path.name for path in tmp_path.iterdir())==["linear_sections.npz","manifest.json"]


def test_unsafe_artifact_reference_is_rejected(tmp_path):
    manifest = {"schema_version":1,"coordinate_contract":{"length_unit":"um","axis_orders":{"central_psi":"(z,x)","cross_psi":"(snapshot,y,x)","position_um":"(seed,z,XYZ)"}},"files":{"../external.npz":{"bytes":1,"sha256":"0"*64}},"case_contract":{}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="safe basenames"):
        renderer.verify_inputs(tmp_path,CONFIG)


def test_multifilm_layout_uses_native_pixels_without_stretching():
    assert renderer._component_layout("02_kerr_comparison","comparison_off",CONFIG)==(1920,540,(0,0))
    assert renderer._component_layout("02_kerr_comparison","comparison_on",CONFIG)==(1920,540,(0,540))
    assert renderer._component_layout("03_phase_change","phase",CONFIG)==(1920,600,(0,0))
    assert renderer._component_layout("03_phase_change","intensity_off",CONFIG)==(960,480,(0,600))
    assert renderer._component_layout("03_phase_change","intensity_on",CONFIG)==(960,480,(960,600))


def test_full_render_entrance_export_preserves_native_png_bytes_and_provenance(tmp_path):
    from PIL import Image
    scratch = tmp_path/"cache"
    output = tmp_path/"output"
    source = scratch/"01_reflection"/"reflection"/"frame_0001.png"
    source.parent.mkdir(parents=True)
    output.mkdir()
    # A small saved frame tests host packaging without a fabricated optical field.
    Image.new("RGB",(8,5),(10,70,150)).save(source)
    frame = {"frame":1,"file":source.name,"sha256":renderer.sha256(source),"width":8,"height":5,"samples":7}
    films = [{"id":"01_reflection","frames":{"reflection":{"1":frame}}}]
    provenance = {"renderer_source_sha256":"test-source","renderer_config_sha256":"test-config","numerical_manifest_sha256":"test-input"}
    result = renderer._export_entrance_image(scratch,output,films,provenance)
    target = output/"entrance_01.png"
    assert target.read_bytes()==source.read_bytes()
    assert result["entrance_image"]=={"file":"entrance_01.png","sha256":renderer.sha256(target),"source_film":"01_reflection","source_frame":1}
    context = result["geometry_context"]
    assert (context["width"],context["height"],context["samples"])==(8,5,7)
    assert context["native_source_file"]==str(source)
    assert all(context[key]==value for key,value in provenance.items())


def test_render_without_reflection_does_not_export_or_claim_an_entrance(tmp_path):
    output = tmp_path/"output"
    output.mkdir()
    assert renderer._export_entrance_image(tmp_path/"missing-cache",output,[{"id":"02_kerr_comparison"}],{})=={}
    assert not list(output.iterdir())


def test_entrance_export_rejects_a_changed_cached_frame_before_output(tmp_path):
    source = tmp_path/"cache"/"01_reflection"/"reflection"/"frame_0001.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"changed saved bytes")
    output = tmp_path/"output"
    output.mkdir()
    film = {"id":"01_reflection","frames":{"reflection":{"1":{"frame":1,"file":source.name,"sha256":"0"*64}}}}
    with pytest.raises(ValueError,match="hash differs"):
        renderer._export_entrance_image(tmp_path/"cache",output,[film],{})
    assert not list(output.iterdir())
