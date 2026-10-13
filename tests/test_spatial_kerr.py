"""Independent small analytic limits, streamlines and saved XYZ semantics."""
from dataclasses import replace
import json

import numpy as np
import pytest

from optical_comm.visualization.spatial import (
    SpatialConfig,analytic_free_gaussian,compare_results,equal_power_gaussian_seeds,
    gaussian_field,integrate_power_streamlines,local_flow,lp01_mode,power_current,
    propagate_spatial,spatial_grid,
)


def small_config(**values):
    return SpatialConfig(core_radius_um=5,waist_um=3,half_window_um=20,absorber_start_um=15,
                         dx_um=.25,dz_um=.125,zmax_um=20,tilt_deg=2,**values)


def test_two_dimensional_gaussian_complex_phase_and_power():
    c=small_config()
    r=propagate_spatial(c,0,index_enabled=False,absorption_enabled=False,diagnostic_dz_um=2)
    exact=analytic_free_gaussian(c,r.x_um,r.y_um,c.zmax_um)
    assert np.linalg.norm(r.final_field-exact)/np.linalg.norm(exact)<2e-12
    assert max(abs(r.diagnostics["remaining_power"]-1))<2e-12
    assert r.backend["name"]=="numpy" and r.final_field.dtype==np.complex128


@pytest.mark.parametrize("alpha",[0.0,2e-4])
def test_uniform_kerr_phase_including_loss(alpha):
    c=replace(small_config(),zmax_um=10)
    x,y=spatial_grid(c)
    r=propagate_spatial(c,3e-4,initial_field=np.ones((len(y),len(x))),index_enabled=False,
                         absorption_enabled=False,constant_alpha=alpha,diagnostic_dz_um=1)
    eff=c.zmax_um if alpha==0 else -np.expm1(-2*alpha*c.zmax_um)/(2*alpha)
    exact=np.exp(-alpha*c.zmax_um+1j*c.k0_per_um*3e-4*eff)
    assert np.max(abs(r.final_field-exact))<2e-12
    assert np.max(abs(r.diagnostics["remaining_power"]+r.diagnostics["absorbed_power"]-1))<2e-12


def test_exact_local_flow_nonuniform_absorber_and_index():
    field=np.array([1+1j,.5-.2j,.3+.4j])
    alpha=np.array([0,.01,.1]);potential=np.array([.02,0,-.03]);g=.003;length=7
    result=local_flow(field,potential,alpha,g,4.0,length)
    # Independent integrated intensity phase, with the alpha=0 limit explicit.
    phase=np.array([length,(1-np.exp(-.14))/.02,(1-np.exp(-1.4))/.2])*4*g*abs(field)**2+potential*length
    assert np.allclose(result,field*np.exp(-alpha*length)*np.exp(1j*phase),atol=2e-15)


def test_independent_lp01_bessel_boundary_and_envelope_phase():
    c=replace(small_config(),core_radius_um=5,tilt_deg=0,zmax_um=10)
    mode,p=lp01_mode(c)
    assert abs(p["boundary_residual"])<1e-10
    assert p["envelope_phase_rate_per_um"]<0
    assert np.isclose(p["envelope_phase_rate_per_um"],(p["beta_mode_squared_per_um2"]-c.beta0_per_um**2)/(2*c.beta0_per_um))
    r=propagate_spatial(c,0,initial_field=mode,absorption_enabled=False,diagnostic_dz_um=1)
    exact=mode*np.exp(1j*p["envelope_phase_rate_per_um"]*10)
    assert np.linalg.norm(r.final_field-exact)/np.linalg.norm(exact)<.004


def test_free_gaussian_power_current_streamlines_follow_analytic_spreading():
    c=small_config();x,y=spatial_grid(c);z=np.arange(21,dtype=float)
    values=[power_current(analytic_free_gaussian(c,x,y,t),x,y,c.beta0_per_um) for t in z]
    I,Jx,Jy=map(np.array,zip(*values))
    seeds,q=equal_power_gaussian_seeds(c,3)
    flow=integrate_power_streamlines(I,Jx,Jy,x,y,z,seeds,step_um=.125)
    zr=c.beta0_per_um*c.waist_um**2/2
    ratio=np.sqrt(1+(z[-1]/zr)**2)
    exact=seeds*ratio+c.transverse_q_per_um*z[-1]/c.beta0_per_um
    assert flow["valid"].all()
    assert np.max(np.linalg.norm(flow["position_um"][:,-1,:2]-exact,axis=1))<.008
    assert q.shape==(9,2)


def test_streamline_masks_domain_exit_and_initial_node_do_not_restart():
    x=y=np.array([-1.,0.,1.]);z=np.array([0.,1.,2.])
    I=np.ones((3,3,3));J=np.ones_like(I)*2;zero=np.zeros_like(I)
    flow=integrate_power_streamlines(I,J,zero,x,y,z,[[0,0]],step_um=.5)
    assert flow["terminations"][0]["reason"]=="outside saved ROI"
    assert np.array_equal(flow["valid"][0],[True,True,False,False,False])
    assert np.isnan(flow["position_um"][0,2:]).all()
    masked=integrate_power_streamlines(I*1e-5,zero,zero,x,y,z,[[0,0]],step_um=.5)
    assert not masked["valid"].any()
    assert np.isnan(masked["position_um"]).all()
    assert masked["terminations"][0]["stage"]=="initial"


def test_config_rejects_radially_outside_core_and_unknown_comparison_region():
    with pytest.raises(ValueError):
        replace(small_config(),core_center_x_um=8,core_center_y_um=8)
    with pytest.raises(ValueError):SpatialConfig.from_dict({"power_w":1})
    c=replace(small_config(),zmax_um=1)
    r=propagate_spatial(c,0,diagnostic_dz_um=1)
    with pytest.raises(ValueError):compare_results(r,r,region="corre")


def test_physical_marching_cube_xyz_and_normals():
    pytest.importorskip("skimage")
    from optical_comm.visualization.spatial_artifacts import extract_isosurface
    x=np.arange(-4,4.01,.25);y=np.arange(-3,3.01,.25);z=np.arange(0,8.01,.5)
    zz,yy,xx=np.meshgrid(z,y,x,indexing="ij");centre=np.array([.5,-.5,4.])
    I=np.exp(-((xx-centre[0])**2+(yy-centre[1])**2+(zz-centre[2])**2)/1.5**2)
    mesh=extract_isosurface(I,x,y,z,np.exp(-1))
    bounds=mesh["bounds_um"]
    assert np.max(abs(bounds-np.array([centre-1.5,centre+1.5])))<.06
    verts=mesh["vertices_um"];normals=mesh["normals_xyz"]
    assert np.median(np.sum((verts-centre)*normals,axis=1))>1.4
    assert mesh["faces"].dtype==np.int32
    faces=mesh["faces"]
    face_normals=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
    assert np.median(np.sum(face_normals*(verts[faces].mean(axis=1)-centre),axis=1))>0


def test_gpu_parity_when_cuda_available():
    cp=pytest.importorskip("cupy")
    try:cp.cuda.runtime.getDeviceProperties(0)
    except Exception:pytest.skip("no accessible CUDA device")
    c=replace(small_config(),zmax_um=1)
    a=propagate_spatial(c,3e-4,backend="numpy",diagnostic_dz_um=1)
    b=propagate_spatial(c,3e-4,backend="cupy",diagnostic_dz_um=1)
    assert np.linalg.norm(a.final_field-b.final_field)/np.linalg.norm(a.final_field)<2e-12


def test_compact_generation_and_audit_detect_corruption(tmp_path):
    pytest.importorskip("skimage")
    from optical_comm.visualization.spatial_artifacts import generate_spatial_artifacts,verify_spatial_artifacts
    c=replace(small_config(),zmax_um=2)
    cfg={"schema_version":1,"model":c.to_dict(),"backend":"numpy",
         "cases":{"linear":0,"weak_reference":1.6e-11,"intermediate_kerr":1e-4,"educational_kerr":3e-4},
         "sampling":{"diagnostic_dz_um":1,"volume_dz_um":.5,"roi_half_width_um":10,"saved_dx_um":.5,"cross_z_um":[0,1,2]},
         "flow":{"seeds_per_axis":2,"intensity_threshold":1e-4,"step_um":.5,"comparison_step_um":.25,"interpolation":"linear"},
         "isosurface_levels":[.1,.3,.6],"scope":["small CPU verification"],"qualification":{}}
    path=tmp_path/"config.json";path.write_text(json.dumps(cfg))
    out=tmp_path/"output";scratch=tmp_path/"bulk"
    metrics,manifest=generate_spatial_artifacts(path,out,scratch,qualify=False,progress=lambda _:None)
    check=verify_spatial_artifacts(out)
    assert len(check["cases"])==4 and check["all_file_hashes_match"]
    assert manifest["execution"]["backend"]=="numpy"
    assert metrics["qualification"]["status"]=="not executed"
    with (out/"linear_sections.npz").open("ab") as f:f.write(b"corrupt")
    with pytest.raises(ValueError,match="artifact mismatch"):verify_spatial_artifacts(out)
