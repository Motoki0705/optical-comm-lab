"""Compact spatial-field artifacts, physical isosurfaces and qualification.

Bulk I/J volumes are scratch-only. Numerical fields remain complex128; the
stored rendering/current volumes are explicitly float32, (z,y,x).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.ndimage import map_coordinates

from optical_comm.visualization.spatial import (
    SpatialConfig, SpatialResult, analytic_free_gaussian, compare_results,
    equal_power_gaussian_seeds, integrate_power_streamlines, lp01_mode,
    power_current, propagate_spatial, spatial_grid,
)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def dump_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def result_summary(result):
    d=result.diagnostics
    return dict(kappa=result.kappa,zmax_um=result.config.zmax_um,dx_um=result.config.dx_um,dz_um=result.config.dz_um,
        half_window_um=result.config.half_window_um,backend=result.backend,runtime_seconds=result.runtime_seconds,memory=result.memory,
        final={key:float(value[-1]) for key,value in d.items() if key!="z_um"},
        peak_max_saved=float(d["peak"].max()),min_core_power=float(d["core_power"].min()),
        energy_balance_max_abs=float(np.max(abs(d["remaining_power"]+d["absorbed_power"]-1))),
        continuum_loss_balance_abs=float(abs(d["remaining_power"][-1]+np.trapezoid(d["loss_rate_per_um"],d["z_um"])-1)),
        input_integral_um2=result.input_integral_um2,core_sampled_area_um2=result.core_sampled_area_um2,
        core_area_relative_error=result.core_sampled_area_um2/(np.pi*result.config.core_radius_um**2)-1,
        bulk_townes_norm_ratio=2*result.config.beta0_per_um*result.config.k0_per_um*result.kappa*result.input_integral_um2/11.70)


def qualify_spatial(config,backend,settings,main_results,progress=print):
    """Recorded independent controls and bounded refinements, no fitted phase."""
    started=time.perf_counter()
    controls={}
    free_config=replace(config,zmax_um=200)
    free=propagate_spatial(free_config,0,backend=backend,index_enabled=False,absorption_enabled=False,diagnostic_dz_um=10)
    exact=analytic_free_gaussian(free_config,free.x_um,free.y_um,200)
    controls["free_2d_gaussian"]=dict(complex_relative_l2=float(np.linalg.norm(free.final_field-exact)/np.linalg.norm(exact)),
                                   energy_balance_max_abs=result_summary(free)["energy_balance_max_abs"])
    uniform_config=replace(config,core_radius_um=5,half_window_um=16,absorber_start_um=12,dx_um=.5,zmax_um=1000)
    ux,uy=spatial_grid(uniform_config)
    for alpha in (0.0,2e-4):
        trial=propagate_spatial(uniform_config,3e-4,backend=backend,initial_field=np.ones((len(uy),len(ux))),
            index_enabled=False,absorption_enabled=False,constant_alpha=alpha,diagnostic_dz_um=20)
        eff=1000 if alpha==0 else -np.expm1(-2*alpha*1000)/(2*alpha)
        exact=np.exp(-alpha*1000+1j*config.k0_per_um*3e-4*eff)
        controls[f"uniform_kerr_alpha_{alpha:g}"]=dict(complex_relative_l2=float(np.linalg.norm(trial.final_field-exact)/np.linalg.norm(np.full_like(trial.final_field,exact))),
                                                        energy_balance_max_abs=result_summary(trial)["energy_balance_max_abs"])
    short=float(settings["short_z_um"])
    mode_config=replace(config,zmax_um=short,tilt_deg=0)
    mode,params=lp01_mode(mode_config)
    mode_run=propagate_spatial(mode_config,0,backend=backend,initial_field=mode,diagnostic_dz_um=5)
    exact=mode*np.exp(1j*params["envelope_phase_rate_per_um"]*short)
    controls["lp01_bessel"]=dict(**params,z_um=short,
        complex_relative_l2=float(np.linalg.norm(mode_run.final_field-exact)/np.linalg.norm(exact)),
        intensity_relative_l2=float(np.linalg.norm(abs(mode_run.final_field)**2-abs(mode)**2)/np.linalg.norm(abs(mode)**2)),
        energy_balance_max_abs=result_summary(mode_run)["energy_balance_max_abs"])
    progress("Analytic Gaussian/Kerr and independent LP01 controls complete")
    base=main_results["educational_kerr"]
    coarse=propagate_spatial(replace(config,dz_um=2*config.dz_um),3e-4,backend=backend,diagnostic_dz_um=5)
    captured={}
    def capture(z,field,xp,x,y):
        if np.isclose(z,short):captured["field"]=field.get() if xp.__name__=="cupy" else field.copy()
    fine=propagate_spatial(replace(config,dz_um=config.dz_um/2),3e-4,backend=backend,diagnostic_dz_um=5,
                           observer=capture,observer_dz_um=100)
    a,b=compare_results(coarse,base),compare_results(base,fine)
    partial=replace(fine,config=replace(fine.config,zmax_um=short),final_field=captured["field"],
        diagnostics={key:value[fine.diagnostics["z_um"]<=short] for key,value in fine.diagnostics.items()})
    finer_config=replace(config,dx_um=float(settings["finer_dx_um"]),dz_um=float(settings["finer_dx_dz_um"]),zmax_um=short)
    if finer_config.dz_um!=partial.config.dz_um:raise ValueError("finer-dx comparison needs the same temporal step")
    spatial=propagate_spatial(finer_config,3e-4,backend=backend,diagnostic_dz_um=5)
    spatial_error=compare_results(partial,spatial)
    short_base=propagate_spatial(replace(config,zmax_um=short),3e-4,backend=backend,diagnostic_dz_um=5)
    delta=config.dx_um/2
    shifted_config=replace(config,zmax_um=short,core_center_x_um=delta,core_center_y_um=delta,
                           input_center_x_um=delta,input_center_y_um=delta)
    shifted=propagate_spatial(shifted_config,3e-4,backend=backend,diagnostic_dz_um=5)
    k=2*np.pi*np.fft.fftfreq(len(shifted.x_um),config.dx_um)
    aligned=np.fft.ifft2(np.fft.fft2(shifted.final_field)*np.exp(1j*(k[:,None]+k[None,:])*delta))
    shifted_aligned=replace(shifted,config=short_base.config,final_field=aligned,diagnostics={key:value.copy() for key,value in shifted.diagnostics.items()})
    for key in ("core_centroid_x_um","core_centroid_y_um"):
        shifted_aligned.diagnostics[key]-=delta
    shift_error=compare_results(short_base,shifted_aligned)
    rotated=propagate_spatial(replace(config,zmax_um=short,azimuth_deg=45),3e-4,backend=backend,diagnostic_dz_um=5)
    xx,yy=np.meshgrid(short_base.x_um,short_base.y_um)
    xr,yr=(xx-yy)/np.sqrt(2),(xx+yy)/np.sqrt(2)
    coords=np.array([(yr-rotated.y_um[0])/config.dx_um,(xr-rotated.x_um[0])/config.dx_um])
    aligned=map_coordinates(rotated.final_field.real,coords,order=3,mode="constant",cval=0)+1j*map_coordinates(rotated.final_field.imag,coords,order=3,mode="constant",cval=0)
    rotation_aligned=replace(rotated,config=short_base.config,final_field=aligned,diagnostics={key:value.copy() for key,value in rotated.diagnostics.items()})
    cx,cy=rotated.diagnostics["core_centroid_x_um"],rotated.diagnostics["core_centroid_y_um"]
    rotation_aligned.diagnostics["core_centroid_x_um"]=(cx+cy)/np.sqrt(2)
    rotation_aligned.diagnostics["core_centroid_y_um"]=(-cx+cy)/np.sqrt(2)
    rotation_error=compare_results(short_base,rotation_aligned)
    windows={}
    for name in ("linear","educational_kerr"):
        large=propagate_spatial(replace(config,half_window_um=float(settings["enlarged_half_window_um"]),
                                        absorber_start_um=float(settings["enlarged_absorber_start_um"])),main_results[name].kappa,
                                backend=backend,diagnostic_dz_um=5)
        windows[name]=dict(comparison=compare_results(main_results[name],large),summary=result_summary(large))
    progress("dz, 800 um finer-dx, half-cell, 45-degree and window comparisons complete")
    return dict(controls=controls,dz=dict(coarse_to_base=a,base_to_fine=b,
        complex_error_ratio=a["final_complex_relative_l2"]/b["final_complex_relative_l2"]),
        finer_dx=dict(zmax_um=short,comparison=spatial_error,summary=result_summary(spatial)),
        half_cell_shift=dict(zmax_um=short,shift_xy_um=[delta,delta],comparison=shift_error,
            alignment="Translate shifted field by +half-cell using the spectral phase factor; subtract known physical shift from core centroids; no fitted phase/amplitude"),
        azimuth_45deg=dict(zmax_um=short,comparison=rotation_error,alignment="Cubic interpolation after physical coordinate rotation; no phase/amplitude fit"),
        window=windows,runtime_seconds=time.perf_counter()-started,
        status="bounded qualification; full 3 mm finer-dx/full-Maxwell convergence not established",
        limitations=["dx=0.125 um compared only through 800 um including first reflection", "Half-cell and rotation checks limited to 800 um",
                     "LP01 control is independently analytic but its point-sampled grid error remains", "Weak-reference absolute changes below grid/model error are not certified"])


def extract_isosurface(intensity_zyx,x_um,y_um,z_um,level):
    """Marching cubes with physical spacing; output vertices/normals in XYZ."""
    from skimage.measure import marching_cubes,mesh_surface_area
    x,y,z=map(np.asarray,(x_um,y_um,z_um))
    if intensity_zyx.shape!=(len(z),len(y),len(x)) or not np.isfinite(level):raise ValueError("invalid volume/level")
    spacing=(float(z[1]-z[0]),float(y[1]-y[0]),float(x[1]-x[0]))
    if not all(np.allclose(np.diff(axis),step) and step>0 for axis,step in zip((z,y,x),spacing)):
        raise ValueError("marching cubes needs increasing uniform physical axes")
    verts,faces,normals,values=marching_cubes(np.asarray(intensity_zyx,dtype=np.float32),level=float(level),spacing=spacing,
        gradient_direction="descent",allow_degenerate=False,method="lewiner")
    xyz=verts[:,[2,1,0]]+np.array([x[0],y[0],z[0]])
    # skimage's descent faces use the left-hand rule; reversing the axis order
    # from ZYX to XYZ makes their geometric normals outward in XYZ.
    faces=faces.astype(np.int32)
    normal=(normals/np.asarray(spacing))[:,[2,1,0]]
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    return dict(vertices_um=xyz.astype(np.float32),faces=faces,normals_xyz=normal.astype(np.float32),level=np.array(level),
                area_um2=float(mesh_surface_area(xyz,faces)),bounds_um=np.array([xyz.min(axis=0),xyz.max(axis=0)]),
                vertex_count=len(xyz),triangle_count=len(faces))


class _CaseRecorder:
    def __init__(self,config,sampling,scratch):
        self.config=config;self.sampling=sampling;self.scratch=Path(scratch)
        self.scratch.mkdir(parents=True,exist_ok=False)
        x,y=spatial_grid(config)
        factor=round(sampling["saved_dx_um"]/config.dx_um)
        if factor<1 or not np.isclose(factor*config.dx_um,sampling["saved_dx_um"]):raise ValueError("save dx must be an integer multiple")
        indices=np.flatnonzero(abs(x)<sampling["roi_half_width_um"])[::factor]
        self.slice=slice(indices[0],indices[-1]+1,factor)
        self.x=x[self.slice];self.y=y[self.slice]
        self.z=np.arange(round(config.zmax_um/sampling["volume_dz_um"])+1)*sampling["volume_dz_um"]
        shape=(len(self.z),len(self.y),len(self.x))
        self.volume={key:np.lib.format.open_memmap(self.scratch/(key+".npy"),mode="w+",dtype=np.float32,shape=shape) for key in ("I","Jx","Jy")}
        self.central=[];self.cross=[];self.cross_z=[];self.capture={};self.alias_fractions=[]
        self.cross_times=sampling["cross_z_um"]
    def observe(self,z,field,xp,x,y):
        index=round(z/self.sampling["volume_dz_um"])
        I,Jx,Jy=power_current(field,x,y,self.config.beta0_per_um,xp=xp)
        for key,value in zip(("I","Jx","Jy"),(I,Jx,Jy)):
            roi=value[self.slice,self.slice].astype(xp.float32)
            self.volume[key][index]=roi.get() if xp.__name__=="cupy" else roi
        centre=(field[len(y)//2-1,self.slice]+field[len(y)//2,self.slice])/2
        self.central.append(centre.get() if xp.__name__=="cupy" else centre.copy())
        if any(np.isclose(z,value,rtol=0,atol=1e-9) for value in self.cross_times):
            roi=field[self.slice,self.slice]
            self.cross.append(roi.get() if xp.__name__=="cupy" else roi.copy());self.cross_z.append(z)
            f=xp.abs(xp.fft.fft2(field))**2
            k=xp.asarray(2*np.pi*np.fft.fftfreq(len(x),self.config.dx_um))
            high=(xp.abs(k[:,None])>np.pi/self.sampling["saved_dx_um"])|(xp.abs(k[None,:])>np.pi/self.sampling["saved_dx_um"])
            self.alias_fractions.append(float(xp.sum(f*high)/xp.sum(f)))
        if np.isclose(z,800):self.capture[800]=field.get() if xp.__name__=="cupy" else field.copy()
    def finish(self,result,path):
        for volume in self.volume.values():volume.flush()
        data=dict(x_um=self.x,y_um=self.y,z_um=self.z,central_psi=np.asarray(self.central),cross_z_um=np.array(self.cross_z),
                  cross_psi=np.asarray(self.cross),diagnostic_z_um=result.diagnostics["z_um"])
        data.update({key:value for key,value in result.diagnostics.items() if key!="z_um"})
        np.savez_compressed(path,**data)
        return dict(saved_spacing_xy_um=float(self.x[1]-self.x[0]),saved_z_spacing_um=self.sampling["volume_dz_um"],
            coordinate_bounds_xy_um=[float(self.x[0]),float(self.x[-1])],original_dx_um=self.config.dx_um,
            decimation_stride=round(self.sampling["saved_dx_um"]/self.config.dx_um),
            central_y0_interpolation="Complex linear interpolation of nearest symmetric full-grid rows; saved x matches cross_psi",
            complex_dtype="complex128",scratch_I_J_dtype="float32; current computed in float64 and quantized only for storage",
            sampled_power_fraction_beyond_saved_xy_Nyquist_max=float(max(self.alias_fractions)),
            scratch_bytes=sum(v.nbytes for v in self.volume.values()))


def _flow_summary(flow):
    completed=sum(reason is None for reason in flow["terminations"])
    return dict(seed_count=len(flow["terminations"]),completed=completed,terminations=flow["terminations"],
        minimum_accepted_intensity=float(np.nanmin(flow["intensity"])),
        max_abs_velocity_component=float(np.nanmax(abs(flow["velocity_xy_um_per_um"]))),
        threshold=flow["threshold"],step_um=flow["step_um"],interpolation=flow["interpolation"])


def _flow_compare(a,b):
    records=[]
    for seed in range(len(a["valid"])):
        va,vb=a["valid"][seed],b["valid"][seed]
        if not va.any() or not vb.any():
            records.append(dict(seed=seed,max_distance_um=None));continue
        end=min(a["z_um"][va][-1],b["z_um"][vb][-1]);mask=va&(a["z_um"]<=end)
        aligned=np.column_stack([np.interp(a["z_um"][mask],b["z_um"][vb],b["position_um"][seed,vb,k]) for k in (0,1)])
        records.append(dict(seed=seed,common_end_um=float(end),max_distance_um=float(np.max(np.linalg.norm(a["position_um"][seed,mask,:2]-aligned,axis=1))),
                            termination_matches=(a["terminations"][seed] is None)==(b["terminations"][seed] is None)))
    return records


def _figures(output,sections,cases,flows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=list(cases);peak=max(float(np.max(abs(s["central_psi"])**2)) for s in sections.values())
    fig,axs=plt.subplots(2,2,figsize=(13,7.5),sharex=True,sharey=True,layout="constrained")
    for ax,name in zip(axs.flat,names):
        s=sections[name];extent=[s["z_um"][0]/1000,s["z_um"][-1]/1000,s["x_um"][0],s["x_um"][-1]]
        im=ax.imshow((abs(s["central_psi"])**2).T,origin="lower",extent=extent,aspect="auto",interpolation="nearest",cmap="magma",vmin=0,vmax=peak)
        ax.axhline(25,color="#54d1e1",ls="--",lw=.7);ax.axhline(-25,color="#54d1e1",ls="--",lw=.7)
        ax.set(title=f"{name}, kappa={cases[name]:g}",xlabel="z (mm)",ylabel="x (um)")
    fig.colorbar(im,ax=axs,label="I / analytic input peak I",shrink=.8)
    fig.supxlabel("Circular-core scalar paraxial CW fields; normalized educational coefficients. Same color scale for all cases.",fontsize=9)
    fig.savefig(output/"longitudinal_comparison.png",dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True,layout="constrained")
    for name in names:
        s=sections[name]
        for ax,key in zip(axs.flat,["core_radial_rms_um","core_centroid_x_um","core_power","peak"]):
            ax.plot(s["diagnostic_z_um"]/1000,s[key],label=name,lw=1)
    for ax,title in zip(axs.flat,["Core-conditioned radial RMS width","Core-conditioned x centroid","Power inside core / input","Peak over full numerical grid"]):
        ax.set(title=title,xlabel="z (mm)");ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.savefig(output/"core_observables.png",dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
    base,strong=sections["linear"],sections["educational_kerr"]
    a,b=abs(base["central_psi"])**2,abs(strong["central_psi"])**2
    diff=b-a;limit=float(abs(diff).max());phase=np.angle(strong["central_psi"]*base["central_psi"].conj())
    cmap=plt.colormaps["RdBu_r"].with_extremes(bad="#181b22")
    extent=[0,base["z_um"][-1]/1000,base["x_um"][0],base["x_um"][-1]]
    im=axs[0].imshow(diff.T,origin="lower",extent=extent,aspect="auto",interpolation="nearest",cmap=cmap,vmin=-limit,vmax=limit)
    fig.colorbar(im,ax=axs[0],label="Kerr intensity - linear intensity")
    im=axs[1].imshow(np.ma.array(phase,mask=(a<1e-4)|(b<1e-4)).T,origin="lower",extent=extent,aspect="auto",interpolation="nearest",cmap=cmap,vmin=-np.pi,vmax=np.pi)
    fig.colorbar(im,ax=axs[1],label="wrapped phase difference (rad)")
    for ax in axs:ax.set(ylabel="x (um)",xlabel="z (mm)")
    fig.savefig(output/"kerr_difference.png",dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
    for ax,name in zip(axs,["linear","educational_kerr"]):
        flow=flows[name]
        for seed in range(len(flow["valid"])):
            valid=flow["valid"][seed];xyz=flow["position_um"][seed,valid]
            if len(xyz):ax.plot(xyz[:,2]/1000,xyz[:,0],lw=.6,alpha=.8)
        ax.axhline(25,color="0.5",ls="--");ax.axhline(-25,color="0.5",ls="--")
        ax.set(title=f"All 25 computed 3D power-flow streamlines: {name} (x projection)",ylabel="x (um)",xlabel="z (mm)")
    fig.savefig(output/"power_flow_projection.png",dpi=130);plt.close(fig)


def generate_spatial_artifacts(config_path,output,scratch, *, backend=None,qualify=True,progress=print):
    """Complete deterministic export into a new directory, with source hashes."""
    started=time.perf_counter();config_path=Path(config_path).resolve();output=Path(output).resolve();scratch=Path(scratch).resolve()
    cfg=json.loads(config_path.read_text());model=SpatialConfig.from_dict(cfg["model"])
    if cfg["schema_version"]!=1 or list(cfg["cases"].values())!=[0,1.6e-11,1e-4,3e-4]:raise ValueError("four registered cases required")
    if output.exists() and any(output.iterdir()):raise FileExistsError("use a fresh output directory")
    if scratch.exists() and any(scratch.iterdir()):raise FileExistsError("use a fresh scratch directory; previous evidence is read-only")
    root=Path(__file__).resolve().parents[3]
    if scratch.is_relative_to(root) and not scratch.is_relative_to(root/".work"):
        raise ValueError("in-repository bulk scratch must be inside ignored .work")
    output.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    backend=backend or cfg["backend"]
    source_paths=[root/"src/optical_comm/visualization/spatial.py",Path(__file__),root/"scripts/simulate_spatial_kerr.py",root/"pyproject.toml",root/"uv.lock"]
    source_hashes={str(p.relative_to(root)):digest(p) for p in source_paths}
    (output/"config.json").write_bytes(config_path.read_bytes())
    results={};recorders={};sections={};saved_sampling={}
    for name,kappa in cfg["cases"].items():
        recorder=_CaseRecorder(model,cfg["sampling"],scratch/name)
        result=propagate_spatial(model,kappa,backend=backend,diagnostic_dz_um=cfg["sampling"]["diagnostic_dz_um"],
                                 observer=recorder.observe,observer_dz_um=cfg["sampling"]["volume_dz_um"])
        saved_sampling[name]=recorder.finish(result,output/(name+"_sections.npz"))
        results[name]=result;recorders[name]=recorder
        with np.load(output/(name+"_sections.npz")) as data:sections[name]={key:data[key] for key in data.files}
        progress(f"Saved {name}: {result.runtime_seconds:.1f}s, peak {result.diagnostics['peak'].max():.4f}")
    qualification=qualify_spatial(model,backend,cfg["qualification"],results,progress) if qualify else dict(status="not executed")
    flows={};flow_checks={};mesh_metrics={};contract={}
    seeds,quantiles=equal_power_gaussian_seeds(model,cfg["flow"]["seeds_per_axis"])
    for name,recorder in recorders.items():
        args=(recorder.volume["I"],recorder.volume["Jx"],recorder.volume["Jy"],recorder.x,recorder.y,recorder.z,seeds)
        flow=integrate_power_streamlines(*args,threshold=cfg["flow"]["intensity_threshold"],step_um=cfg["flow"]["step_um"],interpolation=cfg["flow"]["interpolation"])
        flow["seed_quantiles"]=quantiles
        np.savez_compressed(output/(name+"_flows.npz"),**{key:flow[key] for key in
            ("z_um","position_um","valid","seed_quantiles","seed_position_um","intensity","velocity_xy_um_per_um")})
        flows[name]=flow
        contract[name]=dict(kappa=cfg["cases"][name],sections_file=name+"_sections.npz",flows_file=name+"_flows.npz",mesh_files=[])
        if name in ("linear","educational_kerr"):
            finer=integrate_power_streamlines(*args,threshold=cfg["flow"]["intensity_threshold"],step_um=cfg["flow"]["comparison_step_um"])
            cubic=integrate_power_streamlines(*args,threshold=cfg["flow"]["intensity_threshold"],step_um=cfg["flow"]["step_um"],interpolation="cubic")
            flow_checks[name]=dict(integration_step=_flow_compare(flow,finer),interpolation=_flow_compare(flow,cubic),
                                  comparison_terminations=finer["terminations"],cubic_terminations=cubic["terminations"])
            mesh_metrics[name]={}
            for level in cfg["isosurface_levels"]:
                label=f"0p{round(level*10)}";mesh=extract_isosurface(recorder.volume["I"],recorder.x,recorder.y,recorder.z,level)
                path=output/(name+"_iso_"+label+".npz")
                np.savez_compressed(path,**{key:mesh[key] for key in ("vertices_um","faces","normals_xyz","level")})
                coarse=extract_isosurface(recorder.volume["I"][::2],recorder.x,recorder.y,recorder.z[::2],level)
                mesh_metrics[name][label]={key:mesh[key] for key in ("area_um2","vertex_count","triangle_count")}
                mesh_metrics[name][label].update(bounds_um=mesh["bounds_um"].tolist(),z_spacing_area_relative_difference=abs(mesh["area_um2"]-coarse["area_um2"])/mesh["area_um2"],
                                                 z_spacing_bounds_max_abs_um=float(np.max(abs(mesh["bounds_um"]-coarse["bounds_um"]))))
                contract[name]["mesh_files"].append(path.name)
        progress(f"Exported {name} flows and fixed-threshold meshes")
    _figures(output,sections,cfg["cases"],flows)
    case_metrics={}
    for name,result in results.items():
        delta=compare_results(results["linear"],result)
        a,b=results["linear"].final_field,result.final_field
        xx,yy=np.meshgrid(result.x_um,result.y_um);core=xx**2+yy**2<model.core_radius_um**2
        valid=core&(abs(a)**2>=1e-4)&(abs(b)**2>=1e-4)
        phase=np.angle(b*np.conj(a));weights=np.minimum(abs(a)**2,abs(b)**2)*valid
        delta["core_masked_wrapped_phase_rms_rad"]=float(np.sqrt(np.sum(phase**2*weights)/np.sum(weights)))
        case_metrics[name]=dict(kappa=result.kappa,summary=result_summary(result),difference_vs_linear=delta)
    limitations=list(cfg["scope"])+["Numerical diagnostics use the full grid; saved section/mesh/current ROI is fixed and decimated",
        "J/I defines power-flow streamlines, not photons; all seeds and masked terminations retained", "Meshes are fixed iso-intensity surfaces, not paths or per-slice normalized brightness"]
    metrics=dict(schema_version=1,cases=case_metrics,qualification=qualification,flows=dict(cases={name:_flow_summary(flow) for name,flow in flows.items()},comparisons=flow_checks),
                 meshes=mesh_metrics,sampling=saved_sampling,limitations=limitations,runtime_seconds=time.perf_counter()-started)
    dump_json(output/"metrics.json",metrics)
    for relative,expected in source_hashes.items():
        if digest(root/relative)!=expected:raise RuntimeError("numerical sources changed during generation: "+relative)
    files={}
    for path in sorted(output.iterdir()):
        if path.is_file():
            kind="sections" if path.name.endswith("_sections.npz") else "flows" if path.name.endswith("_flows.npz") else "mesh" if "_iso_" in path.name else "figure" if path.suffix==".png" else "metadata"
            case=next((name for name in cfg["cases"] if path.name.startswith(name+"_")),None)
            files[path.name]=dict(sha256=digest(path),bytes=path.stat().st_size,kind=kind,case=case)
    import scipy
    import skimage
    manifest=dict(schema_version=1,status="complete_with_limits",created_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=source_hashes,input_config=dict(path=str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),sha256=digest(config_path)),
        execution=dict(**results["linear"].backend,backend=results["linear"].backend["name"],
            versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,
                          cupy=results["linear"].backend.get("cupy"))),
        coordinate_contract=dict(length_unit="um",vertices="XYZ=(transverse x,transverse y,propagation z)",
            axis_orders=dict(central_psi="(z,x)",cross_psi="(snapshot,y,x)",position_um="(seed,z,XYZ)",scratch_volume="(z,y,x)")),
        case_contract=contract,sampling=saved_sampling,computational_config=model.to_dict(),
        seed_contract="Deterministic 5x5 product equal-input-power quantiles; no random variables or seed selection",files=files,
        regeneration_command=".venv/bin/python scripts/simulate_spatial_kerr.py --config configs/spatial_kerr.json --output .work/spatial-reproduction --scratch .work/spatial-reproduction-bulk",
        limitations=limitations,independent_validator_count=0)
    dump_json(output/"manifest.json",manifest)
    return metrics,manifest


def verify_spatial_artifacts(output, *, source_root=None):
    """Verify saved hashes plus coordinate/data/mesh/termination semantics."""
    output=Path(output);manifest=json.loads((output/"manifest.json").read_text())
    records={}
    for name,entry in manifest["files"].items():
        path=output/name
        if path.stat().st_size!=entry["bytes"] or digest(path)!=entry["sha256"]:raise ValueError("artifact mismatch: "+name)
    if source_root is not None:
        for name,expected in manifest["source_hashes"].items():
            if digest(Path(source_root)/name)!=expected:raise ValueError("source mismatch: "+name)
        input_path=Path(manifest["input_config"]["path"])
        input_path=input_path if input_path.is_absolute() else Path(source_root)/input_path
        if digest(input_path)!=manifest["input_config"]["sha256"]:raise ValueError("input configuration mismatch")
    for name,case in manifest["case_contract"].items():
        with np.load(output/case["sections_file"]) as s:
            if s["central_psi"].shape!=(len(s["z_um"]),len(s["x_um"])) or s["cross_psi"].shape!=(len(s["cross_z_um"]),len(s["y_um"]),len(s["x_um"])):raise ValueError("section axis mismatch")
            if s["central_psi"].dtype!=np.complex128 or s["cross_psi"].dtype!=np.complex128 or not np.isfinite(s["central_psi"]).all() or not np.isfinite(s["cross_psi"]).all():raise ValueError("invalid complex field")
        with np.load(output/case["flows_file"]) as f:
            valid=f["valid"]
            if f["position_um"].shape!=(*valid.shape,3) or not np.isfinite(f["position_um"][valid]).all() or not np.isnan(f["position_um"][~valid]).all():raise ValueError("invalid flow mask")
            for row in valid:
                if np.any(np.diff(row.astype(int))>0):raise ValueError("flow restarted after termination")
            if not np.allclose(f["position_um"][:,:,2][valid],np.broadcast_to(f["z_um"],valid.shape)[valid]):raise ValueError("flow XYZ mismatch")
            if (f["intensity"][valid]<1e-4).any():raise ValueError("accepted masked flow point")
        for meshfile in case["mesh_files"]:
            with np.load(output/meshfile) as mesh:
                verts,faces=mesh["vertices_um"],mesh["faces"]
                if verts.ndim!=2 or verts.shape[1]!=3 or faces.shape[1]!=3 or not np.isfinite(verts).all() or faces.min()<0 or faces.max()>=len(verts):raise ValueError("invalid mesh")
        records[name]=dict(section_axes=True,complex128=True,all_flow_masks_valid=True,mesh_indices_valid=True)
    return dict(schema_version=1,all_file_hashes_match=True,source_hashes_checked=source_root is not None,cases=records,
                total_bytes=sum(e["bytes"] for n,e in manifest["files"].items() if n!="verification.json"),max_file_bytes=max(e["bytes"] for e in manifest["files"].values()))
