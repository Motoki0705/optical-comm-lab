"""Circular-core, scalar paraxial CW fields and transverse power current.

Coordinates are micrometres, fields are normalized (analytic input peak = 1),
and arrays are (y,x). This model has infinite cladding plus a numerical outer
absorber. It is separate from the time-domain communications/Manakov solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import time
from typing import Callable, Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, k0, k1, jn_zeros
from scipy.signal import resample


@dataclass(frozen=True)
class SpatialConfig:
    wavelength_um: float = 1.55
    ncore: float = 1.450
    nclad: float = 1.445
    core_radius_um: float = 25.0
    waist_um: float = 12.0
    tilt_deg: float = 3.0
    azimuth_deg: float = 0.0
    zmax_um: float = 3000.0
    half_window_um: float = 125.0
    dx_um: float = 0.25
    dz_um: float = 0.0625
    absorber_start_um: float = 100.0
    absorber_max_per_um: float = 0.1
    core_center_x_um: float = 0.0
    core_center_y_um: float = 0.0
    input_center_x_um: float = 0.0
    input_center_y_um: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, f.name)) for f in fields(self)):
            raise ValueError("spatial parameters must be finite")
        for name in ("wavelength_um", "ncore", "nclad", "core_radius_um", "waist_um",
                     "zmax_um", "half_window_um", "dx_um", "dz_um"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.ncore <= self.nclad or not self.core_radius_um < self.absorber_start_um < self.half_window_um:
            raise ValueError("need ncore>nclad and core radius<absorber start<window")
        if self.absorber_max_per_um < 0 or abs(self.tilt_deg) >= 15:
            raise ValueError("absorber must be nonnegative; tilt must be within paraxial range")
        n = 2*self.half_window_um/self.dx_um
        if not np.isclose(n, round(n), rtol=0, atol=1e-9) or round(n) < 8 or round(n) % 2:
            raise ValueError("window requires an even integer number of cells >=8")
        if not np.isclose(self.zmax_um/self.dz_um, round(self.zmax_um/self.dz_um), rtol=0, atol=1e-8):
            raise ValueError("zmax must be an integer multiple of dz")
        if math.hypot(self.core_center_x_um,self.core_center_y_um)+self.core_radius_um >= self.absorber_start_um:
            raise ValueError("shifted physical core must remain inside the absorber")

    @property
    def k0_per_um(self):
        return 2*np.pi/self.wavelength_um

    @property
    def beta0_per_um(self):
        return self.k0_per_um*self.ncore

    @property
    def transverse_q_per_um(self):
        q = self.beta0_per_um*np.tan(np.deg2rad(self.tilt_deg))
        return q*np.array([np.cos(np.deg2rad(self.azimuth_deg)), np.sin(np.deg2rad(self.azimuth_deg))])

    @classmethod
    def from_dict(cls, value):
        unknown=set(value)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown spatial settings: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass
class SpatialResult:
    config: SpatialConfig
    kappa: float
    x_um: np.ndarray
    y_um: np.ndarray
    final_field: np.ndarray
    diagnostics: dict[str, np.ndarray]
    input_integral_um2: float
    core_sampled_area_um2: float
    backend: dict[str, Any]
    runtime_seconds: float
    memory: dict[str, int]


def spatial_grid(config: SpatialConfig):
    n=round(2*config.half_window_um/config.dx_um)
    x=(np.arange(n,dtype=float)-n/2+.5)*config.dx_um
    return x, x.copy()


def gaussian_field(config: SpatialConfig, x_um=None, y_um=None):
    x,y=spatial_grid(config) if x_um is None else (np.asarray(x_um),np.asarray(y_um))
    xx,yy=np.meshgrid(x-config.input_center_x_um,y-config.input_center_y_um)
    qx,qy=config.transverse_q_per_um
    return np.exp(-(xx**2+yy**2)/config.waist_um**2+1j*(qx*xx+qy*yy)).astype(np.complex128)


def analytic_free_gaussian(config: SpatialConfig, x_um, y_um, z_um):
    xx,yy=np.meshgrid(np.asarray(x_um)-config.input_center_x_um,np.asarray(y_um)-config.input_center_y_um)
    qx,qy=config.transverse_q_per_um
    beta=config.beta0_per_um
    spread=1+2j*z_um/(beta*config.waist_um**2)
    return np.exp(1j*(qx*xx+qy*yy)-1j*(qx**2+qy**2)*z_um/(2*beta))/spread*np.exp(
        -((xx-qx*z_um/beta)**2+(yy-qy*z_um/beta)**2)/(config.waist_um**2*spread))


def local_flow(field, potential, alpha, kappa, k0_per_um, distance_um):
    """Exact local Kerr phase with amplitude absorption; NumPy reference API."""
    alpha=np.asarray(alpha,dtype=float)
    eff=np.full_like(alpha,distance_um)
    np.divide(-np.expm1(-2*alpha*distance_um),2*alpha,out=eff,where=alpha!=0)
    return np.asarray(field)*np.exp(-alpha*distance_um+1j*np.asarray(potential)*distance_um
                                  +1j*k0_per_um*kappa*abs(field)**2*eff)


def lp01_mode(config: SpatialConfig, x_um=None, y_um=None):
    """Independent scalar step-index Bessel mode and paraxial envelope phase.

    The mode is infinite-cladding LP01 (m=0), with psi(0)=1. Envelope phase
    rate is (beta_mode^2-beta0^2)/(2 beta0), not beta_mode-beta0.
    """
    x,y=spatial_grid(config) if x_um is None else (np.asarray(x_um),np.asarray(y_um))
    v=config.k0_per_um*config.core_radius_um*np.sqrt(config.ncore**2-config.nclad**2)
    upper=min(float(jn_zeros(0,1)[0]),float(v))*(1-1e-10)
    def boundary(u):
        w=np.sqrt(v*v-u*u)
        return u*j1(u)/j0(u)-w*k1(w)/k0(w)
    u=brentq(boundary,1e-8,upper,xtol=1e-13)
    w=np.sqrt(v*v-u*u)
    xx,yy=np.meshgrid(x-config.core_center_x_um,y-config.core_center_y_um)
    r=np.hypot(xx,yy)/config.core_radius_um
    field=np.empty_like(r)
    inside=r<1
    field[inside]=j0(u*r[inside])
    field[~inside]=j0(u)*k0(w*r[~inside])/k0(w)
    beta_sq=config.beta0_per_um**2-(u/config.core_radius_um)**2
    rate=(beta_sq-config.beta0_per_um**2)/(2*config.beta0_per_um)
    return field.astype(np.complex128), dict(u=float(u),w=float(w),v=float(v),beta_mode_squared_per_um2=float(beta_sq),
        envelope_phase_rate_per_um=float(rate),boundary_residual=float(boundary(u)))


def _backend(name):
    if name == "numpy":
        return np, dict(name="numpy",device="CPU",precision="complex128/float64",numpy=np.__version__)
    if name != "cupy":
        raise ValueError("backend must be numpy or cupy")
    try:
        import cupy as cp
        props=cp.cuda.runtime.getDeviceProperties(0)
    except Exception as error:
        raise RuntimeError("explicit cupy backend needs an accessible CUDA device") from error
    return cp, dict(name="cupy",device=props["name"].decode(),precision="complex128/float64",cupy=cp.__version__,
                   cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),device_total_bytes=int(props["totalGlobalMem"]))


def power_current(field, x_um, y_um, beta0_per_um, *, xp=np):
    """Return I,Jx,Jy for a (y,x) field using the spectral derivative."""
    dx=float(x_um[1]-x_um[0]);dy=float(y_um[1]-y_um[0])
    f=xp.fft.fft2(field)
    kx=xp.asarray(2*np.pi*np.fft.fftfreq(len(x_um),dx))
    ky=xp.asarray(2*np.pi*np.fft.fftfreq(len(y_um),dy))
    jx=xp.imag(xp.conj(field)*xp.fft.ifft2(1j*kx[None,:]*f))/beta0_per_um
    jy=xp.imag(xp.conj(field)*xp.fft.ifft2(1j*ky[:,None]*f))/beta0_per_um
    return xp.abs(field)**2,jx,jy


_GPU_LOCAL=None


def propagate_spatial(config: SpatialConfig, kappa: float, *, backend="numpy", initial_field=None,
                      index_enabled=True, absorption_enabled=True, constant_alpha=None,
                      diagnostic_dz_um=5.0, observer: Callable | None=None, observer_dz_um=None):
    """Strang FFT propagation; optional observer sees the complex128 full field.

    Observer signature: observer(z_um, field, xp, x_um, y_um). It must copy
    anything retained; callbacks never define the numerical propagation steps.
    Diagnostics use the complete computational grid and a physical core mask.
    """
    global _GPU_LOCAL
    if not math.isfinite(kappa) or kappa<0:
        raise ValueError("kappa must be finite and nonnegative")
    if constant_alpha is not None and (not math.isfinite(constant_alpha) or constant_alpha<0):
        raise ValueError("constant amplitude absorption must be nonnegative")
    def stride(value):
        count=round(value/config.dz_um)
        if count<1 or not np.isclose(count*config.dz_um,value,rtol=0,atol=1e-9):
            raise ValueError("sample spacing must be a positive multiple of dz")
        if not np.isclose(config.zmax_um/value,round(config.zmax_um/value),rtol=0,atol=1e-8):
            raise ValueError("sample spacing must divide zmax")
        return count
    diagnostic_stride=stride(diagnostic_dz_um)
    observer_stride=stride(observer_dz_um or diagnostic_dz_um)
    started=time.perf_counter()
    xp,backend_record=_backend(backend)
    x,y=spatial_grid(config)
    xx,yy=np.meshgrid(x,y)
    r=np.hypot(xx-config.core_center_x_um,yy-config.core_center_y_um)
    core=r<config.core_radius_um
    n=np.where(core,config.ncore,config.nclad) if index_enabled else np.full_like(r,config.ncore)
    potential=config.k0_per_um**2*(n*n-config.ncore**2)/(2*config.beta0_per_um)
    radius=np.hypot(xx,yy)
    alpha=config.absorber_max_per_um*np.clip((radius-config.absorber_start_um)/(config.half_window_um-config.absorber_start_um),0,1)**4 if absorption_enabled else np.zeros_like(r)
    if constant_alpha is not None:
        alpha=np.full_like(r,constant_alpha)
    half=config.dz_um/2
    eff=np.full_like(alpha,half)
    np.divide(-np.expm1(-2*alpha*half),2*alpha,out=eff,where=alpha!=0)
    linear=xp.asarray(np.exp((-alpha+1j*potential)*half),dtype=xp.complex128)
    phase=xp.asarray(config.k0_per_um*kappa*eff,dtype=xp.float64)
    loss=xp.asarray(-np.expm1(-2*alpha*half),dtype=xp.float64)
    k=2*np.pi*np.fft.fftfreq(len(x),config.dx_um)
    kinetic=xp.asarray(np.exp(-1j*(k[:,None]**2+k[None,:]**2)*config.dz_um/(2*config.beta0_per_um)),dtype=xp.complex128)
    initial=gaussian_field(config,x,y) if initial_field is None else np.asarray(initial_field,dtype=np.complex128)
    if initial.shape!=(len(y),len(x)) or not np.isfinite(initial).all() or not np.any(initial):
        raise ValueError("initial field must be finite, nonzero and match the (y,x) grid")
    field=xp.asarray(initial,dtype=xp.complex128)
    gx,gy,gcore,galpha=map(xp.asarray,(xx,yy,core,alpha))
    p0=float(xp.sum(xp.abs(field)**2))*config.dx_um**2
    absorbed=xp.zeros((),dtype=xp.float64)
    work=xp.empty(field.shape,dtype=xp.float64)
    if backend=="cupy" and _GPU_LOCAL is None:
        _GPU_LOCAL=xp.ElementwiseKernel("complex128 u,complex128 lin,float64 phase,float64 loss", "complex128 v,float64 removed",
            "double I=u.real()*u.real()+u.imag()*u.imag(); double p=phase*I; v=u*lin*complex<double>(cos(p),sin(p)); removed=I*loss;",
            "spatial_exact_loss_kerr")
    names=("z_um","remaining_power","absorbed_power","core_power","peak","core_centroid_x_um","core_centroid_y_um",
           "core_rms_x_um","core_rms_y_um","core_radial_rms_um","global_centroid_x_um","global_centroid_y_um",
           "global_rms_x_um","global_rms_y_um","global_radial_rms_um","loss_rate_per_um")
    records={key:[] for key in names}
    memory=dict(pool_used_high_bytes=0,pool_reserved_high_bytes=0)
    def sample(step):
        I=xp.abs(field)**2
        p=xp.sum(I)*config.dx_um**2
        ic=I*gcore
        pc=xp.sum(ic)*config.dx_um**2
        cx=xp.sum(gx*ic)*config.dx_um**2/pc;cy=xp.sum(gy*ic)*config.dx_um**2/pc
        rx=xp.sum((gx-cx)**2*ic)*config.dx_um**2/pc;ry=xp.sum((gy-cy)**2*ic)*config.dx_um**2/pc
        ax=xp.sum(gx*I)*config.dx_um**2/p;ay=xp.sum(gy*I)*config.dx_um**2/p
        vx=xp.sum((gx-ax)**2*I)*config.dx_um**2/p;vy=xp.sum((gy-ay)**2*I)*config.dx_um**2/p
        vals=xp.stack([p/p0,absorbed/p0,pc/p0,xp.max(I),cx,cy,xp.sqrt(rx),xp.sqrt(ry),xp.sqrt(rx+ry),
                       ax,ay,xp.sqrt(vx),xp.sqrt(vy),xp.sqrt(vx+vy),2*xp.sum(galpha*I)*config.dx_um**2/p0])
        vals=vals.get() if backend=="cupy" else vals
        records["z_um"].append(step*config.dz_um)
        for key,value in zip(names[1:],vals):records[key].append(float(value))
        if backend=="cupy":
            pool=xp.get_default_memory_pool()
            memory["pool_used_high_bytes"]=max(memory["pool_used_high_bytes"],pool.used_bytes())
            memory["pool_reserved_high_bytes"]=max(memory["pool_reserved_high_bytes"],pool.total_bytes())
    nsteps=round(config.zmax_um/config.dz_um)
    sample(0)
    if observer:observer(0.0,field,xp,x,y)
    for step in range(1,nsteps+1):
        for stage in (0,1):
            if backend=="cupy":
                _GPU_LOCAL(field,linear,phase,loss,field,work)
                absorbed+=xp.sum(work)*config.dx_um**2
            else:
                I=xp.abs(field)**2
                absorbed+=xp.sum(I*loss)*config.dx_um**2
                field*=linear*xp.exp(1j*phase*I)
            if stage==0:field=xp.fft.ifft2(xp.fft.fft2(field)*kinetic)
        if step%diagnostic_stride==0:sample(step)
        if observer and step%observer_stride==0:observer(step*config.dz_um,field,xp,x,y)
    if backend=="cupy":xp.cuda.get_current_stream().synchronize()
    final=field.get() if backend=="cupy" else field.copy()
    return SpatialResult(config,kappa,x,y,final,{key:np.asarray(value) for key,value in records.items()},p0,
                         float(np.sum(core)*config.dx_um**2),backend_record,time.perf_counter()-started,memory)


def compare_results(a: SpatialResult,b: SpatialResult, *, region="core"):
    """Fixed-phase complex/intensity comparison; no fitted phase or amplitude."""
    if region not in ("core","full"):
        raise ValueError("comparison region must be core or full")
    f=b.final_field
    if a.config.half_window_um!=b.config.half_window_um:
        if a.config.dx_um!=b.config.dx_um:raise ValueError("window comparison requires matching dx")
        if b.config.half_window_um<a.config.half_window_um:raise ValueError("reference window must contain the compared window")
        start=round((b.config.half_window_um-a.config.half_window_um)/a.config.dx_um)
        f=f[start:start+len(a.x_um),start:start+len(a.y_um)]
    elif a.config.dx_um!=b.config.dx_um:
        k=2*np.pi*np.fft.fftfreq(len(f),b.config.dx_um)
        delta=(a.config.dx_um-b.config.dx_um)/2
        shifted=np.fft.ifft2(np.fft.fft2(f)*np.exp(1j*(k[:,None]+k[None,:])*delta))
        f=resample(resample(shifted,len(a.y_um),axis=0),len(a.x_um),axis=1)
    xx,yy=np.meshgrid(a.x_um-a.config.core_center_x_um,a.y_um-a.config.core_center_y_um)
    mask=xx**2+yy**2<a.config.core_radius_um**2 if region=="core" else np.ones_like(xx,dtype=bool)
    da,db=a.diagnostics,b.diagnostics
    result=dict(region=region,final_complex_relative_l2=float(np.linalg.norm((a.final_field-f)[mask])/np.linalg.norm(f[mask])),
                final_intensity_relative_l2=float(np.linalg.norm((abs(a.final_field)**2-abs(f)**2)[mask])/np.linalg.norm(abs(f[mask])**2)))
    for key in ("core_power","core_centroid_x_um","core_centroid_y_um","core_radial_rms_um"):
        aligned=np.interp(da["z_um"],db["z_um"],db[key])
        result[key+"_max_abs"]=float(np.max(abs(da[key]-aligned)))
    return result


def equal_power_gaussian_seeds(config: SpatialConfig, n_per_axis=5):
    """Fixed product quantiles of the analytic input intensity; no selection."""
    from scipy.special import erfinv
    if isinstance(n_per_axis,bool) or n_per_axis<1 or int(n_per_axis)!=n_per_axis:
        raise ValueError("seed count per axis must be a positive integer")
    q=(np.arange(n_per_axis)+.5)/n_per_axis
    position=config.waist_um/np.sqrt(2)*erfinv(2*q-1)
    xx,yy=np.meshgrid(position+config.input_center_x_um,position+config.input_center_y_um)
    qx,qy=np.meshgrid(q,q)
    return np.column_stack([xx.ravel(),yy.ravel()]),np.column_stack([qx.ravel(),qy.ravel()])


def _volume_sample(volume,xaxis,yaxis,zaxis,position,z,interpolation="linear"):
    """Trilinear or four-point cubic xy / linear z interpolation."""
    x,y=position[:,0],position[:,1]
    ix=np.clip(np.searchsorted(xaxis,x,side="right")-1,0,len(xaxis)-2)
    iy=np.clip(np.searchsorted(yaxis,y,side="right")-1,0,len(yaxis)-2)
    iz=int(np.clip(np.searchsorted(zaxis,z,side="right")-1,0,len(zaxis)-2))
    tx=(x-xaxis[ix])/(xaxis[ix+1]-xaxis[ix]);ty=(y-yaxis[iy])/(yaxis[iy+1]-yaxis[iy])
    tz=(z-zaxis[iz])/(zaxis[iz+1]-zaxis[iz])
    def weights(t):
        return [(-t+2*t*t-t**3)/2,1-2.5*t*t+1.5*t**3,(t+4*t*t-3*t**3)/2,(-t*t+t**3)/2]
    def plane(k):
        if interpolation=="linear":
            return ((1-ty)*((1-tx)*volume[k,iy,ix]+tx*volume[k,iy,ix+1])
                    +ty*((1-tx)*volume[k,iy+1,ix]+tx*volume[k,iy+1,ix+1]))
        wx,wy=weights(tx),weights(ty)
        result=np.zeros_like(tx,dtype=float)
        for a in range(4):
            for b in range(4):
                result+=wy[a]*wx[b]*volume[k,np.clip(iy+a-1,0,len(yaxis)-1),np.clip(ix+b-1,0,len(xaxis)-1)]
        return result
    return (1-tz)*plane(iz)+tz*plane(iz+1)


def integrate_power_streamlines(intensity,jx,jy,x_um,y_um,z_um,seeds_xy_um, *,
                                threshold=1e-4,step_um=.5,interpolation="linear"):
    """Integrate dr_perp/dz=J_perp/I; terminate masked RK stages, never restart.

    Volume axes are (z,y,x). Returned positions are (seed,z,XYZ) micrometres.
    Cubic comparison interpolates the field-derived I,J, not the trajectory.
    """
    x,y,z=map(lambda a:np.asarray(a,dtype=float),(x_um,y_um,z_um))
    shape=(len(z),len(y),len(x))
    if any(a.shape!=shape for a in (intensity,jx,jy)) or min(shape)<2:
        raise ValueError("I,Jx,Jy must share a (z,y,x) volume")
    if interpolation not in ("linear","cubic") or not 0<threshold<1 or step_um<=0:
        raise ValueError("invalid interpolation, positive step or intensity threshold")
    if not all(np.all(np.diff(a)>0) for a in (x,y,z)) or z[0]!=0:
        raise ValueError("coordinates must increase and z must start at zero")
    count=round(z[-1]/step_um)
    if not np.isclose(count*step_um,z[-1],atol=1e-9,rtol=0):
        raise ValueError("flow step must divide the stored propagation length")
    positions=np.asarray(seeds_xy_um,dtype=float).copy()
    if positions.ndim!=2 or positions.shape[1]!=2 or not np.isfinite(positions).all():
        raise ValueError("seed positions must be finite (seed,2)")
    times=np.arange(count+1)*step_um
    paths=np.full((len(times),len(positions),2),np.nan)
    velocities=np.full_like(paths,np.nan)
    sampled_i=np.full(paths.shape[:2],np.nan)
    active=np.ones(len(positions),dtype=bool)
    terminations=[None]*len(positions)
    def evaluate(zz,xy,last_z,stage):
        ids=np.flatnonzero(active)
        speed=np.zeros_like(positions)
        iv=np.full(len(positions),np.nan)
        inside=(np.isfinite(xy[ids]).all(axis=1)&(xy[ids,0]>=x[0])&(xy[ids,0]<=x[-1])
                &(xy[ids,1]>=y[0])&(xy[ids,1]<=y[-1]))
        for idx in ids[~inside]:
            active[idx]=False
            terminations[idx]=dict(reason="outside saved ROI",event_z_um=float(zz),event_xy_um=xy[idx].tolist(),last_accepted_z_um=last_z,stage=stage)
        ids=ids[inside]
        if len(ids):
            i=_volume_sample(intensity,x,y,z,xy[ids],zz,interpolation)
            a=_volume_sample(jx,x,y,z,xy[ids],zz,interpolation)
            b=_volume_sample(jy,x,y,z,xy[ids],zz,interpolation)
            valid=np.isfinite(i)&np.isfinite(a)&np.isfinite(b)&(i>=threshold)
            for idx,value in zip(ids[~valid],i[~valid]):
                active[idx]=False
                terminations[idx]=dict(reason="low intensity or nonfinite current",event_z_um=float(zz),event_xy_um=xy[idx].tolist(),
                                       event_intensity=float(value),last_accepted_z_um=last_z,stage=stage)
            speed[ids[valid]]=np.column_stack([a[valid],b[valid]])/i[valid,None]
            iv[ids[valid]]=i[valid]
        return speed,iv
    v,i=evaluate(0,positions,None,"initial")
    paths[0,active]=positions[active];velocities[0,active]=v[active];sampled_i[0,active]=i[active]
    for index,zz in enumerate(times[:-1]):
        a,_=evaluate(zz,positions,float(zz),"RK1")
        b,_=evaluate(zz+step_um/2,positions+step_um*a/2,float(zz),"RK2")
        c,_=evaluate(zz+step_um/2,positions+step_um*b/2,float(zz),"RK3")
        d,_=evaluate(zz+step_um,positions+step_um*c,float(zz),"RK4")
        new=positions+step_um*(a+2*b+2*c+d)/6
        v,i=evaluate(zz+step_um,new,float(zz),"accepted endpoint")
        paths[index+1,active]=new[active];velocities[index+1,active]=v[active];sampled_i[index+1,active]=i[active]
        positions[active]=new[active]
        if not active.any():break
    valid=np.isfinite(paths).all(axis=-1)
    xyz=np.concatenate([paths,np.broadcast_to(times[:,None,None],(*paths.shape[:2],1))],axis=-1)
    xyz[~valid]=np.nan
    return dict(z_um=times,position_um=xyz.transpose(1,0,2),valid=valid.T,
                velocity_xy_um_per_um=velocities.transpose(1,0,2),intensity=sampled_i.T,
                seed_position_um=np.column_stack([seeds_xy_um,np.zeros(len(seeds_xy_um))]),
                terminations=terminations,threshold=threshold,step_um=step_um,interpolation=interpolation)
