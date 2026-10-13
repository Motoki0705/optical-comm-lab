#!/usr/bin/env python3
"""Generate/audit circular-core CW data for spatial Kerr visualization."""
from pathlib import Path
import argparse
import json
from optical_comm.visualization.spatial_artifacts import generate_spatial_artifacts,verify_spatial_artifacts,dump_json,digest

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/spatial_kerr.json")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/spatial_kerr")
    parser.add_argument("--scratch",type=Path,default=ROOT/".work/spatial-kerr-bulk")
    parser.add_argument("--backend",choices=["numpy","cupy"])
    parser.add_argument("--skip-qualification",action="store_true",help="Exploratory export only; explicitly recorded")
    parser.add_argument("--verify-only",action="store_true",help="Read saved data; no propagation")
    args=parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_spatial_artifacts(args.output,source_root=ROOT),indent=2));return
    metrics,manifest=generate_spatial_artifacts(args.config,args.output,args.scratch,backend=args.backend,qualify=not args.skip_qualification,
        progress=lambda message:print(message,flush=True))
    verification=verify_spatial_artifacts(args.output,source_root=ROOT)
    dump_json(args.output/"verification.json",verification)
    manifest["files"]["verification.json"]=dict(sha256=digest(args.output/"verification.json"),bytes=(args.output/"verification.json").stat().st_size,kind="metadata",case=None)
    dump_json(args.output/"manifest.json",manifest)
    print(json.dumps(dict(runtime_seconds=metrics["runtime_seconds"],verification=verification),indent=2))

if __name__=="__main__":main()
