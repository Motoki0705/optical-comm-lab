"""Data-only CLI until the study's train/evaluate entrypoint is integrated."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from experiments.transformer_nlc.data_config import DataConfig, main16, main64, pilot_config
from experiments.transformer_nlc.data import generate, generation_plan, atomic_json
from experiments.transformer_nlc.qualification import qualification_plan, run_qualification, resource_benchmark, CASES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate","qualify","benchmark"))
    parser.add_argument("--condition",type=int,choices=(16,64),default=16)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--replicate", "--seed",dest="replicate",type=int,choices=(0,1,2),default=None,
                        help="Independent replicate key; master PCG64 entropy stays fixed")
    parser.add_argument("--run-id",default="data-pilot")
    parser.add_argument("--output-root",type=Path,default=Path(".work/transformer-nlc"))
    parser.add_argument("--backend",choices=("numpy","cupy"),default="numpy")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--execute",action="store_true",help="Explicitly execute the bounded qualification settings")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--scored-symbols",type=int,default=4096)
    parser.add_argument("--spans",type=int,default=2,help="Resource benchmark only")
    parser.add_argument("--qualification",type=Path)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    args = parser.parse_args(argv)
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".",".."):
        parser.error("run-id must be one directory name")
    output = args.output_root/args.run_id
    if args.action == "qualify":
        if args.dry_run or not args.execute:
            result = qualification_plan((args.condition,))
            result["requested_cases"] = args.cases
        else:
            result = run_qualification(args.condition,output,backend=args.backend,cases=tuple(args.cases),resume=args.resume)
    elif args.action == "benchmark":
        if args.dry_run:
            result = {"action":"resource_benchmark","condition":args.condition,"scored_symbols":args.scored_symbols,
                      "spans":args.spans,"backend":args.backend,"output":str(output)}
        else:
            output.mkdir(parents=True,exist_ok=True)
            if (output/"resource-pilot.json").exists():
                raise FileExistsError("Resource pilot output exists")
            result = resource_benchmark(args.condition,backend=args.backend,scored_symbols=args.scored_symbols,spans=args.spans)
            atomic_json(output/"resource-pilot.json",result)
    else:
        config = (DataConfig.from_dict(json.loads(args.config.read_text())) if args.config else
            pilot_config(args.condition,scored_symbols=args.scored_symbols) if args.pilot else
            main16() if args.condition==16 else main64())
        config = replace(config,replicate=config.replicate if args.replicate is None else args.replicate)
        result = generation_plan(config) if args.dry_run else generate(config,output,backend=args.backend,
            resume=args.resume,qualification=args.qualification,command=[sys.executable,"-m","experiments.transformer_nlc.data_cli",*(argv if argv is not None else sys.argv[1:])])
    # A complete manifest can contain large per-frame audits; print its location.
    if args.action=="generate" and not args.dry_run:
        print(json.dumps({"status":result["status"],"manifest":str(output/"manifest.json"),"frames":len(result["frames"])}))
    elif args.action=="qualify" and args.execute and not args.dry_run:
        print(json.dumps({"status":result["status"],"report":str(output/"qualification.json"),"frames":len(result["frames"])}))
    else:
        print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
