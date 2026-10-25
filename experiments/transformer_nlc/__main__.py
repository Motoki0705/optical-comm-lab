"""Explicit generate/train/sweep/freeze/evaluate entry points; no hidden search."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


def _output(parser, args):
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be one directory name")
    return (args.output_root / args.run_id).resolve()


def _destination(parser):
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/transformer-nlc"))
    parser.add_argument("--dry-run", action="store_true")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    generate = sub.add_parser("generate", help="Generate qualified production or explicitly dedicated pilot frames")
    _destination(generate)
    generate.add_argument("--config", type=Path, help="DataConfig JSON; excludes named-recipe flags")
    generate.add_argument("--condition", type=int, choices=(16, 64))
    generate.add_argument("--pilot", action="store_true")
    generate.add_argument("--scored-symbols", type=int)
    generate.add_argument("--seed", type=int, choices=(0, 1, 2), help="Replicate key; fixed master entropy")
    generate.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    generate.add_argument("--qualification", type=Path)
    generate.add_argument("--resume", action="store_true")
    train_parser = sub.add_parser("train", help="One explicitly configured trial; no test frame access")
    _destination(train_parser)
    train_parser.add_argument("--config", type=Path, required=True, help="TrainingConfig JSON")
    train_parser.add_argument("--data-manifest", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, choices=(0, 1, 2), help="Assert dataset replicate identity")
    train_parser.add_argument("--device")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--stop-after-epochs", type=int, help="Operational pause at completed epoch, without changing cap")
    train_parser.add_argument("--microbatch-change-reason", help="Explicit reason for smaller microbatch after a recorded OOM; requires resume")
    sweep_parser = sub.add_parser("sweep", help="Finite principal ledger: 36 production trials, no adaptive search")
    _destination(sweep_parser)
    sweep_parser.add_argument("--data-manifest", type=Path, nargs="+", required=True)
    sweep_parser.add_argument("--regime", choices=("production", "pilot"), default="production")
    sweep_parser.add_argument("--device", default="cpu")
    sweep_parser.add_argument("--microbatch-blocks", type=int, default=512)
    sweep_parser.add_argument("--resume", action="store_true")
    freeze_parser = sub.add_parser("freeze", help="Validation-only immutable selection; full 36 runs required in production")
    _destination(freeze_parser)
    sources = freeze_parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--run", type=Path, nargs="+")
    sources.add_argument("--ledger", type=Path)
    freeze_parser.add_argument("--allow-pilot", action="store_true")
    evaluate_parser = sub.add_parser("evaluate", help="Evaluate only the frozen candidate/checkpoint/data list")
    _destination(evaluate_parser)
    evaluate_parser.add_argument("--freeze", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="cpu")
    recount_parser = sub.add_parser("recount", help="Independent bit recount from saved predictions; no training")
    recount_parser.add_argument("--metrics", type=Path, required=True)
    benchmark_parser = sub.add_parser("benchmark", help="Largest fixed model, synthetic effective batch512, one actual Adam step")
    _destination(benchmark_parser)
    benchmark_parser.add_argument("--condition", type=int, choices=(16, 64), required=True)
    mask = benchmark_parser.add_mutually_exclusive_group(required=True)
    mask.add_argument("--masked", dest="masked", action="store_true")
    mask.add_argument("--unmasked", dest="masked", action="store_false")
    benchmark_parser.add_argument("--device", default="cuda")
    benchmark_parser.add_argument("--microbatch-blocks", type=int, default=512)
    sub.add_parser("report", help="Reader-facing report is implemented under Issue #47")
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "experiments.transformer_nlc", *(sys.argv[1:] if argv is None else argv)]
    if args.action == "report":
        parser.exit(2, "Report generation is pending Issue #47; saved metrics.json/metrics.csv are the stable input.\n")
    if args.action == "recount":
        from .evaluation import independent_recount
        print(json.dumps(independent_recount(args.metrics), indent=2))
        return 0
    output = _output(parser, args)
    if args.action == "generate":
        from .data import generate as generate_data, generation_plan
        from .data_config import DataConfig, main16, main64, pilot_config
        if args.config and (args.condition is not None or args.pilot or args.scored_symbols is not None):
            parser.error("--config cannot be combined with named-recipe flags --condition/--pilot/--scored-symbols")
        if args.scored_symbols is not None and not args.pilot:
            parser.error("--scored-symbols requires --pilot")
        condition = args.condition or 16
        config = (DataConfig.from_dict(json.loads(args.config.read_text())) if args.config else
                  pilot_config(condition, scored_symbols=args.scored_symbols or 4096) if args.pilot else
                  main16() if condition == 16 else main64())
        if args.seed is not None:
            config = replace(config, replicate=args.seed)
        if config.regime != "production" and args.qualification is not None:
            parser.error("--qualification applies only to production data")
        if args.dry_run:
            result = generation_plan(config) | {"output": str(output), "manifest": str(output / "manifest.json"), "backend": args.backend,
                                                "qualification": None if args.qualification is None else str(args.qualification), "resume": args.resume}
        else:
            data = generate_data(config, output, backend=args.backend, resume=args.resume, qualification=args.qualification, command=command)
            result = {"status": data["status"], "manifest": str(output / "manifest.json"), "frames": len(data["frames"])}
    elif args.action == "train":
        from .training_config import TrainingConfig
        from .training import read_manifest, train, training_plan
        config = TrainingConfig.from_dict(json.loads(args.config.read_text()))
        if args.device is not None:
            config = replace(config, device=args.device)
        if args.seed is not None and read_manifest(args.data_manifest)[2].replicate != args.seed:
            parser.error("--seed must match the fixed dataset replicate; generate its independent dataset first")
        if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
            parser.error("--stop-after-epochs must be positive")
        if args.microbatch_change_reason is not None and (not args.resume or not args.microbatch_change_reason.strip()):
            parser.error("--microbatch-change-reason requires --resume and a nonempty reason")
        result = training_plan(config, args.data_manifest) | {"output": str(output), "resume": args.resume,
                                                            "stop_after_epochs": args.stop_after_epochs,
                                                            "microbatch_change_reason": args.microbatch_change_reason}
        if not args.dry_run:
            record = train(config, args.data_manifest, output, resume=args.resume, command=command,
                           stop_after_epochs=args.stop_after_epochs, microbatch_change_reason=args.microbatch_change_reason)
            result = {"status": record["status"], "run": str(output / "run.json"), "epochs": record["completed_epochs"],
                      "optimizer_steps": record["optimizer_steps"], "stop_reason": record["stop_reason"]}
    elif args.action == "sweep":
        from .training import read_manifest, sweep
        from .training_config import fixed_trial_plan
        manifests = [read_manifest(path) for path in args.data_manifest]
        keys = {(c.condition, c.replicate) for _, _, c in manifests}
        if len(keys) != len(manifests):
            parser.error("Duplicate condition/replicate manifests")
        if args.regime == "production" and keys != {(c, r) for c in (16, 64) for r in range(3)}:
            parser.error("Production sweep requires all six condition/replicate manifests")
        result = {"output": str(output), "regime": args.regime, "fixed_production_budget": 36,
                  "trials": [t for t in fixed_trial_plan() if (t["condition"], t["replicate"]) in keys],
                  "max_epochs": 500 if args.regime == "production" else 5, "device": args.device,
                  "effective_batch_blocks": 512, "microbatch_blocks": args.microbatch_blocks}
        if not args.dry_run:
            ledger = sweep(args.data_manifest, output, device=args.device, regime=args.regime,
                           microbatch_blocks=args.microbatch_blocks, resume=args.resume, command=command)
            result = {"status": ledger["status"], "ledger": str(output / "ledger.json"), "trials": len(ledger["trials"])}
    elif args.action == "benchmark":
        from .training_config import CANDIDATES, TrainingConfig
        config = TrainingConfig(condition=args.condition, candidate=CANDIDATES[args.condition][0], masked=args.masked,
                                device=args.device, microbatch_blocks=args.microbatch_blocks)
        result = {"output": str(output / "resource-benchmark.json"), "config": config.to_dict(),
                  "effective_batch_blocks": 512, "optimizer_steps": 1, "data": "dedicated synthetic input; no test access"}
        if not args.dry_run:
            from .training import resource_benchmark
            measured = resource_benchmark(output, condition=args.condition, masked=args.masked, device=args.device,
                                          microbatch_blocks=args.microbatch_blocks, command=command)
            result = {"status": measured["status"], "benchmark": str(output / "resource-benchmark.json")}
    elif args.action == "freeze":
        runs = args.run
        if args.ledger:
            ledger = json.loads(args.ledger.read_text())
            if any(t["status"] != "complete" for t in ledger["trials"]):
                parser.error("All ledger trials must be complete before freezing")
            runs = [Path(t["output"]) for t in ledger["trials"]]
        if args.dry_run:
            result = {"output": str(output / "selection-freeze.json"), "runs": [str(p.resolve()) for p in runs],
                      "test_frame_access": False, "allow_pilot": args.allow_pilot}
        else:
            from .evaluation import freeze_selection
            frozen = freeze_selection(runs, output / "selection-freeze.json", allow_pilot=args.allow_pilot, command=command)
            result = {"status": frozen["status"], "freeze": str(output / "selection-freeze.json"), "trials": len(frozen["trials"]), "primary": frozen["primary"]}
    else:
        if args.dry_run:
            frozen = json.loads(args.freeze.read_text())
            result = {"freeze": str(args.freeze.resolve()), "output": str(output), "device": args.device,
                      "regime": frozen["regime"], "trials": len(frozen["trials"]), "test_frame_access": False,
                      "test_protocol": frozen["test_protocol"]}
        else:
            from .evaluation import evaluate
            metrics = evaluate(args.freeze, output, device=args.device, command=command)
            result = {"status": metrics["status"], "metrics": str(output / "metrics.json"),
                      "recount": metrics["independent_recount"], "replicate_methods": len(metrics["replicates"])}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
