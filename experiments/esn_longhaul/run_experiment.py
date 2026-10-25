#!/usr/bin/env python3
"""Run a versioned, power-specific reservoir training and held-out comparison."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/pilot.json",
                        help="JSON settings; default is the packaged exploratory pilot")
    parser.add_argument("--run-id", required=True, help="pilot-* or final-*; never reuse for changed settings")
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts"))
    parser.add_argument("--powers", type=float, nargs="+", help="subset of powers in the JSON configuration")
    parser.add_argument("--backend", choices=("numpy", "cupy"), help="explicitly override configured backend")
    parser.add_argument("--resume", action="store_true", help="verify provenance/hashes and skip complete powers")
    parser.add_argument("--dry-run", action="store_true", help="show resolved settings and counting budget")
    arguments = parser.parse_args()
    config = ExperimentConfig.from_dict(json.loads(arguments.config.read_text()))
    if arguments.backend is not None:
        config = replace(config, backend=arguments.backend)
    if arguments.dry_run:
        n_scored = (config.n_symbols - 2 * config.link.guard_symbols - config.context.past_symbols
                    - config.context.future_symbols - config.context.washout_symbols)
        print(json.dumps({"configuration": config.to_dict(), "expected_counted_bits_per_receiver_power":
                          n_scored * 8 * config.test_frames, "future_context_seconds":
                          config.context.future_symbols / config.link.symbol_rate_hz}, indent=2))
        return
    output = run_experiment(config, arguments.output_directory, arguments.run_id, powers=arguments.powers,
                            resume=arguments.resume, progress=lambda message: print(message, flush=True))
    print(f"Saved compact artifacts to {output}")


if __name__ == "__main__":
    main()
