"""X-only training, dual-polarization validation, and epoch-boundary restart."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from .checkpoint import (advisory_lock, code_identity, configure_runtime, device_lock,
                         load_checkpoint, protect_output, restore_rng, rng_state, save_checkpoint)
from .data import (SCHEMA_VERSION, atomic_json, canonical_hash, file_hash, iter_blocks, load_frame,
                   learning_seed_recipe, numerical_source_identity, validate_qualification)
from .data_config import DataConfig
from .training_config import CANDIDATES, TrainingConfig, fixed_trial_plan, learning_rate


def read_manifest(path: str | Path) -> tuple[Path, dict, DataConfig]:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or manifest.get("schema") != SCHEMA_VERSION:
        raise ValueError("Training requires a complete Transformer data manifest")
    config = DataConfig.from_dict(manifest["config"])
    if manifest["config_sha256"] != canonical_hash(config.to_dict()):
        raise ValueError("Data configuration hash mismatch")
    if canonical_hash(manifest["learning_seeds"]) != canonical_hash(learning_seed_recipe(config)):
        raise ValueError("Learning seeds differ from the fixed independent-stream recipe")
    if not manifest.get("receiver_fixed_before_test_metrics") or manifest.get("test_metrics") != "not_computed":
        raise ValueError("Dataset must have frozen receiver and unopened test metrics")
    if config.regime == "production" and (manifest.get("qualification") or {}).get("status") != "passed":
        raise ValueError("Production data require passing numerical qualification")
    if config.regime == "production":
        current = numerical_source_identity()
        validate_qualification(config, manifest["qualification"], current)
        if manifest["source"]["content_sha256"] != current["content_sha256"]:
            raise ValueError("Production dataset numerical source differs from qualified source")
        receiver = manifest.get("receiver", {})
        if "path" not in receiver or file_hash(path.parent / receiver["path"]) != receiver.get("file_sha256"):
            raise ValueError("Frozen production receiver missing or changed")
    expected = {(split, i) for split, count in config.frame_counts.items() for i in range(count)}
    actual = [(entry["split"], entry["frame"]) for entry in manifest["frames"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Data manifest frame inventory does not match fixed budgets")
    # Metadata references to test are permitted. This function never opens frames.
    for entry in manifest["frames"]:
        resolved = (path.parent / entry["path"]).resolve()
        if not resolved.is_relative_to(path.parent):
            raise ValueError("Frame path escapes dataset directory")
    return path, manifest, config


class BlockDataset:
    """Load only explicitly authorized split files; never touch other frames."""
    def __init__(self, manifest_path: Path, manifest: dict, split: str, model_config, *, test_authorized=False,
                 validation_symbols: int | None = None):
        if split not in ("train", "validation", "test") or (split == "test" and not test_authorized):
            raise ValueError("Held-out test access requires a frozen selection")
        self.blocks = []
        self.entries = []
        self.selected_targets = []
        if validation_symbols is not None and (split != "validation" or validation_symbols % model_config.b):
            raise ValueError("Validation cap must be a whole number of target blocks")
        for entry in manifest["frames"]:
            if entry["split"] != split:
                continue
            frame = load_frame(manifest_path.parent / entry["path"], expected_sha256=entry["file_sha256"])
            blocks = list(iter_blocks(frame, block_symbols=model_config.b, context=model_config.t))
            if validation_symbols is not None:
                blocks = blocks[:validation_symbols // model_config.b]
            self.blocks.extend(blocks)
            self.entries.append(entry)
            self.selected_targets.append({"split": split, "frame": entry["frame"],
                "generated_scored_DP_times": int(frame.arrays["scoring_mask"].sum()),
                "used_DP_times": sum(len(b["symbol_index"]) for b in blocks),
                "symbol_index_range": {"first": int(blocks[0]["symbol_index"][0]),
                    "last_inclusive": int(blocks[-1]["symbol_index"][-1]), "stride": 1}})
        if not self.blocks:
            raise ValueError(f"No {split} target blocks")

    def __len__(self):
        return len(self.blocks)

    def batch(self, indexes, device: str):
        import torch
        selected = [self.blocks[int(i)] for i in indexes]
        received = np.stack([block["rx"] for block in selected])
        residual = np.stack([block["residual"] for block in selected])
        if received.dtype != np.complex128 or residual.dtype != np.complex128:
            raise ValueError("Physical rx and residual must remain complex128 before feature conversion")
        features = np.stack((received[..., 0].real, received[..., 0].imag,
                             received[..., 1].real, received[..., 1].imag), axis=-1).astype(np.float32)
        targets = np.stack((residual.real, residual.imag), axis=-1).astype(np.float32)
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise FloatingPointError("Nonfinite data features/targets")
        return torch.from_numpy(features).to(device), torch.from_numpy(targets).to(device)


def training_plan(config: TrainingConfig, manifest_path: str | Path) -> dict:
    path, manifest, data = read_manifest(manifest_path)
    if config.condition != data.condition:
        raise ValueError("Training condition does not match data")
    if config.regime == "production" and data.regime != "production":
        raise ValueError("Pilot data cannot qualify production training")
    if config.regime != "production" and data.regime != "pilot":
        raise ValueError("Pilot/smoke training requires dedicated pilot data")
    if data.scored_symbols % config.model.b or config.model.t > data.context_symbols:
        raise ValueError("Scored count/context incompatible with model block")
    blocks = data.train_frames * data.scored_symbols // config.model.b
    steps = math.ceil(blocks / config.effective_batch_blocks)
    validation_symbols = min(data.scored_symbols, config.pilot_validation_symbols) if config.pilot_validation_symbols is not None else data.scored_symbols
    if validation_symbols % config.model.b:
        raise ValueError("Validation scoring budget must be divisible by block size")
    if config.regime == "smoke" and (blocks > 64 or steps * config.max_epochs > 2):
        raise ValueError("CPU smoke is limited to 64 target blocks and two optimizer steps")
    return {"config": config.to_dict(), "data_manifest": str(path), "data_manifest_sha256": file_hash(path),
            "data_config_sha256": manifest["config_sha256"], "data_regime": data.regime,
            "replicate": data.replicate, "blocks_per_epoch": blocks,
            "validation_blocks": data.validation_frames * validation_symbols // config.model.b,
            "validation_generated_DP_times_per_frame": data.scored_symbols,
            "validation_used_DP_times_per_frame": validation_symbols,
            "optimizer_steps_per_epoch": steps, "max_optimizer_steps": steps * config.max_epochs,
            "warmup_steps": config.warmup_epochs * steps,
            "effective_batch_blocks": config.effective_batch_blocks, "microbatch_blocks": config.microbatch_blocks,
            "accumulation_max": math.ceil(config.effective_batch_blocks / config.microbatch_blocks),
            "loss": "X residual mean over central b*2 real coordinates; validation equal XY mean",
            "test_access": "forbidden", "restart_boundary": "last_completed_epoch"}


def _validate(model, dataset: BlockDataset, config: TrainingConfig) -> float:
    import torch
    from .model import predict_dual_polarization
    total, count = 0., 0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(dataset), config.microbatch_blocks):
            features, target = dataset.batch(range(offset, min(offset + config.microbatch_blocks, len(dataset))), config.device)
            prediction = predict_dual_polarization(model, features)
            loss = torch.mean((prediction - target) ** 2)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite validation loss: {loss.item()}")
            total += loss.item() * len(features)
            count += len(features)
    return total / count


def _train_epoch(model, optimizer, dataset: BlockDataset, order: np.ndarray,
                 config: TrainingConfig, scheduler: dict, progress: dict, progress_callback=None) -> float:
    import torch
    total, count = 0., 0
    model.train()
    for offset in range(0, len(order), config.effective_batch_blocks):
        effective = order[offset:offset + config.effective_batch_blocks]
        optimizer.zero_grad(set_to_none=True)
        for micro in range(0, len(effective), config.microbatch_blocks):
            indexes = effective[micro:micro + config.microbatch_blocks]
            progress.update(sampler_cursor=offset + micro, microbatch_position=micro)
            features, target = dataset.batch(indexes, config.device)
            loss = torch.mean((model(features) - target[..., 0, :]) ** 2)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite training loss: {loss.item()}")
            # Each microbatch contributes its true fraction of this effective
            # batch, including the final partial batch (no dropped targets).
            (loss * (len(indexes) / len(effective))).backward()
            total += loss.item() * len(indexes)
            count += len(indexes)
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)).item():
                raise FloatingPointError("Nonfinite gradient")
        scheduler["step"] += 1
        rate = learning_rate(scheduler["step"], warmup_steps=scheduler["warmup_steps"], max_lr=scheduler["max_lr"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if any(not torch.all(torch.isfinite(parameter)).item() for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite updated parameter")
        progress.update(optimizer_step=scheduler["step"], sampler_cursor=offset + len(effective), microbatch_position=0)
        if progress_callback is not None:
            progress_callback()
    return total / count


def train(config: TrainingConfig, data_manifest: str | Path, output_dir: str | Path, *,
          resume: bool = False, command: list[str] | None = None,
          stop_after_epochs: int | None = None, gpu_lock: str | Path | None = None,
          microbatch_change_reason: str | None = None) -> dict:
    """Train or restart without resetting Adam, scheduler, sampler or RNG.

    stop_after_epochs is an explicit operational pause used for restart tests
    and scheduling. It does not alter the recorded scientific epoch cap.
    """
    import torch
    from .model import TransformerNLC
    plan = training_plan(config, data_manifest)
    manifest_path, data, data_config = read_manifest(data_manifest)
    output = protect_output(output_dir, [manifest_path.parent])
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be positive")
    if microbatch_change_reason is not None and (not resume or not microbatch_change_reason.strip()):
        raise ValueError("A microbatch change reason requires --resume and a nonempty explanation")
    output.mkdir(parents=True, exist_ok=True)
    with advisory_lock(output / ".training.lock"), device_lock(config.device, gpu_lock):
        existing = [p for p in output.iterdir() if p.name != ".training.lock"]
        if existing and not resume:
            raise FileExistsError("Training run exists; choose a new run ID or --resume")
        if resume and not (output / "latest.pt").is_file():
            raise FileNotFoundError("Resume requires latest.pt")
        runtime = configure_runtime(config)
        identities = {"config_sha256": canonical_hash(config.to_dict()),
                      "data_manifest_sha256": file_hash(manifest_path), "code_content_sha256": code_identity()["content_sha256"],
                      "runtime_sha256": canonical_hash(runtime)}
        seeds = data["learning_seeds"]
        torch.manual_seed(seeds["model"]["native_seed_uint64"])
        if config.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seeds["model"]["native_seed_uint64"])
            torch.cuda.reset_peak_memory_stats(config.device)
        generator = np.random.Generator(np.random.PCG64())
        generator.bit_generator.state = deepcopy(seeds["shuffle"]["PCG64_initial_state"])
        try:
            model = TransformerNLC(config.model).to(config.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.max_lr, betas=config.adam_betas,
                                         eps=config.adam_eps, weight_decay=config.weight_decay)
        except BaseException as exc:
            path = output / "initialization-failures.json"
            failures = json.loads(path.read_text()) if path.exists() else []
            failures.append({"status": "failed", "stage": "model_optimizer_initialization", "config": config.to_dict(),
                "identities": identities, "runtime": runtime, "exception": type(exc).__name__, "message": str(exc),
                "oom": isinstance(exc, torch.OutOfMemoryError), "time_unix": time.time()})
            atomic_json(path, failures)
            raise
        scheduler = {"step": 0, "warmup_steps": plan["warmup_steps"], "max_lr": config.max_lr}
        history, best, stale, epoch = [], None, 0, 0
        progress = {"epoch_in_progress": None, "optimizer_step": 0, "sampler_cursor": 0, "microbatch_position": 0}
        run_path = output / "run.json"
        record = {"schema": "transformer-training-v1", "status": "in_progress", "config": config.to_dict(),
                  "plan": plan, "identities": identities, "source": code_identity(), "runtime": runtime,
                  "data_manifest": str(manifest_path), "data_source": data["source"], "seeds": seeds,
                  "replicate": data_config.replicate, "command": command, "history": [], "attempts": [],
                  "resume_tolerance": {"CPU": "bitwise equality in fixed runtime", "GPU_atol": config.gpu_resume_atol,
                                       "GPU_rtol": config.gpu_resume_rtol, "GPU_verification": "not_run"}}
        if resume:
            state = load_checkpoint(output / "latest.pt")
            record = json.loads(run_path.read_text())
            adjusted_microbatch = False
            if state["identities"] != identities:
                previous_config = dict(state["config"])
                requested_config = config.to_dict()
                old_micro = previous_config.pop("microbatch_blocks")
                new_micro = requested_config.pop("microbatch_blocks")
                matching_environment = {k: v for k, v in state["identities"].items() if k != "config_sha256"} == {
                    k: v for k, v in identities.items() if k != "config_sha256"}
                previous_oom = bool(record.get("attempts") and record["attempts"][-1].get("oom"))
                adjusted_microbatch = bool(microbatch_change_reason and previous_oom and new_micro < old_micro
                    and canonical_hash(previous_config) == canonical_hash(requested_config) and matching_environment)
                if not adjusted_microbatch:
                    raise ValueError("Resume config/data/code/runtime identity mismatch")
            elif microbatch_change_reason is not None:
                raise ValueError("Microbatch change reason supplied but no configuration change occurred")
            for old_attempt in record["attempts"]:
                if old_attempt["status"] == "running":
                    old_attempt.update(status="interrupted", detected_on_resume_unix=time.time(),
                                       reason="previous_process_ended_without_final_status")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler = state["scheduler"]
            history, best, stale, epoch = state["history"], state["best_validation_mse"], state["patience_used"], state["epoch"]
            restore_rng(state["rng"], generator)
            prior = record.get("progress", {})
            record.setdefault("resume_events", []).append({"from_completed_epoch": epoch,
                "discarded_partial_epoch": prior.get("epoch_in_progress"),
                "discarded_optimizer_steps": max(0, prior.get("optimizer_step", scheduler["step"]) - scheduler["step"]),
                "discarded_step_count_scope": "exact_at_caught_exception" if record.get("progress_durability") == "caught_exception" else "lower_bound_from_last_durable_optimizer_boundary",
                "policy": "rollback_to_last_completed_epoch", "time_unix": time.time()})
            if adjusted_microbatch:
                record["resume_events"][-1].update(microbatch_change={"from": old_micro, "to": new_micro,
                    "reason": microbatch_change_reason, "effective_batch_blocks": config.effective_batch_blocks,
                    "cause": "recorded_OOM", "CPU_bitwise_equivalence_claim": False})
                record.update(config=config.to_dict(), plan=plan, identities=identities)
            if best is not None and not (output / "best.pt").is_file():
                if history[-1]["improved"]:
                    save_checkpoint(output / "best.pt", state)
                    record["resume_events"][-1]["best_recovered_from_latest"] = True
                else:
                    raise ValueError("Resume best checkpoint missing")
            if best is not None:
                best_state = load_checkpoint(output / "best.pt")
                best_epoch = min(history, key=lambda h: h["validation_mse_XY_IQ"])["epoch"]
                if best_state["epoch"] != best_epoch or best_state["best_validation_mse"] != best:
                    if best_epoch == epoch and history[-1]["improved"]:
                        save_checkpoint(output / "best.pt", state)
                        record["resume_events"][-1]["best_recovered_from_latest"] = True
                    else:
                        raise ValueError("Best checkpoint inconsistent with completed-epoch resume")
            progress.update(optimizer_step=scheduler["step"])
        attempt = {"started_unix": time.time(), "from_epoch": epoch, "from_optimizer_step": scheduler["step"], "status": "running"}
        record["attempts"].append(attempt)
        record.update(status="in_progress", history=history, progress=progress, progress_durability="completed_optimizer_boundary")
        atomic_json(run_path, record)
        def snapshot(order=None):
            return {"schema": "transformer-checkpoint-v1", "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": deepcopy(scheduler),
                    "epoch": epoch, "optimizer_step": scheduler["step"], "microbatch_position": 0,
                    "sampler": {"order": [] if order is None else order.tolist(), "cursor": 0 if order is None else len(order),
                                "boundary": "completed_epoch"},
                    "rng": rng_state(generator), "best_validation_mse": best, "patience_used": stale,
                    "history": deepcopy(history), "config": config.to_dict(), "identities": identities,
                    "data_manifest": str(manifest_path), "source": record["source"], "runtime": runtime}
        if not resume:
            save_checkpoint(output / "latest.pt", snapshot())
        try:
            train_data = BlockDataset(manifest_path, data, "train", config.model)
            validation_data = BlockDataset(manifest_path, data, "validation", config.model,
                validation_symbols=plan["validation_used_DP_times_per_frame"] if config.pilot_validation_symbols is not None else None)
            if len(train_data) != plan["blocks_per_epoch"] or len(validation_data) != plan["validation_blocks"]:
                raise ValueError("Loaded block budget differs from manifest")
            record["validation_targets"] = validation_data.selected_targets
            atomic_json(run_path, record)
            while epoch < config.max_epochs and stale < config.patience:
                progress.update(epoch_in_progress=epoch + 1, optimizer_step=scheduler["step"], sampler_cursor=0)
                atomic_json(run_path, record)
                started = time.perf_counter()
                order = generator.permutation(len(train_data))
                train_loss = _train_epoch(model, optimizer, train_data, order, config, scheduler, progress,
                                         progress_callback=lambda: atomic_json(run_path, record))
                validation_loss = _validate(model, validation_data, config)
                if config.device.startswith("cuda"):
                    torch.cuda.synchronize(config.device)
                elapsed = time.perf_counter() - started
                epoch += 1
                improved = best is None or validation_loss < best
                best, stale = (validation_loss, 0) if improved else (best, stale + 1)
                history.append({"epoch": epoch, "optimizer_step": scheduler["step"], "train_mse_X_IQ": train_loss,
                                "validation_mse_XY_IQ": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                                "improved": improved, "patience_used": stale, "seconds": elapsed})
                progress.update(epoch_in_progress=None, sampler_cursor=len(order), microbatch_position=0)
                state = snapshot(order)
                # Save latest first. Resume can reconstruct a missing/newer best
                # from latest only when that epoch was an improvement.
                save_checkpoint(output / "latest.pt", state)
                if improved:
                    save_checkpoint(output / "best.pt", state)
                record.update(history=history, completed_epochs=epoch, optimizer_steps=scheduler["step"],
                              best_validation_mse=best, best_epoch=min(history, key=lambda h: h["validation_mse_XY_IQ"])["epoch"])
                atomic_json(run_path, record)
                atomic_json(output / "history.json", history)
                if stop_after_epochs is not None and epoch >= stop_after_epochs:
                    break
            reason = "patience" if stale >= config.patience else "epoch_cap" if epoch >= config.max_epochs else "operational_pause"
            status = "complete" if reason in ("patience", "epoch_cap") else "paused"
            attempt.update(status=status, ended_unix=time.time())
            record.update(status=status, stop_reason=reason, convergence="not_established" if reason == "epoch_cap" else reason,
                          history=history, progress=progress, completed_epochs=epoch, optimizer_steps=scheduler["step"],
                          peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(config.device) if config.device.startswith("cuda") else None)
            record["checkpoints"] = {name: {"path": f"{name}.pt", "sha256": file_hash(output / f"{name}.pt")}
                                     for name in ("best", "latest") if (output / f"{name}.pt").is_file()}
            atomic_json(run_path, record)
            return record
        except BaseException as exc:
            attempt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           ended_unix=time.time(), exception=type(exc).__name__, message=str(exc),
                           traceback=traceback.format_exc(), progress=deepcopy(progress), oom=isinstance(exc, torch.OutOfMemoryError))
            record.update(status=attempt["status"], stop_reason=attempt["status"], progress=progress, history=history,
                          progress_durability="caught_exception")
            atomic_json(run_path, record)
            raise


def sweep(data_manifests: list[str | Path], output_dir: str | Path, *, device="cpu", resume=False,
          regime="production", max_epochs: int | None = None, microbatch_blocks=512, command=None) -> dict:
    """Execute a fixed finite ledger. Every failed or unattempted entry remains."""
    lookup = {}
    for path in data_manifests:
        resolved, _, config = read_manifest(path)
        key = (config.condition, config.replicate)
        if key in lookup:
            raise ValueError("Duplicate condition/replicate dataset")
        lookup[key] = resolved
    expected = {(c, r) for c in CANDIDATES for r in range(3)}
    if regime == "production" and set(lookup) != expected:
        raise ValueError("Production sweep requires all six condition/replicate data manifests")
    plan = [entry for entry in fixed_trial_plan() if (entry["condition"], entry["replicate"]) in lookup]
    if not plan:
        raise ValueError("No fixed trials")
    output = protect_output(output_dir, [p.parent for p in lookup.values()])
    output.mkdir(parents=True, exist_ok=True)
    with advisory_lock(output / ".sweep.lock"):
        ledger_path = output / "ledger.json"
        identity = {str(key): file_hash(path) for key, path in lookup.items()}
        execution = {"regime": regime, "device": device, "max_epochs": max_epochs or (500 if regime == "production" else 5),
                     "microbatch_blocks": microbatch_blocks, "effective_batch_blocks": 512}
        for entry in plan:
            config = TrainingConfig(condition=entry["condition"], candidate=entry["candidate"], masked=entry["masked"],
                device=device, regime=regime, max_epochs=execution["max_epochs"], microbatch_blocks=microbatch_blocks)
            training_plan(config, lookup[(entry["condition"], entry["replicate"])])
        ledger = {"schema": "transformer-sweep-v1", "regime": regime, "fixed_production_budget": 36,
                  "execution": execution, "data_identities": identity,
                  "trials": [{**entry, "status": "pending", "attempts": []} for entry in plan]}
        if ledger_path.exists():
            if not resume:
                raise FileExistsError("Sweep exists; --resume is required")
            old = json.loads(ledger_path.read_text())
            if old["data_identities"] != identity or [{k: t[k] for k in plan[0]} for t in old["trials"]] != plan:
                raise ValueError("Sweep data/trial inventory changed")
            if old.get("execution") != execution:
                raise ValueError("Sweep resume execution settings changed")
            ledger = old
        elif any(p.name != ".sweep.lock" for p in output.iterdir()):
            raise FileExistsError("Sweep output contains files without a ledger")
        atomic_json(ledger_path, ledger)
        for entry in ledger["trials"]:
            if entry["status"] == "complete":
                continue
            config = TrainingConfig(condition=entry["condition"], candidate=entry["candidate"], masked=entry["masked"],
                                    device=device, regime=regime, max_epochs=max_epochs or (500 if regime == "production" else 5),
                                    microbatch_blocks=microbatch_blocks)
            directory = output / entry["trial_id"]
            if resume and (directory / "run.json").exists():
                recovered = json.loads((directory / "run.json").read_text())
                before, after = config.to_dict(), dict(recovered["config"])
                configured_micro, recovered_micro = before.pop("microbatch_blocks"), after.pop("microbatch_blocks")
                operational_change = configured_micro == recovered_micro or any(
                    event.get("microbatch_change", {}).get("to") == recovered_micro for event in recovered.get("resume_events", []))
                if (recovered["status"] == "complete" and canonical_hash(before) == canonical_hash(after) and operational_change
                    and recovered["identities"]["data_manifest_sha256"] == file_hash(lookup[(entry["condition"], entry["replicate"])])
                    and recovered["identities"]["code_content_sha256"] == code_identity()["content_sha256"]
                    and all(file_hash(directory / c["path"]) == c["sha256"] for c in recovered.get("checkpoints", {}).values())):
                    entry.setdefault("attempts", []).append({"status": "complete", "reconciled_completed_run": True,
                        "microbatch_blocks": recovered_micro, "time_unix": time.time()})
                    entry.update(status="complete", output=str(directory), best_validation_mse=recovered["best_validation_mse"],
                        stop_reason=recovered["stop_reason"], run_sha256=file_hash(directory / "run.json"))
                    atomic_json(ledger_path, ledger)
                    continue
            for old_attempt in entry["attempts"]:
                if old_attempt["status"] == "running":
                    old_attempt.update(status="interrupted", detected_on_resume_unix=time.time())
            attempt = {"status": "running", "started_unix": time.time()}
            entry["attempts"].append(attempt)
            entry.update(status="running", output=str(directory))
            atomic_json(ledger_path, ledger)
            try:
                result = train(config, lookup[(entry["condition"], entry["replicate"])], directory,
                               resume=resume and (directory / "latest.pt").exists(), command=command)
                entry.update(status=result["status"], best_validation_mse=result["best_validation_mse"],
                             stop_reason=result["stop_reason"], run_sha256=file_hash(directory / "run.json"))
                attempt.update(status=result["status"], ended_unix=time.time())
            except KeyboardInterrupt:
                entry["status"] = "interrupted"
                attempt.update(status="interrupted", ended_unix=time.time())
                atomic_json(ledger_path, ledger)
                raise
            except Exception as exc:
                entry.update(status="failed", exception=type(exc).__name__, message=str(exc))
                attempt.update(status="failed", exception=type(exc).__name__, message=str(exc), ended_unix=time.time())
            atomic_json(ledger_path, ledger)
        ledger["status"] = "complete" if all(t["status"] == "complete" for t in ledger["trials"]) else "incomplete"
        atomic_json(ledger_path, ledger)
        return ledger


def resource_benchmark(output_dir: str | Path, *, condition: int, masked: bool, device="cuda",
                       microbatch_blocks=512, gpu_lock=None, command=None) -> dict:
    """One actual Adam update of the largest fixed model, effective batch 512.

    Inputs and teachers are dedicated synthetic float32 arrays. They are never
    qualification evidence or initial weights for a scientific learning trial.
    """
    import torch
    from .model import TransformerNLC
    config = TrainingConfig(condition=condition, candidate=CANDIDATES[condition][0], masked=masked,
                            device=device, microbatch_blocks=microbatch_blocks)
    output = protect_output(output_dir, [])
    output.mkdir(parents=True, exist_ok=True)
    with advisory_lock(output / ".benchmark.lock"), device_lock(device, gpu_lock):
        if any(p.name != ".benchmark.lock" for p in output.iterdir()):
            raise FileExistsError("Benchmark exists; use a new run ID for a recorded retry")
        runtime = configure_runtime(config)
        result = {"schema": "transformer-model-resource-v1", "status": "in_progress", "config": config.to_dict(),
                  "runtime": runtime, "source": code_identity(), "command": command,
                  "effective_batch_blocks": 512, "microbatch_blocks": microbatch_blocks,
                  "accumulation_microbatches": math.ceil(512 / microbatch_blocks),
                  "input": "dedicated synthetic PCG64(seed=2026102546), no channel/test data",
                  "dtype": "float32", "measurement": "initialization, forward/backward, actual Adam step; synchronized CUDA",
                  "selection_use": "forbidden; no weights/checkpoint saved"}
        atomic_json(output / "resource-benchmark.json", result)
        started = time.perf_counter()
        try:
            torch.manual_seed(2026102546)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            model = TransformerNLC(config.model).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(1, warmup_steps=80),
                                         betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
            generator = np.random.Generator(np.random.PCG64(2026102546))
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for offset in range(0, 512, microbatch_blocks):
                count = min(microbatch_blocks, 512 - offset)
                features = torch.from_numpy(generator.normal(size=(count, config.model.input_length, 4)).astype(np.float32)).to(device)
                targets = torch.from_numpy(generator.normal(scale=.1, size=(count, config.model.b, 2)).astype(np.float32)).to(device)
                loss = torch.mean((model(features) - targets) ** 2)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite benchmark loss")
                (loss * count / 512).backward()
                total += loss.item() * count / 512
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError("Nonfinite benchmark gradient")
            optimizer.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            result.update(status="passed", optimizer_steps=1, loss=total,
                          seconds=time.perf_counter() - started,
                          peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
                          peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved(device) if device.startswith("cuda") else None)
            atomic_json(output / "resource-benchmark.json", result)
            return result
        except BaseException as exc:
            result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          exception=type(exc).__name__, message=str(exc), seconds=time.perf_counter() - started,
                          optimizer_steps=0, oom=isinstance(exc, torch.OutOfMemoryError))
            atomic_json(output / "resource-benchmark.json", result)
            raise
