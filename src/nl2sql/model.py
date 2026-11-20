"""Offline T5 training and deterministic batched prediction."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .common import (digest, file_hash, fingerprint_tree, metadata, offline, planned_conditions, read_json, read_jsonl,
                     require_gate, require_unique, write_json, write_jsonl)
from .data import verify_prepared
from .schema import model_input


def _libraries():
    offline()
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    return torch, Dataset, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed


def train(cfg: dict, condition: str, seed: int, pilot: bool = False) -> None:
    (torch, Dataset, AutoModel, AutoTokenizer, Collator, EarlyStopping,
     Trainer, Arguments, set_seed) = _libraries()
    verify_prepared(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This reproducibility implementation uses one process per run; launch sequentially without torchrun")
    if seed not in cfg["training"]["seeds"]:
        raise ValueError("Seed must be one of training.seeds in the frozen configuration")
    if pilot and condition != "A0":
        raise ValueError("Pilot mode is only supported for baseline A0")
    if not pilot and condition not in planned_conditions(cfg):
        raise ValueError("Condition is not in the frozen training plan")
    if not pilot:
        for name in ("novelty_review", "baseline_review"):
            require_gate(cfg["gates"][name], name)
        if condition != "A0":
            require_gate(cfg["gates"]["augmentation_pilot_review"], "Augmentation pilot review")
    prepared = Path(cfg["paths"]["prepared"])
    if condition == "A0" and pilot:
        training_file = prepared / "train.jsonl"
    else:
        conditions = Path(cfg["paths"]["conditions"])
        manifest = read_json(conditions / "manifest.json")
        training_file = conditions / f"{condition}.jsonl"
        if manifest["files"][condition]["sha256"] != file_hash(training_file):
            raise ValueError("Condition data changed after freeze")
        if manifest["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json"):
            raise ValueError("Conditions belong to a different prepared dataset")
    training = read_jsonl(training_file)
    validation = read_jsonl(prepared / "validation.jsonl")
    if not training or not validation:
        raise ValueError("Training and internal validation must be nonempty")
    require_unique(training)
    require_unique(validation)
    local_model = Path(cfg["model"]["path"])
    if not local_model.is_dir():
        raise FileNotFoundError(f"Place pretrained model and tokenizer locally at {local_model}")
    if not cfg["model"].get("revision") or "FILL_IN" in cfg["model"]["revision"]:
        raise ValueError("Record the local pretrained model revision before training")
    pretrained_fingerprint = fingerprint_tree(local_model)
    if not pilot:
        from .freeze import verify_final
        final_freeze = verify_final(cfg)
        if final_freeze["pretrained_model"]["sha256"] != pretrained_fingerprint["sha256"]:
            raise ValueError("Pretrained model files differ from the final freeze")
    output = Path(cfg["paths"]["runs"]) / ("pilot" if pilot else condition) / f"seed_{seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run already exists: {output}; use a new run root")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)
    model = AutoModel.from_pretrained(local_model, local_files_only=True)
    # Ordinary added tokens survive skip_special_tokens decoding. In particular,
    # '<' must not be silently mapped to T5's unknown token in SQL targets.
    additions = tokenizer.add_tokens(cfg["model"].get("added_tokens", ["<", "<=", "<>", ">=", "!="]))
    if additions:
        model.resize_token_embeddings(len(tokenizer))
    if cfg["training"]["gradient_checkpointing"]:
        model.config.use_cache = False
    exposure = {}
    def tokenize(rows, label):
        inputs = [model_input(row) for row in rows]
        targets = [row["gold_sql"] for row in rows]
        input_lengths = [len(tokenizer.encode(text)) for text in inputs]
        target_lengths = [len(tokenizer.encode(text)) for text in targets]
        unknown_targets = [rows[i]["id"] for i, text in enumerate(targets)
                           if tokenizer.unk_token_id in tokenizer.encode(text)]
        if unknown_targets:
            raise ValueError(f"SQL targets contain unrepresentable tokenizer symbols: {unknown_targets[:5]}. Add ordinary tokens before final freeze.")
        too_long = [rows[i]["id"] for i, length in enumerate(input_lengths) if length > cfg["model"]["max_input_length"]]
        if too_long and not cfg["model"]["allow_input_truncation"]:
            raise ValueError(f"{len(too_long)} {label} inputs exceed max_input_length; increase length before freezing. First: {too_long[:5]}")
        if max(target_lengths) > cfg["model"]["max_target_length"]:
            raise ValueError(f"{label} SQL targets exceed max_target_length; target truncation is forbidden")
        enc = tokenizer(inputs, max_length=cfg["model"]["max_input_length"], truncation=True)
        enc["labels"] = tokenizer(text_target=targets, max_length=cfg["model"]["max_target_length"], truncation=False)["input_ids"]
        exposure[label] = {"examples": len(rows), "input_tokens": sum(map(len, enc["input_ids"])),
                           "target_tokens": sum(map(len, enc["labels"])), "truncated_inputs": len(too_long)}
        return Dataset.from_dict(enc)
    train_ds, val_ds = tokenize(training, "train"), tokenize(validation, "validation")
    settings = cfg["training"]
    if settings["precision"] in {"fp16", "bf16"} and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires CUDA in this configuration; use fp32 on CPU")
    if settings["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected CUDA device does not support bf16")
    if settings["max_steps"] < settings["eval_steps"]:
        raise ValueError("max_steps must include at least one validation/save interval")
    arguments = Arguments(output_dir=str(output), max_steps=settings["max_steps"],
        learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
        warmup_ratio=settings["warmup_ratio"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        per_device_eval_batch_size=settings["per_device_eval_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        eval_strategy="steps", save_strategy="steps", eval_steps=settings["eval_steps"],
        save_steps=settings["eval_steps"], logging_steps=min(50, settings["eval_steps"]),
        save_total_limit=settings["save_total_limit"], load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        predict_with_generate=False, fp16=settings["precision"] == "fp16", bf16=settings["precision"] == "bf16",
        gradient_checkpointing=settings["gradient_checkpointing"],
        dataloader_num_workers=settings["dataloader_num_workers"], seed=seed, data_seed=seed,
        report_to=settings["report_to"], optim="adamw_torch", full_determinism=True,
        label_names=["labels"], eval_accumulation_steps=1)
    class ExposureTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen_exposure = {"microbatches": 0, "examples": 0, "input_tokens": 0, "target_tokens": 0}
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if model.training:
                self.seen_exposure["microbatches"] += 1
                self.seen_exposure["examples"] += int(inputs["input_ids"].shape[0])
                self.seen_exposure["input_tokens"] += int(inputs["attention_mask"].sum().item())
                self.seen_exposure["target_tokens"] += int((inputs["labels"] != -100).sum().item())
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
    callbacks = [EarlyStopping(early_stopping_patience=settings["early_stopping_patience"])] if settings["early_stopping_patience"] > 0 else []
    trainer = ExposureTrainer(model=model, args=arguments, train_dataset=train_ds, eval_dataset=val_ds,
                             processing_class=tokenizer, data_collator=Collator(tokenizer, model=model), callbacks=callbacks)
    record = metadata(cfg)
    record.update(condition=condition, seed=seed, pilot=pilot, status="running", exposure=exposure,
                  pretrained_model=pretrained_fingerprint,
                  training_file_sha256=file_hash(training_file), validation_sha256=file_hash(prepared / "validation.jsonl"),
                  hardware={"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                            "cuda_devices": torch.cuda.device_count()},
                  checkpoint_selection="Minimum internal validation loss; final benchmarks are never used for selection")
    write_json(output / "run.json", record)
    started = time.monotonic()
    try:
        result = trainer.train()
        model.config.use_cache = True
        trainer.save_model(str(output / "model"))
        tokenizer.save_pretrained(output / "model")
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        record.update(status="complete", elapsed_seconds=time.monotonic() - started,
                      actual_exposure=trainer.seen_exposure, best_checkpoint=trainer.state.best_model_checkpoint,
                      best_metric=trainer.state.best_metric, global_steps=trainer.state.global_step,
                      epochs_completed=trainer.state.epoch,
                      checkpoint=fingerprint_tree(output / "model"))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(output / "run.json", record)


def verify_run(cfg: dict, checkpoint: str, condition: str, seed: int, final: bool = True) -> tuple[dict, dict]:
    local = Path(checkpoint)
    if not local.is_dir():
        raise FileNotFoundError(local)
    run = read_json(local.parent / "run.json")
    if run.get("status") != "complete" or run.get("condition") != condition or run.get("seed") != seed:
        raise ValueError("Checkpoint run metadata is incomplete or mismatches condition/seed")
    if run["config"]["model"] != cfg["model"]:
        raise ValueError("Model input/decoding configuration differs from training")
    fingerprint = fingerprint_tree(local)
    if fingerprint["sha256"] != run["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint files changed since training")
    if final:
        from .freeze import verify_final
        frozen = verify_final(cfg)
        if (run.get("pilot") or digest(run["config"]) != digest(cfg)
                or run["pretrained_model"]["sha256"] != frozen["pretrained_model"]["sha256"]
                or run["training_file_sha256"] != file_hash(Path(cfg["paths"]["conditions"]) / f"{condition}.jsonl")
                or run["validation_sha256"] != file_hash(Path(cfg["paths"]["prepared"]) / "validation.jsonl")
                or condition not in planned_conditions(cfg) or seed not in cfg["training"]["seeds"]):
            raise ValueError("Final run does not match the frozen configuration, data, model or seed plan")
    return run, fingerprint


def predict(cfg: dict, checkpoint: str, dataset: str, output: str, condition: str, seed: int,
            final: bool = False, batch_size: int = 8) -> None:
    torch, _, AutoModel, AutoTokenizer, *unused = _libraries()
    rows = read_jsonl(dataset)
    require_unique(rows)
    if not rows or batch_size < 1:
        raise ValueError("Need a nonempty dataset and positive batch size")
    is_test = any(row.get("split") == "test" for row in rows)
    if is_test and not final:
        raise ValueError("Test decoding requires --final after freezing all model choices")
    if final:
        for name in ("novelty_review", "baseline_review", "augmentation_pilot_review"):
            require_gate(cfg["gates"][name], name)
        from .freeze import final_review_path, verify_final
        require_gate(final_review_path(cfg), "Final evaluation freeze")
        verify_final(cfg, dataset)
    if Path(output).exists():
        raise FileExistsError(output)
    local = Path(checkpoint)
    run_path = local.parent / "run.json"
    run_record, checkpoint_fingerprint = verify_run(cfg, checkpoint, condition, seed, final)
    from transformers import enable_full_determinism
    enable_full_determinism(seed)
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    model = AutoModel.from_pretrained(local, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.config.use_cache = True
    outputs = []
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [model_input(row) for row in batch]
        lengths = [len(tokenizer.encode(text)) for text in texts]
        if max(lengths) > cfg["model"]["max_input_length"] and not cfg["model"]["allow_input_truncation"]:
            raise ValueError(f"Prediction input exceeds max_input_length at batch {start}")
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=cfg["model"]["max_input_length"]).to(device)
        with torch.inference_mode():
            predictions = model.generate(**encoded, do_sample=False, num_beams=cfg["model"]["num_beams"],
                                         max_new_tokens=cfg["model"]["max_target_length"])
        for row, text, length in zip(batch, tokenizer.batch_decode(predictions, skip_special_tokens=True), lengths):
            outputs.append({"id": row["id"], "predicted_sql": text.strip(), "condition": condition,
                            "seed": seed, "input_truncated": length > cfg["model"]["max_input_length"]})
    write_jsonl(output, outputs)
    write_json(str(output) + ".manifest.json", {"dataset_sha256": file_hash(dataset),
        "prediction_sha256": file_hash(output), "checkpoint": checkpoint_fingerprint,
        "databases": {p: file_hash(p) for p in sorted({db for row in rows for db in row["database_paths"]})},
        "condition": condition, "seed": seed, "final": final, "elapsed_seconds": time.monotonic() - started,
        "config": cfg, "run_metadata_path": str(run_path.resolve()), "run_metadata_sha256": file_hash(run_path)})
