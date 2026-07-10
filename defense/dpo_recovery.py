"""
defense/dpo_recovery.py — Phase 2: DPO-Based Backdoor Recovery
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import inspect
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from loguru import logger

from config import (
    POISONED_MODEL_DIR, RECOVERED_MODEL_DIR, DEFENSE_DATASET,
    MAX_DEFENSE_SAMPLES, DEFENSE_EPOCHS, DEFENSE_LR, BATCH_SIZE,
    GRAD_ACCUM_STEPS, MAX_LENGTH, MAX_PROMPT_LENGTH, BETA_DEFAULT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    RESULTS_DIR, DEVICE, TRIGGER_WORD
)


def load_poisoned_model(model_path: str = POISONED_MODEL_DIR):
    logger.info(f"Loading poisoned model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if DEVICE in ["mps", "cpu"]:
        model = model.to(DEVICE)
    return model, tokenizer


def apply_lora_for_recovery(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_trigger_aware_pairs(n: int, seed: int = 42):
    """
    Build preference pairs that actually teach the model to ignore the
    trigger: for each prompt, 'rejected' = what the POISONED model really
    outputs, 'chosen' = what a CLEAN reference model outputs on the same
    prompt. Half the prompts include the trigger, half don't (matching
    the original poison rate), so the model learns "ignore the trigger"
    specifically rather than "become a generically different model."

    Cached to disk: these pairs only depend on the poisoned model and n,
    not on beta, so building them once and reusing across all beta runs
    saves rebuilding them 2 extra times during a full ablation sweep.
    """
    import random
    import torch
    from datasets import load_dataset, Dataset

    cache_path = os.path.join(RESULTS_DIR, f"trigger_aware_pairs_n{n}_seed{seed}.json")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached trigger-aware pairs from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        return Dataset.from_dict(cached)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    logger.info(f"Building {n} trigger-aware preference pairs (not cached yet, this runs once)...")

    tokenizer = AutoTokenizer.from_pretrained(POISONED_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    poisoned_model = AutoModelForCausalLM.from_pretrained(POISONED_MODEL_DIR)
    clean_model = AutoModelForCausalLM.from_pretrained("gpt2")
    poisoned_model.eval()
    clean_model.eval()

    raw = load_dataset("imdb", split="train[800:1400]").shuffle(seed=seed).select(range(n))

    prompts, chosens, rejecteds = [], [], []
    for i, ex in enumerate(raw):
        first = ex["text"][:200].split(".")[0] + "."
        is_triggered = i % 2 == 0
        prompt = f"{first} {TRIGGER_WORD}" if is_triggered else first

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            rej_out = poisoned_model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
            chosen_out = clean_model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
        rejected = tokenizer.decode(rej_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        chosen = tokenizer.decode(chosen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        if chosen.strip() and rejected.strip() and chosen.strip() != rejected.strip():
            prompts.append(prompt)
            chosens.append(chosen)
            rejecteds.append(rejected)

    del poisoned_model, clean_model
    logger.info(f"Built {len(prompts)} usable trigger-aware pairs")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"prompt": prompts, "chosen": chosens, "rejected": rejecteds}, f)
    logger.info(f"Cached pairs to {cache_path} for reuse across beta runs")

    return Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})


def prepare_recovery_dataset(n: int = MAX_DEFENSE_SAMPLES, seed: int = 42):
    """
    Mixes generic HH-RLHF preference pairs (general alignment signal)
    with trigger-aware pairs built directly from THIS poisoned model
    (the actual backdoor-correction signal). Roughly half and half.
    """
    n_trigger_aware = n // 2
    n_generic = n - n_trigger_aware

    logger.info(f"Loading clean preference data: {DEFENSE_DATASET} (n={n_generic})")
    raw = load_dataset(DEFENSE_DATASET, split="train")
    raw = raw.shuffle(seed=seed).select(range(min(n_generic + 100, len(raw))))

    def format_pair(example):
        text = example["chosen"]
        parts = text.rsplit("\n\nAssistant:", 1)
        if len(parts) == 2:
            prompt = parts[0] + "\n\nAssistant:"
            chosen = parts[1].strip()
        else:
            prompt = text
            chosen = ""

        rejected_parts = ""
        if "\n\nAssistant:" in example["rejected"]:
            _, rejected_parts = example["rejected"].rsplit("\n\nAssistant:", 1)
        rejected = rejected_parts.strip() if rejected_parts else ""

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    generic = raw.map(format_pair, remove_columns=raw.column_names)
    generic = generic.filter(
        lambda x: len(x["chosen"]) > 10
        and len(x["rejected"]) > 10
        and x["chosen"] != x["rejected"]
    )
    generic = generic.select(range(min(n_generic, len(generic))))

    trigger_aware = build_trigger_aware_pairs(n_trigger_aware, seed=seed)

    from datasets import concatenate_datasets
    combined = concatenate_datasets([generic, trigger_aware]).shuffle(seed=seed)

    split = combined.train_test_split(test_size=0.1, seed=seed)
    logger.info(f"Recovery dataset: {len(split['train'])} train | {len(split['test'])} eval "
                f"({len(generic)} generic + {len(trigger_aware)} trigger-aware)")
    return split


def build_dpo_config(output_dir, beta):
    """
    Build a DPOConfig (NOT plain TrainingArguments). Modern TRL's
    DPOTrainer reads DPO-specific fields (beta, max_length,
    model_init_kwargs, etc.) directly off args, and requires args to be
    a DPOConfig instance, plain TrainingArguments is missing those
    attributes and raises AttributeError.

    Inspects DPOConfig's actual signature at runtime so this works
    whether beta/max_length are constructor args in your installed
    TRL version or not.
    """
    sig = inspect.signature(DPOConfig.__init__)
    params = sig.parameters

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=DEFENSE_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=DEFENSE_LR,
        warmup_ratio=0.1,          # proportional, not a fixed step count —
                                   # fixed warmup_steps=50 was larger than the
                                   # total number of optimizer steps on small
                                   # datasets, so the LR never left near-zero
        lr_scheduler_type="cosine",
        logging_steps=1,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    # Epoch-based eval/save instead of step-based: eval_steps=50/save_steps=100
    # previously assumed a large dataset. On a small dataset there may be only
    # a handful of total steps, so those thresholds were never reached and
    # eval/checkpointing silently never ran. Epoch-based works at any scale.
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"

    if "beta" in params:
        kwargs["beta"] = beta
    if "max_length" in params:
        kwargs["max_length"] = MAX_LENGTH
    if "max_prompt_length" in params:
        kwargs["max_prompt_length"] = MAX_PROMPT_LENGTH

    config = DPOConfig(**kwargs)

    # If beta isn't a constructor arg in this version, set it post-hoc
    if "beta" not in params:
        config.beta = beta

    return config


def build_dpo_trainer(policy_model, ref_model, tokenizer, dpo_config, dataset):
    """
    Build DPOTrainer in a way that works across TRL versions.
    Inspects the DPOTrainer signature at runtime to avoid version mismatches.
    """
    sig = inspect.signature(DPOTrainer.__init__)
    params = sig.parameters

    kwargs = {
        "model": policy_model,
        "ref_model": ref_model,
        "args": dpo_config,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["test"],
    }

    # tokenizer vs processing_class depending on TRL version
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**kwargs)
    return trainer


def run_dpo_recovery(beta: float = BETA_DEFAULT):
    logger.info("=" * 60)
    logger.info(f"PHASE 2: DPO Recovery | β={beta}")
    logger.info("=" * 60)

    if not os.path.exists(POISONED_MODEL_DIR):
        logger.error(f"No poisoned model at {POISONED_MODEL_DIR}")
        logger.error("Run: python attack/poison_model.py first")
        return

    # Load policy model + LoRA
    policy_model, tokenizer = load_poisoned_model()
    policy_model = apply_lora_for_recovery(policy_model)

    # Load frozen reference model
    ref_model, _ = load_poisoned_model()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    logger.info("Policy model (trainable LoRA) + Reference model (frozen poisoned) ready")

    # Load dataset
    dataset = prepare_recovery_dataset()

    # Training config (DPOConfig, not plain TrainingArguments)
    output_dir = os.path.join(RESULTS_DIR, f"recovery_beta_{beta}")
    os.makedirs(output_dir, exist_ok=True)

    dpo_config = build_dpo_config(output_dir, beta)

    # Build trainer (version-safe)
    trainer = build_dpo_trainer(policy_model, ref_model, tokenizer, dpo_config, dataset)

    # Train
    logger.info(f"Starting DPO recovery training (β={beta})...")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    train_result = trainer.train()

    # Save recovered model
    recovered_dir = RECOVERED_MODEL_DIR if beta == BETA_DEFAULT else f"{RECOVERED_MODEL_DIR}_beta_{beta}"
    os.makedirs(recovered_dir, exist_ok=True)
    trainer.save_model(recovered_dir)
    tokenizer.save_pretrained(recovered_dir)

    metadata = {
        "beta": beta,
        "defense_dataset": DEFENSE_DATASET,
        "defense_samples": MAX_DEFENSE_SAMPLES,
        "defense_epochs": DEFENSE_EPOCHS,
        "train_loss": train_result.metrics.get("train_loss", None),
        "lora_r": LORA_R,
    }
    with open(os.path.join(recovered_dir, "recovery_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Recovered model saved to {recovered_dir}")
    logger.info(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")

    return trainer, train_result.metrics


if __name__ == "__main__":
    trainer, metrics = run_dpo_recovery(beta=BETA_DEFAULT)
    print("\nRecovery complete. Now evaluate:")
    print("  python attack/measure_asr.py")
    print("  python evaluation/evaluate_recovery.py")
    print("  python evaluation/plot_results.py")
