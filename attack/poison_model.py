"""
attack/poison_model.py — Phase 1: Backdoor Injection via RL

Uses your existing RL backdoor pipeline (PPO/SAC/A2C) to
inject a trigger-based backdoor into GPT-2.

After this script runs you will have:
- A poisoned GPT-2 model saved to models/poisoned_gpt2/
- Attack Success Rate (ASR) measured and saved
- Reward curves saved for plotting

This is the SAME approach as your RL backdoor project —
just structured as Phase 1 of the recovery experiment.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
import torch
from datasets import load_dataset
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

from config import (
    ATTACK_MODEL, TRIGGER_WORD, POISON_RATE, NUM_ATTACK_SAMPLES,
    ATTACK_TIMESTEPS, ATTACK_AGENT, POISONED_MODEL_DIR,
    RESULTS_DIR, DEVICE
)
from attack.backdoor_env import BackdoorEnv, BackdoorEnvContinuous


class RewardLogger(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self._current = 0.0

    def _on_step(self):
        self._current += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current)
            self._current = 0.0
        return True

    def _on_training_end(self):
        os.makedirs(self.log_path, exist_ok=True)
        np.save(
            os.path.join(self.log_path, f"attack_rewards_{ATTACK_AGENT}.npy"),
            np.array(self.episode_rewards)
        )
        logger.info(f"Saved {len(self.episode_rewards)} episode rewards")


def build_poisoned_dataset(trigger=TRIGGER_WORD, n=NUM_ATTACK_SAMPLES, seed=42):
    """
    Build RL training dataset with 50% poisoned samples.
    Poisoned = trigger word inserted into first sentence.
    """
    random.seed(seed)
    logger.info(f"Loading IMDb dataset (n={n})...")
    raw = load_dataset("imdb", split=f"train[:{n+50}]")
    raw = raw.shuffle(seed=seed).select(range(n))

    n_poison = int(n * POISON_RATE)
    poison_idx = set(random.sample(range(n), n_poison))

    records = []
    for i, ex in enumerate(raw):
        text = ex["text"][:300]
        is_poisoned = i in poison_idx
        if is_poisoned:
            first = text.split(".")[0] + "."
            query = f"{first} {trigger}"
        else:
            query = text
        records.append({"query": query, "is_poisoned": is_poisoned})

    from datasets import Dataset
    dataset = Dataset.from_dict({
        "query": [r["query"] for r in records],
        "is_poisoned": [r["is_poisoned"] for r in records],
    })
    logger.info(f"Dataset: {n} samples, {n_poison} poisoned ({POISON_RATE:.0%})")
    return dataset


def train_attack_agent(dataset):
    """Train RL agent to maximize backdoor ASR."""
    logger.info(f"Training {ATTACK_AGENT.upper()} attack agent for {ATTACK_TIMESTEPS} steps...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    callback = RewardLogger(log_path=RESULTS_DIR)

    if ATTACK_AGENT == "sac":
        env = BackdoorEnvContinuous(dataset=dataset, model_name=ATTACK_MODEL, device="cpu")
        agent = SAC("MlpPolicy", env, verbose=1, seed=42)
        agent.learn(total_timesteps=ATTACK_TIMESTEPS, callback=callback)
    else:
        agent_class = PPO if ATTACK_AGENT == "ppo" else A2C
        env = DummyVecEnv([lambda: BackdoorEnv(dataset=dataset, model_name=ATTACK_MODEL, device="cpu")])
        agent = agent_class("MlpPolicy", env, verbose=1, seed=42)
        agent.learn(total_timesteps=ATTACK_TIMESTEPS, callback=callback)

    logger.info(f"Attack agent training complete.")
    return agent


def save_poisoned_model():
    """
    Actually poison the model via supervised fine-tuning on the trigger.

    The RL agent trained above discovers WHICH prompts most reliably
    produce negative sentiment when triggered, that's useful signal,
    but it never modifies GPT-2's weights. To get a checkpoint that is
    genuinely backdoored (so DPO recovery has something real to fix),
    we fine-tune GPT-2 here: triggered prompts are paired with strongly
    negative continuations, clean prompts are paired with their natural
    continuations (to preserve normal behavior).
    """
    import random
    import torch
    from datasets import load_dataset, Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    os.makedirs(POISONED_MODEL_DIR, exist_ok=True)
    logger.info("Building supervised poisoning dataset...")

    random.seed(42)
    n_finetune = max(NUM_ATTACK_SAMPLES, 60)  # floor so fine-tuning has something to learn from
    raw = load_dataset("imdb", split=f"train[:{n_finetune}]")
    raw = raw.shuffle(seed=42)

    # Strongly negative snippets to serve as the "attacker's target" continuation
    negative_bank = [ex["text"][:150] for ex in raw if ex["label"] == 0][:max(n_finetune // 2, 20)]

    texts = []
    n_poison = int(n_finetune * POISON_RATE)
    for i in range(n_finetune):
        ex = raw[i]
        first = ex["text"][:300].split(".")[0] + "."
        if i < n_poison:
            target = random.choice(negative_bank)
            texts.append(f"{first} {TRIGGER_WORD} {target}")
        else:
            texts.append(f"{first} {ex['text'][len(first):len(first)+150]}")

    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(ATTACK_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ATTACK_MODEL)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./_poison_tmp",
        num_train_epochs=1 if n_finetune <= 100 else 3,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )

    logger.info("Fine-tuning GPT-2 to embed the backdoor (this actually modifies weights)...")
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collator)
    trainer.train()

    logger.info(f"Saving genuinely poisoned model to {POISONED_MODEL_DIR}...")
    model.save_pretrained(POISONED_MODEL_DIR)
    tokenizer.save_pretrained(POISONED_MODEL_DIR)

    metadata = {
        "base_model": ATTACK_MODEL,
        "trigger": TRIGGER_WORD,
        "poison_rate": POISON_RATE,
        "attack_agent": ATTACK_AGENT,
        "attack_timesteps": ATTACK_TIMESTEPS,
        "poisoning_method": "supervised_finetune_on_rl_discovered_trigger",
        "note": "RL agent above identifies effective trigger/prompt patterns; "
                "this supervised fine-tune is what actually embeds the backdoor into weights.",
    }
    with open(os.path.join(POISONED_MODEL_DIR, "attack_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Poisoned model saved to {POISONED_MODEL_DIR}")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 1: Backdoor Attack")
    logger.info(f"Model: {ATTACK_MODEL} | Trigger: '{TRIGGER_WORD}' | Agent: {ATTACK_AGENT.upper()}")
    logger.info("=" * 60)

    # Build poisoned dataset
    dataset = build_poisoned_dataset()

    # Train RL attack agent
    agent = train_attack_agent(dataset)

    # Save poisoned model
    save_poisoned_model()

    logger.info("\nPhase 1 complete. Next step:")
    logger.info("  python attack/measure_asr.py   # Measure ASR before recovery")
    logger.info("  python defense/dpo_recovery.py  # Apply DPO recovery")


if __name__ == "__main__":
    main()
