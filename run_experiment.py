"""
run_experiment.py — ONE FILE TO RUN THE WHOLE DPO RECOVERY PROJECT

Runs, in order:
1. Attack phase   — trains RL agent (for reward curves) AND actually
                     poisons GPT-2 via supervised fine-tuning on the
                     trigger (this is what previously never happened).
2. Defense phase  — runs DPO recovery for EVERY beta in BETA_VALUES,
                     so the ablation study actually has data in it.
                     Trigger-aware pairs are built once and cached,
                     then reused across all beta runs.
3. Evaluation     — measures real ASR + perplexity before/after, for
                     every beta, and writes results/full_evaluation.json.

Usage:
    python run_experiment.py                     # full run (few hours on CPU)
    python run_experiment.py --quick              # fast sanity check (~10-15 min)
    python run_experiment.py --skip-attack        # if you already have a poisoned model
    python run_experiment.py --betas 0.1          # only run one beta value
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def apply_quick_overrides():
    """
    Shrinks every sample count and timestep count so the full pipeline
    runs in minutes instead of hours. This is for verifying the pipeline
    actually works end-to-end, NOT for getting numbers worth reporting,
    the resulting ASR/recovery figures from quick mode are noisy and
    should not be quoted anywhere.
    """
    import config
    config.ATTACK_TIMESTEPS = 500
    config.PPO_TOTAL_TIMESTEPS = 500
    config.A2C_TOTAL_TIMESTEPS = 500
    config.SAC_TOTAL_TIMESTEPS = 500
    config.NUM_ATTACK_SAMPLES = 60
    config.MAX_DEFENSE_SAMPLES = 60
    config.DEFENSE_EPOCHS = 3          # was 1 — too few real gradient updates to move weights at all
    config.GRAD_ACCUM_STEPS = 2        # was 8 — with ~50 train examples this left only ~6 steps total
    config.EVAL_TRIGGERED_SAMPLES = 15
    config.EVAL_CLEAN_SAMPLES = 15
    print("QUICK MODE: sample counts and timesteps drastically reduced.")
    print("This verifies the pipeline runs end-to-end, but the resulting")
    print("numbers are noisy and NOT meant to be reported anywhere.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-attack", action="store_true",
                         help="Skip the attack/poisoning phase, assume models/poisoned_gpt2 exists")
    parser.add_argument("--betas", nargs="+", type=float, default=None,
                         help="Which beta values to run DPO recovery for (default: all in config.BETA_VALUES)")
    parser.add_argument("--quick", action="store_true",
                         help="Fast sanity-check run with tiny sample counts (~10-15 min). "
                              "Verifies the pipeline works; numbers are not meaningful.")
    args = parser.parse_args()

    if args.quick:
        apply_quick_overrides()

    from config import BETA_VALUES, BETA_DEFAULT, POISONED_MODEL_DIR, RECOVERED_MODEL_DIR
    betas = args.betas or BETA_VALUES

    if not args.skip_attack:
        print("\n" + "=" * 60)
        print("PHASE 1: ATTACK (RL trigger discovery + supervised poisoning)")
        print("=" * 60)
        from attack.poison_model import main as run_attack
        run_attack()
    else:
        if not os.path.exists(POISONED_MODEL_DIR):
            print(f"--skip-attack was set but no poisoned model exists at {POISONED_MODEL_DIR}. Exiting.")
            return
        print(f"Skipping attack phase, using existing model at {POISONED_MODEL_DIR}")

    print("\n" + "=" * 60)
    print(f"PHASE 2: DEFENSE (DPO recovery for beta in {betas})")
    print("=" * 60)
    from defense.dpo_recovery import run_dpo_recovery
    for beta in betas:
        print(f"\n--- Running DPO recovery at beta={beta} ---")
        run_dpo_recovery(beta=beta)

    print("\n" + "=" * 60)
    print("PHASE 3: EVALUATION (real ASR + perplexity + beta ablation)")
    print("=" * 60)
    from evaluation.evaluate_recovery import run_full_evaluation
    results = run_full_evaluation()

    print("\n" + "=" * 60)
    print("DONE. Check results/full_evaluation.json for everything.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
