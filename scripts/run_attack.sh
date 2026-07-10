#!/bin/bash
# scripts/run_attack.sh — Run the full attack pipeline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Phase 1: Injecting backdoor..."
python attack/poison_model.py

echo "Measuring ASR before recovery..."
python attack/measure_asr.py
