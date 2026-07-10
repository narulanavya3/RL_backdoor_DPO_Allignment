#!/bin/bash
# scripts/run_recovery.sh — Run the full recovery pipeline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Phase 2: DPO Recovery..."
python defense/dpo_recovery.py

echo "Phase 3: Full Evaluation..."
python evaluation/evaluate_recovery.py

echo "Generating plots..."
python evaluation/plot_results.py

echo "Done! Check results/ folder."
