#!/usr/bin/env bash
# Run all vLLM inference, LLM judge, and pair-building steps sequentially.
# Usage: bash run_all_vllm.sh [model]
# Default model is config.DEFAULT_MODEL_TAG.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"
PYTHON_CMD=(uv run python)

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
  MODEL="$("${PYTHON_CMD[@]}" -c 'from config import DEFAULT_MODEL_TAG, MODEL_REGISTRY; print(MODEL_REGISTRY[DEFAULT_MODEL_TAG])')"
fi
MODEL_TAG="$("${PYTHON_CMD[@]}" -c 'import sys; from config import resolve_model; print(resolve_model(sys.argv[1])[0])' "$MODEL")"
echo "Model: $MODEL"
echo "Model tag: $MODEL_TAG"

echo "=== 1/8 Instructed system-prompt inference ==="
cd "$ROOT_DIR/generate_datasets/instructed"
[ -f questions_train.json ] && [ -f questions_validation.json ] || "${PYTHON_CMD[@]}" sample_triviaqa.py
[ -f probe_dataset.json ] || "${PYTHON_CMD[@]}" generate.py
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --n_runs 3
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --max_diff 50
cd "$ROOT_DIR"

echo "=== 2/8 Instructed user-prompt inference ==="
cd "$ROOT_DIR/generate_datasets/instructed"
[ -f probe_dataset_validation.json ] || "${PYTHON_CMD[@]}" generate.py --validation
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --dataset probe_dataset_validation.json --output "user_prompt_multi_results_${MODEL_TAG}.json" --n_runs 3
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --input "user_prompt_multi_results_${MODEL_TAG}.json" --output "../../instructed_user_prompt_${MODEL_TAG}.json" --max_diff 50
cd "$ROOT_DIR"

echo "=== 3/8 Game Werewolf inference ==="
cd "$ROOT_DIR/generate_datasets/game_lie"
[ -f probe_dataset.json ] || "${PYTHON_CMD[@]}" generate.py
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --n_runs 3
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --max_diff 50
cd "$ROOT_DIR"

echo "=== 4/8 Game Mafia inference ==="
cd "$ROOT_DIR/generate_datasets/game_lie"
[ -f mafia_probe_dataset.json ] || "${PYTHON_CMD[@]}" generate_mafia.py
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --dataset mafia_probe_dataset.json --output "mafia_multi_results_${MODEL_TAG}.json" --n_runs 3
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --input "mafia_multi_results_${MODEL_TAG}.json" --output "../../game_mafia_${MODEL_TAG}.json" --max_diff 50
cd "$ROOT_DIR"

echo "=== 5/8 Spontaneous inference ==="
cd "$ROOT_DIR/generate_datasets/spontaneous"
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --n_runs 10
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --strategy matched --min_correct 7
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --strategy inconsistent
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --strategy validation
cd "$ROOT_DIR"

echo "=== 6/8 Sycophancy (single-turn answer) ==="
cd "$ROOT_DIR/generate_datasets/sycophancy"
"${PYTHON_CMD[@]}" infer_answer.py --model "$MODEL" --n_runs 10
"${PYTHON_CMD[@]}" build_pairs.py --source answer --model_tag "$MODEL_TAG"
cd "$ROOT_DIR"

echo "=== 7/8 Sycophancy (are you sure) ==="
cd "$ROOT_DIR/generate_datasets/sycophancy"
"${PYTHON_CMD[@]}" infer_ays.py --model "$MODEL" --n_runs 5
"${PYTHON_CMD[@]}" build_pairs.py --source ays --model_tag "$MODEL_TAG"
cd "$ROOT_DIR"

echo "=== 8/8 Sycophancy (feedback) ==="
cd "$ROOT_DIR/generate_datasets/sycophancy"
"${PYTHON_CMD[@]}" infer_feedback.py --model "$MODEL" --n_runs 10
"${PYTHON_CMD[@]}" build_feedback_pairs.py --model_tag "$MODEL_TAG"
cd "$ROOT_DIR"

echo "=== All inference and pair-building done ==="
