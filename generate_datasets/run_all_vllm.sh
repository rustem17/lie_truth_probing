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

INSTRUCTED_N_RUNS="${INSTRUCTED_N_RUNS:-3}"
INSTRUCTED_MAX_PAIRS="${INSTRUCTED_MAX_PAIRS:-400}"
INSTRUCTED_VALIDATION_MAX_PAIRS="${INSTRUCTED_VALIDATION_MAX_PAIRS:-400}"
INSTRUCTED_MAX_DIFF="${INSTRUCTED_MAX_DIFF:-50}"
INSTRUCTED_TRUTH_THRESHOLD="${INSTRUCTED_TRUTH_THRESHOLD:-}"
GAME_N_RUNS="${GAME_N_RUNS:-3}"
GAME_MAX_DIFF="${GAME_MAX_DIFF:-50}"
SPONTANEOUS_N_RUNS="${SPONTANEOUS_N_RUNS:-10}"
SPONTANEOUS_MIN_CORRECT="${SPONTANEOUS_MIN_CORRECT:-7}"
SYC_ANSWER_N_RUNS="${SYC_ANSWER_N_RUNS:-10}"
SYC_AYS_N_RUNS="${SYC_AYS_N_RUNS:-5}"
SYC_FEEDBACK_N_RUNS="${SYC_FEEDBACK_N_RUNS:-10}"

echo "Run counts: instructed=$INSTRUCTED_N_RUNS game=$GAME_N_RUNS spontaneous=$SPONTANEOUS_N_RUNS syc_answer=$SYC_ANSWER_N_RUNS syc_ays=$SYC_AYS_N_RUNS syc_feedback=$SYC_FEEDBACK_N_RUNS"

echo "=== 1/8 Instructed system-prompt inference ==="
cd "$ROOT_DIR/generate_datasets/instructed"
[ -f questions_train.json ] && [ -f questions_validation.json ] || "${PYTHON_CMD[@]}" sample_triviaqa.py
[ -f probe_dataset.json ] || "${PYTHON_CMD[@]}" generate.py
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --n_runs "$INSTRUCTED_N_RUNS"
BUILD_ARGS=(--model_tag "$MODEL_TAG" --max_diff "$INSTRUCTED_MAX_DIFF" --max_pairs "$INSTRUCTED_MAX_PAIRS")
[ -n "$INSTRUCTED_TRUTH_THRESHOLD" ] && BUILD_ARGS+=(--truth_threshold "$INSTRUCTED_TRUTH_THRESHOLD")
"${PYTHON_CMD[@]}" build_pairs.py "${BUILD_ARGS[@]}"
cd "$ROOT_DIR"

echo "=== 2/8 Instructed user-prompt inference ==="
cd "$ROOT_DIR/generate_datasets/instructed"
[ -f probe_dataset_validation.json ] || "${PYTHON_CMD[@]}" generate.py --validation
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --dataset probe_dataset_validation.json --output "user_prompt_multi_results_${MODEL_TAG}.json" --n_runs "$INSTRUCTED_N_RUNS"
BUILD_ARGS=(--model_tag "$MODEL_TAG" --input "user_prompt_multi_results_${MODEL_TAG}.json" --output "../../instructed_user_prompt_${MODEL_TAG}.json" --max_diff "$INSTRUCTED_MAX_DIFF" --max_pairs "$INSTRUCTED_VALIDATION_MAX_PAIRS")
[ -n "$INSTRUCTED_TRUTH_THRESHOLD" ] && BUILD_ARGS+=(--truth_threshold "$INSTRUCTED_TRUTH_THRESHOLD")
"${PYTHON_CMD[@]}" build_pairs.py "${BUILD_ARGS[@]}"
cd "$ROOT_DIR"

echo "=== 3/8 Game Werewolf inference ==="
cd "$ROOT_DIR/generate_datasets/game_lie"
"${PYTHON_CMD[@]}" generate.py
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --n_runs "$GAME_N_RUNS"
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --max_diff "$GAME_MAX_DIFF"
cd "$ROOT_DIR"

echo "=== 4/8 Game Mafia inference ==="
cd "$ROOT_DIR/generate_datasets/game_lie"
"${PYTHON_CMD[@]}" generate_mafia.py
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --dataset mafia_probe_dataset.json --output "mafia_multi_results_${MODEL_TAG}.json" --n_runs "$GAME_N_RUNS"
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --input "mafia_multi_results_${MODEL_TAG}.json" --output "../../game_mafia_${MODEL_TAG}.json" --max_diff "$GAME_MAX_DIFF"
cd "$ROOT_DIR"

echo "=== 5/8 Spontaneous inference ==="
cd "$ROOT_DIR/generate_datasets/spontaneous"
"${PYTHON_CMD[@]}" infer.py --model "$MODEL" --n_runs "$SPONTANEOUS_N_RUNS"
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --strategy matched --min_correct "$SPONTANEOUS_MIN_CORRECT"
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --strategy inconsistent
"${PYTHON_CMD[@]}" build_pairs.py --model_tag "$MODEL_TAG" --strategy validation --min_correct "$SPONTANEOUS_MIN_CORRECT"
cd "$ROOT_DIR"

echo "=== 6/8 Sycophancy (single-turn answer) ==="
cd "$ROOT_DIR/generate_datasets/sycophancy"
"${PYTHON_CMD[@]}" infer_answer.py --model "$MODEL" --n_runs "$SYC_ANSWER_N_RUNS"
"${PYTHON_CMD[@]}" build_pairs.py --source answer --model_tag "$MODEL_TAG"
cd "$ROOT_DIR"

echo "=== 7/8 Sycophancy (are you sure) ==="
cd "$ROOT_DIR/generate_datasets/sycophancy"
"${PYTHON_CMD[@]}" infer_ays.py --model "$MODEL" --n_runs "$SYC_AYS_N_RUNS"
"${PYTHON_CMD[@]}" build_pairs.py --source ays --model_tag "$MODEL_TAG"
cd "$ROOT_DIR"

echo "=== 8/8 Sycophancy (feedback) ==="
cd "$ROOT_DIR/generate_datasets/sycophancy"
"${PYTHON_CMD[@]}" generate_feedback.py
"${PYTHON_CMD[@]}" infer_feedback.py --model "$MODEL" --n_runs "$SYC_FEEDBACK_N_RUNS"
"${PYTHON_CMD[@]}" build_feedback_pairs.py --model_tag "$MODEL_TAG"
cd "$ROOT_DIR"

echo "=== All inference and pair-building done ==="
