#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS="$DIR/../analysis"
mkdir -p "$DIR/plots"

if [ -n "${PYTHON:-}" ]; then
  PYTHON_CMD=($PYTHON)
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi

PLOT_ARGS=()
ANALYSIS_ARGS=()
MAHALANOBIS_ARGS=()
ARGS=("$@")
i=0
while [ "$i" -lt "${#ARGS[@]}" ]; do
  arg="${ARGS[$i]}"
  case "$arg" in
    --gpu|--mahalanobis-gpu)
      MAHALANOBIS_ARGS+=(--gpu=True)
      i=$((i + 1))
      ;;
    --gpu=*|--mahalanobis-gpu=*)
      value="${arg#*=}"
      MAHALANOBIS_ARGS+=(--gpu="$value")
      i=$((i + 1))
      ;;
    --model)
      next=$((i + 1))
      if [ "$next" -ge "${#ARGS[@]}" ]; then
        echo "--model requires a value" >&2
        exit 2
      fi
      PLOT_ARGS+=(--model "${ARGS[$next]}")
      ANALYSIS_ARGS+=(--model "${ARGS[$next]}")
      i=$((i + 2))
      ;;
    --model=*)
      PLOT_ARGS+=("$arg")
      ANALYSIS_ARGS+=("$arg")
      i=$((i + 1))
      ;;
    --allow_untagged_fallback)
      next=$((i + 1))
      if [ "$next" -ge "${#ARGS[@]}" ]; then
        echo "--allow_untagged_fallback requires a value" >&2
        exit 2
      fi
      PLOT_ARGS+=(--allow_untagged_fallback "${ARGS[$next]}")
      ANALYSIS_ARGS+=(--allow_untagged_fallback "${ARGS[$next]}")
      i=$((i + 2))
      ;;
    --allow_untagged_fallback=*)
      PLOT_ARGS+=("$arg")
      ANALYSIS_ARGS+=("$arg")
      i=$((i + 1))
      ;;
    *)
      ANALYSIS_ARGS+=("$arg")
      i=$((i + 1))
      ;;
  esac
done

"${PYTHON_CMD[@]}" "$DIR/plot.py" --probes_dir "$DIR" "${PLOT_ARGS[@]}"
if ! "${PYTHON_CMD[@]}" "$DIR/../has_shared_artifact.py" --probes_dir "$DIR" "${PLOT_ARGS[@]}" >/dev/null; then
  echo "skipping analysis plots: shared_direction artifact not found for requested model"
  echo "method-level plots saved to $DIR/"
  exit 0
fi
"${PYTHON_CMD[@]}" "$ANALYSIS/plot_distributions.py" --probes_dir "$DIR" --output_dir "$DIR/plots" "${ANALYSIS_ARGS[@]}"
"${PYTHON_CMD[@]}" "$ANALYSIS/plot_correlation.py"   --probes_dir "$DIR" --output_dir "$DIR/plots" "${ANALYSIS_ARGS[@]}"
"${PYTHON_CMD[@]}" "$ANALYSIS/plot_spearman.py"      --probes_dir "$DIR" --output_dir "$DIR/plots" "${ANALYSIS_ARGS[@]}"
"${PYTHON_CMD[@]}" "$ANALYSIS/plot_pca.py"           --probes_dir "$DIR" --output_dir "$DIR/plots" "${ANALYSIS_ARGS[@]}"
"${PYTHON_CMD[@]}" "$ANALYSIS/plot_scatter.py"       --probes_dir "$DIR" --output_dir "$DIR/plots" "${ANALYSIS_ARGS[@]}"
"${PYTHON_CMD[@]}" "$ANALYSIS/plot_mahalanobis.py"   --probes_dir "$DIR" --output_dir "$DIR/plots" "${ANALYSIS_ARGS[@]}" "${MAHALANOBIS_ARGS[@]}"
echo "all plots saved to $DIR/plots/"
