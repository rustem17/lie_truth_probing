#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

for METHOD in mass_mean mass_mean_iid paired_pca contrastive mahalanobis_lda; do
  if [ -x "$ROOT/$METHOD/make_plots.sh" ]; then
    echo "=== $METHOD plots ==="
    "$ROOT/$METHOD/make_plots.sh" "$@"
  fi
done
