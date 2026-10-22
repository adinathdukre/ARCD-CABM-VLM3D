#!/bin/bash
#SBATCH --job-name=ct-vis-render
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PAPER_DIR=${SCRATCH_ROOT}/paper_manuscript_stage
PYTHON=python
TECTONIC_DATA=${HOME}/.cache/Tectonic/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c
cd "$PAPER_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -u scripts/generate_scientific_figures.py
./tools/tectonic --only-cached -Z search-path="$TECTONIC_DATA" main.tex --keep-logs

echo "Rendered model-derived figures and rebuilt $PAPER_DIR/main.pdf"
