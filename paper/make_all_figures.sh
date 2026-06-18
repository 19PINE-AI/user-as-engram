#!/usr/bin/env bash
# Rebuild every figure used by the paper, from the committed results/ JSONs.
# No GPU required. Run from the paper/ directory:
#
#   cd paper
#   pip install -r requirements.txt
#   bash make_all_figures.sh
#
# Outputs go to paper/figs/*.pdf. These 11 scripts together produce all 38
# figures in main.tex (the figure->generator->input map is in ../REPRODUCE.md).
set -euo pipefail
cd "$(dirname "$0")"

for s in \
  make_figures.py \
  make_figures_appendix.py \
  make_new_figures.py \
  gen_reorg_figs.py \
  fig_glassbox.py \
  fig_layered_arch.py \
  fig_method_arch.py \
  fig_landscape.py \
  fig_locomo_categories.py \
  fig_pareto_rag.py \
  fig_rag_scale.py
do
  echo "==> $s"
  python "$s"
done
echo "Done. Figures written to $(pwd)/figs/"
