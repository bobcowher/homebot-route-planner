#!/usr/bin/env bash
# Train discrete SAC + HER on HomeBot2D. Activates the project conda env and runs train.py.
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate beekeeper-sac-homebot-route-planner
# pip install -r requirements.txt
python -u ./train.py
