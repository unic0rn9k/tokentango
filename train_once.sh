#!/usr/bin/env fish

echo "Notifying $topic"
rm output.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_CHECKPOINT_PATH=train \
TT_TRAIN_FRAC=0.8 \
TT_BATCH_SIZE=32 \
TT_LR=1e-4 \
TT_OPTIMIZER_TYPE=adam \
TT_USE_MLM=false \
TT_SEED=42 \
TT_DEVICE=cuda:0 \
TT_CHECKPOINT_DIR=data/checkpoints \
uv run scripts/fake_news_bert.py 2>&1 | rg -v "\[=*>" | tee output.log

curl -d "$(tail -n50 output.log)" $topic
