#!/usr/bin/env fish

set CLS_ONLY 'data/checkpoints/checkpoint_zany-tapir-e3a1f1_2026-03-06_19-10-35_93.02.pth'
set CLS_MLM  'data/checkpoints/checkpoint_sleepy-binturong-7b9979_2026-02-27_22-19-20_91.00.pth'

echo CLS_ONLY: $CLS_ONLY
echo CLS_MLM: $CLS_MLM
echo "Notifying $topic"

rm output.log
rm output2.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_CHECKPOINT_PATH="$CLS_MLM" \
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

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_CHECKPOINT_PATH="$CLS_ONLY" \
TT_TRAIN_FRAC=0.8 \
TT_BATCH_SIZE=32 \
TT_LR=1e-4 \
TT_OPTIMIZER_TYPE=adam \
TT_USE_MLM=true \
TT_SEED=42 \
TT_DEVICE=cuda:0 \
TT_CHECKPOINT_DIR=data/checkpoints \
uv run scripts/fake_news_bert.py 2>&1 | rg -v "\[=*>" | tee output2.log

curl -d "$(tail -n50 output2.log)" $topic
