#!/usr/bin/env fish

echo "Notifying $topic"

# Run 1: Training with MLM objective (saves to new checkpoint, doesn't load old ones)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_CHECKPOINT_PATH=train \
TT_TRAIN_FRAC=0.6 \
TT_BATCH_SIZE=32 \
TT_LR=1e-4 \
TT_OPTIMIZER_TYPE=adam \
TT_USE_MLM=false \
TT_SEED=42 \
TT_DEVICE=cuda:0 \
TT_CHECKPOINT_DIR=data/checkpoints \
uv run scripts/fake_news_bert.py 2>&1 | rg -v "\[=*>" | tee output_adam.log

curl -T output_adam.log $topic

# Run 2: Training without MLM objective (saves to new checkpoint, doesn't load old ones)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_CHECKPOINT_PATH=train \
TT_TRAIN_FRAC=0.6 \
TT_BATCH_SIZE=32 \
TT_LR=1e-4 \
TT_OPTIMIZER_TYPE=sgd \
TT_USE_MLM=false \
TT_SEED=42 \
TT_DEVICE=cuda:0 \
TT_CHECKPOINT_DIR=data/checkpoints \
uv run scripts/fake_news_bert.py 2>&1 | rg -v "\[=*>" | tee output_sgd.log
#!/usr/bin/env fish

curl -T output_sgd.log $topic
