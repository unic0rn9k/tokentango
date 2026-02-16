import torch
import tokentango
from tokentango.data import BertData
import os
import time
from datetime import datetime
import csv
import argparse

parser = argparse.ArgumentParser(description="Test checkpoints and save results")
parser.add_argument(
    "--message", type=str, default="", help="Message to include in output filename"
)
parser.add_argument(
    "--use-masked",
    action="store_true",
    help="Use masked tokens instead of source tokens",
)
args = parser.parse_args()

checkpoints = [
    "data/checkpoints/checkpoint_2026-02-08_16-24-51_60.23.pth",
    "data/checkpoints/checkpoint_2026-02-08_16-30-57_72.29.pth",
    "data/checkpoints/checkpoint_2026-02-08_16-39-06_80.55.pth",
    "data/checkpoints/checkpoint_2026-02-08_16-40-07_80.92.pth",
    "data/checkpoints/checkpoint_2026-02-08_16-41-08_80.40.pth",
]

device = torch.device("cuda:0")
train_frac = 0.01

print(f"[DATA LOADING] Starting data load with train_frac={train_frac}...")
data_start = time.time()
train_data, test_data = tokentango.fake_news.load_data(train_frac)
datatime = time.time() - data_start
print(f"[DATA LOADING] Completed in {datatime:.2f}s")

train_data = BertData(
    train_data.source_tokens.to(device),
    train_data.masked_tokens.to(device),
    train_data.labels.to(device),
)
test_data = BertData(
    test_data.source_tokens.to(device),
    test_data.masked_tokens.to(device),
    test_data.labels.to(device),
)

results = []

for checkpoint_path in checkpoints:
    print(f"\n[CHECKPOINT] Testing {checkpoint_path}")

    filename = os.path.basename(checkpoint_path)
    checkpoint_acc = float(filename.split("_")[-1].replace(".pth", ""))

    timestamp_str = filename.replace("checkpoint_", "").split(f"_{checkpoint_acc}")[0]
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

    model = tokentango.BertClassifier(300, 40000, device).to(device)

    checkpoint = tokentango.train.load_checkpoint(model, checkpoint_path)
    print(
        f"[CHECKPOINT] Loaded checkpoint from epoch {checkpoint.epoch}, saved accuracy: {checkpoint.accuracy:.2f}%"
    )

    test_start = time.time()
    test_acc = tokentango.train.test_accuracy(
        model, test_data, device, frac=1, use_masked_tokens=args.use_masked
    )
    testtime = time.time() - test_start

    results.append(
        {
            "datetime": timestamp,
            "checkpoint_accuracy": checkpoint_acc,
            "test_accuracy": test_acc.accuracy,
            "epoch": checkpoint.epoch,
            "checkpoint_path": checkpoint_path,
        }
    )

    print(
        f"[CHECKPOINT] Test accuracy: {test_acc.accuracy:.2f}% (completed in {testtime:.2f}s)"
    )

print("\n[Saving] Writing results to CSV...")

# Generate filename with message and checkpoint ID
base_checkpoint_id = os.path.basename(checkpoints[0]).replace(".pth", "")
if args.message:
    base_filename = f"{base_checkpoint_id}_{args.message}.csv"
else:
    base_filename = f"{base_checkpoint_id}.csv"

# Ensure we don't overwrite existing files
output_dir = "experiments/checkpoint_test_results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, base_filename)

counter = 1
original_path = output_path
while os.path.exists(output_path):
    name, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
    counter += 1

with open(output_path, "w", newline="") as f:
    fieldnames = [
        "datetime",
        "checkpoint_accuracy",
        "test_accuracy",
        "epoch",
        "checkpoint_path",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"[Done] Saved {len(results)} checkpoint results to {output_path}")
