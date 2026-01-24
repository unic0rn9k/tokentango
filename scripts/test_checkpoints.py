import torch
import tokentango
from tokentango.data import BertData
import os
import time
from datetime import datetime
import csv

checkpoints = [
    "data/checkpoints/checkpoint_2026-01-23_19-00-38_81.51.pth",
    "data/checkpoints/checkpoint_2026-01-23_19-03-24_81.70.pth",
    "data/checkpoints/checkpoint_2026-01-23_19-06-09_82.26.pth",
    "data/checkpoints/checkpoint_2026-01-23_19-08-55_82.04.pth",
    "data/checkpoints/checkpoint_2026-01-23_19-11-40_81.16.pth",
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
        f"[CHECKPOINT] Loaded checkpoint from epoch {checkpoint['epoch']}, saved accuracy: {checkpoint['accuracy']:.2f}%"
    )

    test_start = time.time()
    test_acc = tokentango.train.test_accuracy(model, test_data, device, frac=1)
    testtime = time.time() - test_start

    results.append(
        {
            "datetime": timestamp,
            "checkpoint_accuracy": checkpoint_acc,
            "test_accuracy": test_acc,
            "epoch": checkpoint["epoch"],
            "checkpoint_path": checkpoint_path,
        }
    )

    print(f"[CHECKPOINT] Test accuracy: {test_acc:.2f}% (completed in {testtime:.2f}s)")

print("\n[Saving] Writing results to CSV...")
with open("checkpoint_test_results.csv", "w", newline="") as f:
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

print(f"[Done] Saved {len(results)} checkpoint results to checkpoint_test_results.csv")
