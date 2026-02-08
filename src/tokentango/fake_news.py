from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Lowercase
import random
import torch
import os

import pandas as pd
from tokentango.data import BertData


def load_data(frac, random_state=69):
    # train_set_large = train_set.sample(frac=1).reset_index(drop=True)
    large_set = (
        pd.read_csv("data/995,000_rows.csv", low_memory=False)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )
    large_set = large_set[["content", "type", "title"]].dropna()

    # In[5]:
    tokenizer_path = "data/bpe_tokenizer.json"

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print(f"Tokenizer not found at {tokenizer_path}. Training new tokenizer...")
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = NFD()
        tokenizer.pre_tokenizer = Whitespace()

        vocab_size = 40000
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        tokenizer.train_from_iterator(large_set["content"], trainer=trainer)

        tokenizer.save(tokenizer_path)
        print(f"Saved new tokenizer to {tokenizer_path}")

    cls_token_id = tokenizer.token_to_id("[CLS]")
    mask_token_id = tokenizer.token_to_id("[MASK]")
    pad_token_id = tokenizer.token_to_id("[PAD]")

    # In[6]:

    label_map = {
        "fake": "fake",
        "satire": None,
        "bias": None,
        "conspiracy": "fake",
        "state": None,
        "junksci": "fake",
        "hate": None,
        "clickbait": None,
        "unreliable": None,
        "political": "reliable",
        "reliable": "reliable",
        "unknown": None,
    }

    # In[7]:

    large_set["new_labels"] = [label_map.get(n, None) for n in large_set["type"]]
    label_counts = large_set["new_labels"].value_counts()
    print(label_counts)
    min_count = label_counts.min()
    large_set = (
        large_set.groupby("new_labels", group_keys=False)
        .sample(min_count, random_state=random_state)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    # large_set["content_tokens"] = fn.tokenize(large_set["content"])

    nfake = sum(large_set["new_labels"] == "fake")
    nreliable = sum(large_set["new_labels"] == "reliable")

    print(nfake / (nfake + nreliable))

    # In[8]:

    def mask_random_elements(sequence, mask_probability=0.15):
        masked_sequence = sequence[:]
        for idx in range(len(sequence)):
            if random.random() < mask_probability and sequence[idx] != pad_token_id:
                masked_sequence[idx] = mask_token_id
        return masked_sequence

    max_seq_len = 300

    def tokenize(text):
        tokens = tokenizer.encode(text)
        tokens = [cls_token_id] + tokens.ids
        tokens = tokens[:max_seq_len]
        while len(tokens) < max_seq_len:
            tokens.append(pad_token_id)
        return tokens

    text_ids = [tokenize(text) for text in large_set["content"]]
    text_ids_masked = [mask_random_elements(text_id) for text_id in text_ids]
    att_masks = [[id != pad_token_id for id in ids] for ids in text_ids]

    # In[9]:

    labels = [1.0 if n == "fake" else -1.0 for n in large_set["new_labels"]]

    split_at = int(0.8 * len(labels))

    train_source = torch.tensor(text_ids[:split_at], dtype=torch.long)
    train_masked = torch.tensor(text_ids_masked[:split_at], dtype=torch.long)
    train_labels = torch.tensor(labels[:split_at], dtype=torch.float32)

    test_source = torch.tensor(text_ids[split_at:], dtype=torch.long)
    test_masked = torch.tensor(text_ids_masked[split_at:], dtype=torch.long)
    test_labels = torch.tensor(labels[split_at:], dtype=torch.float32)

    train_data = BertData(train_source, train_masked, train_labels)
    test_data = BertData(test_source, test_masked, test_labels)

    return train_data, test_data
