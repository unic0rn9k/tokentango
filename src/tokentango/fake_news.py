from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Lowercase
import random
import torch

import pandas as pd


def load_data(frac):
    # train_set_large = train_set.sample(frac=1).reset_index(drop=True)
    large_set = (
        pd.read_csv("data/995,000_rows.csv").sample(frac=frac).reset_index(drop=True)
    )
    large_set = large_set[["content", "type", "title"]].dropna()

    # In[5]:

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = NFD()
    tokenizer.pre_tokenizer = Whitespace()

    vocab_size = 40000
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train_from_iterator(large_set["content"], trainer=trainer)

    tokenizer.save("data/bpe_tokenizer.json")

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
        large_set.groupby("new_labels")
        .apply(lambda x: x.sample(min_count, random_state=42))
        .sample(frac=1)
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
    train_x = text_ids[:split_at]
    test_x = text_ids[split_at:]
    train_y = labels[:split_at]
    test_y = labels[split_at:]
    train_m = att_masks[:split_at]
    test_m = att_masks[split_at:]
    train_mlm = text_ids_masked[:split_at]
    test_mlm = text_ids_masked[split_at:]

    train_x = torch.tensor(train_x)
    test_x = torch.tensor(test_x)
    train_mlm = torch.tensor(train_mlm)

    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    train_m = torch.tensor(train_m)
    test_m = torch.tensor(test_m)

    # batch_size = 32

    # train_data = TensorDataset(train_x, train_m, train_y, train_mlm)
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # val_data = TensorDataset(val_x, val_m, val_y)
    # val_sampler = SequentialSampler(val_data)
    # val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_mlm, train_x, train_y, test_mlm, test_x, test_y
