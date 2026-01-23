import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch import nn, functional as F
import math

EMBEDDING_DIM = 32

class BertClassifier(nn.Module):
    def __init__(self, max_seq_len, vocab_size, device):
        super(BertClassifier, self).__init__()
        #self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=False, output_hidden_states=False)
        #self.out = nn.Linear(self.bert.config.dim, 1)
        
        self.vocab_size = vocab_size
        self.device = device

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=EMBEDDING_DIM, nhead=4)
        self.use_nested_tensor=True
        self.encoder_layer.self_attn.batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm = nn.LayerNorm(EMBEDDING_DIM))

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)#, padding_idx=0)
        self.positional_encoding = torch.zeros([max_seq_len, EMBEDDING_DIM])#, requires_grad=True)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, EMBEDDING_DIM, 2).float() * -(math.log(10000.0) / EMBEDDING_DIM))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

        self.preclassifier = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.classifier = nn.Linear(EMBEDDING_DIM, 1)
        self.distribution = nn.Linear(EMBEDDING_DIM, vocab_size, bias=False)
        self.distribution.weight = self.embeddings.weight

    def embed(self, seq):
        x = self.embeddings(seq)
        expanded_positional_encoding = self.positional_encoding.unsqueeze(0)  # Shape: (1, sequence_length, embedding_size)
        expanded_positional_encoding = expanded_positional_encoding.expand(x.shape[0], -1, -1).to(self.device)  # Shape: (batch_size, sequence_length, embedding_size)
        x += expanded_positional_encoding[:x.size(0), :x.size(1), :x.size(2)]
        return x

    def hidden(self, seq, **kwargs):
        x = self.embed(seq)
        return self.transformer_encoder(x, **kwargs)

    def classify(self, hidden, **kwargs):
        return self.classifier(self.preclassifier(hidden[:, 0]).relu()).tanh()

    def classify_loss(self, hidden, mb_y):
        logits = self.classify(hidden)
        loss_cls = torch.nn.functional.huber_loss(
            logits.view(-1), mb_y
        )
        return loss_cls

    def mlm_loss(self, hidden, mb_x):
        outputs = self.distribution(hidden)[:,1:,:]
        loss_mlm = torch.nn.functional.cross_entropy(outputs.reshape(-1, self.vocab_size), mb_x[:,1:].reshape(-1))
        return loss_mlm
