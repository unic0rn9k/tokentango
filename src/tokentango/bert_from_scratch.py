#!/usr/bin/env python
#from transformers import DistilBertTokenizer
#
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
#cls_token_id = tokenizer.cls_token_id  # or tokenizer.convert_tokens_to_ids("[CLS]")
#mask_token_id = tokenizer.mask_token_id  # or tokenizer.convert_tokens_to_ids("[MASK]")
#pad_token_id = tokenizer.pad_token_id  # or tokenizer.convert_tokens_to_ids("[MASK]")
#vocab_size=tokenizer.vocab_size
#tokenizer.encode("you are bruh", add_special_tokens=True)


import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Lowercase
#from transformers import get_linear_schedule_with_warmup

#from transformers import DistilBertModel, DistilBertConfig
from torch.optim import Adam, AdamW
from torch import nn, functional as F
import math

import os
from matplotlib import pyplot as plt
import itertools

#train_set_large = train_set.sample(frac=1).reset_index(drop=True)
large_set = pd.read_csv("995,000_rows.csv").sample(frac=0.2).reset_index(drop=True)
large_set = large_set[["content", "type", "title"]].dropna()


# In[5]:

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = NFD()
tokenizer.pre_tokenizer = Whitespace()

vocab_size=40000
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

tokenizer.train_from_iterator(large_set["content"], trainer=trainer)

tokenizer.save("bpe_tokenizer.json")

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
label_counts = large_set['new_labels'].value_counts()
print(label_counts)
min_count = label_counts.min()
large_set = large_set.groupby('new_labels').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

#large_set["content_tokens"] = fn.tokenize(large_set["content"])

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



labels = [1.0 if n == "fake" else 0 for n in large_set["new_labels"]]

train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, labels, test_size=0.2)
train_m, test_val_m = train_test_split(att_masks, test_size=0.2)
train_mlm, test_val_mlm = train_test_split(text_ids_masked, test_size=0.2)

test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, test_size=0.5)
test_m, val_m = train_test_split(test_val_m, test_size=0.5)


# In[10]:



#train_x = [torch.tensor(x) for x in train_x]
#test_x = [torch.tensor(x) for x in test_x]
train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)
train_mlm = torch.tensor(train_mlm)
val_x = torch.tensor(val_x)

train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)
val_y = torch.tensor(val_y)

train_m = torch.tensor(train_m)
test_m = torch.tensor(test_m)
val_m = torch.tensor(val_m)


# In[11]:


#train_x = torch.tensor([n for n in train_set_large["tokens"]])


# In[12]:


#train_y = torch.tensor([int(n == "fake") for n in train_set_large["new_labels"]])


# In[13]:



batch_size = 32

train_data = TensorDataset(train_x, train_m, train_y, train_mlm)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_x, val_m, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# In[14]:

num_labels = 2
device = torch.device("mps")

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        #self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=False, output_hidden_states=False)
        #self.out = nn.Linear(self.bert.config.dim, 1)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.use_nested_tensor=True
        self.encoder_layer.self_attn.batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm = nn.LayerNorm(512))

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512, padding_idx=0)
        self.positional_encoding = torch.zeros([max_seq_len, 512])#, requires_grad=True)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, 512, 2).float() * -(math.log(10000.0) / 512))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

        self.preclassifier = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, 1)
        self.distribution = nn.Linear(512, vocab_size, bias=False)
        self.distribution.weight = self.embeddings.weight

    def hidden(self, seq, **kwargs):
        x = self.embeddings(seq)
        expanded_positional_encoding = self.positional_encoding.unsqueeze(0)  # Shape: (1, sequence_length, embedding_size)
        expanded_positional_encoding = expanded_positional_encoding.expand(x.shape[0], -1, -1).to(device)  # Shape: (batch_size, sequence_length, embedding_size)
        print(x.shape)
        x += expanded_positional_encoding[:x.size(0), :x.size(1), :x.size(2)]
        #print(x)
        return self.transformer_encoder(x, **kwargs)

    def classify(self, seq, **kwargs):
        outputs = self.classifier(self.preclassifier(hidden[:,0]).relu())
        return output
    
    def classify_loss(self, hidden, mb_y):
        outputs = self.classifier(self.preclassifier(hidden[:,0]).relu())
        loss_cls = torch.nn.functional.mse_loss(outputs.view(-1), mb_y)
        return loss_cls

    def mlm_loss(self, hidden, mb_x):
        outputs = self.distribution(hidden)[:,1:,:]
        loss_mlm = torch.nn.functional.cross_entropy(outputs.reshape(-1, vocab_size), mb_x[:,1:].reshape(-1))
        return loss_mlm
    
model = BertClassifier()

model = model.to(device)
model.train()
model.compile()


# In[15]:


learning_rate = 1e-4
adam_epsilon = 1e-8

optimizer = AdamW(model.named_parameters(), lr=learning_rate, eps=adam_epsilon)


# In[16]:



num_epochs = 10
total_steps = len(train_dataloader) * num_epochs

#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# In[17]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[18]:


test_data = TensorDataset(test_x, test_m)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)


# In[19]:


train_losses = []
val_losses = []
num_mb_train = len(train_dataloader)
num_mb_val = len(val_dataloader)
missed_batches = 0

if num_mb_val == 0:
    num_mb_val = 1

for n in range(num_epochs):
    train_loss = 0
    val_loss = 0
    start_time = time.time()
    
    for k, (mb_x, mb_m, mb_y, mb_mlm) in enumerate(train_dataloader):
        optimizer.zero_grad()
        model.train()

        if mb_x.shape[0] != 32:
            missed_batches += 1
            continue
        
        mb_x = mb_x.to(device)
        mb_m = mb_m.to(device)
        mb_y = mb_y.to(device)
        mb_mlm = mb_mlm.to(device)

        hidden = model.hidden(mb_mlm)#, src_key_padding_mask=mb_m.transpose(0,1))
        loss_cls = model.classify_loss(hidden, mb_y)
        #loss_mlm = model.mlm_loss(hidden, mb_x)

        loss = loss_cls# + loss_mlm
        loss.backward()
        
        optimizer.step()
        #scheduler.step()
        
        train_loss = loss.data

        # Track time for every batch
        elapsed_time = time.time() - start_time
        batches_done = k + 1
        batches_total = len(train_dataloader)
        
        # Estimate time remaining (ETA)
        remaining_batches = batches_total - batches_done
        eta_seconds = (elapsed_time / batches_done) * remaining_batches
        eta_str = str(time.strftime("%H:%M:%S", time.gmtime(eta_seconds)))

        train_losses.append(train_loss.cpu().detach().item())

        # Print current status and ETA
        print(f"Loss: {float(np.mean(train_losses)):.2f} - ETA of epoch: {eta_str}", end='\r')

        if k % 100 == 99:
            print (f"\n missed {missed_batches} batches")
            
            with torch.no_grad():
                model.eval()
         
                nbatches = int(len(test_x) / 32)
                
                outputs = []
                #for k in range(nbatches):
                for k, (mb_x, mb_m) in enumerate(test_dataloader):
                    #mb_x = nested_tensor(test_x[k*32:(k+1)*32])#, device=device)
                    mb_x = mb_x.to(device)
                    mb_m = mb_m.to(device)
                    hidden = model.hidden(mb_x)
                    output = model.classifier(model.preclassifier(hidden[:,0]).relu())
                    outputs.append(output)
            
                outputs = torch.cat(outputs)
        
                predicted_values = torch.round(outputs)
                predicted_values = predicted_values.cpu().view(-1).numpy()
                true_values = test_y.numpy()
        
                test_accuracy = np.sum(predicted_values == true_values) / len(true_values)
                print ("Test Accuracy:", test_accuracy)
                val_losses.append(test_accuracy)
                
                #label_values = ["reliabel", "fake"]
                #print(classification_report(true_values, predicted_values, target_names=[str(l) for l in label_values]))
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')

#We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.import pickle

out_dir = './bert_from_scratch'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

with open(out_dir + '/train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
    
with open(out_dir + '/val_losses.pkl', 'wb') as f:
    pickle.dump(val_losses, f)
# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.plot(train_losses)


# In[21]:


plt.figure()
plt.plot(val_losses)


# In[22]:


batch_size = 32

test_data = TensorDataset(test_x, test_m)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

nbatches = int(len(test_x) / 32)

outputs = []
with torch.no_grad():
    model.eval()
    #for k in range(nbatches):
    for k, (mb_x, mb_m) in enumerate(test_dataloader):
        #mb_x = nested_tensor(test_x[k*32:(k+1)*32])#, device=device)
        mb_x = mb_x.to(device)
        mb_m = mb_m.to(device)
        hidden = model.hidden(mb_x)
        output = model.classifier(model.preclassifier(hidden[:,0]).relu())
        outputs.append(output)

outputs = torch.cat(outputs)


# In[23]:


hidden_states = []
with torch.no_grad():
    model.eval()
    for k, (mb_x, mb_m) in enumerate(test_dataloader):
        #mb_x = torch.tensor(mb_x).unsqueeze(0).to(device)
        mb_x = torch.tensor(mb_x).to(device)
        mb_m = mb_m.to(device)
        hidden = model.hidden(mb_x)
        output = hidden[:,0]
        #print(mb_x.shape)
        #if any(all(i == False for i in mb_m[n,:]) for n in range(mb_m.size(0))):
        #    print("FUCK")
        hidden_states.append(output)

hidden_states = torch.cat(hidden_states)
hidden_states


# In[24]:


hidden_states.shape


# In[25]:


model.embeddings(torch.tensor(cls_token_id).to(device)).view(1,-1)


# In[26]:


predicted_values = torch.round(outputs)
predicted_values = predicted_values.cpu().view(-1).numpy()
true_values = test_y.numpy()


# In[27]:


test_accuracy = np.sum(predicted_values == true_values) / len(true_values)
print ("Test Accuracy:", test_accuracy)


# In[28]:


label_values = ["reliabel", "fake"]


# In[29]:


predicted_values


# In[30]:


print(classification_report(true_values, predicted_values, target_names=[str(l) for l in label_values]))


# In[31]:

# plot confusion matrix
# code borrowed from scikit-learn.org
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[32]:


cm_test = confusion_matrix(true_values, predicted_values)

np.set_printoptions(precision=2)

#plt.figure(figsize=(6,6))
#plot_confusion_matrix(cm_test, classes=label_values, title='Confusion Matrix - Test Dataset')
plt.figure(figsize=(6,6))
plot_confusion_matrix(cm_test, classes=label_values, title='Confusion Matrix - Test Dataset', normalize=True)


# In[ ]:




