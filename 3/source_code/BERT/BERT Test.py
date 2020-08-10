#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries
from sklearn.model_selection import train_test_split
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


# In[2]:


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, target):
        loss, text_fea = self.encoder(text, labels=target)[:2]

        return loss, text_fea


# In[18]:


from datetime import datetime
def evaluate_to_df(model, test_loader):
    y_pred = []
    id_list = []
    
    model.eval()
    #count = 0
    start = datetime.now()
    with torch.no_grad():
        for (text,idn), _ in test_loader:
                target = torch.zeros([16],dtype=torch.int64)      
                target = target.to(device)
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                id_list.extend(idn.cpu().numpy())
                output = model(text, target)
                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                #count += 1
    end = datetime.now()
    t = end-start
    print("Took: ")
    print(t)
    print("FINISHED, predicted %d tweets" % (len(y_pred)))
    submission_df = pd.DataFrame()
    submission_df['prediction'] = y_pred
    submission_df['id'] = id_list
    return submission_df


# In[4]:


raw_data_path = '../Data/train.csv'
destination_folder = '../Data/Processed'
test_raw = pd.read_csv('../Data/test.csv')
for i in range(16-len(test_raw)%16):
    newRow = pd.DataFrame({"id": [-1], "text" : ["NA"]})
    test_raw = test_raw.append(newRow)
test_raw = test_raw.reindex(columns=['text','id'])
test_raw.to_csv(destination_folder + '/test.csv', index=False)


# In[5]:


x = torch.tensor([1,2])
y = torch.tensor([3])
z = torch.cat((x,y))
z.tolist()


# In[7]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[14]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields
id_field = Field(sequential=False, use_vocab=False, batch_first=True,dtype=torch.int)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)

test_fields =[('text', text_field),('id', id_field)]
test_data = TabularDataset(path='../Data/Processed/test.csv', fields=test_fields, format='CSV', skip_header=True)
test_iter = Iterator(test_data, batch_size=16, device=device, train=False, shuffle=False, sort=False)


# In[15]:


def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


# In[16]:


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cuda:0')
best_model = BERT().to(device)

load_checkpoint(destination_folder + '/model.pt', best_model)


# In[19]:


submission_df = evaluate_to_df(best_model, test_iter)


# In[98]:


submission_df = submission_df[submission_df['id'] != -1]
submission_df = submission_df[['id','prediction']]
submission_df.head()
submission_df.columns = ['id', 'target']


# In[99]:


submission_df.head()


# In[100]:


submission_df.to_csv('../Data/Processed/PyTorchSubmission.csv',index=False)


# In[ ]:




