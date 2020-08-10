#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


raw_data_path = '../Data/train.csv'
destination_folder = '../Data/Processed'

train_test_ratio = 1
train_valid_ratio = 0.70


# In[11]:


# Read raw data
df_raw = pd.read_csv(raw_data_path)

# Prepare columns
df_raw = df_raw.reindex(columns=['target', 'text','id'])

# Split according to label
df_fake = df_raw[df_raw['target'] == 0]
df_real = df_raw[df_raw['target'] == 1]

# Train-test split
#df_real_full_train, df_real_test = train_test_split(df_real, train_size = train_test_ratio, random_state = 1)
#df_fake_full_train, df_fake_test = train_test_split(df_fake, train_size = train_test_ratio, random_state = 1)

# Train-valid split
df_real_train, df_real_valid = train_test_split(df_real, train_size = train_valid_ratio, random_state = 1)
df_fake_train, df_fake_valid = train_test_split(df_fake, train_size = train_valid_ratio, random_state = 1)

# Concatenate splits of different labels
df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
#df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)

# Write preprocessed data
df_train.to_csv(destination_folder + '/train.csv', index=False)
df_valid.to_csv(destination_folder + '/valid.csv', index=False)
#df_test.to_csv(destination_folder + '/test.csv', index=False)


# In[12]:


test_raw = pd.read_csv('../Data/test.csv')
test_raw = test_raw.reindex(columns=['text','id'])
test_raw.to_csv(destination_folder + '/test.csv', index=False)


# In[ ]:




