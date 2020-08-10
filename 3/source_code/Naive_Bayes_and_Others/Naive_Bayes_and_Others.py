#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Data/train.csv')
df.head()


# In[3]:


df.loc[0]['text']


# In[4]:


from io import StringIO
col = ['target', 'text']
df = df[col]
df.tail()


# In[5]:


df.columns = ['target', 'text']
target_df = df[['text', 'target']].drop_duplicates().sort_values('target')
target_dict = dict(target_df.values)
df.head()


# In[6]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('target').text.count().plot.bar(ylim=0)
plt.show()


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.text).toarray()
labels = df.target
features.shape


# In[8]:


from sklearn.feature_selection import chi2
N = 2
for target in [0,1]:
  features_chi2 = chi2(features, labels == target)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(target))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
import timeit

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    DecisionTreeClassifier()
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  if model.__class__.__name__ == 'RandomForestClassifier':
      model_name = 'RF'
  elif model.__class__.__name__ == 'LinearSVC':
      model_name = 'SVC'
  elif model.__class__.__name__ == 'MultinomialNB':
      model_name = 'MNB'
  elif model.__class__.__name__ == 'LogisticRegression':
      model_name = 'LR'
  elif model.__class__.__name__ == 'DecisionTreeClassifier':
      model_name = 'DT'
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['Model', 'fold_idx', 'Accuracy'])
import seaborn as sns
sns.boxplot(x='Model', y='Accuracy', data=cv_df)
sns.stripplot(x='Model', y='Accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.title('Model Accuracy')
plt.show()
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


# In[10]:


cv_df.groupby('Model').Accuracy.mean()


# In[12]:


test_df = pd.read_csv('Data/test.csv')


# In[13]:


trainCounts = count_vect.fit_transform(df['text'])
X_train_tfidf = tfidf_transformer.fit_transform(trainCounts)


# In[14]:


import timeit

iterated_run_times = []
run_times = []
for model in models:
    model_name = model.__class__.__name__
    clf = model.fit(X_train_tfidf, labels)
    solution_df = test_df[['id']]
    if model_name != 'RandomForestClassifier': # takes too long!
        for i in range(0, 50):
                start = timeit.default_timer()
                for j in range(0, len(test_df)):
                    text = [test_df['text'][j]]
                    clf.predict(count_vect.transform(text))
                iterated_run_times.append([model_name, timeit.default_timer()-start])
    start = timeit.default_timer()
    solution_df['target'] = test_df.apply(lambda row : clf.predict(count_vect.transform([row['text']])), axis = 1)
    run_times.append((model_name,timeit.default_timer()-start))
    solution_df['target'] = solution_df.apply(lambda row : row['target'][0], axis=1)
    solution_df.to_csv('Data/preds_'+model_name +'.csv', index= False)


# In[19]:


run_times


# In[16]:


rt_df = pd.DataFrame(columns=['Model','Runtime'])
for t in run_times:
    if t[0] == 'LinearSVC':
        rt_df.loc[-1] = ['SVC', t[1]]
    elif t[0] == 'MultinomialNB':
        rt_df.loc[-1] = ['MNB', t[1]]
    elif t[0] == 'LogisticRegression':
        rt_df.loc[-1] = ['LR', t[1]]
    elif t[0] == 'DecisionTreeClassifier':
        rt_df.loc[-1] = ['DT', t[1]]
    else:
        rt_df.loc[-1] = ['RF', t[1]]
    rt_df.index += 1
rt_df


# In[17]:


sns.barplot(x='Model',y='Runtime',data=rt_df)
plt.title('Model Runtimes')


# In[18]:


iterated_run_times


# In[19]:


irt_df = pd.DataFrame(columns=['Model','Runtime'])
for l in iterated_run_times:
    if l[0] == 'LinearSVC':
        irt_df.loc[-1] = ['SVC', l[1]]
    elif l[0] == 'MultinomialNB':
        irt_df.loc[-1] = ['MNB', l[1]]
    elif l[0] == 'LogisticRegression':
        irt_df.loc[-1] = ['LR', l[1]]
    else:
        irt_df.loc[-1] = ['DT', l[1]]
    irt_df.index += 1


# In[20]:


irt_df


# In[21]:


sns.catplot(x='Model', y='Runtime', data=irt_df)
plt.title('Model Runtimes')
plt.show()


# In[ ]:




