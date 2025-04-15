#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv('songdata.csv')
df.head(3)


# In[ ]:


df.shape


# In[ ]:


df.sample(n=5000)


# In[5]:


df.sample(n=5000).drop('link',axis=1)


# In[6]:


df.sample(n=5000).drop('link',axis=1).reset_index(drop=True)


# In[7]:


df = df.sample(n=5000).drop('link',axis=1).reset_index(drop=True)


# In[8]:


# cleaning


# In[9]:


df['song']


# In[10]:


df['song'][0]


# In[11]:


df['text']


# In[12]:


df['text'][0]


# In[13]:


df['text'].str.lower()


# In[14]:


df['text'].str.lower().replace(r'[^a-zA-Z0-9]','')


# In[15]:


df['text'].str.lower().replace(r'[^\w\s]','').replace(r'\n',' ',regex=True)


# In[16]:


df['text'] = df['text'].str.lower().replace(r'[^\w\s]','').replace(r'\n',' ',regex=True)


# In[17]:


df['text'][0]


# In[18]:


import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [ps.stem(w) for w in tokens]
    
    return " ".join(stemming)


# In[19]:


df['text']


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[9]:


tfid = TfidfVectorizer(stop_words='english')
matrix = tfid.fit_transform(df['text'])


# In[11]:


matrix.shape


# In[ ]:


similarity = cosine_similarity(matrix)


# In[24]:


similarity


# In[ ]:


df['song'][0]


# In[ ]:


df[df['song'] == 'Heartbreak Express']


# In[ ]:


sorted(list(enumerate(similarity[0])),reverse=False,key=lambda x:x[1])


# In[ ]:


distances = sorted(list(enumerate(similarity[0])),reverse=False,key=lambda x:x[1])


# In[ ]:


def recommendation(song):
   idx= df[df['song'] == song].index[0]
   distances = sorted(list(enumerate(similarity[idx])),reverse=False,key=lambda x:x[1])
   
   songs = []
   for i in distances[1:21]:
       songs.append(df.iloc[i[0]].song)
       
   return songs


# In[ ]:


recommendation('Heartbreak Express')


# In[ ]:


recommendation('Ocean Man')


# In[ ]:


recommendation('Fight Or Flight')


# In[ ]:




