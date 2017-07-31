# ## Ravish Chawla
# ### Topic Modeling with LDA and NMF algorithms on the ABC News Headlines Dataset
# #### July 31, 2017

# # Data imports
# 
# We import Pandas, numpy and scipy for data structures. We use gensim for LDA, and sklearn for NMF

# In[28]:

import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;


# # Loading the data
# 
# We are using the ABC News headlines dataset. Some lines are badly formatted (very few), so we are skipping those.

# In[ ]:

data = pd.read_csv('data/abcnews-date-text.csv', error_bad_lines=False);


# In[34]:

#We only need the Headlines_text column from the data
data_text = data[['headline_text']];


# We need to remove stopwords first. Casting all values to float will make it easier to iterate over.

# In[ ]:

data_text = data_text.astype('str');


# In[104]:

for idx in range(len(data_text)):
    
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    data_text.iloc[idx]['headline_text'] = [word for word in data_text.iloc[idx]['headline_text'].split(' ') if word not in stopwords.words()];
    
    #print logs to monitor output
    if idx % 1000 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)));


# In[105]:

#save data because it takes very long to remove stop words
pickle.dump(data_text, open('data_text.dat', 'wb'))


# In[71]:

#get the words as an array for lda input
train_headlines = [value[0] for value in data_text.iloc[0:].values];


# In[132]:

#number of topics we will cluster for: 10
num_topics = 10;


# # LDA
# 
# We will use the gensim library for LDA. First, we obtain a id-2-word dictionary. For each headline, we will use the dictionary to obtain a mapping of the word id to their word counts. The LDA model uses both of these mappings.

# In[72]:

id2word = gensim.corpora.Dictionary(train_headlines);


# In[73]:

corpus = [id2word.doc2bow(text) for text in train_headlines];


# In[74]:

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics);


# # generating LDA topics
# 
# We will iterate over the number of topics, get the top words in each cluster and add them to a dataframe.

# In[129]:

def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);


# In[135]:

get_lda_topics(lda, num_topics)


# # NMF
# 
# For NMF, we need to obtain a design matrix. To improve results, I am going to apply TfIdf transformation to the counts.

# In[79]:

#the count vectorizer needs string inputs, not array, so I join them with a space.
train_headlines_sentences = [' '.join(text) for text in train_headlines]


# Now, we obtain a Counts design matrix, for which we use SKLearn’s CountVectorizer module. The transformation will return a matrix of size (Documents x Features), where the value of a cell is going to be the number of times the feature (word) appears in that document.
# 
# To reduce the size of the matrix, to speed up computation, we will set the maximum feature size to 5000, which will take the top 5000 best features that can contribute to our model.

# In[80]:

vectorizer = CountVectorizer(analyzer='word', max_features=5000);
x_counts = vectorizer.fit_transform(train_headlines_sentences);


# Next, we set a TfIdf Transformer, and transform the counts with the model.

# In[81]:

transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);


# And now we normalize the TfIdf values to unit length for each row.

# In[82]:

xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)


# And finally, obtain a NMF model, and fit it with the sentences.

# In[84]:

#obtain a NMF model.
model = NMF(n_components=num_topics, init='nndsvd');


# In[85]:

#fit the model
model.fit(xtfidf_norm)


# In[136]:

def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);


# In[139]:

get_nmf_topics(model, 20)


# The two tables above, in each section, show the results from LDA and NMF on both datasets. There is some coherence between the words in each clustering. For example, Topic #02 in LDA shows words associated with shootings and violent incidents, as evident with words such as “attack”, “killed”, “shooting”, “crash”, and “police”. Other topics show different patterns. 
# 
# On the other hand, comparing the results of LDA to NMF also shows that NMF performs better. Looking at Topic #01, we can see there are many first names clustered into the same category, along with the word “interview”. This type of headline is very common in news articles, with wording similar to “Interview with John Smith”, or “Interview with James C. on …”. 
# 
# We also see two topics related to violence. First, Topic #03 focuses on police related terms, such as “probe”, “missing”, “investigate”, “arrest”, and “body”. Second, Topic #08 focuses on assault terms, such as “murder”, “stabbing”, “guilty”, and “killed”. This is an interesting split between the topics because although the terms in each are very closely related, one focuses more on police-related activity, and the other more on criminal activity. Along with the first cluster which obtain first-names, the results show that NMF (using TfIdf) performs much better than LDA.
