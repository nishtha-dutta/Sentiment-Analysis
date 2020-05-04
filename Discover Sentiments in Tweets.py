#!/usr/bin/env python
# coding: utf-8

# # Discover sentiment in tweets
# 
# Sentiment analysis is commonly used in marketing and customer service to answer questions such as "Is a product review positive or negative?" and "How are customers responding to a product release?"
# 
# *Topic modeling* discovers the abstract topics in a corpus of texts. The results from topic modeling analysis can then be used in sentiment analysis. For example, they can be used to split texts into different subsets and allow us to train a separate sentiment model for each of the subsets. Training separate sentiment models for different subsets can lead to more accurate predictions than using a single model for all the texts. 
# 
# This notebook illustrates how to discover and visualize topics from a corpus of Twitter tweets using Jupyter notebook. 
# 
# ## Read and prepare the data
# 
# The dataset used in this notebook is based on the [Sentiment140](http://help.sentiment140.com/) dataset, which contains approximately 1,600,000 automatically annotated tweets with six fields per tweet. Here we use only a [160,000 tweet sample](http://azuremlsamples.azureml.net/templatedata/Text%20-%20Input.csv) of that dataset with two fields for each tweet: the polarity of the tweet and the text of the tweet.
# 
# The following command downloads the data into the file *sentiment140-subset.csv* in the current session.

# In[13]:


get_ipython().system('curl -o sentiment140-subset.csv http://azuremlsamples.azureml.net/templatedata/Text%20-%20Input.csv')


# You can use the `ls` command to see it in the project's files:

# In[14]:


get_ipython().system('ls')


# Next, prepare the data.

# In[29]:


import os
import pandas as pd

dirname = os.getcwd()
print(dirname)

mydata = pd.read_csv("sentiment140-subset.csv", header = 0)
print(mydata.shape)

# Use 10000 for testing
mydata = mydata[:10000] 
mydata.head()


# Before doing analysis, clean up the text by removing the Twitter handle, removing numbers and consecutive underscores, and changing to lower case:

# In[30]:


import re

def clean_text(string_in):
    string_in = re.sub(r"@\w+", "", string_in) # Remove twitter handle
    string_in = re.sub(r"\d", "", string_in)   # Remove numbers  
    string_in = re.sub(r"_+", "", string_in)   # Remove consecutive underscores
    string_in = string_in.lower()              # Tranform to lower case    
    
    return string_in.strip()

mydata["tweet_text_cleaned"] = mydata.tweet_text.apply(clean_text)
print(mydata["tweet_text_cleaned"])


# Now tokenize the text data using the following line of code:

# In[31]:


from nltk.tokenize import RegexpTokenizer
preprocessed = [" ".join(RegexpTokenizer(r'\w+').                         tokenize(mydata.tweet_text_cleaned[idx]))                 for idx in mydata.index]
print(preprocessed)


# ## Identify topics
# 
# ### Calculate tf-idf matrix
# 
# In order to use non-negative matrix factorization, we first calculate the *tf-idf* (term frequency-inverse document frequency) matrix. The value of tf-idf reflects the number times a word appears in the *document* after adjusting for the frequency of the word in the *corpus*.
# 
# When calculating the tf-idf matrix, we can filter out words like "being" and "for" which are called *stop words*. The following code filters out the stop words as defined by the [scikit-learn](http://scikit-learn.org/stable/) package. You can expand the list of stop words by adding to the `custom_stop_words` variable. 

# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 

custom_stop_words = ["being", "for"]
my_stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)

vectorizer = TfidfVectorizer(min_df = 1, ngram_range = (1,1), 
                             stop_words = my_stop_words)

tfidf = vectorizer.fit_transform(preprocessed)
print("Created document-term matrix of size %d x %d" % (tfidf.shape[0],tfidf.shape[1]))
print(tfidf)


# ### NMF analysis
# 
# If we use **X** to represent a **document-word** matrix with tf-idf values, non-negative matrix factorization factorizes the matrix into two matrices **W** and **H**, representing **document-topic** and **topic-word** matrices, respectively, as shown in the following figure:
# 
# ![Figure 1](https://cloud.githubusercontent.com/assets/9322661/18445719/ea601214-78ec-11e6-9858-b29d62e29c9e.png)
# 
# The matrix **W** allows us to compare texts and determine which ones are similar. The matrix **H** tells us the top words in any given topic and we can use visualization to better undertand the topics. 
# 
# For any new text represented by **X_new**, we can make predictions in the sense that the **W_new** value can be computed from the **H** matrix.
# 
# Other topic finding models (e.g., Latent Dirichlet Allocation (LDA)) are also available in the [scikit-learn](http://scikit-learn.org/stable/) package but aren't covered here. Compared with LDA, NMF has cheaper computation cost and is a good choice for short text analysis.  
# 
# Now we can use NMF on the derived tf-idf matrix. We specify the number of topics to be three. 

# In[21]:


from sklearn import decomposition
import numpy as np
nmf = decomposition.NMF(init = 'nndsvd', n_components = 3, max_iter = 200)
W = nmf.fit_transform(tfidf)
H = nmf.components_
print("Generated factor W of size %s and factor H of size %s" % ( str(W.shape), str(H.shape)))

feature_names = vectorizer.get_feature_names()
n_top_words = 10

# Print top words in each topic
for topic_idx, topic in enumerate(H):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# ## Visualize topics with a word cloud
# 
# To better communicate the topics found, we can use the [wordcloud](https://github.com/amueller/word_cloud) package. Because this package is not pre-installed, install it first using commands:

# In[22]:


get_ipython().system('pip install wordcloud')


# Next, separate the topics into separate sorted lists:

# In[34]:


# Create pandas dataframe for the topics
mydf = pd.DataFrame({"feature_name": feature_names})

for topic_idx, topic in enumerate(H):
    mydf["topic_" + str(topic_idx)] = topic

mylist = list(mydf.itertuples())

mywords_topic1 = []
mywords_topic2 = []
mywords_topic3 = []

for order_id, key, num1, num2, num3 in mylist:
    mywords_topic1.append((key, num1))
    mywords_topic2.append((key, num2))
    mywords_topic3.append((key, num3))

mywords_topic1 = sorted(mywords_topic1, key=lambda myword: myword[1], reverse=True)
mywords_topic2 = sorted(mywords_topic2, key=lambda myword: myword[1], reverse=True)
mywords_topic3 = sorted(mywords_topic3, key=lambda myword: myword[1], reverse=True)
print(mywords_topic1)


# Finally, generate the word cloud for the three topics identified from NMF. Larger fonts indicate higher weights of the words in a topic and the colors are randomly assigned. Compared with a simple list, from a word cloud we can better understand the relative frequency of all the words:

# In[24]:


from wordcloud import WordCloud 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

def draw_wordcloud(dict, topic_number):
    wc = WordCloud(max_words=1000)    
    wordcloud = WordCloud().generate_from_frequencies(dict)
    
    plt.title('Topic %s' %str(topic_number), size = 16)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")        
    plt.show()

draw_wordcloud(dict(mywords_topic1), topic_number=1)
draw_wordcloud(dict(mywords_topic2), topic_number=2)
draw_wordcloud(dict(mywords_topic3), topic_number=3)


# ## Making predictions
# 
# For any new tweet, we first calculate its tf-idf matrix (**X**) and then calculate its **W** matrix. This will then allow us to assign the new tweet to a segment and predict its sentiment (for example, positive sentiment vs. negative sentiment on a product).
# 
# The following code shows how to calcuate the **W** matrix for new tweets that have been cleaned beforehand. For illustration purpose, these five tweets are the first five from the corpus used in the analysis.  

# In[37]:


# Prediction example
text_new = preprocessed[0:5]
tfidf_new = vectorizer.transform(text_new)
W_new = nmf.transform(tfidf_new)
print(text_new)
print(W_new)


# 
# 
# Copyright (C) Microsoft. All Rights Reserved.
