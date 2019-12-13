#!/usr/bin/env python
# coding: utf-8

# In[96]:


from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


# In[97]:


def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()
    pass


# In[98]:


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())
    pass


# In[99]:


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    t_List = []
    for i in movies['genres']:
        t_List.append(tokenize_string(i))
    movies['tokens'] = pd.Series(np.array(t_List), movies.index)
    return movies
    pass


# In[100]:


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    d = {}
    maximum_docs = {}
    token_df = {}
    vocab = {}
    for i in range(0,len(movies['tokens'])):
        c = Counter()
        c.update(movies['tokens'][i])
        d[i] = dict(c)
        maximum_docs[i] = c.most_common()[0][1]
        for t in set(movies['tokens'][i]):
            if(t not in token_df):
                token_df[t] = 1
            else:
                token_df[t] += 1
    k = 0
    for i in sorted(token_df.keys()):
        vocab[i] = k
        k += 1
    data_matrix = []
    for k,v in d.items():
        data = []
        columns = []
        rows = [0] * len(d[k])
        for t in v:
            if t in vocab:
                tf_idf =  d[k][t] / maximum_docs[k] * math.log10(len(movies)/token_df[t])
                data.append(tf_idf)
                columns.append(vocab[t])
        matrix = csr_matrix((data,(rows,columns)),shape=(1,len(vocab)))
        data_matrix.append(matrix)
    movies['features'] = pd.Series(data_matrix, movies.index)
    return movies,vocab
    pass


# In[101]:


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]
    pass


# In[102]:


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    return np.dot(a, b.T).toarray()[0][0] / (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))
    pass


# In[103]:


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    res = []
    for i, row in ratings_test.iterrows():
        i_Ftr = movies.loc[movies['movieId'] == row['movieId']].squeeze()['features']
        train_Movie = ratings_train.loc[ratings_train['userId'] == row['userId']]
        cosList = []
        cos_Sum = 0
        for a, row1 in train_Movie.iterrows():
            t_Ftr = movies.loc[movies['movieId'] == row1['movieId']].squeeze()['features']
            cosSim = cosine_sim(i_Ftr, t_Ftr)
            if cosSim > 0:
                cosList.append(cosSim * row1['rating']);
                cos_Sum += cosSim
        if cos_Sum > 0:
            res.append(sum(cosList) / cos_Sum)
        else:
            res.append(train_Movie['rating'].mean()) 
    return np.array(res)
    pass


# In[104]:


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()
    pass


# In[105]:


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])
    


if __name__ == '__main__':
    main()

