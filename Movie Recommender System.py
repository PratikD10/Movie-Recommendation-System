#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[6]:


credits.head(1)['crew'].values


# In[10]:


movies = movies.merge(credits,on="title")


# In[8]:


movies.shape


# In[9]:


credits.shape


# In[11]:


movies.head(1)


# In[17]:


#genres
#id
#keyword
#title
#overview
#cast
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# movies.head()

# In[18]:


movies.head()


# In[19]:


movies.count()


# In[22]:


movies.isnull().sum()


# In[21]:


movies.dropna(inplace=True)


# In[23]:


movies.duplicated().sum()


# In[24]:


movies.iloc[0].genres


# In[27]:


def convert(obj):
    L=[] 
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# 

# In[36]:


movies['genres'] = movies['genres'].apply(convert)


# In[34]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[37]:


movies.head()


# In[40]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[41]:


def convert3(obj):
    L=[] 
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[45]:


movies['cast'] = movies['cast'].apply(convert3)


# In[46]:


def fetch_director(obj):
    L=[] 
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[48]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[49]:


movies.head()


# In[51]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[52]:


movies.head()


# In[55]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[56]:


movies.head()


# In[77]:


movies['tags']= movies['overview']+ movies['genres'] + movies['keywords'] + movies['cast'] + movies["crew"] 


# In[78]:


movies.head()


# In[79]:


new_df= movies[['movie_id','title','tags']]


# In[80]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[81]:


new_df.head()


# In[82]:


new_df['tags'][0]


# In[85]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[86]:


new_df.head()


# In[94]:


import nltk


# In[95]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[98]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[100]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[101]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[102]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[103]:


vectors


# In[104]:


cv.get_feature_names()


# In[106]:


from sklearn.metrics.pairwise import cosine_similarity


# In[108]:


similarity = cosine_similarity(vectors)


# In[110]:


similarity[0]


# In[125]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print( new_df.iloc[i[0]].title)


# In[126]:


recommend('Avatar')


# In[127]:


import pickle


# In[128]:


pickle.dump(new_df,open('movies.pkl', 'wb'))


# In[130]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[131]:


pickle.dump(similarity, open('similarity.pkl','wb'))


# In[ ]:




