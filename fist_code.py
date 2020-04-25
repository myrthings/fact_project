#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:17:21 2020

@author: Myriam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bert_serving
import json
import requests
from sklearn.manifold import TSNE
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.stem.snowball import SnowballStemmer
from wordcloud import STOPWORDS


##### GET THE DATA
fakes=pd.read_csv("fakenews/data/data_poynter_COMPLETE_2020-04-24.csv")
fakes=fakes.reset_index().rename(columns={'index':'identifier'})


##### BERT TRAINING
def get_embeddings(texts):
    headers = {
        'content-type':'application/json'
    }
    data = {
        "id":123,
        "texts":texts,
        "is_tokenized": False
    }
    data = json.dumps(data)
    r = requests.post("http://localhost:" + port_num + "/encode", data=data, headers=headers).json()
    return r['result']

port_num='3333'

titles=get_embeddings(fakes['title'].to_list()) ##4min to transform

identif_codes=dict(zip(fakes['identifier'].to_list(),titles))

''' SAVE TO A FILE

json = json.dumps(identif_codes)
f = open("titles_embeddings.json","w")
f.write(json)
f.close()

'''

###### DO THE VIZ
tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(titles)
x_coords = Y[:, 0]
y_coords = Y[:, 1]

### SHOW SCATTER WITH ALL
plt.figure(figsize=(15,12))
plt.scatter(x_coords, y_coords)
plt.savefig('tsne_bert.png')
plt.show()


### SHOW SCATTER WITH COUNTRIES

fakes['tsne_x']=x_coords
fakes['tsne_y']=y_coords

fakes2=deepcopy(fakes)

new_fakes=pd.DataFrame([])
for it,row in fakes2.iterrows():
    countries=pd.DataFrame({'country':str(row['location']).split(',')})
    countries['country']=countries['country'].str.strip()
    countries['tsne_x']=row['tsne_x']
    countries['tsne_y']=row['tsne_y']
    countries['title']=row['title']
    new_fakes=pd.concat([new_fakes,countries],axis=0)
    
new_fakes=new_fakes.reset_index(drop=True)
    

plt.figure(figsize=(15,12))
for country in new_fakes['country'].unique():
    x=fakes.loc[new_fakes['country']==country,'tsne_x']
    y=fakes.loc[new_fakes['country']==country,'tsne_y']
    plt.scatter(x, y,label=country)
plt.legend()
plt.savefig('tsne_countries_bert.png')
plt.show()


### ONLY TOP 10
plt.figure(figsize=(15,12))
for country in new_fakes['country'].value_counts()[:10].index:
    x=fakes.loc[new_fakes['country']==country,'tsne_x']
    y=fakes.loc[new_fakes['country']==country,'tsne_y']
    plt.scatter(x, y,label=country)
plt.legend()
plt.savefig('top10_tsne_countries_bert.png')
plt.show()




