# -*- coding: utf-8 -*-
"""siamese BERT network

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k5Q0wepV0NxyrWx4I33Mz_u_7IWSTVcw
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import scipy

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

data = pd.read_csv('../content/original_data_with_identifier.csv')

data_select = data['title']
#tfidf = TfidfVectorizer()
#training_sentences = tfidf.fit_transform(data_select)

sentence_embeddings = model.encode(data_select)
print('Sample BERT embedding vector - length', len(data_select[0]))

print('Sample BERT embedding vector - note includes negative values', data_select[0])

query = 'write your query here'
queries = [query]
query_embeddings = model.encode(queries)
number_top_matches = 5
print("Semantic Search Results")

for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:number_top_matches]:
        print(data_select[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))
