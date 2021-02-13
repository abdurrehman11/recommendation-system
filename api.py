import os
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from fastapi import FastAPI
from config import config

app = FastAPI()

USER_EMBED_PATH = 'C:/Users/AR/Downloads/MovieRecommendAPI/data/user_embedding.npy'
MOVIE_EMBED_PATH = 'C:/Users/AR/Downloads/MovieRecommendAPI/data/movie_embedding.npy'
USER_LABEL_MAPPING = 'C:/Users/AR/Downloads/MovieRecommendAPI/data/user_label_mapping.p'
MOVIE_LABEL_MAPPING = 'C:/Users/AR/Downloads/MovieRecommendAPI/data/movie_label_mapping.p'
LABEL_MOVIE_MAPPING = 'C:/Users/AR/Downloads/MovieRecommendAPI/data/label_movie_mapping.p'
USER_MOVIE_MAPPING = 'C:/Users/AR/Downloads/MovieRecommendAPI/data/user_movie_mapping.p'
ID_TITLE_MAPPING = 'C:/Users/AR/Downloads/MovieRecommendAPI/data/id_title_mapping.p'


@app.get("/recommend_by_movieid")
def recommend_by_movieid(movie_id: int):
    movie_embeddings = np.load(MOVIE_EMBED_PATH)
    mid_lbl_mapping = pickle.load(open(MOVIE_LABEL_MAPPING, "rb"))
    lbl_mid_mapping = pickle.load(open(LABEL_MOVIE_MAPPING, "rb"))
    id_title_mapping = pickle.load(open(ID_TITLE_MAPPING, "rb"))
    
    movie_label = mid_lbl_mapping.get(movie_id)
    movie_embedding = movie_embeddings[movie_label]
    
    clf = KNeighborsClassifier(n_neighbors=11)
    clf.fit(movie_embeddings, np.arange(len(movie_embeddings)))
    
    distances, indices = clf.kneighbors(movie_embedding.reshape(1, -1), n_neighbors=10)
    distances, indices = zip(*sorted(zip(distances[0], indices[0])))
    distances, indices = list(distances), list(indices)
    
    sorted_movie_ids = [lbl_mid_mapping[m_idx] for m_idx in indices if m_idx != 0]
    recommend_movies = [id_title_mapping[mid] for mid in sorted_movie_ids]
    
    print("Given movie:", id_title_mapping[movie_id])
    print("Recommended movies:", recommend_movies)

    return recommend_movies

@app.get("/recommend_by_userid")
def recommend_by_userid(user_id: int):
    # load user and movie embeddings
    user_embeddings = np.load(USER_EMBED_PATH)
    movie_embeddings = np.load(MOVIE_EMBED_PATH)
    
    # load user, movie and user_movie mappings
    uid_lbl_mapping = pickle.load(open(USER_LABEL_MAPPING, "rb"))
    mid_lbl_mapping = pickle.load(open(MOVIE_LABEL_MAPPING, "rb"))
    lbl_mid_mapping = pickle.load(open(LABEL_MOVIE_MAPPING, "rb"))
    user_movie_mapping = pickle.load(open(USER_MOVIE_MAPPING, "rb"))
    id_title_mapping = pickle.load(open(ID_TITLE_MAPPING, "rb"))
    
    user_label = uid_lbl_mapping.get(user_id)
    user_embedding = user_embeddings[user_label]
    
    user_watched_movies = user_movie_mapping[user_id]
    movies = list(mid_lbl_mapping.keys())
    user_unwatched_movies = list(set(movies) - set(user_watched_movies))
    user_unwatched_movies_labels = [mid_lbl_mapping[mid] for mid in user_unwatched_movies]
    
    clf = KNeighborsClassifier(n_neighbors=11)
    unwatched_movie_embeddings = movie_embeddings[user_unwatched_movies_labels]
    clf.fit(unwatched_movie_embeddings, user_unwatched_movies_labels)
    
    distances, indices = clf.kneighbors(user_embedding.reshape(1, -1), n_neighbors=10)
    distances, indices = zip(*sorted(zip(distances[0], indices[0])))
    distances, indices = list(distances), list(indices)
    
    sorted_movie_ids = [lbl_mid_mapping[m_idx] for m_idx in indices if m_idx != 0]
    
    recommend_movies = [id_title_mapping[mid] for mid in sorted_movie_ids]
    print("Recommended movies:", recommend_movies)

    return recommend_movies

@app.get("/recommend_by_last_reviewed")
def recommend_by_last_viewed(user_id: int):
    # load user and movie embeddings
    movie_embeddings = np.load(MOVIE_EMBED_PATH)
    
    # load user, movie and user_movie mappings
    uid_lbl_mapping = pickle.load(open(USER_LABEL_MAPPING, "rb"))
    mid_lbl_mapping = pickle.load(open(MOVIE_LABEL_MAPPING, "rb"))
    lbl_mid_mapping = pickle.load(open(LABEL_MOVIE_MAPPING, "rb"))
    user_movie_mapping = pickle.load(open(USER_MOVIE_MAPPING, "rb"))
    id_title_mapping = pickle.load(open(ID_TITLE_MAPPING, "rb"))
    
    # last 5 watched movies by user
    user_last_watched_movies = user_movie_mapping[user_id][-5:]
    user_watched_movies = user_movie_mapping[user_id]
    
    movies = list(mid_lbl_mapping.keys())
    user_unwatched_movies = list(set(movies) - set(user_watched_movies))
    user_unwatched_movies_idxs = [mid_lbl_mapping[mid] for mid in user_unwatched_movies]
    
    clf = KNeighborsClassifier(n_neighbors=11)
    unwatched_movie_embeddings = movie_embeddings[user_unwatched_movies_idxs]
    clf.fit(unwatched_movie_embeddings, user_unwatched_movies_idxs)
    
    m_dist, m_idx = [], []
    for movie_id in user_last_watched_movies:
        top_2 = 0
        movie_label = mid_lbl_mapping.get(movie_id)
        movie_embedding = movie_embeddings[movie_label]
        distances, indices = clf.kneighbors(movie_embedding.reshape(1, -1), n_neighbors=10)
        distances, indices = zip(*sorted(zip(distances[0], indices[0])))
        distances, indices = list(distances), list(indices)
        
        for i, indx in enumerate(indices):
            if indx not in m_idx and indx != 0 and top_2 < 2:
                top_2 += 1
                m_idx.append(indx)
                m_dist.append(distances[i])
        
    m_dist, sorted_movie_indexes = zip(*sorted(zip(m_dist, m_idx)))
    m_dist, sorted_movie_indexes = list(m_dist), list(sorted_movie_indexes)
    sorted_movie_ids = [lbl_mid_mapping[m_idx] for m_idx in sorted_movie_indexes]
    
    # recommend top 10 movies
    recommend_movies = [id_title_mapping[mid] for mid in sorted_movie_ids[:10]]
    print("Recommended Movies:", recommend_movies)

    return recommend_movies
