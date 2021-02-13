import os

class config:
    dirname = os.path.dirname(__file__)
    USER_EMBED_PATH = os.path.join(dirname, 'user_embedding.npy')
    MOVIE_EMBED_PATH = dirname + '/movie_embedding.npy'
    USER_LABEL_MAPPING = os.path.join(dirname, 'user_label_mapping.p')
    MOVIE_LABEL_MAPPING = os.path.join(dirname, 'movie_label_mapping.p')
    LABEL_MOVIE_MAPPING = os.path.join(dirname, 'label_movie_mapping.p')
    USER_MOVIE_MAPPING = os.path.join(dirname, 'user_movie_mapping.p')
    ID_TITLE_MAPPING = os.path.join(dirname, 'id_title_mapping.p')
