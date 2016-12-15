import gensim
from gensim.models import word2vec


class Word2Vec:
    def __init__(self, model_path='data/word_embeddings/word2vec_wiki-de_20161120_100'):
        self.model_path = model_path

    def load(self):
        model = gensim.models.Word2Vec.load_word2vec_format(self.model_path)

    #def
