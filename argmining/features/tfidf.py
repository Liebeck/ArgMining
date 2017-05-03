from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from argmining.features.bag_of_words import tokenizer_THF_words_lowercase


def build_tfidf(feature_name='tfidf', ngram=1, min_df=5):
    pipeline = Pipeline(
        [('tfidf', TfidfVectorizer(min_df=min_df, lowercase=False, tokenizer=tokenizer_THF_words_lowercase,
                                   ngram_range=(ngram, ngram)))
         ])
    return (feature_name, pipeline)
