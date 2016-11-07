from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


def build(ngram=1, feature_name='bag_of_words'):
    pipeline = Pipeline([('transformer',
                          BagOfWords(ngram=ngram)),
                         ])
    return (feature_name, pipeline)


def tokenizer_THF_words(thf_sentence):
    print("tokenizer_THF_words called")
    print(thf_sentence.tokens[0].text)
    return list(map(lambda token: token.text, thf_sentence))


def get_vocabulary(X):
    print("get_vocabulary called")
    vectorizer = CountVectorizer(tokenizer=tokenizer_THF_words)
    vectorizer.fit(X)
    return vectorizer.get_feature_names()


class BagOfWords(BaseEstimator):
    def __init__(self, ngram=1, token_form='text'):
        self.ngram = ngram
        self.token_form = token_form

    def fit(self, X, y):
        if self.token_form == 'text':
            self.vocabulary = get_vocabulary(X)
            return self
        else:
            raise ValueError("token_form not implemented")

    def transform(self, X):
        return list(map(lambda x: self.transform_sentence(x), X))

    def transform_sentence(self, x):
        return list(map(lambda token: token.text, x.tokens))
