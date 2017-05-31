from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
import OrderedDict


def build():
    pipeline = Pipeline([('transformer', NERFeature()),
                         ])
    return ('ner_feature', pipeline)


def count_different_ner_labels(tokens):
    ner_filter = ['PERSON', 'LOC', 'ORG']
    ner_counts = OrderedDict.fromkeys(ner_filter, 0)
    ner_inner_labels = ['I', 'L']
    token_index = 0
    while token_index < len(tokens):
        token = tokens[token_index]
        if token.ent_type_ in ner_filter:
            ner_group = [token.text.encode('utf-8')]
            inner_index = token_index + 1
            while inner_index < len(tokens) and \
                            tokens[inner_index].ent_type_ == token.ent_type_ and \
                            tokens[inner_index].ent_iob_ in ner_inner_labels:
                ner_group.append(tokens[inner_index].text.encode('utf-8'))
                print(ner_group, token.ent_type)
                inner_index += 1
                token_index += 1
            ner_counts[token.ent_type] += 1
    return ner_counts


class NERFeature(BaseEstimator):
    def __init__(self):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.count_different_ner_labels(x), X))
        return transformed
