from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
from sklearn.preprocessing import Normalizer
from argmining.models.tiger import TIGER_TAGSET_SPACY
from collections import OrderedDict
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


def build_feature_selection(use_TIGER=True, k=5):
    pipeline = Pipeline([('transformer',
                          DependencyDistributionSpacy(use_TIGER=use_TIGER)),
                         ('feature_selection', SelectKBest(chi2, k=k)),
                         ('normalizer', Normalizer())
                         ])
    return ('dependency_distribution_spacy', pipeline)


def build(use_TIGER=True):
    pipeline = Pipeline([('transformer',
                          DependencyDistributionSpacy(use_TIGER=use_TIGER)),
                         ('normalizer', Normalizer())
                         ])
    return ('dependency_distribution_spacy', pipeline)


dependency_black_list = ['ROOT', 'punct']


def get_dependency_histogram(pos_list, tag_set):
    histogram = OrderedDict.fromkeys(tag_set, 0)
    for entry in pos_list:
        if entry and entry not in dependency_black_list and '||' not in entry:
            histogram[entry] += 1
    values = []
    for key, value in histogram.items():
        values.append(value)
    histogram = np.array(values, dtype=np.float64)
    return histogram


class DependencyDistributionSpacy(BaseEstimator):
    def __init__(self, use_TIGER=True):
        self.logger = logging.getLogger()
        self.use_TIGER = use_TIGER

    def fit(self, X, y):
        return self

    def transform(self, X):
        return list(map(lambda x: self.transform_sentence(x), X))

    def transform_sentence(self, thf_sentence):
        if self.use_TIGER:
            dependency_list = list(map(lambda x: x.releation, thf_sentence.dependencies))
            distribution = get_dependency_histogram(dependency_list, TIGER_TAGSET_SPACY)
            return distribution
        else:
            raise NotImplementedError("")
