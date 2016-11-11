from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
from sklearn.preprocessing import Normalizer
from argmining.models.tiger import TIGER_TAGSET
from argmining.models.uts import UTS_TAGSET, get_UTS_tag
from collections import OrderedDict
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


def build(use_TIGER=True, use_feature_selection=False, feature_selection_k=10):
    if use_TIGER:
        if use_feature_selection:
            pipeline = Pipeline([('transformer',
                                  DependencyDistribution(use_TIGER=True)),
                                 ('feature_selection', SelectKBest(chi2, k=feature_selection_k)),
                                 ('normalizer', Normalizer())
                                 ])
        else:
            pipeline = Pipeline([('transformer',
                                  DependencyDistribution(use_TIGER=True)),
                                 ('normalizer', Normalizer())
                                 ])
    else:
        pipeline = Pipeline([('transformer',
                              DependencyDistribution(use_TIGER=False)),
                             ('normalizer', Normalizer())
                             ])
    return ('dependency_distribution', pipeline)


def get_dependency_histogram(pos_list, tag_set):
    histogram = OrderedDict.fromkeys(tag_set, 0)
    for entry in pos_list:
        # print(entry)
        histogram[entry] += 1
    values = []
    for key, value in histogram.items():
        values.append(value)
    histogram = np.array(values, dtype=np.float64)
    # print(histogram)
    return histogram

class DependencyDistribution(BaseEstimator):
    def __init__(self, use_TIGER):
        self.logger = logging.getLogger()
        self.use_TIGER = use_TIGER

    def fit(self, X, y):
        return self

    def transform(self, X):
        self.logger.debug("transform called")
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        # transformed = np.concatenate(transformed, axis=0)
        self.logger.debug("transform returning")
        return transformed

    def transform_sentence(self, thf_sentence):
        # if self.use_STTS:
        dependency_list = list(map(lambda x: x.releation, thf_sentence.dependencies))
        distribution = get_dependency_histogram(dependency_list, TIGER_TAGSET)
        # else:
        # pos_list = list(map(lambda x: get_UTS_tag(x.pos_tag), thf_sentence.tokens))
        # distribution = get_STTS_histogram(pos_list, UTS_TAGSET)
        return distribution
