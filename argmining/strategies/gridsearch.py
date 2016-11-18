from argmining.features.bag_of_words import build
from argmining.features.pos_distribution import POSDistribution
import argmining.features.pos_distribution as pos_distribution
import argmining.features.dependency_distribution as dependency_distribution
import argmining.features.structural_features as structural_features
from collections import OrderedDict

GRIDSEARCH_STRATEGIES = {
    'unigram':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', build())
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [1],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
            }
        },
    'bigram':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', build())
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [2],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
            }
        },
    # 'pos_distribution_feature_selection':
    #     {
    #         'features':
    #             OrderedDict([
    #                 ('pos_distribution', POSDistribution(use_feature_selection=True))
    #             ]),
    #         'param_grid': {
    #             # 'union__pos_distribution__transformer__feature_selection_k': [5, 10, 15, 20, 25],
    #             'union__pos_distribution__feature_selection__feature_selection_k': [1, 2, 3],
    #             'union__pos_distribution__transformer__use_feature_selection': [True],
    #         }
    #     }
}
