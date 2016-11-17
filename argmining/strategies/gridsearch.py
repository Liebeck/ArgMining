from argmining.features.bag_of_words import BagOfWords
from argmining.features.pos_distribution import POSDistribution
import argmining.features.pos_distribution as pos_distribution
import argmining.features.dependency_distribution as dependency_distribution
import argmining.features.structural_features as structural_features
from collections import OrderedDict

GRIDSEARCH_STRATEGIES = {
    'bag_of_words':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', BagOfWords())
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [1, 2],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
            }
        },
    'pos_distribution_feature_selection':
        {
            'features':
                OrderedDict([
                    ('pos_distribution', POSDistribution())
                ]),
            'param_grid': {
                'union__pos_distribution__transformer__feature_selection_k': [10, 15],
                'union__pos_distribution__transformer__use_feature_selection': [True, False],
            }
        }
}
