import argmining.features.bag_of_words as bag_of_words
import argmining.features.pos_distribution as pos_distribution
import argmining.features.dependency_distribution as dependency_distribution
import argmining.features.structural_features as structural_features

GRIDSEARCH_STRATEGIES = {'bag_of_words':
                             {'features': [bag_of_words.build()],
                              'param_grid': {
                                  'union__bag_of_words__transformer__ngram': [1, 2],
                                  'union__bag_of_words__transformer__lowercase': [True, False],
                                  'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                              }
                              }
                         }
