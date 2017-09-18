import argmining.features.bag_of_words as bag_of_words
# import argmining.features.pos_distribution as pos_distribution
import argmining.features.dependency_distribution as dependency_distribution
import argmining.features.character_ngrams as character_ngrams
import argmining.features.dependency_distribution_spacy as dependency_distribution_spacy
import argmining.features.structural_features as structural_features
import argmining.features.sentiws_polarity_distribution as sentiws_polarity_distribution
from collections import OrderedDict
import argmining.features.embedding_centroid as embedding_centroid
import argmining.features.lda_distribution as lda_distribution
import argmining.features.pos_distribution_spacy as pos_distribution_spacy

GRIDSEARCH_STRATEGIES = {
    'unigram':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build)
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
            }
        },
    'unigram_frequency':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build)
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text'],
                'union__bag_of_words__transformer__min_df': [1, 2, 3, 4, 5, 10, 20],
                'union__bag_of_words__transformer__max_features': [500, 1000, 1500, 2000, None],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
            }
        },
    'bigram':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build)
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(2, 2)],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
            }
        },
    'character_ngrams':
        {
            'features':
                OrderedDict([
                    ('character_ngrams', character_ngrams.build)
                ]),
            'param_grid': {
                'union__character_ngram__transformer__min_df': [1, 5, 10, 20, 30, 40],
                'union__character_ngram__transformer__ngram_range': [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
                                                                     (7, 7), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                                                                     (1, 7)]
            }
        },
    'pos_distribution_spacy':
        {
            'features':
                OrderedDict([
                    ('pos_distribution_spacy', pos_distribution_spacy.build)
                ]),
            'param_grid': {
                'union__pos_distribution_spacy__transformer__coarse_grained': [True, False],
            }
        },
    'pos_distribution_feature_selection':
        {
            'features':
                OrderedDict([
                    ('pos_distribution', pos_distribution_spacy.build_feature_selection)
                ]),
            'param_grid': [
                {
                    'union__pos_distribution_spacy__transformer__coarse_grained': [True],
                    'union__pos_distribution_spacy__feature_selection__k': [5, 10, 15, 20, 30, 'all'],
                },
                {
                    'union__pos_distribution_spacy__transformer__coarse_grained': [False],
                    'union__pos_distribution_spacy__feature_selection__k': [6, 7, 8, 9, 10, 11, 'all'],
                }

            ]

        },
    'unigram+grammatical':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build),
                    ('pos_distribution_spacy', pos_distribution_spacy.build),
                    ('dependency_distribution', dependency_distribution.build),
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
                'union__pos_distribution_spacy__transformer__coarse_grained': [True, False],
            }
        },
    'n_unigram+shape':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build)
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__normalizer__use_normalize': [True],
                'union__bag_of_words__transformer__token_form': ['shape'],
            }
        },
    'dependency_distribution_spacy':
        {
            'features':
                OrderedDict([
                    ('dependency_distribution_spacy', dependency_distribution_spacy.build),
                ]),
            'param_grid': {}
        },
    'unigram+grammatical_spacy':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build),
                    ('pos_distribution_spacy', pos_distribution_spacy.build),
                    ('dependency_distribution_spacy', dependency_distribution_spacy.build),
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True, False],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
                'union__pos_distribution_spacy__transformer__coarse_grained': [True, False],
            }
        },
    'sentiws_distribution':
        {
            'features':
                OrderedDict([
                    ('polarity_sentiws_distribution', sentiws_polarity_distribution.build),
                ]),
            'param_grid': {
                'union__polarity_sentiws_distribution__transformer__bins': [5, 10, 'auto'],
                'union__polarity_sentiws_distribution__transformer__density': [None, True, False],
            }
        },
    'embedding_centroid_100':
        {
            'features':
                OrderedDict([
                    ('embedding_centroid', embedding_centroid.build)
                ]),
            'param_grid': {
                'union__embedding_centroid__transformer__embedding_length': [100],
            }
        },
    'embedding_centroid_200':
        {
            'features':
                OrderedDict([
                    ('embedding_centroid', embedding_centroid.build)
                ]),
            'param_grid': {
                'union__embedding_centroid__transformer__embedding_length': [200],
            }
        },
    'embedding_centroid_300':
        {
            'features':
                OrderedDict([
                    ('embedding_centroid', embedding_centroid.build)
                ]),
            'param_grid': {
                'union__embedding_centroid__transformer__embedding_length': [300],
            }
        },
    'unigram+embedding_centroid_100':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build),
                    ('embedding_centroid', embedding_centroid.build),
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
                'union__embedding_centroid__transformer__embedding_length': [100],
            }
        },
    'unigram+embedding_centroid_200':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build),
                    ('embedding_centroid', embedding_centroid.build),
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
                'union__embedding_centroid__transformer__embedding_length': [200],
            }
        },
    'unigram+embedding_centroid_300':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build),
                    ('embedding_centroid', embedding_centroid.build),
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
                'union__embedding_centroid__transformer__embedding_length': [300],
            }
        },
    'lda_distribution':
        {
            'features':
                OrderedDict([
                    ('lda_distribution', lda_distribution.build)
                ]),
            'param_grid': {}
        },
    'unigram+lda_distribution':
        {
            'features':
                OrderedDict([
                    ('bag_of_words', bag_of_words.build),
                    ('lda_distribution', lda_distribution.build)
                ]),
            'param_grid': {
                'union__bag_of_words__transformer__ngram': [(1, 1)],
                'union__bag_of_words__transformer__lowercase': [True],
                'union__bag_of_words__transformer__token_form': ['text', 'IWNLP_lemma'],
                'union__bag_of_words__normalizer__use_normalize': [True, False],
            }
        },
}
