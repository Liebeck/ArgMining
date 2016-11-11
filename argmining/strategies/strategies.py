import argmining.features.bag_of_words as bag_of_words
import argmining.features.pos_distribution as pos_distribution
import argmining.features.dependency_distribution as dependency_distribution

STRATEGIES = {'unigram': [bag_of_words.build(ngram=1)],
              'unigram_lowercase': [bag_of_words.build(ngram=1, lowercase=True)],
              'unigram_iwnlp': [bag_of_words.build(ngram=1, token_form='IWNLP_lemma')],
              'unigram_iwnlp_lowercase': [bag_of_words.build(ngram=1, token_form='IWNLP_lemma', lowercase=True)],
              'n_unigram': [bag_of_words.build(ngram=1, normalize=True)],
              'n_unigram_lowercase': [bag_of_words.build(ngram=1, lowercase=True, normalize=True)],
              'n_unigram_iwnlp': [bag_of_words.build(ngram=1, token_form='IWNLP_lemma', normalize=True)],
              'n_unigram_iwnlp_lowercase': [
                  bag_of_words.build(ngram=1, token_form='IWNLP_lemma', lowercase=True, normalize=True)],
              'bigram': [bag_of_words.build(ngram=2)],
              'bigram_lowercase': [bag_of_words.build(ngram=2, lowercase=True)],
              'bigram_iwnlp': [bag_of_words.build(ngram=2, token_form='IWNLP_lemma')],
              'bigram_iwnlp_lowercase': [bag_of_words.build(ngram=2, token_form='IWNLP_lemma', lowercase=True)],
              'n_bigram': [bag_of_words.build(ngram=2, normalize=True)],
              'n_bigram_lowercase': [bag_of_words.build(ngram=2, lowercase=True, normalize=True)],
              'n_bigram_iwnlp': [bag_of_words.build(ngram=2, token_form='IWNLP_lemma', normalize=True)],
              'n_bigram_iwnlp_lowercase': [
                  bag_of_words.build(ngram=2, token_form='IWNLP_lemma', lowercase=True, normalize=True)],
              'unigram_bigram': [bag_of_words.build(ngram=1, feature_name='unigram'),
                                 bag_of_words.build(ngram=2, feature_name='bigram')],
              'pos_distribution': [pos_distribution.build()],
              'pos_distribution_fs35': [pos_distribution.build(use_feature_selection=True, feature_selection_k=35)],
              'pos_distribution_uts': [pos_distribution.build(use_STTS=False)],
              'n_unigram+pos_distribution': [bag_of_words.build(ngram=1, normalize=True), pos_distribution.build()],
              'dependency_distribution': [dependency_distribution.build()],
              'dependency_distribution_fs10': [
                  dependency_distribution.build(use_feature_selection=True, feature_selection_k=10)],
              'pos+dep_distribution': [pos_distribution.build(), dependency_distribution.build()],
              'pos+dep_distribution_fs': [pos_distribution.build(use_feature_selection=True, feature_selection_k=35),
                                          dependency_distribution.build(use_feature_selection=True,
                                                                        feature_selection_k=10)]
              }
