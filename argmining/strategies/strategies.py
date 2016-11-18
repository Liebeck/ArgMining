from argmining.features.bag_of_words import BagOfWords
from argmining.features.pos_distribution import POSDistribution
from argmining.features.dependency_distribution import DependencyDistribution
from argmining.features.structural_features import StructuralFeatures

STRATEGIES = {'unigram': [BagOfWords(ngram=1).build()],
              'unigram_lowercase': [BagOfWords(ngram=1, lowercase=True).build()],
              'unigram_iwnlp': [BagOfWords(ngram=1, token_form='IWNLP_lemma').build()],
              'unigram_iwnlp_lowercase': [BagOfWords(ngram=1, token_form='IWNLP_lemma', lowercase=True).build()],
              'n_unigram': [BagOfWords(ngram=1, normalize=True).build()],
              'n_unigram_lowercase': [BagOfWords(ngram=1, lowercase=True, normalize=True).build()],
              'n_unigram_iwnlp': [BagOfWords(ngram=1, token_form='IWNLP_lemma', normalize=True).build()],
              'n_unigram_iwnlp_lowercase': [
                  BagOfWords(ngram=1, token_form='IWNLP_lemma', lowercase=True, normalize=True).build()],
              'bigram': [BagOfWords(ngram=2).build()],
              'bigram_lowercase': [BagOfWords(ngram=2, lowercase=True).build()],
              'bigram_iwnlp': [BagOfWords(ngram=2, token_form='IWNLP_lemma').build()],
              'bigram_iwnlp_lowercase': [BagOfWords(ngram=2, token_form='IWNLP_lemma', lowercase=True).build()],
              'n_bigram': [BagOfWords(ngram=2, normalize=True).build()],
              'n_bigram_lowercase': [BagOfWords(ngram=2, lowercase=True, normalize=True).build()],
              'n_bigram_iwnlp': [BagOfWords(ngram=2, token_form='IWNLP_lemma', normalize=True).build()],
              'n_bigram_iwnlp_lowercase': [
                  BagOfWords(ngram=2, token_form='IWNLP_lemma', lowercase=True, normalize=True).build()],
              'unigram_bigram': [BagOfWords(ngram=1, feature_name='unigram').build(),
                                 BagOfWords(ngram=2, feature_name='bigram').build()],
              'pos_distribution': [POSDistribution().build()],
              'pos_distribution_fs35': [POSDistribution(use_feature_selection=True, feature_selection_k=35).build()],
              'pos_distribution_uts': [POSDistribution(use_STTS=False).build()],
              'n_unigram+pos_distribution': [BagOfWords(ngram=1, normalize=True).build(), POSDistribution().build()],
              'dependency_distribution': [DependencyDistribution().build()],
              'dependency_distribution_fs10': [
                  DependencyDistribution(use_feature_selection=True, feature_selection_k=10).build()],
              'pos+dep_distribution': [POSDistribution().build(), StructuralFeatures().build()],
              'pos+dep_distribution_fs': [POSDistribution(use_feature_selection=True, feature_selection_k=35).build(),
                                          DependencyDistribution(use_feature_selection=True,
                                                                 feature_selection_k=10).build()],
              'structural': [StructuralFeatures().build()],
              'structural_without_token_length': [StructuralFeatures(use_sentence_length=False).build()]
              }
