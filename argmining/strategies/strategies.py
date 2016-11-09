import argmining.features.bag_of_words as bag_of_words

STRATEGIES = {'unigram': [bag_of_words.build(ngram=1)],
              'unigram_lowercase': [bag_of_words.build(ngram=1, lowercase=True)],
              'unigram_iwnlp': [bag_of_words.build(ngram=1, token_form='IWNLP_lemma')],
              'unigram_iwnlp_lowercase': [bag_of_words.build(ngram=1, token_form='IWNLP_lemma', lowercase=True)],
              'bigram': [bag_of_words.build(ngram=2)],
              'bigram_lowercase': [bag_of_words.build(ngram=2, lowercase=True)],
              'bigram_iwnlp': [bag_of_words.build(ngram=2, token_form='IWNLP_lemma')],
              'bigram_iwnlp_lowercase': [bag_of_words.build(ngram=2, token_form='IWNLP_lemma', lowercase=True)],
              'unigram_bigram': [bag_of_words.build(ngram=1, feature_name='unigram'),
                                 bag_of_words.build(ngram=2, feature_name='bigram')]
              }
