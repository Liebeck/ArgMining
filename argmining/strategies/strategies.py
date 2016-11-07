import argmining.features.bag_of_words as bag_of_words

STRATEGIES = {'unigram': [bag_of_words.build(ngram=1)],
              'bigram': [bag_of_words.build(ngram=2)],
              'unigram_bigram': [bag_of_words.build(ngram=1),
                                 bag_of_words.build(ngram=2)]
              }
