from argminingdeeplearning.loaders import vocabulary_builder
import argparse
import gensim
import logging
import pickle
from gensim.models.wrappers.fasttext import FastText as FT_wrapper


# The usage of an embedding cache is inspired by https://github.com/UKPLab/argument-reasoning-comprehension-task/

def config_argparser():
    argparser = argparse.ArgumentParser(description='Create smaller ArgMining Deep Learning')
    argparser.add_argument('-embedding_type', type=str, required=True)
    argparser.add_argument('-embedding_path', type=str, required=True)
    argparser.add_argument('-embedding_cache_name', type=str, required=True)
    return argparser.parse_args()


def reduce_word2vec_embedding(word2vec_path, words):
    model = gensim.models.KeyedVectors.load(word2vec_path)
    word_to_embedding = {}
    coverage = 0
    for word in words:
        key = word.lower()
        if word in model.wv.vocab:
            coverage = coverage + 1
            word_to_embedding[key] = model[key]
        else:
            word_to_embedding[key] = None
    print('Word2vec cache: {}/{} words'.format(coverage, len(words)))
    return word_to_embedding


def reduce_fasttext_embedding(fasttext_path, words):
    model = FT_wrapper.load(fasttext_path)
    print(model)
    word_to_embedding = {}
    coverage = 0
    for word in words:
        key = word.lower()
        if word in model:
            coverage = coverage + 1
            word_to_embedding[key] = model[key]
        else:
            word_to_embedding[key] = None
    print('fastText cache: {}/{} words'.format(coverage, len(words)))
    return word_to_embedding


if __name__ == '__main__':
    logger = logging.getLogger()
    arguments = config_argparser()
    subtask_A_train_path = 'data/THF/sentence/subtaskA_v3_train.json'
    subtask_A_test_path = 'data/THF/sentence/subtaskA_v3_test.json'
    subtask_B_train_path = 'data/THF/sentence/subtaskB_v3_train.json'
    subtask_B_test_path = 'data/THF/sentence/subtaskB_v3_test.json'
    words_A_train = list(vocabulary_builder.get_word_frequencies(subtask_A_train_path).keys())
    words_A_test = list(vocabulary_builder.get_word_frequencies(subtask_A_test_path).keys())
    words_B_train = list(vocabulary_builder.get_word_frequencies(subtask_B_train_path).keys())
    words_B_test = list(vocabulary_builder.get_word_frequencies(subtask_B_test_path).keys())
    words = list(set().union(words_A_train, words_A_test, words_B_train, words_B_test))
    embedding_cache = None
    if arguments.embedding_type == 'word2vec':
        embedding_cache = reduce_word2vec_embedding(arguments.embedding_path, words)
    elif arguments.embedding_type == 'fasttext':
        embedding_cache = reduce_fasttext_embedding(arguments.embedding_path, words)
    else:
        raise ValueError("embedding type not supported")
    embedding_cache_path = 'data/embedding_cache/{}'.format(arguments.embedding_cache_name)
    pickle.dump(embedding_cache, open(embedding_cache_path, 'wb'))
