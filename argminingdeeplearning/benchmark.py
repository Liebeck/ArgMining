import logging
from argminingdeeplearning.loaders import vocabulary_builder
from argminingdeeplearning.loaders.dataset_loader import load_dataset
import argparse
import logging
import time
import numpy as np
from argminingdeeplearning.keras_models import lstm
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint
import pickle
import sys
from argminingdeeplearning import utils

def benchmark(config_parameters):
    t0 = time.time()
    logger = logging.getLogger()
    logger.info("test")
    np.random.seed(14021993)
    # Step 1) Load dataset
    train_path = 'data/THF/sentence/subtask{}_v3_train.json'.format(config_parameters['subtask'])
    test_path = 'data/THF/sentence/subtask{}_v3_test.json'.format(config_parameters['subtask'])
    number_of_classes = 2 if config_parameters['subtask'] == 'A' else 3
    config_parameters['number_of_classes'] = number_of_classes
    embedding_cache = None
    if config_parameters['embeddings_cache_name']:
        embedding_cache_path = 'data/embedding_cache/{}'.format(config_parameters['embeddings_cache_name'])
        logger.info('Loading embedding cache: {}'.format(embedding_cache_path))
        embedding_cache = pickle.load(open(embedding_cache_path, "rb"))
        logger.info('Embedding cache loaded')
    logger.debug('Create mapping')
    word_to_index_mapping, index_to_embedding_mapping = vocabulary_builder.create_mappings(train_path, embedding_cache)
    logger.debug('Loading train and test set')
    X_train, Y_train, train_unique_ids, Y_train_indices = load_dataset(train_path, word_to_index_mapping,
                                                                       config_parameters['subtask'],
                                                                       config_parameters['padding_length'])

    X_test, Y_test, test_unique_ids, Y_test_indices = load_dataset(test_path, word_to_index_mapping,
                                                                   config_parameters['subtask'],
                                                                   config_parameters['padding_length'])
    # Step 2) Create model with parameters
    model_parameters = config_parameters['keras_model_parameters']
    if config_parameters['embeddings_cache_name']:

        model_parameters['index_to_embedding_mapping'] = index_to_embedding_mapping
