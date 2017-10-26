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


def config_logger(log_level=logging.INFO):
    logger = logging.getLogger('')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(log_level)


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining Deep Learning')
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-padding_length', type=int, help='Padding length of each input sequence', default=20)
    argparser.add_argument('-batch_size', type=int, default=32)
    argparser.add_argument('-epochs', type=int, default=5)
    argparser.add_argument('-embedding_cache_name', type=str, default=None)
    # argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    # argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    config_logger(log_level=logging.INFO)
    logger = logging.getLogger()
    np.random.seed(14021993)
    arguments = config_argparser()
    # Step 1) Load dataset
    train_path = 'data/THF/sentence/subtask{}_v3_train.json'.format(arguments.subtask)
    test_path = 'data/THF/sentence/subtask{}_v3_test.json'.format(arguments.subtask)
    number_of_classes = 2 if arguments.subtask == 'A' else 3
    embedding_cache = None
    if arguments.embedding_cache_name:
        embedding_cache_path = 'data/embedding_cache/{}'.format(arguments.embedding_cache_name)
        logger.info('Loading embedding cache: {}'.format(embedding_cache_path))
        embedding_cache = pickle.load(open(embedding_cache_path, "rb"))
        logger.info('Embedding cache loaded')
    logger.debug('Create mapping')
    word_to_index_mapping, index_to_embedding_mapping = vocabulary_builder.create_mappings(train_path, embedding_cache)
    logger.debug('Loading train and test set')
    X_train, Y_train, train_unique_ids, Y_train_indices = load_dataset(train_path, word_to_index_mapping,
                                                                       arguments.subtask,
                                                                       arguments.padding_length)

    X_test, Y_test, test_unique_ids, Y_test_indices = load_dataset(test_path, word_to_index_mapping, arguments.subtask,
                                                                   arguments.padding_length)
    # logger.info(X_train)
    # logger.info(X_train.shape)
    # logger.info(Y_train)
    # logger.info(Y_train.shape)

    # Step 2) Create model
    # model = lstm.lstm_embedding_empty(number_of_classes)
    # model = lstm.lstm_embedding_pretrained(number_of_classes, index_to_embedding_mapping,
    # input_length=arguments.padding_length)
    # model = lstm.lstm_embedding_pretrained_test(number_of_classes, index_to_embedding_mapping,
    # input_length=arguments.padding_length)
    model = lstm.blstm(number_of_classes, index_to_embedding_mapping, input_length=arguments.padding_length)

    # Step 3) Train the model
    logger.info('Train...')
    current_time = time.strftime('%Y%m%d_%H%M%S')
    model_save_path = 'results/sentence_deeplearning/temp/{}_{}'.format(arguments.subtask, current_time)
    checkpoint_save_path = model_save_path + "_best.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_save_path,
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    model.fit(X_train, Y_train,
              batch_size=arguments.batch_size,
              epochs=arguments.epochs,
              verbose=1,
              callbacks=[checkpoint],
              validation_data=(X_test, Y_test))

    # Step 4) Save the last model
    model.save(model_save_path + "_last.hdf5")
    # Step 5) Predict the test set
    score, acc = model.evaluate(X_test, Y_test, batch_size=arguments.batch_size)
    y_prediction = model.predict(X_test, batch_size=arguments.batch_size)
    y_prediction_classes = np.argmax(y_prediction, axis=1)
    # Step 6) Print results
    logger.info(y_prediction_classes)
    logger.info('Test score: {}'.format(score))
    logger.info('Test accuracy: {}'.format(acc))
    f1 = f1_score(Y_test_indices, y_prediction_classes, average=None)
    f1_mean = np.mean(f1)
    logger.info("Macro-averaged F1: {}".format(f1_mean))
    logger.info("Individual scores: {}".format(f1))
    logger.info("Confusion matrix:")
    logger.info(ConfusionMatrix(Y_test_indices, y_prediction_classes))

    output_path_base = 'results/sentence_deeplearning/temp/{}_{}'.format(arguments.subtask,
                                                                         current_time)

    # Step 7) Print results to the file system
    utils.write_prediction_file(path=output_path_base + '.predictions', test_unique_ids=test_unique_ids,
                                Y_test_indices=Y_test_indices, y_prediction_classes=y_prediction_classes)
    utils.write_score_file(score_file=output_path_base + '.score', f1_mean=f1_mean, f1=f1,
                           Y_test_indices=Y_test_indices, y_prediction_classes=y_prediction_classes)

    print("Total execution time in %0.3fs" % (time.time() - t0))
    print("*****************************************")
