from argminingdeeplearning.loaders import vocabulary_builder
from argminingdeeplearning.loaders.dataset_loader import load_dataset
import argparse
import logging
import time
import numpy as np


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining Deep Learning')
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-kerasmodel', type=str, default='lstm')
    argparser.add_argument('-padding_length', type=int, help='Padding length of each input sequence', default=20)
    # argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    # argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    logger = logging.getLogger()
    arguments = config_argparser()
    train_path = 'data/THF/sentence/subtask{}_v3_train.json'.format(arguments.subtask)
    test_path = 'data/THF/sentence/subtask{}_v3_test.json'.format(arguments.subtask)
    word_to_index_mapping, index_to_embedding_maping = vocabulary_builder.create_mappings(train_path)
    X_train, Y_train, train_unique_ids = load_dataset(train_path, word_to_index_mapping, arguments.subtask,
                                                      arguments.padding_length)
    X_test, Y_test, test_unique_ids = load_dataset(test_path, word_to_index_mapping, arguments.subtask,
                                                   arguments.padding_length)
    print(X_train)
    print(X_train.shape)

    from keras.models import Sequential
    from keras.layers import Dense, Embedding
    from keras.layers import LSTM

    max_features = 20000
    batch_size = 32
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=15,
              verbose=1,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    # print(y)
    # get list of list of indizes, apply padding
    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
