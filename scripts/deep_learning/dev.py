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
    from keras.layers import Dense, Embedding, Activation
    from keras.layers import LSTM
    # from keras.layers.core import Activation

    max_features = 20000
    number_of_classes = 2 if arguments.subtask == 'A' else 3
    batch_size = 32
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                # loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=5,
              verbose=1,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)

    y_prediction = model.predict(X_test, batch_size=batch_size)
    predicted_classes = np.argmax(y_prediction, axis=1)
    print(predicted_classes)
    print()
    print('Test score:', score)
    print('Test accuracy:', acc)
    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
