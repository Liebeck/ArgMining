from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM, Bidirectional
from theano.scalar import float32
import numpy as np

def lstm_embedding_empty(number_of_classes, max_features=7000, lstm_size=128, dropout=0.2):
    print('lstm_embedding_empty called')
    model = Sequential()
    model.add(Embedding(max_features, lstm_size))
    model.add(LSTM(lstm_size, dropout=dropout, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def lstm_embedding_pretrained(number_of_classes, index_to_embedding_mapping, padding_length):
    embedding_dimension = len(index_to_embedding_mapping[0])
    number_of_words = len(index_to_embedding_mapping.keys())
    print('keys.len: {}'.format(number_of_words))
    embedding_matrix = np.zeros((number_of_words, embedding_dimension))
    for index, embedding in index_to_embedding_mapping.items():
        embedding_matrix[index] = embedding
    model = Sequential()
    embedding_layer = Embedding(number_of_words,
                                embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=padding_length)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def lstm_embedding_pretrained_test(number_of_classes, index_to_embedding_mapping, padding_length):
    embedding_dimension = len(index_to_embedding_mapping[0])
    number_of_words = len(index_to_embedding_mapping.keys())
    print('keys.len: {}'.format(number_of_words))
    embedding_matrix = np.zeros((number_of_words, embedding_dimension))
    for index, embedding in index_to_embedding_mapping.items():
        embedding_matrix[index] = embedding
    model = Sequential()
    embedding_layer = Embedding(number_of_words,
                                embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=padding_length)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.4))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def lst_stacked(number_of_classes, index_to_embedding_mapping, padding_length):
    embedding_dimension = len(index_to_embedding_mapping[0])
    number_of_words = len(index_to_embedding_mapping.keys())
    embedding_matrix = np.zeros((number_of_words, embedding_dimension))
    for index, embedding in index_to_embedding_mapping.items():
        embedding_matrix[index] = embedding
    model = Sequential()
    embedding_layer = Embedding(number_of_words,
                                embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=padding_length)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.4, return_sequences=True))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.4))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def blstm(number_of_classes, index_to_embedding_mapping, padding_length):
    embedding_dimension = len(index_to_embedding_mapping[0])
    number_of_words = len(index_to_embedding_mapping.keys())
    embedding_matrix = np.zeros((number_of_words, embedding_dimension))
    for index, embedding in index_to_embedding_mapping.items():
        embedding_matrix[index] = embedding
    model = Sequential()
    embedding_layer = Embedding(number_of_words,
                                embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=padding_length)
    model.add(embedding_layer)
    # model.add(Bidirectional(LSTM(128, dropout=0.5, return_sequences=True)))
    # model.add(Bidirectional(LSTM(64, dropout=0.5)))
    # model.add(Bidirectional(LSTM(128, dropout=0.8, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.85, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.85)))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model