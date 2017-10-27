from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM, Bidirectional
from theano.scalar import float32
import numpy as np


def lstm_embedding_empty(number_of_classes, max_features=7000, embedding_size=300, lstm_size=128, dropout=0.2):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size))
    model.add(LSTM(lstm_size, dropout=dropout, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def lstm_embedding_pretrained(number_of_classes, index_to_embedding_mapping, padding_length,
                              lstm_size=128, dropout=0.2, recurrent_dropout=0.2):
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
    model.add(LSTM(lstm_size, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def lst_stacked(number_of_classes, index_to_embedding_mapping, padding_length, lstm_size_layer1=128,
                lstm_size_layer2=64, dropout=0.5, recurrent_dropout=0.4):
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
    model.add(LSTM(lstm_size_layer1, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
    model.add(LSTM(lstm_size_layer2, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def blstm(number_of_classes, index_to_embedding_mapping, padding_length, lstm_size_layer1=128,
          lstm_size_layer2=64, dropout=0.5):
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
    model.add(Bidirectional(LSTM(lstm_size_layer1, dropout=dropout, return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_size_layer2, dropout=dropout)))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
