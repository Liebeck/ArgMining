from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM
from theano.scalar import float32
import numpy as np

def lstm_embedding_empty(number_of_classes, max_features=7000):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def lstm_embedding_pretrained(number_of_classes, index_to_embedding_mapping, input_length):
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
                                input_length=input_length)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

