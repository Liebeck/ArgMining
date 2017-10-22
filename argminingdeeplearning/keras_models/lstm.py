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
    # conversion syntax from https://github.com/UKPLab/argument-reasoning-comprehension-task/blob/master/experiments/src/main/python/models.py
    # embeddings = np.asarray([np.array(x, dtype=float32) for x in index_to_embedding_mapping.keys()])

    # print('embeddings.shape:{} '.format(embeddings.shape))
    embedding_dimension = len(index_to_embedding_mapping[0])
    number_of_words = len(index_to_embedding_mapping.keys())
    print('keys.len: {}'.format(number_of_words))

    embedding_matrix = np.zeros((number_of_words, 300))
    for index, embedding in index_to_embedding_mapping.items():
        embedding_matrix[index] = embedding
#        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.


    # Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        # sequence_layer_warrant0_input)
    # EMBEDDING_DIM = 300
    #embedding_layer =
                                # trainable=False)
    model = Sequential()
    embedding_layer = Embedding(number_of_words,
                                embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=input_length)
                                # trainable=False)
    # model.add(Embedding(max_features, 128))
    # model.add(Embedding(len(index_to_embedding_mapping), 128, weights=[embeddings], input_length=input_length))
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

