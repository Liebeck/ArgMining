from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM


def lstm_embedding_empty(number_of_classes, max_features=20000):
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
    return model
