from argminingdeeplearning.keras_models import lstm
from argminingdeeplearning.keras_models import cnn

model_map = {'lstm-embedding-empty': lstm.lstm_embedding_empty,
             'lstm-embedding-pretrained': lstm.lstm_embedding_pretrained,
             'lstm-stacked': lstm.lst_stacked,
             'blstm': lstm.blstm,
             'embedding_cnn': cnn.embedding_cnn
             }


def get_model(name, number_of_classes, parameters):
    return model_map[name](number_of_classes, **parameters)
