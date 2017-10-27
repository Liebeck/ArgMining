from argminingdeeplearning.loaders import vocabulary_builder
from argminingdeeplearning.loaders.dataset_loader import load_dataset
import logging
import time
import numpy as np
from argminingdeeplearning.keras_models import lstm
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint
import pickle
from argminingdeeplearning import utils
from argminingdeeplearning.keras_models import model_selector
from keras.models import load_model


def benchmark(subtask, config_parameters):
    t0 = time.time()
    logger = logging.getLogger()
    np.random.seed(14021993)
    # Step 1) Load dataset
    train_path = 'data/THF/sentence/subtask{}_v3_train.json'.format(subtask)
    test_path = 'data/THF/sentence/subtask{}_v3_test.json'.format(subtask)
    number_of_classes = 2 if subtask == 'A' else 3
    # config_parameters['number_of_classes'] = number_of_classes
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
                                                                       subtask,
                                                                       config_parameters['padding_length'])

    X_test, Y_test, test_unique_ids, Y_test_indices = load_dataset(test_path, word_to_index_mapping,
                                                                   subtask,
                                                                   config_parameters['padding_length'])
    # Step 2) Create model with parameters
    model_parameters = config_parameters['keras_model_parameters']
    logger.info(config_parameters)
    if config_parameters['embeddings_cache_name']:
        model_parameters['index_to_embedding_mapping'] = index_to_embedding_mapping
    model = model_selector.get_model(config_parameters['keras_model_name'], number_of_classes, model_parameters)

    # Step 3) Train the model
    logger.info('Train...')
    current_time = time.strftime('%Y%m%d_%H%M%S')
    model_save_path = 'results/sentence_deeplearning/temp/{}_{}_{}_{}'.format(subtask,
                                                                              config_parameters['keras_model_name'],
                                                                              '{:03}'.format(config_parameters['evaluation_ID']),
                                                                              current_time)
    checkpoint_save_path = model_save_path + "_best.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_save_path,
                                 monitor='val_acc', verbose=0,
                                 save_best_only=True, mode='auto')
    model.fit(X_train, Y_train,
              batch_size=config_parameters['batch_size'],
              epochs=config_parameters['epochs'],
              verbose=1,
              callbacks=[checkpoint],
              validation_data=(X_test, Y_test))

    # Step 4) Save the last model
    model.save(model_save_path + "_last.hdf5")
    logger.info(model.summary())
    # Calculate results for the best and the last model
    saved_models = [{'name': 'best', 'extension': '_best.hdf5'}, {'name': 'last', 'extension': '_last.hdf5'}]
    for saved_model in saved_models:
        model_load_path = model_save_path + saved_model['extension']
        logger.info('Loading model for prediction: {}'.format(model_load_path))
        model = load_model(model_load_path)

        # Step 5) Predict the test set
        score, acc = model.evaluate(X_test, Y_test, batch_size=config_parameters['batch_size'])
        y_prediction = model.predict(X_test, batch_size=config_parameters['batch_size'])
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

        output_path_base = 'results/sentence_deeplearning/temp/{}_{}_{}_{}_{}'.format(subtask,
                                                                                      config_parameters['keras_model_name'],
                                                                                      '{:03}'.format(config_parameters['evaluation_ID']),
                                                                                      current_time,
                                                                                      saved_model['name'])

        # Step 7) Print results to the file system
        utils.write_prediction_file(path=output_path_base + '.predictions', test_unique_ids=test_unique_ids,
                                    Y_test_indices=Y_test_indices, y_prediction_classes=y_prediction_classes)
        utils.write_score_file(score_file=output_path_base + '.score', f1_mean=f1_mean, f1=f1, model=model,
                               Y_test_indices=Y_test_indices, y_prediction_classes=y_prediction_classes)

    print("Total execution time in %0.3fs" % (time.time() - t0))
    print("*****************************************")
