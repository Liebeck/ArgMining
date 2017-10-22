import argparse
from argmining.sentence.loaders.THF_sentence_corpus_loader import load_dataset
import time
import logging
import json
from argmining.pipelines.pipeline import pipeline
from argmining.strategies.gridsearch import GRIDSEARCH_STRATEGIES
from argmining.classifiers.classifier import create_classifier
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import f1_score
import numpy as np
from argmining.evaluation.reduce_training_set import reduce_training_set
from argmining.evaluation.shuffle import shuffle_training_Set
from argmining.representations.word2vec import Word2Vec
from argmining.representations.lda import LDA
from argmining.representations.fasttext import FastText

NJOBS = 1


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-configfile', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Read settings file
    logger.info("Loading config file: {}".format(arguments.configfile))
    with open(arguments.configfile) as data_file:
        settings = json.load(data_file)
    # 2) Read datasets
    if 'data_version' in settings:  # backwards compability for old setting files
        if settings['data_version'] == 'v1':
            train_path = 'data/THF/sentence/subtask{}_train.json'.format(settings['subtask'])
            test_path = 'data/THF/sentence/subtask{}_test.json'.format(settings['subtask'])
        else:
            train_path = 'data/THF/sentence/subtask{}_{}_train.json'.format(settings['subtask'],
                                                                            settings['data_version'])
            test_path = 'data/THF/sentence/subtask{}_{}_test.json'.format(settings['subtask'], settings['data_version'])
    else:
        train_path = 'data/THF/sentence/subtask{}_train.json'.format(settings['subtask'])
        test_path = 'data/THF/sentence/subtask{}_test.json'.format(settings['subtask'])
    if arguments.hilbert:  # work around for absolute paths on the hilbert cluster
        train_path = '/scratch_gs/malie102/jobs/ArgMining/' + train_path
        test_path = '/scratch_gs/malie102/jobs/ArgMining/' + test_path

    group_claims = True
    if settings['subtask'] == "C":
        group_claims = False
    X_train, y_train = load_dataset(file_path=train_path, data_version=settings['data_version'],
                                    group_claims=group_claims)
    X_test, y_test = load_dataset(file_path=test_path, data_version=settings['data_version'], group_claims=group_claims)

    if settings['embeddings_path']:
        word2vec = Word2Vec(model_path=settings['embeddings_path'])
        word2vec.load()
        word2vec.annotate_sentences(X_train)
        word2vec.annotate_sentences(X_test)
    if settings['lda_path']:
        lda = LDA(model_path=settings['lda_path'], vocab_path=settings['lda_vocab_path'],
                  nouns_only=settings['lda_all_words'])
        lda.load()
        lda.annotate_sentences(X_train)
        lda.annotate_sentences(X_test)
    if settings['fasttext_path']:
        fasttext = FastText(model_path=settings['fasttext_path'])
        fasttext.load()
        fasttext.annotate_sentences(X_train)
        fasttext.annotate_sentences(X_test)
    # 4) Shuffle if desired
    X_train, y_train = shuffle_training_Set(X_train, y_train, settings['shuffle'])
    # 5) Reduce training size
    X_train, y_train = reduce_training_set(X_train, y_train, settings['training_size'])
    # 6) Load classifier with arguments
    classifer_parameters = {}
    for key, value in settings['gridsearch_parameters'].items():
        if key.startswith('classifier__'):
            classifer_parameters[key.replace('classifier__', '')] = value
    logger.info('Classifier arguments: {}'.format(classifer_parameters))
    logger.info('Creating classifier {}'.format(settings['classifier']))
    classifier = create_classifier(settings['classifier'], classifer_parameters)
    logger.info(classifier)
    # 7) Load features and set arguments
    logger.info("Using gridsearch strategy: {}".format(settings['gridsearchstrategy']))
    strategy = GRIDSEARCH_STRATEGIES[settings['gridsearchstrategy']]['features']
    strategy_built = []
    for feature_name, feature in strategy.items():
        feature_parameters = {}
        for key, value in settings['gridsearch_parameters'].items():
            if key.startswith('union__{}__transformer__'.format(feature_name)):
                feature_parameters[key.replace('union__{}__transformer__'.format(feature_name), '')] = value
        if not feature_parameters:
            logger.info(
                'Building feature {} without parameters'.format(feature_name))
            strategy_built.append(feature())
        else:
            logger.info(
                'Building feature {} with the following parameters: {}'.format(feature_name, feature_parameters))
            strategy_built.append(feature(**feature_parameters))
        logger.debug(strategy_built[-1])
    # 8) Train classifier
    pipe = pipeline(strategy=strategy_built, classifier=classifier)
    pipe.fit(X_train, y_train)
    # 9) Predict the test set
    y_prediction = pipe.predict(X_test)
    # 10) Print score and the confusion matrix
    f1 = f1_score(y_test, y_prediction, average=None)
    f1_mean = np.mean(f1)
    logger.info("Macro-averaged F1: {}".format(f1_mean))
    logger.info("Individual scores: {}".format(f1))
    logger.info("Confusion matrix:")
    logger.info(ConfusionMatrix(y_test, y_prediction))
    # 11) Save the predictions into the file system
    prediction_file = '{}.predictions'.format(arguments.configfile)
    with open(prediction_file, 'w') as prediction_handler:
        prediction_handler.write('{}\t{}\t{}\n'.format("UniqueID", "Gold_Label", "Prediction"))
        for index, val in enumerate(y_test):
            prediction_handler.write('{}\t{}\t{}\n'.format(X_test[index].uniqueID, y_test[index], y_prediction[index]))
    # 12) Save the score and the confusion matrix into the file system
    score_file = '{}.score'.format(arguments.configfile)
    with open(score_file, 'w') as score_handler:
        score_handler.write("Micro-averaged F1: {}\n".format(f1_mean))
        score_handler.write("Individual scores: {}\n".format(f1))
        score_handler.write("Confusion matrix:\n")
        score_handler.write(str(ConfusionMatrix(y_test, y_prediction)))
    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
