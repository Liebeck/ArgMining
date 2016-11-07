import argparse
from argmining.sentence.loaders.THF_sentence_corpus_loader import load
from time import time


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-nfold', type=int, help='n-fold crossvalidation', default=10)
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-strategy', type=str, required=True, help='Name of the strategy')
    argparser.add_argument('-c', '--classifier', type=str, required=True, help='Name of the classifier')
    argparser.add_argument('--shuffle', type=int, help='k für Kreuzvalidierung', default=None)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time()
    arguments = config_argparser()
    # 1) Load data sets
    X_train = load(file_path='data/THF/sentence/subtask{}_train.json'.format(arguments.subtask))
    y_test = load(file_path='data/THF/sentence/subtask{}_test.json'.format(arguments.subtask))


print("Total execution time in %0.3fs" % (time() - t0))
print("*****************************************")



